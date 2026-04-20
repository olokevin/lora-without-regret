# Math-verify post-training eval + merged checkpoints — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--enable-merged-ckpt` (default on) so `run_rl.py` saves plain HF checkpoints, add `--enable-math-verify` (default on) so it runs MATH-500/AIME-24/AIME-25/AMC23/Minerva eval after training, and add a standalone `eval_rl.py`.

**Architecture:** Two new modules (`eval_datasets.py`, `math_verify_eval.py`) shared by `run_rl.py` and `eval_rl.py`. The merge logic lives in `run_rl.py` since it handles each `--train-mode` case. Eval calls into `math-verify` (a HF library) for grading.

**Tech Stack:** Python 3.13, PyTorch, HuggingFace transformers, PEFT, vLLM 0.10.2, the `math-verify` HuggingFace library, `unittest` for tests, `uv` for dependency management.

**Spec:** `docs/superpowers/specs/2026-04-19-math-verify-eval-and-merged-ckpt-design.md`

---

## File Structure

**Create:**
- `eval_datasets.py` — frozen dataset registry + loader that wraps each dataset in `boxed.prompt` + chat template.
- `math_verify_eval.py` — vLLM-based eval driver. One public function `math_verify_eval(...)` returning per-dataset results.
- `eval_rl.py` — standalone CLI for evaluating any merged checkpoint or HF model ID.
- `tests/test_eval_datasets.py` — registry sanity. Default tests are offline; full HF-hub round-trip gated behind `RUN_HF_TESTS=1`.
- `tests/test_math_verify_eval.py` — grader smoke test using mocked vLLM output.
- `tests/test_eval_rl_cli.py` — argparse + pre-flight legacy-format detection.
- `tests/test_run_rl_merged_ckpt.py` — argparse for the two new flags + `save_merged_checkpoint` round-trip per mode using a tiny model.
- `tests/test_run_rl_math_verify_cli.py` — argparse for the math-verify flags + the warn-but-don't-block combination.

**Modify:**
- `pyproject.toml` — add `math-verify>=0.5` dependency.
- `run_rl.py` — add the two flags, the `save_merged_checkpoint` helper, the post-training eval hook, and rename the rollout adapter dir to `lora_adapters/step={N}/`.
- `tests/test_run_rl_cli.py` — add `--no-enable-math-verify --no-enable-merged-ckpt` to existing argparse paths that get further than parse_args. (Most existing tests stop at parse_args and don't need changes, but verify case-by-case.)
- Existing shell scripts (`run_rl.sh`, `run_rl_dapo.sh` if it exists, `run_temp.sh`) — add `--no-enable-math-verify` to the smoke / debug invocations that aren't intended to do post-training eval.

**Do not modify:**
- `math_utils.py` — the GRPO reward grader and in-loop training val keep using `is_equiv` (out of scope per spec §1).
- `legacy/run_rl_dapo.py` — the `legacy/` directory is reference-only; no eval hook there.
- `boxed.prompt` — reused as-is.

---

## Task 1: Add `math-verify` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read the current `pyproject.toml`**

Run: `cat pyproject.toml`
Expected output: dependency block with `peft>=0.17.1`, `vllm==0.10.2`, etc.

- [ ] **Step 2: Add `math-verify>=0.5` to the dependency list**

Edit `pyproject.toml`. Locate the `dependencies = [...]` array and add the line in alphabetical order (between `kernels` and `numpy`):

```toml
[project]
name = "lora-without-regret-draft"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "datasets>=4.2.0",
    "kernels>=0.10.4",
    "math-verify>=0.5",
    "numpy>=2.2.6",
    "peft>=0.17.1",
    "torch>=2.8.0",
    "tqdm>=4.67.1",
    "transformers>=4.57.1",
    "vllm==0.10.2",
    "wandb>=0.22.2",
]
```

- [ ] **Step 3: Run `uv sync` to install the new dependency**

Run: `uv sync`
Expected: lockfile updated, `math-verify` installed.

- [ ] **Step 4: Verify the import works**

Run: `uv run python -c "from math_verify import parse, verify; print('ok')"`
Expected: `ok`. (If `math_verify` exposes the API differently — e.g., `math_verify.metric` — adjust the import in later tasks; the symbol names are confirmed by this step.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add math-verify for post-training eval"
```

---

## Task 2: Create `eval_datasets.py` — registry + loader (offline-friendly)

**Files:**
- Create: `eval_datasets.py`
- Test: `tests/test_eval_datasets.py`

The registry is a constant; the loader wraps `datasets.load_dataset` and the chat template. Tests must work without HF Hub access (default) but allow opting in to a network round-trip via `RUN_HF_TESTS=1`.

- [ ] **Step 1: Write the failing offline test for the registry**

Create `tests/test_eval_datasets.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestRegistry(unittest.TestCase):
    def test_registry_has_exactly_five_expected_datasets(self):
        from eval_datasets import REGISTRY

        self.assertEqual(
            set(REGISTRY.keys()),
            {"MATH-500", "AIME-24", "AIME-25", "AMC23", "Minerva"},
        )

    def test_aime_datasets_use_multi_sample_defaults(self):
        from eval_datasets import REGISTRY

        for name in ("AIME-24", "AIME-25"):
            spec = REGISTRY[name]
            self.assertEqual(spec.n_samples, 8, f"{name} should default to 8 samples")
            self.assertAlmostEqual(spec.temperature, 0.6)
            self.assertAlmostEqual(spec.top_p, 0.95)

    def test_greedy_datasets_use_temperature_zero(self):
        from eval_datasets import REGISTRY

        for name in ("MATH-500", "AMC23", "Minerva"):
            spec = REGISTRY[name]
            self.assertEqual(spec.n_samples, 1, f"{name} should default to 1 sample")
            self.assertEqual(spec.temperature, 0.0)
            self.assertEqual(spec.top_p, 1.0)

    def test_dataset_spec_is_frozen(self):
        from eval_datasets import REGISTRY
        import dataclasses

        spec = REGISTRY["MATH-500"]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.hf_id = "other"


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_eval_datasets -v`
Expected: ImportError ("No module named 'eval_datasets'").

- [ ] **Step 3: Create `eval_datasets.py` with the registry**

```python
"""Registry of math-reasoning eval datasets used by math_verify_eval.

Each entry pins a specific HuggingFace dataset ID and the sampling defaults
(n_samples / temperature / top_p) that match published evaluation conventions
(greedy@1 for MATH-500/AMC23/Minerva, avg@8 with T=0.6 for AIME).
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DatasetSpec:
    hf_id: str
    split: str
    problem_field: str
    answer_field: str
    n_samples: int
    temperature: float
    top_p: float


REGISTRY: dict[str, DatasetSpec] = {
    "MATH-500": DatasetSpec(
        hf_id="HuggingFaceH4/MATH-500",
        split="test",
        problem_field="problem",
        answer_field="answer",
        n_samples=1,
        temperature=0.0,
        top_p=1.0,
    ),
    "AIME-24": DatasetSpec(
        hf_id="HuggingFaceH4/aime_2024",
        split="train",
        problem_field="problem",
        answer_field="answer",
        n_samples=8,
        temperature=0.6,
        top_p=0.95,
    ),
    "AIME-25": DatasetSpec(
        hf_id="yentinglin/aime_2025",
        split="train",
        problem_field="problem",
        answer_field="answer",
        n_samples=8,
        temperature=0.6,
        top_p=0.95,
    ),
    "AMC23": DatasetSpec(
        hf_id="math-ai/amc23",
        split="test",
        problem_field="question",
        answer_field="answer",
        n_samples=1,
        temperature=0.0,
        top_p=1.0,
    ),
    "Minerva": DatasetSpec(
        hf_id="math-ai/minerva-math",
        split="test",
        problem_field="question",
        answer_field="answer",
        n_samples=1,
        temperature=0.0,
        top_p=1.0,
    ),
}


def known_dataset_names() -> list[str]:
    """Return the canonical ordering of dataset names for CLI defaults."""
    return ["MATH-500", "AIME-24", "AIME-25", "AMC23", "Minerva"]
```

- [ ] **Step 4: Run the offline tests to verify they pass**

Run: `uv run python -m unittest tests.test_eval_datasets -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Add the `load_eval_dataset` function (not yet exercised by tests)**

Append to `eval_datasets.py`:

```python
def load_eval_dataset(
    name: str,
    tokenizer,
    prompt_template: str,
) -> list[dict]:
    """Load a registered eval dataset and wrap each problem in
    `boxed.prompt` + chat template (same wrapping as run_rl.py training).

    Returns a list of dicts: {"prompt": str, "gold_answer": str,
                               "n_samples": int, "temperature": float,
                               "top_p": float}.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown eval dataset: {name!r}. Known: {sorted(REGISTRY.keys())}"
        )

    from datasets import load_dataset

    spec = REGISTRY[name]
    raw = load_dataset(spec.hf_id, split=spec.split)

    out = []
    for example in raw:
        problem_text = example[spec.problem_field]
        gold_answer = example[spec.answer_field]
        with_template = prompt_template.replace("{question}", problem_text)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": with_template}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(
            {
                "prompt": prompt,
                "gold_answer": str(gold_answer),
                "n_samples": spec.n_samples,
                "temperature": spec.temperature,
                "top_p": spec.top_p,
            }
        )
    return out
```

- [ ] **Step 6: Add the network-gated round-trip test**

Append to `tests/test_eval_datasets.py` (above `if __name__ == "__main__":`):

```python
@unittest.skipUnless(
    os.environ.get("RUN_HF_TESTS") == "1",
    "Set RUN_HF_TESTS=1 to enable HF Hub round-trip tests.",
)
class TestLoadEvalDatasetHFRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        with open(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "boxed.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            cls.template = f.read().strip()

    def test_load_each_dataset_returns_nonempty(self):
        from eval_datasets import load_eval_dataset, known_dataset_names

        for name in known_dataset_names():
            with self.subTest(dataset=name):
                items = load_eval_dataset(name, self.tokenizer, self.template)
                self.assertGreater(len(items), 0, f"{name} returned 0 items")
                first = items[0]
                self.assertIn("prompt", first)
                self.assertIn("gold_answer", first)
                self.assertIn("\\boxed", first["prompt"])
```

- [ ] **Step 7: Run the offline tests once more to confirm nothing regressed**

Run: `uv run python -m unittest tests.test_eval_datasets -v`
Expected: 4 PASS, 1 SKIPPED.

- [ ] **Step 8: Commit**

```bash
git add eval_datasets.py tests/test_eval_datasets.py
git commit -m "eval: add eval-datasets registry and loader"
```

---

## Task 3: Create `math_verify_eval.py` — grader with mocked-vLLM smoke test

**Files:**
- Create: `math_verify_eval.py`
- Test: `tests/test_math_verify_eval.py`

The function takes a tokenizer, dataset names, and either an existing vLLM `LLM` instance (passed in `vllm_kwargs={"llm": ...}`) or kwargs to construct a fresh one. Tests mock the vLLM call so they run on CPU.

- [ ] **Step 1: Write the failing test using a fake "vLLM" callable**

Create `tests/test_math_verify_eval.py`:

```python
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class _FakeTokenizer:
    """Stand-in tokenizer; chat-template is unused since we patch the dataset loader."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


def _fake_dataset(_name, _tokenizer, _template):
    # Two problems, gold = "42" each.
    return [
        {"prompt": "p1", "gold_answer": "42", "n_samples": 1, "temperature": 0.0, "top_p": 1.0},
        {"prompt": "p2", "gold_answer": "42", "n_samples": 1, "temperature": 0.0, "top_p": 1.0},
    ]


class TestMathVerifyEval(unittest.TestCase):
    def test_grades_two_problems_one_correct(self):
        import math_verify_eval

        # Fake LLM returns: problem 1 → correct boxed{42}, problem 2 → wrong boxed{0}.
        fake_llm = MagicMock()
        fake_outputs = [
            MagicMock(outputs=[MagicMock(text=r"reasoning... \boxed{42}")]),
            MagicMock(outputs=[MagicMock(text=r"reasoning... \boxed{0}")]),
        ]
        fake_llm.generate.return_value = fake_outputs

        results = math_verify_eval.math_verify_eval(
            model=None,
            tokenizer=_FakeTokenizer(),
            datasets=["MATH-500"],
            max_tokens=128,
            prompt_template_path=None,
            vllm_kwargs={"llm": fake_llm},
            _dataset_loader=_fake_dataset,
        )

        self.assertIn("MATH-500", results["datasets"])
        ds = results["datasets"]["MATH-500"]
        self.assertEqual(ds["n_total"], 2)
        self.assertEqual(ds["n_correct"], 1)
        self.assertAlmostEqual(ds["accuracy"], 0.5)

    def test_unparseable_counts_as_incorrect(self):
        import math_verify_eval

        fake_llm = MagicMock()
        # No \boxed{} in either output.
        fake_outputs = [
            MagicMock(outputs=[MagicMock(text="some prose")]),
            MagicMock(outputs=[MagicMock(text="more prose")]),
        ]
        fake_llm.generate.return_value = fake_outputs

        results = math_verify_eval.math_verify_eval(
            model=None,
            tokenizer=_FakeTokenizer(),
            datasets=["MATH-500"],
            max_tokens=128,
            prompt_template_path=None,
            vllm_kwargs={"llm": fake_llm},
            _dataset_loader=_fake_dataset,
        )

        ds = results["datasets"]["MATH-500"]
        self.assertEqual(ds["n_correct"], 0)
        self.assertEqual(ds["n_unparseable"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_math_verify_eval -v`
Expected: ImportError ("No module named 'math_verify_eval'").

- [ ] **Step 3: Create `math_verify_eval.py`**

```python
"""Post-training math-verify evaluation driver.

Used by run_rl.py (after the GRPO loop) and by eval_rl.py (standalone).
Generates with vLLM, grades with HuggingFace math-verify.
"""

import datetime
import time
from typing import Callable, Optional

from eval_datasets import REGISTRY, load_eval_dataset


def _build_or_get_llm(model, vllm_kwargs):
    """Return an LLM-like object exposing a `.generate(prompts, sampling_params)` method.

    If `vllm_kwargs` contains "llm", reuse it (for in-loop reuse from run_rl.py
    or for tests). Otherwise construct a fresh vLLM `LLM` from `vllm_kwargs`.
    """
    if vllm_kwargs is None:
        vllm_kwargs = {}
    if "llm" in vllm_kwargs:
        return vllm_kwargs["llm"], False  # not owned (caller manages)
    import os

    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM

    return LLM(**vllm_kwargs), True  # owned (we constructed it)


def _verify_one(gold: str, model_text: str) -> tuple[bool, bool, bool]:
    """Return (is_correct, was_unparseable, grader_raised).

    Uses `math_verify.parse` + `math_verify.verify`. An empty parse counts as
    unparseable (and incorrect). A grader exception counts as `grader_raised`
    (and incorrect).
    """
    from math_verify import parse, verify

    try:
        pred = parse(model_text)
    except Exception:
        return False, True, True
    if not pred:
        return False, True, False

    try:
        gold_parsed = parse(gold)
        ok = bool(verify(gold_parsed, pred))
    except Exception:
        return False, False, True
    return ok, False, False


def math_verify_eval(
    model,                               # may be None when vllm_kwargs["llm"] is supplied
    tokenizer,
    datasets: list[str],
    *,
    n_samples_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    max_tokens: int = 2048,
    prompt_template_path: Optional[str] = "boxed.prompt",
    vllm_kwargs: Optional[dict] = None,
    _dataset_loader: Callable = load_eval_dataset,  # injected for tests
) -> dict:
    """Run math-verify eval over the requested dataset names.

    Returns:
        {
          "datasets": {name: {"accuracy", "n_correct", "n_total",
                              "n_samples_per_problem", "temperature",
                              "max_tokens", "wall_time_sec",
                              "n_unparseable", "n_grader_errors"}},
          "errors": {name: reason_str},
          "math_verify_version": str,
          "timestamp": ISO8601 str,
        }
    """
    from importlib import metadata as _metadata
    from vllm import SamplingParams

    template = ""
    if prompt_template_path is not None:
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            template = f.read().strip()

    llm, owned = _build_or_get_llm(model, vllm_kwargs)
    try:
        try:
            mv_version = _metadata.version("math-verify")
        except _metadata.PackageNotFoundError:
            mv_version = "unknown"

        out = {
            "datasets": {},
            "errors": {},
            "math_verify_version": mv_version,
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        for name in datasets:
            try:
                items = _dataset_loader(name, tokenizer, template)
            except Exception as exc:
                out["errors"][name] = f"dataset load failed: {exc!r}"
                continue

            spec = REGISTRY[name]
            n_samples = (
                n_samples_override if n_samples_override is not None else spec.n_samples
            )
            temperature = (
                temperature_override
                if temperature_override is not None
                else spec.temperature
            )
            top_p = spec.top_p

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n_samples,
            )

            prompts = [item["prompt"] for item in items]
            gold_answers = [item["gold_answer"] for item in items]

            t0 = time.time()
            outputs = llm.generate(prompts, sampling_params)
            wall = time.time() - t0

            n_correct = 0
            n_unparseable = 0
            n_grader_errors = 0
            n_total = 0

            for problem_output, gold in zip(outputs, gold_answers):
                for completion in problem_output.outputs:
                    n_total += 1
                    ok, unparseable, raised = _verify_one(gold, completion.text)
                    if ok:
                        n_correct += 1
                    if unparseable:
                        n_unparseable += 1
                    if raised:
                        n_grader_errors += 1

            out["datasets"][name] = {
                "accuracy": n_correct / n_total if n_total else 0.0,
                "n_correct": n_correct,
                "n_total": n_total,
                "n_samples_per_problem": n_samples,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "wall_time_sec": wall,
                "n_unparseable": n_unparseable,
                "n_grader_errors": n_grader_errors,
            }

        return out
    finally:
        if owned:
            # vLLM doesn't expose a clean shutdown in 0.10.2; rely on GC.
            del llm
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_math_verify_eval -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add math_verify_eval.py tests/test_math_verify_eval.py
git commit -m "eval: add math_verify_eval driver"
```

---

## Task 4: Add the two new flags to `run_rl.py` argparse + validation

**Files:**
- Modify: `run_rl.py` — argparse section (around line 367), `validate_mode_specific_flags` (around line 415).
- Test: `tests/test_run_rl_math_verify_cli.py`

This task only adds the flags and validators. Subsequent tasks wire them to behavior.

- [ ] **Step 1: Write the failing argparse tests**

Create `tests/test_run_rl_math_verify_cli.py`:

```python
import io
import os
import sys
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestRunRlMathVerifyCli(unittest.TestCase):
    def test_defaults_are_on(self):
        import run_rl

        args = run_rl.parse_args(["--train-mode", "full"])
        self.assertTrue(args.enable_merged_ckpt)
        self.assertTrue(args.enable_math_verify)
        self.assertEqual(
            args.math_verify_datasets,
            ["MATH-500", "AIME-24", "AIME-25", "AMC23", "Minerva"],
        )
        self.assertIsNone(args.math_verify_n_samples)
        self.assertIsNone(args.math_verify_temperature)
        self.assertEqual(args.math_verify_max_tokens, 2048)

    def test_no_flags_disable_features(self):
        import run_rl

        args = run_rl.parse_args(
            ["--train-mode", "full", "--no-enable-merged-ckpt", "--no-enable-math-verify"]
        )
        self.assertFalse(args.enable_merged_ckpt)
        self.assertFalse(args.enable_math_verify)

    def test_math_verify_datasets_parses_csv(self):
        import run_rl

        args = run_rl.parse_args(
            ["--train-mode", "full", "--math-verify-datasets", "MATH-500,AIME-24"]
        )
        self.assertEqual(args.math_verify_datasets, ["MATH-500", "AIME-24"])

    def test_math_verify_datasets_rejects_unknown_name(self):
        import run_rl

        argv = ["--train-mode", "full", "--math-verify-datasets", "BOGUS,MATH-500"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError) as cm:
            run_rl.validate_mode_specific_flags(args, argv)
        self.assertIn("BOGUS", str(cm.exception))
        self.assertIn("MATH-500", str(cm.exception))  # known names cited

    def test_math_verify_n_samples_zero_rejected(self):
        import run_rl

        argv = ["--train-mode", "full", "--math-verify-n-samples", "0"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

    def test_no_merge_with_eval_emits_warning(self):
        import run_rl

        argv = [
            "--train-mode",
            "blocktt",
            "--no-enable-merged-ckpt",
            "--enable-math-verify",
        ]
        args = run_rl.parse_args(argv)

        buf = io.StringIO()
        with redirect_stderr(buf):
            # validate should not raise; warning goes to stderr.
            run_rl.validate_mode_specific_flags(args, argv)

        self.assertIn("--no-enable-merged-ckpt", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_run_rl_math_verify_cli -v`
Expected: failures because `args.enable_merged_ckpt` doesn't exist yet.

- [ ] **Step 3: Add the new argparse flags to `parse_args`**

In `run_rl.py`, locate the `--no-wandb` flag (around line 367) and **insert immediately above** `add_calibrated_btt_args(parser, hyphen_style=True)`:

```python
    # Merged-checkpoint and post-training math-verify eval flags
    parser.add_argument(
        "--enable-merged-ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save checkpoints in plain HF format (LoRA merged into base, "
            "BlockTT/SVD materialized to nn.Linear). Default: enabled."
        ),
    )
    parser.add_argument(
        "--enable-math-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After training, evaluate the final checkpoint on math reasoning "
            "benchmarks via the math-verify library. Default: enabled."
        ),
    )
    parser.add_argument(
        "--math-verify-datasets",
        type=str,
        default="MATH-500,AIME-24,AIME-25,AMC23,Minerva",
        help=(
            "Comma-separated list of eval dataset names. "
            "Default: MATH-500,AIME-24,AIME-25,AMC23,Minerva."
        ),
    )
    parser.add_argument(
        "--math-verify-n-samples",
        type=int,
        default=None,
        help=(
            "Override per-dataset n_samples. Default: registry per-dataset "
            "(8 for AIME-24/25, 1 for the rest)."
        ),
    )
    parser.add_argument(
        "--math-verify-temperature",
        type=float,
        default=None,
        help=(
            "Override per-dataset sampling temperature. Default: registry "
            "per-dataset (0.6 for AIME-24/25, 0.0 for the rest)."
        ),
    )
    parser.add_argument(
        "--math-verify-max-tokens",
        type=int,
        default=2048,
        help="Max tokens per eval generation. Default: 2048.",
    )
```

- [ ] **Step 4: Convert the comma-separated datasets string to a list immediately after parsing**

In `parse_args`, just before `return parser.parse_args(argv)`, the existing code returns the namespace. We need post-processing. Replace the existing return at the end of `parse_args` with:

```python
    args = parser.parse_args(argv)
    args.math_verify_datasets = [
        s.strip() for s in args.math_verify_datasets.split(",") if s.strip()
    ]
    return args
```

(If `parse_args` already does post-processing, fold this in.)

- [ ] **Step 5: Add the validation logic to `validate_mode_specific_flags`**

In `run_rl.py`, locate the bottom of `validate_mode_specific_flags` (just before the existing `validate_calibrated_btt_args(args, argv=argv, hyphen_style=True)` call near the end of the function) and **insert above it**:

```python
    # Math-verify validation
    from eval_datasets import REGISTRY as _MV_REGISTRY

    unknown = [d for d in args.math_verify_datasets if d not in _MV_REGISTRY]
    if unknown:
        known = sorted(_MV_REGISTRY.keys())
        raise ValueError(
            f"Unknown --math-verify-datasets entries: {unknown}. "
            f"Known names: {known}"
        )
    if args.math_verify_n_samples is not None and args.math_verify_n_samples <= 0:
        raise ValueError(
            f"--math-verify-n-samples must be > 0, got {args.math_verify_n_samples}"
        )
    if args.math_verify_max_tokens <= 0:
        raise ValueError(
            f"--math-verify-max-tokens must be > 0, got {args.math_verify_max_tokens}"
        )

    if args.enable_math_verify and not args.enable_merged_ckpt and args.train_mode != "full":
        import sys as _sys

        print(
            "WARNING: --enable-math-verify with --no-enable-merged-ckpt may fail at "
            "eval time for non-full modes (the saved checkpoint isn't loadable by "
            "vanilla AutoModelForCausalLM.from_pretrained).",
            file=_sys.stderr,
        )
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `uv run python -m unittest tests.test_run_rl_math_verify_cli -v`
Expected: 6 PASS.

- [ ] **Step 7: Run existing CLI tests to confirm no regressions**

Run: `uv run python -m unittest tests.test_run_rl_cli -v`
Expected: all existing tests PASS (the new flags have defaults so they don't break existing argv).

- [ ] **Step 8: Commit**

```bash
git add run_rl.py tests/test_run_rl_math_verify_cli.py
git commit -m "run_rl: add --enable-merged-ckpt and --enable-math-verify flags"
```

---

## Task 5: Implement `save_merged_checkpoint` helper

**Files:**
- Modify: `run_rl.py` — add `save_merged_checkpoint`, route `save_checkpoint` through it when `args.enable_merged_ckpt` is true.
- Test: `tests/test_run_rl_merged_ckpt.py`

The helper handles all five modes. For `lora`/`lora_full` it merges PEFT adapters; for `blocktt`/`svd` it builds a dense state_dict from `materialize_dense_weight()` outputs without mutating the model object.

- [ ] **Step 1: Write the failing test for the `full` mode no-op path**

Create `tests/test_run_rl_merged_ckpt.py`:

```python
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch


class _Tokenizer:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")


class TestSaveMergedCheckpointFull(unittest.TestCase):
    def test_full_mode_calls_save_pretrained(self):
        import run_rl

        model = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "full", args)
            model.save_pretrained.assert_called_once_with(ckpt)
            self.assertTrue(os.path.exists(os.path.join(ckpt, "tokenizer.json")))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.test_run_rl_merged_ckpt -v`
Expected: AttributeError ("module 'run_rl' has no attribute 'save_merged_checkpoint'").

- [ ] **Step 3: Add the helper to `run_rl.py`**

In `run_rl.py`, **insert immediately above** the existing `def save_checkpoint(...)` at line 681:

```python
def _build_factored_dense_state_dict(model):
    """Return a state_dict where every BTTLayer / SVDLayer is replaced by a
    dense `{prefix}.weight` (and `{prefix}.bias` if present) materialized from
    the factored cores. Other parameters are kept verbatim. The model object
    is never mutated.
    """
    factored_prefixes = []
    extra = {}

    for module_name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            factored_prefixes.append(module_name + ".")
            dense = module.materialize_dense_weight().detach().clone()
            extra[f"{module_name}.weight"] = dense
            if module.bias is not None:
                extra[f"{module_name}.bias"] = module.bias.detach().clone()
        elif isinstance(module, SVDLayer):
            factored_prefixes.append(module_name + ".")
            dense = module.materialize_dense_weight().detach().clone()
            extra[f"{module_name}.weight"] = dense
            if module.bias is not None:
                extra[f"{module_name}.bias"] = module.bias.detach().clone()

    new_sd = {}
    for name, tensor in model.state_dict().items():
        if any(name.startswith(p) for p in factored_prefixes):
            continue
        new_sd[name] = tensor
    new_sd.update(extra)
    return new_sd


def save_merged_checkpoint(model, tokenizer, ckpt_dir: str, train_mode: str, args):
    """Save model in plain HuggingFace format. The on-disk result contains
    only nn.Linear layers (no LoRA adapters, no BTT/SVD factored cores).
    The in-memory model object is never mutated; training can resume.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    if train_mode == "full":
        model.save_pretrained(ckpt_dir)
    elif train_mode in {"lora", "lora_full"}:
        model.merge_adapter()
        try:
            base = model.get_base_model()
            base.save_pretrained(ckpt_dir)
        finally:
            model.unmerge_adapter()
    elif train_mode in {"blocktt", "svd"}:
        if train_mode == "blocktt" and getattr(args, "calib_mode", "none") != "none":
            save_calibrated_btt_hf_pretrained(model, ckpt_dir)
        else:
            state_dict = _build_factored_dense_state_dict(model)
            model.save_pretrained(ckpt_dir, state_dict=state_dict)
    else:
        raise ValueError(f"Unknown train_mode for save_merged_checkpoint: {train_mode}")

    tokenizer.save_pretrained(ckpt_dir)
```

- [ ] **Step 4: Run the test**

Run: `uv run python -m unittest tests.test_run_rl_merged_ckpt -v`
Expected: PASS.

- [ ] **Step 5: Add the test for the `lora`/`lora_full` merge path**

Append to `tests/test_run_rl_merged_ckpt.py`:

```python
class TestSaveMergedCheckpointLora(unittest.TestCase):
    def test_lora_calls_merge_then_base_save_then_unmerge(self):
        import run_rl

        base = MagicMock()
        model = MagicMock()
        model.get_base_model.return_value = base

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "lora", args)

            model.merge_adapter.assert_called_once()
            base.save_pretrained.assert_called_once_with(ckpt)
            model.unmerge_adapter.assert_called_once()
            model.save_pretrained.assert_not_called()

    def test_lora_unmerges_even_if_save_raises(self):
        import run_rl

        base = MagicMock()
        base.save_pretrained.side_effect = RuntimeError("disk full")
        model = MagicMock()
        model.get_base_model.return_value = base

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            with self.assertRaises(RuntimeError):
                run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "lora", args)
            model.merge_adapter.assert_called_once()
            model.unmerge_adapter.assert_called_once()

    def test_lora_full_mode_uses_same_path(self):
        import run_rl

        base = MagicMock()
        model = MagicMock()
        model.get_base_model.return_value = base

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "lora_full", args)

            model.merge_adapter.assert_called_once()
            base.save_pretrained.assert_called_once_with(ckpt)
            model.unmerge_adapter.assert_called_once()
```

- [ ] **Step 6: Run the new tests**

Run: `uv run python -m unittest tests.test_run_rl_merged_ckpt -v`
Expected: 4 PASS.

- [ ] **Step 7: Add the BTT factored→dense state_dict test using a real `BTTLayer`**

Append to `tests/test_run_rl_merged_ckpt.py`:

```python
class TestBuildFactoredDenseStateDict(unittest.TestCase):
    """Exercises the factored→dense conversion using real BTTLayer/SVDLayer
    instances. CPU-only, tiny dimensions."""

    def _make_btt_module(self):
        # Build a model with one nn.Linear, then convert it to BTTLayer using
        # the public training-time API. Only public API is used.
        import torch.nn as nn

        torch.manual_seed(0)
        in_features, out_features = 8, 8
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)

        class _Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features, bias=True)

        wrap = _Wrap()
        with torch.no_grad():
            wrap.linear.weight.copy_(weight)
            wrap.linear.bias.copy_(bias)

        from btt_layer import convert_linear_to_btt

        convert_linear_to_btt(
            wrap,
            btt_rank="full",
            decomp_mode="input_one_block",
            init_mode="default",
            include_names=("linear",),
            skip_names=(),
            lr_act=False,
            s_merged_to="frozen",
            train_position="small",
            factorize_by_head=False,
            model_config=None,
        )
        # The materialized weight may differ from the original by reconstruction
        # error of the BTT factorization at this rank; the test compares
        # _build_factored_dense_state_dict's output to materialize_dense_weight()
        # of the live module rather than to the original weight.
        return wrap, weight, bias

    def test_btt_factored_state_dict_matches_dense(self):
        import run_rl

        model, _, _ = self._make_btt_module()

        # Snapshot the live materialized weight before calling the helper.
        with torch.no_grad():
            expected = model.linear.materialize_dense_weight().detach().clone()

        sd = run_rl._build_factored_dense_state_dict(model)

        # No factored core keys leaked into the new state_dict.
        for k in sd.keys():
            self.assertNotIn(".btt_l", k)
            self.assertNotIn(".btt_r", k)
            self.assertNotIn(".btt_s", k)

        # The dense weight in the state_dict matches what the BTTLayer
        # currently materializes.
        self.assertIn("linear.weight", sd)
        torch.testing.assert_close(
            sd["linear.weight"].float(), expected.float(), atol=1e-5, rtol=1e-5
        )

        # Model object is unchanged: the BTTLayer is still in place.
        from btt_layer import BTTLayer

        self.assertIsInstance(model.linear, BTTLayer)
```

- [ ] **Step 8: Run the BTT test**

Run: `uv run python -m unittest tests.test_run_rl_merged_ckpt.TestBuildFactoredDenseStateDict -v`
Expected: PASS. If the import `_solve_btt_factors_from_weight` fails (it's a private helper that may not exist), remove that line — the test only relies on the public `convert_linear_to_btt` path.

- [ ] **Step 9: Add the SVD analogue test**

Append to `tests/test_run_rl_merged_ckpt.py`:

```python
class TestBuildFactoredDenseStateDictSVD(unittest.TestCase):
    def test_svd_factored_state_dict_matches_dense(self):
        import torch.nn as nn
        from svd_layer import SVDLayer, convert_linear_to_svd

        torch.manual_seed(0)
        in_features, out_features = 8, 8
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)

        class _Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features, bias=True)

            def save_pretrained(self, path, state_dict=None):
                os.makedirs(path, exist_ok=True)
                from safetensors.torch import save_file

                save_file(state_dict or self.state_dict(), os.path.join(path, "model.safetensors"))
                with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
                    f.write("{}")

        wrap = _Wrap()
        with torch.no_grad():
            wrap.linear.weight.copy_(weight)
            wrap.linear.bias.copy_(bias)

        convert_linear_to_svd(
            wrap,
            include_names=("linear",),
            skip_names=(),
            s_merged_to="frozen",
            train_position="output",
        )

        with torch.no_grad():
            expected = wrap.linear.materialize_dense_weight().detach().clone()

        import run_rl

        sd = run_rl._build_factored_dense_state_dict(wrap)
        for k in sd.keys():
            self.assertNotIn(".svd_a", k)
            self.assertNotIn(".svd_b", k)
            self.assertNotIn(".svd_s", k)

        self.assertIn("linear.weight", sd)
        torch.testing.assert_close(
            sd["linear.weight"].float(), expected.float(), atol=1e-5, rtol=1e-5
        )
        self.assertIsInstance(wrap.linear, SVDLayer)
```

- [ ] **Step 10: Run all merged-ckpt tests**

Run: `uv run python -m unittest tests.test_run_rl_merged_ckpt -v`
Expected: 6 PASS.

- [ ] **Step 11: Wire `save_checkpoint` (run_rl.py:681) to route through `save_merged_checkpoint` when enabled**

Replace the body of `save_checkpoint` (run_rl.py:681) with:

```python
def save_checkpoint(model, tokenizer, run_dir: str, step_num: int, args=None):
    ckpt_dir = os.path.join(run_dir, f"step={step_num}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving checkpoint to {ckpt_dir}")
    if args is not None and getattr(args, "enable_merged_ckpt", True):
        save_merged_checkpoint(model, tokenizer, ckpt_dir, args.train_mode, args)
    elif args is not None and getattr(args, "calib_mode", "none") != "none":
        save_calibrated_btt_hf_pretrained(model, ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
    else:
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
    print(f"Checkpoint saved to {ckpt_dir}")
```

- [ ] **Step 12: Run all CLI + checkpoint tests once more**

Run: `uv run python -m unittest tests.test_run_rl_cli tests.test_run_rl_merged_ckpt tests.test_run_rl_math_verify_cli -v`
Expected: all PASS.

- [ ] **Step 13: Commit**

```bash
git add run_rl.py tests/test_run_rl_merged_ckpt.py
git commit -m "run_rl: implement save_merged_checkpoint helper and route saves through it"
```

---

## Task 6: Move LoRA rollout adapter saves to `lora_adapters/step={N}/`

**Files:**
- Modify: `run_rl.py` — `save_lora` function (around line 815).

The current `save_lora` writes to `{run_dir}/step={N}` which collides with merged-ckpt saves. Per spec §3, rollout adapters move to `{run_dir}/lora_adapters/step={N}/`.

- [ ] **Step 1: Locate `save_lora` (run_rl.py around line 815)**

Read the function. Current body:

```python
    def save_lora(step):
        lora_name = f"{run_dir}/step={step}"
        if not os.path.exists(lora_name):
            model.save_pretrained(lora_name)
        return lora_name
```

- [ ] **Step 2: Update `save_lora` to use the new subdirectory**

Replace the body with:

```python
    def save_lora(step):
        lora_name = f"{run_dir}/lora_adapters/step={step}"
        os.makedirs(os.path.dirname(lora_name), exist_ok=True)
        if not os.path.exists(lora_name):
            model.save_pretrained(lora_name)
        return lora_name
```

(The `os.makedirs(..., exist_ok=True)` ensures the parent `lora_adapters/` directory exists before `save_pretrained` is called.)

- [ ] **Step 3: Run the existing CLI tests to confirm no regression**

Run: `uv run python -m unittest tests.test_run_rl_cli -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add run_rl.py
git commit -m "run_rl: move HTTP-lora rollout adapter saves to lora_adapters/ subdir"
```

---

## Task 7: Wire post-training math-verify eval into `run_rl.py`'s `main`

**Files:**
- Modify: `run_rl.py` — at the end of `main`, after the GRPO loop completes.

After training, save the final merged checkpoint, then run `math_verify_eval`. The eval needs access to the existing in-process vLLM `LLM` instance (where present) for weight reuse, otherwise it spins up a fresh one.

- [ ] **Step 1: Refactor the local-vLLM generator builders to expose the `LLM` instance**

In `run_rl.py`, locate `build_local_vllm_generators` (around line 910) and `build_lora_local_generators` (around line 844). Both currently construct an `LLM(...)` and return only the two generator closures.

Modify each to also return the `LLM` instance. Change the `return` lines from:

```python
    return generate_for_train, generate_for_eval
```

to:

```python
    return generate_for_train, generate_for_eval, vllm_model
```

And update the corresponding `build_lora_http_generators` (around line 786) to return `None` for the third element (no in-process LLM in HTTP mode):

```python
    return generate_for_train, generate_for_eval, None
```

- [ ] **Step 2: Update the call sites in `main` to capture the third return**

In `run_rl.py`'s `main`, locate where these builders are called (around line 1434-1447):

```python
    if args.train_mode in {"lora", "lora_full"}:
        if lora_rollout_backend == "http":
            generate_for_train, generate_for_eval = build_lora_http_generators(...)
        else:
            generate_for_train, generate_for_eval = build_lora_local_generators(...)
    else:
        generate_for_train, generate_for_eval = build_local_vllm_generators(args, model)
```

Change to:

```python
    if args.train_mode in {"lora", "lora_full"}:
        if lora_rollout_backend == "http":
            generate_for_train, generate_for_eval, in_process_llm = build_lora_http_generators(
                args, model, run_dir,
            )
        else:
            generate_for_train, generate_for_eval, in_process_llm = build_lora_local_generators(
                args, model,
            )
    else:
        generate_for_train, generate_for_eval, in_process_llm = build_local_vllm_generators(
            args, model,
        )
```

- [ ] **Step 3: Add the post-training save + eval block at the end of `main`**

In `run_rl.py`, locate the end of `main` (after the existing `if not args.no_wandb: wandb.finish()` line, around line 1702-1703). **Insert immediately above** `if not args.no_wandb: wandb.finish()`:

```python
    # ─── Final merged-checkpoint save ──────────────────────────────────────
    final_step = args.n_grpo_steps
    final_ckpt_dir = None
    if args.enable_merged_ckpt or args.enable_math_verify:
        final_ckpt_dir = os.path.join(run_dir, f"step={final_step}")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        if args.enable_merged_ckpt:
            print(f"Saving final merged checkpoint to {final_ckpt_dir}")
            save_merged_checkpoint(
                model, tokenizer, final_ckpt_dir, args.train_mode, args
            )

    # ─── Post-training math-verify eval ────────────────────────────────────
    if args.enable_math_verify:
        import json as _json
        from math_verify_eval import math_verify_eval as _math_verify_eval

        try:
            if in_process_llm is not None:
                # Reuse the in-memory model + existing vLLM via hot-swap.
                # The eval driver uses the LLM instance; weight load must
                # happen before generate. We push the merged weights now.
                if args.train_mode in {"blocktt", "svd"}:
                    weight_tuples = export_weights_for_vllm(model)
                elif args.train_mode in {"lora", "lora_full"}:
                    model.merge_adapter()
                    try:
                        base = model.get_base_model()
                        weight_tuples = []
                        seen = set()
                        for name, p in base.named_parameters():
                            normalized = normalize_lora_merged_weight_name(name)
                            if normalized is None or normalized in seen:
                                continue
                            seen.add(normalized)
                            weight_tuples.append((normalized, p))
                        in_process_llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                            weight_tuples
                        )
                    finally:
                        model.unmerge_adapter()
                    weight_tuples = None  # already loaded
                else:
                    weight_tuples = [(n, p) for n, p in model.named_parameters()]

                if weight_tuples is not None:
                    in_process_llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                        weight_tuples
                    )

                results = _math_verify_eval(
                    model=None,
                    tokenizer=tokenizer,
                    datasets=args.math_verify_datasets,
                    n_samples_override=args.math_verify_n_samples,
                    temperature_override=args.math_verify_temperature,
                    max_tokens=args.math_verify_max_tokens,
                    prompt_template_path=args.prompt_template,
                    vllm_kwargs={"llm": in_process_llm},
                )
            else:
                # HTTP-lora training: spin up a fresh in-process LLM from disk.
                if final_ckpt_dir is None:
                    raise RuntimeError(
                        "math-verify eval requires either an in-process LLM "
                        "or --enable-merged-ckpt for a saved checkpoint to load."
                    )
                results = _math_verify_eval(
                    model=None,
                    tokenizer=tokenizer,
                    datasets=args.math_verify_datasets,
                    n_samples_override=args.math_verify_n_samples,
                    temperature_override=args.math_verify_temperature,
                    max_tokens=args.math_verify_max_tokens,
                    prompt_template_path=args.prompt_template,
                    vllm_kwargs={
                        "model": final_ckpt_dir,
                        "tensor_parallel_size": 1,
                        "gpu_memory_utilization": args.gpu_memory_utilization,
                        "max_model_len": args.max_model_len,
                        "max_num_batched_tokens": 4096,
                    },
                )

            results["checkpoint"] = final_ckpt_dir or "<in-memory>"
            results["model_id_at_train_time"] = args.model_id

            if final_ckpt_dir is not None:
                results_path = os.path.join(final_ckpt_dir, "eval_results.json")
                with open(results_path, "w", encoding="utf-8") as f:
                    _json.dump(results, f, indent=2)
                print(f"Wrote {results_path}")

            if not args.no_wandb:
                for ds_name, ds_result in results["datasets"].items():
                    wandb.log({f"eval/{ds_name}/accuracy": ds_result["accuracy"]})
                for ds_name, reason in results.get("errors", {}).items():
                    wandb.log({f"eval/{ds_name}/error": reason})

            print("Math-verify results:")
            for ds_name, ds_result in results["datasets"].items():
                print(
                    f"  {ds_name}: {ds_result['accuracy']:.2%} "
                    f"({ds_result['n_correct']}/{ds_result['n_total']})"
                )
        except Exception as exc:
            print(f"ERROR: math-verify eval failed: {exc!r}")
            if not args.no_wandb:
                wandb.log({"eval/error": repr(exc)})
```

- [ ] **Step 4: Run the existing tests to confirm no regressions**

Run: `uv run python -m unittest tests.test_run_rl_cli tests.test_run_rl_merged_ckpt tests.test_run_rl_math_verify_cli -v`
Expected: all PASS. (No new test for the eval hook itself — exercised by the manual smoke run in Task 11.)

- [ ] **Step 5: Commit**

```bash
git add run_rl.py
git commit -m "run_rl: hook post-training math-verify eval into main"
```

---

## Task 8: Create `eval_rl.py` standalone entrypoint

**Files:**
- Create: `eval_rl.py`
- Test: `tests/test_eval_rl_cli.py`

The script is thin: argparse → pre-flight legacy-format check → `AutoModelForCausalLM.from_pretrained` → `math_verify_eval` → write JSON.

- [ ] **Step 1: Write the failing argparse + pre-flight tests**

Create `tests/test_eval_rl_cli.py`:

```python
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestEvalRlCli(unittest.TestCase):
    def test_checkpoint_required(self):
        import eval_rl

        with self.assertRaises(SystemExit):
            eval_rl.parse_args([])

    def test_defaults(self):
        import eval_rl

        args = eval_rl.parse_args(["--checkpoint", "Qwen/Qwen3-1.7B"])
        self.assertEqual(
            args.math_verify_datasets,
            ["MATH-500", "AIME-24", "AIME-25", "AMC23", "Minerva"],
        )
        self.assertEqual(args.math_verify_max_tokens, 2048)
        self.assertEqual(args.prompt_template, "boxed.prompt")
        self.assertEqual(args.max_model_len, 2048)


class TestEvalRlPreflight(unittest.TestCase):
    def test_adapter_only_dir_rejected(self):
        import eval_rl

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "adapter_config.json"), "w") as f:
                f.write("{}")
            with self.assertRaises(ValueError) as cm:
                eval_rl.preflight_checkpoint(d)
            self.assertIn("adapter", str(cm.exception).lower())

    def test_factored_btt_dir_rejected(self):
        import eval_rl

        from safetensors.torch import save_file
        import torch

        with tempfile.TemporaryDirectory() as d:
            save_file(
                {"layer.btt_l": torch.zeros(2, 2), "layer.btt_r": torch.zeros(2, 2)},
                os.path.join(d, "model.safetensors"),
            )
            with self.assertRaises(ValueError) as cm:
                eval_rl.preflight_checkpoint(d)
            self.assertIn("factored", str(cm.exception).lower())

    def test_factored_svd_dir_rejected(self):
        import eval_rl

        from safetensors.torch import save_file
        import torch

        with tempfile.TemporaryDirectory() as d:
            save_file(
                {"layer.svd_a": torch.zeros(2, 2), "layer.svd_b": torch.zeros(2, 2)},
                os.path.join(d, "model.safetensors"),
            )
            with self.assertRaises(ValueError):
                eval_rl.preflight_checkpoint(d)

    def test_plain_hf_dir_passes(self):
        import eval_rl

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                f.write("{}")
            # No exception expected.
            eval_rl.preflight_checkpoint(d)

    def test_hf_id_passes(self):
        import eval_rl

        # Anything not a local directory just passes (will fail later in
        # from_pretrained if invalid, but pre-flight is happy).
        eval_rl.preflight_checkpoint("Qwen/Qwen3-1.7B")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_eval_rl_cli -v`
Expected: ImportError ("No module named 'eval_rl'").

- [ ] **Step 3: Create `eval_rl.py`**

```python
"""Standalone math-verify evaluation entrypoint.

Loads a merged checkpoint (or HuggingFace model ID) and evaluates it on a
fixed set of math reasoning benchmarks. Does not support legacy adapter-only
or factored checkpoints — for those, re-run training with --enable-merged-ckpt
or use the in-loop --enable-math-verify path in run_rl.py.

Examples:
  uv run eval_rl.py --checkpoint Qwen/Qwen3-1.7B
  uv run eval_rl.py --checkpoint /path/to/runs/lora/run-name/step=50
  uv run eval_rl.py --checkpoint <path> --math-verify-datasets MATH-500,AIME-24
"""

import argparse
import json
import os
import sys
import time

import torch

from eval_datasets import REGISTRY, known_dataset_names
from math_verify_eval import math_verify_eval


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a merged checkpoint on math reasoning benchmarks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a merged checkpoint directory or a HuggingFace model ID.",
    )
    parser.add_argument(
        "--math-verify-datasets",
        type=str,
        default=",".join(known_dataset_names()),
        help="Comma-separated dataset names. Default: all five.",
    )
    parser.add_argument(
        "--math-verify-n-samples",
        type=int,
        default=None,
        help="Override per-dataset n_samples. Default: registry per-dataset.",
    )
    parser.add_argument(
        "--math-verify-temperature",
        type=float,
        default=None,
        help="Override per-dataset sampling temperature.",
    )
    parser.add_argument(
        "--math-verify-max-tokens",
        type=int,
        default=2048,
        help="Max tokens per generation. Default: 2048.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="boxed.prompt",
        help="Path to the prompt template. Default: boxed.prompt.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM max_model_len. Default: 2048.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.4,
        help="vLLM gpu_memory_utilization. Default: 0.4.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Where to write eval_results.json. Default: "
            "{checkpoint}/eval_results.json for local paths, "
            "./eval_results.json for HF IDs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    args = parser.parse_args(argv)
    args.math_verify_datasets = [
        s.strip() for s in args.math_verify_datasets.split(",") if s.strip()
    ]
    return args


def validate_args(args):
    unknown = [d for d in args.math_verify_datasets if d not in REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown --math-verify-datasets entries: {unknown}. "
            f"Known names: {sorted(REGISTRY.keys())}"
        )
    if args.math_verify_n_samples is not None and args.math_verify_n_samples <= 0:
        raise ValueError("--math-verify-n-samples must be > 0")
    if args.math_verify_max_tokens <= 0:
        raise ValueError("--math-verify-max-tokens must be > 0")


def preflight_checkpoint(path: str) -> None:
    """Reject legacy adapter-only / factored checkpoints with a clear message.

    HF model IDs and plain HF directories pass through.
    """
    if not os.path.isdir(path):
        return  # HF ID or unknown path → defer to from_pretrained

    if os.path.exists(os.path.join(path, "adapter_config.json")):
        raise ValueError(
            f"Checkpoint at {path} is a legacy adapter-only checkpoint "
            f"(found adapter_config.json). eval_rl.py only supports merged "
            f"checkpoints; re-run training with --enable-merged-ckpt true "
            f"or use the in-loop --enable-math-verify path."
        )

    safetensors_path = os.path.join(path, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt") as f:
            for key in f.keys():
                if any(
                    marker in key
                    for marker in (".btt_l", ".btt_r", ".btt_s", ".svd_a", ".svd_b", ".svd_s")
                ):
                    raise ValueError(
                        f"Checkpoint at {path} is a legacy factored checkpoint "
                        f"(found key {key!r}). eval_rl.py only supports merged "
                        f"checkpoints; re-run training with --enable-merged-ckpt true "
                        f"or use the in-loop --enable-math-verify path."
                    )


def default_output_json_path(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return os.path.join(checkpoint, "eval_results.json")
    return os.path.join(os.getcwd(), "eval_results.json")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    preflight_checkpoint(args.checkpoint)

    output_json = args.output_json or default_output_json_path(args.checkpoint)

    import random

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model: {args.checkpoint}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Running math-verify on: {args.math_verify_datasets}")
    results = math_verify_eval(
        model=None,
        tokenizer=tokenizer,
        datasets=args.math_verify_datasets,
        n_samples_override=args.math_verify_n_samples,
        temperature_override=args.math_verify_temperature,
        max_tokens=args.math_verify_max_tokens,
        prompt_template_path=args.prompt_template,
        vllm_kwargs={
            "model": args.checkpoint,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": 4096,
        },
    )
    results["checkpoint"] = args.checkpoint
    results["model_id_at_train_time"] = args.checkpoint  # best-effort

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output_json}")

    print("Math-verify results:")
    for ds_name, ds_result in results["datasets"].items():
        print(
            f"  {ds_name}: {ds_result['accuracy']:.2%} "
            f"({ds_result['n_correct']}/{ds_result['n_total']})"
        )
    if results.get("errors"):
        print("Errors:")
        for ds_name, reason in results["errors"].items():
            print(f"  {ds_name}: {reason}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_eval_rl_cli -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add eval_rl.py tests/test_eval_rl_cli.py
git commit -m "eval: add eval_rl.py standalone entrypoint"
```

---

## Task 9: Update existing shell scripts to opt out of new defaults

**Files:**
- Modify: `run_rl.sh`, `run_temp.sh` (and any other `*.sh` invoking `run_rl.py`).

The new flags default on, but smoke / debug shell scripts should not unintentionally trigger 5-dataset post-training eval.

- [ ] **Step 1: List affected shell scripts**

Run: `grep -l 'run_rl.py' *.sh 2>/dev/null`
Expected: a list of scripts. Note each one.

- [ ] **Step 2: Inspect each script and decide intent**

For each script, decide: is it a "real" training run (eval should run) or a "smoke / debug" run (eval should be skipped)?

Run: `cat run_rl.sh` (and each other script).

The judgment call: scripts with `--n-grpo-steps` ≥ 30 are real runs; scripts with `--n-grpo-steps` ≤ 5 (or `--stop-after-first-step`) are smoke runs.

- [ ] **Step 3: For smoke / debug scripts, add `--no-enable-math-verify --no-enable-merged-ckpt`**

For each smoke / debug script identified in Step 2, add the two flags to the `run_rl.py` invocation. Real training runs are left unchanged.

Example diff for a smoke script:

```diff
-CUDA_VISIBLE_DEVICES=0 uv run run_rl.py --train-mode lora --n-grpo-steps 1 --no-wandb
+CUDA_VISIBLE_DEVICES=0 uv run run_rl.py --train-mode lora --n-grpo-steps 1 --no-wandb \
+    --no-enable-math-verify --no-enable-merged-ckpt
```

- [ ] **Step 4: Commit**

```bash
git add run_rl.sh run_temp.sh
git commit -m "scripts: opt smoke/debug runs out of math-verify and merged-ckpt"
```

(Adjust the `git add` arguments to whatever scripts you actually changed.)

---

## Task 10: Run the full test suite to confirm no regressions

**Files:** none (verification step).

- [ ] **Step 1: Run the entire test suite**

Run: `uv run python -m unittest discover tests -v`
Expected: all tests PASS, only `TestLoadEvalDatasetHFRoundTrip` SKIPPED (network-gated).

- [ ] **Step 2: If any pre-existing tests now fail, investigate before proceeding**

The two new defaults are designed to be non-breaking for argparse smoke tests, but if a test invokes `main` past parse_args without `--no-enable-math-verify`, it may now try to run vLLM. Add the opt-out flag to such tests.

- [ ] **Step 3: Run the syntax check**

Run: `uv run python -m py_compile run_rl.py eval_rl.py math_verify_eval.py eval_datasets.py`
Expected: no output (no syntax errors).

- [ ] **Step 4: Commit any test fixes**

If you needed to update tests in Step 2:

```bash
git add tests/
git commit -m "tests: add --no-enable-math-verify to existing tests that go past parse_args"
```

---

## Task 11: Manual smoke validation (not automated)

**Files:** none (smoke runs).

These manual runs validate the end-to-end behavior. They require a GPU and HF Hub access. The user should run them and report results; do not mark this task complete until they have.

- [ ] **Step 1: Smoke-test merged-ckpt + eval for `lora` mode**

Run: `CUDA_VISIBLE_DEVICES=0 uv run run_rl.py --train-mode lora --lr 1e-4 --lora-rank 1 --trainable-type all --n-grpo-steps 1 --no-wandb --math-verify-datasets MATH-500`

Expected:
1. Training completes 1 GRPO step.
2. Final merged checkpoint is written to `<run_dir>/step=1/` containing `config.json`, `model.safetensors`, `tokenizer.json` etc. **Not** `adapter_config.json`.
3. `<run_dir>/step=1/eval_results.json` is written with a non-zero `MATH-500.accuracy` figure.
4. Console prints `MATH-500: <pct> (n_correct/500)`.

- [ ] **Step 2: Smoke-test base-model eval via `eval_rl.py`**

Run: `CUDA_VISIBLE_DEVICES=0 uv run eval_rl.py --checkpoint Qwen/Qwen3-1.7B --math-verify-datasets MATH-500 --gpu-memory-utilization 0.7`

Expected:
1. Base-model eval succeeds.
2. `./eval_results.json` is written.
3. `MATH-500.accuracy` matches roughly the published Qwen3-1.7B baseline.

- [ ] **Step 3: Smoke-test merged-ckpt for `blocktt` mode**

Run: `CUDA_VISIBLE_DEVICES=0 uv run run_rl.py --train-mode blocktt --lr 1e-4 --trainable-type all --decomp-mode input_one_block --train-position small --n-grpo-steps 1 --no-wandb --math-verify-datasets MATH-500`

Expected:
1. Final checkpoint at `<run_dir>/step=1/` contains only `nn.Linear`-style weights — `safetensors` keys do NOT include `.btt_l` / `.btt_r`.
2. `eval_results.json` written.

Verification command after the run:

```bash
uv run python -c "from safetensors import safe_open; \
  f = safe_open('<run_dir>/step=1/model.safetensors', framework='pt'); \
  print('factored keys:', [k for k in f.keys() if '.btt' in k])"
```

Expected output: `factored keys: []`.

- [ ] **Step 4: Smoke-test merged-ckpt for `svd` mode**

Run: `CUDA_VISIBLE_DEVICES=0 uv run run_rl.py --train-mode svd --lr 1e-4 --trainable-type all --train-position output --n-grpo-steps 1 --no-wandb --math-verify-datasets MATH-500`

Expected: same as Task 11 Step 3, with `.svd_a` / `.svd_b` absent from the checkpoint.

- [ ] **Step 5: Smoke-test the `--no-enable-merged-ckpt` escape hatch**

Run: `CUDA_VISIBLE_DEVICES=0 uv run run_rl.py --train-mode lora --lr 1e-4 --lora-rank 1 --trainable-type all --n-grpo-steps 1 --no-wandb --no-enable-merged-ckpt --no-enable-math-verify --enable-save-ckpt`

Expected: checkpoint saved as today's adapter-only format (`adapter_config.json` present, no full `model.safetensors`).

- [ ] **Step 6: If all smoke runs pass, you're done with this plan**

No further commits. Report results to the user.

---

## Self-Review

Spec coverage check:

- §1 goal 1 (`--enable-merged-ckpt` default true, plain HF format, escape hatch via `--no-...`): Tasks 4, 5, 11 step 5. ✓
- §1 goal 2 (`--enable-math-verify` default true, post-loop eval, wandb + `eval_results.json`): Tasks 4, 7. ✓
- §1 goal 3 (`eval_rl.py` standalone, supports HF IDs, no legacy support): Task 8. ✓
- §2 four-file architecture (`eval_datasets.py`, `math_verify_eval.py`, `eval_rl.py`, `run_rl.py` modifications): Tasks 2, 3, 8, 4-7. ✓
- §3 vLLM reuse logic (in-process hot-swap vs HTTP-lora fresh load): Task 7 step 3 covers both branches. ✓
- §3 `lora_adapters/` subdir for rollout adapters: Task 6. ✓
- §3 `eval_results.json` schema: Task 7 (in-loop) + Task 8 (standalone). ✓
- §4 tier-1 vs tier-2 failure handling: Task 7 (try/except around the whole eval block = tier 1; per-dataset try/except inside `math_verify_eval` = tier 2). ✓
- §4 unparseable / grader_errors counters: Task 3 step 3 (`_verify_one`). ✓
- §4 `eval_rl.py` pre-flight legacy detection: Task 8 step 3 (`preflight_checkpoint`). ✓
- §4 boundary checks (unknown dataset names, n_samples=0, max_tokens<=0): Task 4 step 5. ✓
- §4 default-on impact on existing scripts/tests: Task 9, Task 10 step 2. ✓
- §5 all five test files: Tasks 2, 3, 4, 5, 8. ✓
- §5 `math-verify>=0.5` dependency: Task 1. ✓
- §5 manual validation steps (3 smoke runs): Task 11. ✓

Placeholder scan: searched plan for "TBD", "TODO", "implement later", "fill in details", "appropriate", "similar to". None found in instruction text (only in test docstrings, which is intentional).

Type / signature consistency: `math_verify_eval` signature is the same in math_verify_eval.py (Task 3), in run_rl.py's call site (Task 7), and in eval_rl.py (Task 8). `save_merged_checkpoint` signature is the same in Tasks 5, 7, 11. `preflight_checkpoint` is defined and called only in Task 8.
