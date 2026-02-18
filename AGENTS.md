# Repository Guidelines

## Project Structure & Module Organization
This repository is a compact Python research codebase for reproducing LoRA vs full fine-tuning results.

- Top-level training scripts:
  - `run_sft.py` for supervised fine-tuning (SFT) with `--train-mode full|lora|blocktt`
  - `rl_full.py`, `rl_lora.py` for GRPO-style reinforcement learning (RL)
- Shared utilities: `math_utils.py` (boxed-answer extraction and equivalence checks)
- Prompt template: `boxed.prompt`
- Experiment artifacts:
  - `figures/` for generated plots
  - `results/` for exported SQLite results and schema notes
- Dependency/config files: `pyproject.toml`, `uv.lock`

Keep new training or analysis scripts at repo root unless a clear module split is introduced.

## Build, Test, and Development Commands
- `uv sync`: install and lock project dependencies.
- `uv run run_sft.py --train-mode lora --lr 2e-4 --lora-rank 1 --lora-type all --no-wandb`: run a baseline LoRA SFT experiment.
- `uv run run_sft.py --train-mode full --lr 2.5e-5 --no-wandb`: run full SFT baseline.
- `uv run rl_lora.py --lr 1e-4 --lora_r 1 --disable_wandb`: run LoRA RL (requires local vLLM server).
- `uv run rl_full.py --lr 1e-5 --disable_wandb`: run full RL baseline.
- `python -m py_compile *.py`: quick syntax sanity check across scripts.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and PEP 8-aligned formatting.
- Prefer `snake_case` for variables/functions/CLI flags, `UPPER_SNAKE_CASE` for constants.
- Match existing argparse style: explicit flags, defaults, and help text.
- Keep scripts runnable as CLIs via `if __name__ == "__main__":`.

No formatter/linter is currently enforced in repo config; keep changes stylistically consistent with nearby code.

## Testing Guidelines
There is no dedicated `tests/` directory yet. For contributions:

- Run `python -m py_compile *.py` before opening a PR.
- Smoke-test the touched script with a minimal run (typically with `--no-wandb` / `--disable_wandb`).
- When changing reward/data logic, validate against small dataset slices first.

If adding tests, use `tests/test_<module>.py` naming and focus on deterministic utility logic (`math_utils.py`) before long-running training paths.

## Commit & Pull Request Guidelines
- Keep commit messages short and imperative (history examples: `add results, edit copy`, `remove main.py`).
- Make one logical change per commit when possible.
- PRs should include:
  - what changed and why,
  - exact command(s) used for validation,
  - any environment assumptions (GPU, vLLM, model IDs),
  - updated figures/results references if outputs changed.
