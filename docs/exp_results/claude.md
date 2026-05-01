# RL Results Collection Notes

_Updated from run artifacts on 2026-04-22 (post bug-fix + LR sweep)._

## Result Snapshot (primary eval/accuracy; best per method)

- fura (blocktt): **0.895** (`blocktt-adamw-lr_1e-5-output_one_block-s_to_keep-train_both-0317-155422`)
- svd: **0.891** (`svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139`)
- full: **0.886** (`full-adamw-lr_2e-5-0325-215533`)
- dora: **0.871** (`dora-adamw-lr_2e-4-rank_64-sweep-0422-021100`)
- milora: **0.856** (`milora-adamw-lr_6e-5-rank_64-sweep-0422-020941`)
- lora_full: **0.856** (`lora_full-adamw-lr_1e-5-rank_64-0319-140945`)
- lift: **0.849** (`lift-adamw-lr_6e-5-sweep-0422-031308`)
- lora: **0.848** (`lora-adamw-lr_6e-5-rank_64-sweep-0422-002231`; after clone-fix)
- randlora: **0.856** (`randlora-adamw-lr_1e-4-rank_64-sweep-0422-041850`)
- pissa: **0.802** (`pissa-adamw-lr_6e-5-rank_64-sweep-0422-032327`)

## Result Snapshot (mean of extended evals MATH-500/AMC23/AIME-24/AIME-25/Minerva)

- full: 35.1 — `full-adamw-lr_2e-5-0420`
- fura: 33.5 (5-metric) — `blocktt-lr_8e-5-output_one_block-s_to_trainable-train_small-sweep-0423` (MATH 63.0, AMC 52.5, AIME-24 13.3, AIME-25 17.5, Minerva 21.0)
- randlora: 34.8 — `randlora-lr_1e-4-rank_64-sweep-0422`
- svd: 31.0 — `svd-lr_1e-5-s_to_keep_trainable-train_input-ext-0421`
- lora: 30.4 — `lora-lr_6e-5-rank_64-sweep-0422`
- dora: 29.7 — `dora-lr_1e-4-rank_64-retry-0422`
- lift: 29.1 — `lift-lr_8e-5-0421`
- milora: 28.1 — `milora-lr_6e-5-rank_64-sweep-0422`
- pissa: 25.4 — `pissa-lr_8e-5-rank_64-0421`

## Bug fixes landed in this session

1. **LoRA regression** (`run_rl.py:1038-1061`): the inner `export_lora_merged_weights` returned live `base_layer.weight` references, which PEFT's `unmerge_adapter()` then rolled back in place before vLLM read them. vLLM was therefore serving the frozen base model every step. Fix: extract into top-level `export_lora_merged_weights_for_vllm(model)` and `.detach().clone()` each tensor before the `finally:` unmerge. Re-used by both the in-loop rollout (`build_lora_local_generators`, `run_rl.py:1143`) and the post-training math-verify hook (`run_rl.py:2088-2093`). DoRA happened to avoid this bug because PEFT's DoRA swaps `.data` to a fresh tensor on unmerge rather than mutating in place.

2. **Minerva dataset ID** (`eval_datasets.py:61`): `math-ai/minerva-math` is gated / removed; the community mirror `math-ai/minervamath` has the same 272-example Minerva Math test split and loads cleanly. Fields (`question`, `answer`) and greedy@1 sampling unchanged.

3. **RandLoRA save-and-eval** (`run_rl.py:903-919`, `2070-2090`): PEFT's RandLoRA shares the `randlora_A` random projection across layers, and safetensors refuses to serialize shared storages. `save_merged_checkpoint` now passes `safe_serialization=False` for `randlora` (falls back to pytorch_model.bin); downstream eval via `AutoModelForCausalLM.from_pretrained` handles either format. The final-checkpoint save at the end of `main()` is also wrapped in `try/except` so that a save failure cannot block the post-training math-verify hook (which uses the in-memory model, not the on-disk checkpoint).

## How Results Were Collected

1. Enumerate summaries: find all `wandb-summary.json` files under `/data/yequan/fura/rl_runs`.
2. Exclude debug experiments: skip any path whose first directory is `debug`.
3. Parse final metrics directly from each summary JSON:
   - Primary: `eval/accuracy`
   - Extended: `eval/MATH-500/accuracy`, `eval/AMC23/accuracy`, `eval/AIME-24/accuracy`, `eval/AIME-25/accuracy`, `eval/Minerva/accuracy`
   - Supporting: `train/accuracy`, `_runtime`, `_step`
4. Mark runs as incomplete when `eval/accuracy` is absent.
5. Build:
   - Group-level aggregates (runs / best / mean primary accuracy)
   - Best extended-eval row per method, ranked by unweighted mean of available extended metrics (missing metrics are skipped rather than counted as 0; this avoids penalising older runs that ran before the Minerva fix)
   - Per-run table sorted by MATH-500 descending

## Repro Command

```sh
uv run python tools/collect_rl_results.py --root /data/yequan/fura/rl_runs
```

# SFT Results Collection Notes

_Updated from run artifacts on 2026-04-15._

## Result Snapshot

- Source roots:
  - `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B`
  - `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B`
- Current best in published SFT result docs:
  - Math (`docs/exp_results/lift_math.md`): `output_one_block + smerge_trainable`, avg **80.47**
  - Commonsense (`docs/exp_results/lift_commonsense.md`): `output_one_block + smerge_keep_trainable`, avg **87.91**

## How SFT Results Were Collected

1. Enumerate candidate run directories under the math and commonsense roots above.
2. Read each task-level `eval.log` file.
3. Extract the final metric from lines matching `Result <value>`.
4. Compute per-run average over task sets:
   - Math tasks: `MultiArith`, `gsm8k`, `AddSub`, `AQuA`, `SingleEq`, `SVAMP`, `mawps`
   - Commonsense tasks: `boolq`, `piqa`, `social_i_qa`, `hellaswag`, `winogrande`, `ARC-Easy`, `ARC-Challenge`, `openbookqa`
5. Exclude runs with missing task logs or missing `Result` values from the main tables, and list them separately as incomplete.

## Repro Command (SFT)

```sh
python - <<'PY'
import re
from pathlib import Path

pat=re.compile(r'Result\\s+([0-9]+(?:\\.[0-9]+)?)')

def get_result(path):
    if not path.exists():
        return None
    m=None
    for m in pat.finditer(path.read_text(errors='ignore')):
        pass
    return float(m.group(1)) if m else None

def collect(root, subdir, tasks):
    rows=[]
    for run in sorted([p for p in Path(root).iterdir() if p.is_dir()]):
        vals=[get_result(run/subdir/t/'eval.log') for t in tasks]
        if all(v is not None for v in vals):
            rows.append((run.name, sum(vals)/len(vals)))
    return sorted(rows, key=lambda x: x[1], reverse=True)

math=collect('/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B','math',
             ['MultiArith','gsm8k','AddSub','AQuA','SingleEq','SVAMP','mawps'])
cs=collect('/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B','commonsense',
           ['boolq','piqa','social_i_qa','hellaswag','winogrande','ARC-Easy','ARC-Challenge','openbookqa'])
print('MATH:', math[:5])
print('COMMONSENSE:', cs[:5])
PY
```
