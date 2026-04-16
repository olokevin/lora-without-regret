# RL Results Collection Notes

_Updated from run artifacts on 2026-04-15._

## Result Snapshot

- Best overall eval/accuracy: **0.895** (`blocktt` / `blocktt-adamw-lr_1e-5-output_one_block-s_to_keep-train_both-0317-155422`).
- Best `blocktt`: **0.895** (`blocktt-adamw-lr_1e-5-output_one_block-s_to_keep-train_both-0317-155422`).
- Best `svd`: **0.891** (`svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139`).
- Best `full`: **0.886** (`full-adamw-lr_2e-5-0325-215533`).
- Best `lora_full`: **0.856** (`lora_full-adamw-lr_1e-5-rank_64-0319-140945`).

## How Results Were Collected

1. Enumerate summaries: find all `wandb-summary.json` files under `/data/yequan/fura/rl_runs`.
2. Exclude debug experiments: skip any path whose first directory is `debug`.
3. Parse final metrics directly from each summary JSON:
   - Primary: `eval/accuracy`
   - Supporting: `train/accuracy`, `train/reward_mean`, `train/approx_kl`, `_runtime`, `_step`
4. Mark runs as incomplete when `eval/accuracy` is absent.
5. Build:
   - Group-level aggregates (runs/completed/best/mean accuracy)
   - Per-run table sorted by `eval/accuracy` descending

## Repro Command

```sh
python - <<'PY'
import json
from pathlib import Path
root=Path('/data/yequan/fura/rl_runs')
for p in root.glob('**/wandb-summary.json'):
    rel=p.relative_to(root)
    if rel.parts[0]=='debug':
        continue
    d=json.loads(p.read_text())
    print(rel.parts[0], rel.parts[1], d.get('eval/accuracy'))
PY
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
