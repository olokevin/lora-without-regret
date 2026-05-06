# Running Workloads on NERSC Perlmutter — Reusable Workflow

This guide captures the pattern that worked for the DoRA + BlockTT port
(`ref/DoRA/`) so the same scaffolding can be reused for new training work on
Perlmutter. The companion DoRA-specific docs are
`ref/DoRA/SETUP_GUIDE.md` (generic install) and `ref/DoRA/README_NERSC.md`
(Perlmutter wiring).

---

## 0. The workflow in one picture

```
LOGIN NODE                 INTERACTIVE NODE              BATCH QUEUE              LOGIN NODE
(no GPUs, has internet)    (salloc, GPUs, no internet)   (sbatch, GPUs,           (poll & sync
                                                          no internet)             metrics back)
─────────────                ──────────────────────       ──────────────────       ─────────────
1. Activate env  ────►       3. Smoke test (20 steps) ──► 4. sbatch real job ───►  5. wandb_auto_sync
2. Download data                save → reload → eval         48h DoRA / BTT          tails offline runs
   (background,                  one mini batch                                       and uploads them
   idempotent)                  on real GPUs                                          to wandb cloud
```

**Golden rule:** never go from "code change" straight to a multi-hour `sbatch`.
Always do a 20-step smoke on an interactive node first. The interactive queue
gives you a 4×A100 node in <5 min and surfaces 90 % of the failures (HF cache
missing, NCCL setup wrong, save path bug, OOM at this batch size, etc.) for
near-zero cost.

---

## 1. Perlmutter ground rules (read once, save pain forever)

| Rule | Why |
|---|---|
| Put **everything** under `$DATA_ROOT=/pscratch/sd/l/<user>/...` | `$HOME` is 40 GiB; conda + HF cache will explode it. |
| `/pscratch` is **purged after ~8 weeks idle** | Copy keeper checkpoints to `$HOME` or CFS before you leave. |
| Compute nodes have **no outbound internet** | All HF / git / pip / wandb traffic must happen on login. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `WANDB_MODE=offline` in jobs. |
| Slingshot, not InfiniBand | NCCL needs `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=hsn`. |
| Submit GPU jobs with `--account=<proj>_g --constraint=gpu` | CPU-only flag wastes node hours and gets rejected. |
| One Perlmutter GPU node = **4× A100 40 GB** | Recipes tuned for 8× GPUs need `gradient_accumulation_steps` doubled. |
| Multi-rank caches collide on Lustre | Per-job `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `XDG_CACHE_HOME` under `$TMPDIR`. |

---

## 2. One-time setup (login node)

### 2.1 Pick paths and write `env.sh`

```bash
export DATA_ROOT=/pscratch/sd/l/$USER/<project>
export REPO=/global/homes/l/$USER/<repo-checkout>
mkdir -p $DATA_ROOT/{hf_cache,torch_cache,pip_cache,conda_pkgs,logs,wandb,checkpoints,datasets}
```

Write `$DATA_ROOT/env.sh` (one-liner activation for every shell / job):

```bash
#!/usr/bin/env bash
export DATA_ROOT=/pscratch/sd/l/<user>/<project>
export REPO=/global/homes/l/<user>/<repo-checkout>

source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh

export HF_HOME=$DATA_ROOT/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$DATA_ROOT/torch_cache
export PIP_CACHE_DIR=$DATA_ROOT/pip_cache
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR"

# CUDA toolkit for any source builds (flash-attn etc.). Caller still has to
# `module load cudatoolkit/<ver>` before compiling.
[[ -d /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9 ]] && \
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
```

### 2.2 Build the conda env on scratch

```bash
source $DATA_ROOT/env.sh
conda create -p $DATA_ROOT/envs/<name> python=3.10 -y
conda activate $DATA_ROOT/envs/<name>
# pip install ...
```

For source builds like `flash-attn`, `module load cudatoolkit/12.9` before
`pip install` and use `--no-cache-dir` (pip's tmp dir and `$PIP_CACHE_DIR`
sit on different filesystems → cross-device link error).

### 2.3 Download data on the login node

Login nodes have outbound internet; compute nodes do not. Make the download
script **idempotent** (`wget -c`, HuggingFace hub which resumes by hash) so
you can rerun freely. Launch it in the background and forget:

```bash
nohup bash $DATA_ROOT/download_all.sh \
    >$DATA_ROOT/logs/dl.log 2>&1 &
disown
tail -f $DATA_ROOT/logs/dl.log
```

### 2.4 Pre-warm the HuggingFace cache

Anything `from_pretrained(...)` is going to fetch — pull it once on login:

```bash
python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5'); \
    AutoModel.from_pretrained('lmsys/vicuna-7b-v1.5')"
```

If you skip this, the first compute-node job dies with a confusing offline
error.

---

## 3. Smoke test on an interactive node

Goal: in **30 min or less**, prove that the env + data + code path actually
runs end-to-end on a real Perlmutter GPU node.

### 3.1 Get an interactive 4-GPU node

```bash
salloc --account=<proj>_g --constraint=gpu --qos=interactive \
       --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 --time=0:30:00
```

You land on a real GPU node with the same compute-node restrictions
(no internet) — exactly the environment your batch job will see.

### 3.2 Run a smoke script

Pattern (see `/pscratch/sd/l/liyantan/dora/smoke_train.sh` and `smoke_btt_train.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail
source $DATA_ROOT/env.sh
conda activate $DATA_ROOT/envs/<name>
module load cudatoolkit/12.4 2>/dev/null || true

# Same offline + NCCL knobs as the real sbatch — match what you'll run later.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_PROJECT=<your-project>
export WANDB_DIR=$DATA_ROOT/wandb
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
export OMP_NUM_THREADS=4

RUN_NAME=<task>-smoke-${SLURM_JOB_ID:-local}
OUTPUT_DIR=$DATA_ROOT/checkpoints/$RUN_NAME

# 20 steps, smaller batch, force a save at the end so we can validate the
# materialize/save path matches the real run.
deepspeed --num_gpus=4 <train_entry.py> \
    ... \
    --output_dir $OUTPUT_DIR \
    --max_steps 20 \
    --save_strategy steps --save_steps 20 --save_total_limit 1 \
    --run_name $RUN_NAME --report_to wandb
```

Kick it off inside the `salloc` shell:

```bash
bash $DATA_ROOT/smoke_train.sh 2>&1 | tee $DATA_ROOT/logs/smoke-$(date +%s).log
```

### 3.3 Smoke success checklist

Before submitting the real job, all of these must be true:

- [ ] `nvidia-smi -L` shows 4 GPUs in the smoke log
- [ ] Loss decreases over 20 steps
- [ ] Checkpoint written to `$OUTPUT_DIR/checkpoint-20/` (or equivalent)
- [ ] If your code transforms weights at save (e.g. BTT → dense), reload the
      checkpoint and run a single-batch eval (`smoke_eval_pope.sh` style)
- [ ] `$WANDB_DIR/wandb/offline-run-*` exists for that smoke run
- [ ] No NCCL warnings, no `pkg_resources` warnings, no "tried to fetch from
      hub" errors

If any item fails, fix it on the interactive node — much cheaper to debug here
than after waiting in the regular queue.

---

## 4. Submit the real job to the batch queue

### 4.1 Shared SLURM preamble — `slurm/_common.sh`

Same source-once helper for every sbatch:

```bash
#!/usr/bin/env bash
set -euo pipefail
source $DATA_ROOT/env.sh
module load gpu          2>/dev/null || true
module load nccl         2>/dev/null || true
module load cudatoolkit/12.4 2>/dev/null || true

echo "================================================================"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-interactive}"
echo "GPUs/node: ${SLURM_GPUS_PER_NODE:-?}"
echo "DATA_ROOT=$DATA_ROOT"
echo "================================================================"
nvidia-smi || true
echo "================================================================"
```

### 4.2 Training sbatch template

```bash
#!/usr/bin/env bash
#SBATCH --job-name=<task>_train
#SBATCH --account=<proj>_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --output=/pscratch/sd/l/<user>/<project>/logs/slurm-%x-%j.out

source $REPO/slurm/_common.sh
conda activate $DATA_ROOT/envs/<name>
cd $REPO/<project_dir>

# ---- offline mode — compute nodes have no internet ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- wandb offline — sync from login node after the job runs ----
export WANDB_MODE=offline
export WANDB_PROJECT=<your-project>
export WANDB_DIR=$DATA_ROOT/wandb
export WANDB_NAME=<run-name>-${SLURM_JOB_ID}
mkdir -p $WANDB_DIR

# ---- NCCL: Slingshot-friendly ----
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

# ---- CPU threading ----
export OMP_NUM_THREADS=4

# ---- Per-job cache isolation on node-local scratch ----
LOCAL_SCRATCH="${TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="$LOCAL_SCRATCH/$USER/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_SCRATCH/$USER/torchinductor_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$LOCAL_SCRATCH/$USER/xdg_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash ./<train_recipe>.sh

echo "To upload metrics to wandb, run on login node:"
echo "  wandb sync $WANDB_DIR/wandb/offline-run-*"
```

Submit & watch:

```bash
sbatch $REPO/slurm/<task>_train.sbatch
squeue -u $USER
tail -f $DATA_ROOT/logs/slurm-<task>_train-<jobid>.out
```

### 4.3 Eval sbatch template

Same skeleton, shorter wall time (`--time=08:00:00`), and runs your eval
shell against the produced checkpoint. Submit it after training finishes
**and** the checkpoint directory exists.

---

## 5. wandb sync (login node, background poll)

Compute nodes can't reach wandb, so the sbatch dumps `offline-run-*`
directories under `$WANDB_DIR/wandb/`. A single background poller on the
login node uploads them as they appear, and is incremental — safe to leave
running for weeks.

`$DATA_ROOT/wandb_auto_sync.sh`:

```bash
#!/usr/bin/env bash
# Poll every 5 min, sync every offline-run-*. Idempotent.
set -u
source $DATA_ROOT/env.sh
conda activate $DATA_ROOT/envs/<name>
cd $DATA_ROOT/wandb/wandb

while true; do
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    for d in offline-run-*; do
        [[ -d "$d" ]] || continue
        echo "[$ts] sync $d"
        wandb sync --project <your-project> "$d" 2>&1 | grep -E 'done|error|Error' | tail -3
    done
    sleep 300
done
```

Launch it once on the login node and disown:

```bash
nohup bash $DATA_ROOT/wandb_auto_sync.sh \
    >$DATA_ROOT/logs/wandb_sync.log 2>&1 &
disown
```

Verify with `tail -f $DATA_ROOT/logs/wandb_sync.log`. Run jobs reach the wandb
UI within ~5 min of producing offline data.

**Don't have wandb credentials in the env?** `wandb login` once on the login
node — it writes `~/.netrc` and is picked up by the auto-sync script.

---

## 6. After the job finishes

1. **Pull metrics** — confirm the smoke run + the real run are both in wandb.
2. **Copy keeper checkpoints off scratch** — `cp -r` to `$HOME` or CFS.
   Scratch is purged after ~8 weeks idle.
3. **Tag in git** — record the commit that produced the run, and put the
   wandb run URL in the commit message or a `RUNS.md` log.
4. `$TMPDIR` caches are auto-removed at job end, no action needed.

---

## 7. Reusable skeleton for a new task

```
$DATA_ROOT/
├── env.sh                       # source-once: paths, conda, HF cache
├── envs/<name>/                 # conda env
├── hf_cache/ torch_cache/       # caches on scratch
├── datasets/                    # downloaded inputs
├── checkpoints/                 # training outputs
├── eval_result/                 # eval outputs
├── wandb/                       # offline-run-* dirs
├── logs/                        # slurm + download + wandb_sync logs
├── download_<task>.sh           # idempotent data fetch (login node)
├── smoke_<task>_train.sh        # 20-step smoke (interactive node)
├── smoke_<task>_eval.sh         # 1-batch eval smoke
└── wandb_auto_sync.sh           # 5-min poll uploader

$REPO/
├── <project_dir>/...            # the actual training code
├── <task>_train.sh              # full-run training recipe
├── <task>_eval.sh               # full eval recipe
└── slurm/
    ├── _common.sh               # shared preamble
    ├── <task>_train.sbatch
    └── <task>_eval.sbatch
```

End-to-end checklist for a new task on this skeleton:

1. Create `$DATA_ROOT/<new-task>/` and a fresh `env.sh` if env differs.
2. Build env on login node; pre-warm HF cache for any new model IDs.
3. Write idempotent `download_<task>.sh`; run as `nohup ... &` on login node.
4. Copy `smoke_<existing>_train.sh` → `smoke_<task>_train.sh`, edit hparams,
   `salloc` and run. Confirm the checklist in §3.3.
5. Copy `slurm/<existing>_train.sbatch` → `slurm/<task>_train.sbatch`, edit
   the recipe call, `sbatch` it.
6. Make sure `wandb_auto_sync.sh` is running in the background (check
   `pgrep -af wandb_auto_sync`).
7. After training: submit eval sbatch, then archive ckpt off scratch.

---

## 8. Troubleshooting crib sheet

| Symptom | Fix |
|---|---|
| `conda: command not found` | `source $DATA_ROOT/env.sh` first. |
| `Connection refused` / `Couldn't reach huggingface.co` from a job | Compute node has no internet. Pre-warm HF cache on login; set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. |
| `flash-attn` import fails after a torch upgrade | `module load cudatoolkit/12.9` and rebuild flash-attn with `--no-cache-dir`. |
| `pkg_resources` not found during flash-attn build | `pip install "setuptools<70"`. |
| Cross-device link error during `pip install` | Add `--no-cache-dir`. |
| NCCL hangs at startup | Need `NCCL_IB_DISABLE=1` and `NCCL_SOCKET_IFNAME=hsn`. |
| Triton / inductor "file exists" / "permission denied" with multi-rank | Per-job `TRITON_CACHE_DIR` etc. under `$TMPDIR`, not on Lustre. |
| OOM on 4× A100 40 GB with an 8-GPU recipe | Double `gradient_accumulation_steps` to keep effective batch size. |
| wandb run isn't on the dashboard | Check `pgrep -af wandb_auto_sync`; manually `wandb sync $WANDB_DIR/wandb/offline-run-<id>`. |
| `Job cost estimated … balance is 0.00 node hours` for a CPU job | Use `--account=<proj>_g --constraint=gpu` for GPU nodes; do CPU-only work on the login node instead. |
| Scratch files vanished | Past the 8-week purge. Restore from `$HOME` / CFS. Always copy keepers out. |

---

## 9. Worked example — DoRA + BlockTT (this repo)

| Concern | Where it lives |
|---|---|
| `env.sh` | `/pscratch/sd/l/liyantan/dora/env.sh` |
| Download script | `/pscratch/sd/l/liyantan/dora/download_all.sh` |
| Smoke (DoRA) | `/pscratch/sd/l/liyantan/dora/smoke_train.sh`, `smoke_eval_pope.sh` |
| Smoke (BTT) | `/pscratch/sd/l/liyantan/dora/smoke_btt_train.sh`, `smoke_btt_eval_pope.sh` |
| Train recipe (DoRA) | `ref/DoRA/visual_instruction_tuning/Dora_7b.sh` |
| Train recipe (BTT) | `ref/DoRA/visual_instruction_tuning/BTT_7b.sh` |
| Eval recipe (BTT) | `ref/DoRA/visual_instruction_tuning/7B_eval_btt.sh` |
| BTT integration | `ref/DoRA/visual_instruction_tuning/llava/train/{train_btt.py,train_mem_btt.py}` |
| sbatch (preamble + 4 jobs) | `ref/DoRA/slurm/{_common.sh, llava_train.sbatch, llava_btt_train.sbatch, llava_eval.sbatch, llava_btt_eval.sbatch}` |
| wandb auto-sync | `/pscratch/sd/l/liyantan/dora/wandb_auto_sync.sh` |
| Project-specific docs | `ref/DoRA/SETUP_GUIDE.md`, `ref/DoRA/README_NERSC.md` |

For the next workload, copy this layout and adapt: new `env.sh` (or reuse),
new `download_*.sh`, new `smoke_*.sh`, new pair of sbatch files. Everything
else (offline mode, NCCL flags, cache isolation, wandb sync) is identical.
