# DoRA on NERSC Perlmutter — How to Run

Companion to `SETUP_GUIDE.md`. Captures the Perlmutter-specific wiring:
scratch layout, conda module, SLURM queues, and the small code tweaks the
install needed.

> **Scope.** We are only reproducing **Table 4 / Table 12** (LLaVA-1.5-7B visual
> instruction tuning). VL-BART (Table 2 / Table 3) was dropped because the
> required CLIP-feature data (~130 GB) is no longer publicly downloadable. The
> `vlt5` env and the `image_video_text_understanding/` sub-repo are kept
> installed but unused.

Everything lives under:

```
DATA_ROOT=/pscratch/sd/l/liyantan/dora        # scratch — **purged after ~8 weeks idle**
DORA_REPO=/global/homes/l/liyantan/lora-without-regret/ref/DoRA
```

> ⚠️ **Scratch purge.** After training completes, copy checkpoints you want to
> keep to `$HOME` or CFS. Scratch is high-bandwidth but volatile.

## 1. Activation

```bash
source /pscratch/sd/l/liyantan/dora/env.sh   # sets DATA_ROOT, HF_HOME, conda, CUDA_HOME
module load cudatoolkit/12.9                 # for GPU runs
conda activate /pscratch/sd/l/liyantan/dora/envs/llava
```

## 2. Installed envs

| Env | Python | Key pkgs | Use for |
|---|---|---|---|
| `llava` | 3.10 | torch 2.1.2+cu121, transformers 4.31, deepspeed 0.9.5, flash-attn 2.4.2, DoRA-patched peft 0.4.0 | LLaVA-1.5-7B DoRA (**active**) |
| `vlt5` | 3.8 | torch 1.12.1+cu113, transformers 4.2.1, CLIP | VL-BART (dropped — data not public) |

Smoke-test:

```bash
conda activate /pscratch/sd/l/liyantan/dora/envs/llava
python -c "import torch, flash_attn, peft, deepspeed; print('ok')"
```

## 3. Data download

Running on the login node in the background (no SLURM account needed; I/O
bound). The download is **idempotent** — rerun it to resume.

```bash
# Launch (already started — check status first)
nohup bash /pscratch/sd/l/liyantan/dora/download_all.sh \
    >/pscratch/sd/l/liyantan/dora/logs/dl.log 2>&1 &

# Monitor
tail -f /pscratch/sd/l/liyantan/dora/logs/dl.log
du -sh /pscratch/sd/l/liyantan/dora/datasets/llava/
```

What it fetches for LLaVA (~75 GB images + ~1 GB annotations + 41 MB projector):

| Dataset | Size | Source |
|---|---|---|
| `llava_v1_5_mix665k.json` | ~1 GB | HuggingFace `liuhaotian/LLaVA-Instruct-150K` |
| `mm_projector.bin` | 41 MB | HuggingFace `liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` |
| COCO `train2017` | 18 GB | `images.cocodataset.org` |
| GQA images | 20 GB | Stanford NLP |
| TextVQA train+val images | 7 GB | FAIR CDN |
| Visual Genome (2 parts) | 15 GB | Stanford VL |
| OCR-VQA (`loadDataset.py` + ~207K Amazon covers) | ~12 GB | gdown folder + Amazon (best-effort; some 404s expected) |
| LLaVA `eval.zip` (annotations) | 23 MB | gdown |
| VQAv2 `test2015` images | 6 GB | COCO |
| TextVQA val JSON, MMBench TSV | < 1 MB | FAIR / OpenMMLab |

## 4. Code tweaks made during setup

- **`visual_instruction_tuning/Dora_7b.sh`** — `gradient_accumulation_steps`
  bumped from `4` → `8`. Upstream is tuned for 8 GPUs; Perlmutter nodes have
  4× A100, so double grad accumulation keeps effective batch = 128. Matches
  `SETUP_GUIDE.md` §5.

That is the only upstream edit kept for the active (LLaVA) path. The
VL-BART `scripts/*/dora.sh` edits (env-var `BACKBONE_BART`) are harmless and
left in place in case anyone re-enables VL-BART later.

## 5. SLURM job templates

Under `ref/DoRA/slurm/` — single Perlmutter GPU node (4× A100 40 GB), account
`m4788_g`. Logs land in `$DATA_ROOT/logs/`.

| File | Purpose | Wall time |
|---|---|---|
| `llava_train.sbatch` | LLaVA-1.5-7B DoRA fine-tune (Table 4 / Table 12) | 24 h |
| `llava_eval.sbatch` | Eval suite (GQA/SQA/TextVQA/POPE local; VQAv2/VizWiz/MMBench upload) | 8 h |

```bash
sbatch ref/DoRA/slurm/llava_train.sbatch
squeue -u $USER
tail -f /pscratch/sd/l/liyantan/dora/logs/slurm-dora_llava_train-<jobid>.out
```

Submit the eval job after training finishes and the checkpoint
`checkpoints/llava-v1.5-7b-dora-r128-alpha-256/` exists:

```bash
sbatch ref/DoRA/slurm/llava_eval.sbatch
```

VQAv2 / VizWiz / MMBench final numbers require uploading the answer files to
their respective eval servers. See `SETUP_GUIDE.md` §9.3 for URLs.

## 6. Troubleshooting crib sheet

| Symptom | Fix |
|---|---|
| `conda: command not found` in a fresh shell | `source /pscratch/sd/l/liyantan/dora/env.sh` first. |
| Download fails on Amazon URLs (OCR-VQA) | Expected — many product listings expire. LLaVA training tolerates missing OCR-VQA images; ignore. |
| New run hangs on HF download | Check `HF_HOME` points at scratch (`echo $HF_HOME`); home quota is only 40 GiB. |
| Deepspeed OOM on 4× A100 40 GB | Ensure `gradient_accumulation_steps=8` in `Dora_7b.sh`. |
| `flash-attn` import error after torch upgrade | Re-run `bash /pscratch/sd/l/liyantan/dora/build_flash_attn.sh`. |
| SLURM: `Job cost estimated … balance is 0.00 node hours` for CPU jobs | Use `#SBATCH --account=m4788_g --constraint=gpu` for GPU work; downloads run on login node instead. |
