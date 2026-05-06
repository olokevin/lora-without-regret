# DoRA Reproduction Setup Guide

**Goal:** Reproduce Table 2/3 (VL-BART image/video-text) and Table 4/12 (LLaVA-1.5-7B) from the DoRA paper (ICML 2024 Oral).  
**Compute needed:** 4× A100 40GB (or equivalent)  
**Total storage needed:** ~280GB

> **Storage note:** `/home` is likely small. Put all conda envs, data, and HF cache on a large disk (e.g. `/raid0-data/$USER/`). Commands below use `$DATA_ROOT` as a placeholder — set it to your large disk path.

```bash
export DATA_ROOT=/raid0-data/$USER   # adjust to your cluster
export REPO=/path/to/DoRA            # absolute path to this repo
```

---

## 1. System Requirements

- CUDA driver ≥ 12.x (tested with 12.6)
- CUDA toolkit ≥ 12.0 installed at `/usr/local/cuda` (needed to compile flash-attn)
- Miniconda or Anaconda
- ~280GB free storage on the data disk

---

## 2. Install Miniconda (if not already available)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $DATA_ROOT/miniconda3
$DATA_ROOT/miniconda3/bin/conda init bash
source ~/.bashrc
```

---

## 3. Environment A: `vlt5` (VL-BART, Python 3.8)

```bash
conda create -n vlt5 python=3.8 -y
conda activate vlt5

cd $REPO/image_video_text_understanding
pip install -r requirements.txt

# CLIP is not in requirements.txt — install manually
pip install git+https://github.com/openai/CLIP.git

# Download BART backbone
python download_backbones.py

# COCO captioning evaluation support
python -c "import language_evaluation; language_evaluation.download('coco')"
```

**Verify:**
```bash
python -c "import torch, clip, transformers; print('OK')"
```

---

## 4. Environment B: `llava` (LLaVA-1.5, Python 3.10)

> **Critical notes:**
> - The local `./peft` folder contains DoRA patches — do NOT use `pip install peft`.
> - The upstream `torch==2.0.1+cu117` in `pyproject.toml` is **incompatible** with CUDA 12.x toolkits. Upgrade to `torch==2.1.2+cu121`.
> - `setuptools` must be `<70` for `pkg_resources` support needed by flash-attn build.
> - `numpy` must be `<2` for torch 2.1.x compatibility.
> - Use `--no-cache-dir` for flash-attn install to avoid cross-device link errors (pip temp dir vs. cache on different filesystems).

```bash
conda create -n llava python=3.10 -y
conda activate llava

cd $REPO/visual_instruction_tuning
pip install -e .
pip install -e ".[train]"

# Fix dependency issues before flash-attn
pip install "setuptools<70"
pip install "numpy<2"

# Upgrade torch to match CUDA 12.x toolkit (original 2.0.1+cu117 won't compile flash-attn)
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn (takes 10–20 min to compile; use --no-cache-dir to avoid cross-device error)
CUDA_HOME=/usr/local/cuda MAX_JOBS=4 \
    pip install flash-attn==2.4.2 --no-build-isolation --no-cache-dir

# Install local DoRA-patched peft (MUST be this, not upstream peft)
pip install -U ./peft --no-cache-dir
```

**Verify:**
```bash
python -c "
import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
import flash_attn; print('flash_attn:', flash_attn.__version__)
import peft; print('peft:', peft.__version__)
import deepspeed; print('deepspeed:', deepspeed.__version__)
"
# Expected: torch 2.1.2+cu121, flash_attn 2.4.2, peft 0.4.0, deepspeed 0.9.5
```

---

## 5. Code Change: Patch for 4-GPU Training

The `Dora_7b.sh` script was written for 8 GPUs. For 4 GPUs, double `gradient_accumulation_steps` to preserve effective batch size of 128.

```bash
# Already committed in this repo (commit 8d05df8), no action needed.
# For reference, the diff is:
# -    --gradient_accumulation_steps 4 \
# +    --gradient_accumulation_steps 8 \
```

---

## 6. Data: VL-BART (Image/Video-Text)

### 6.1 Image-Text CLIP Features (~130GB total)

Already present in this repo at:
```
$REPO/image_video_text_understanding/datasets/
    COCO/          (62GB — includes clip_features/)
    GQA/           (279MB)
    VG/            (21GB)
    nlvr/          (47GB)
    vqa/           (814MB)
    lxmert/        (968KB)
```

If missing, download the CLIP features zip from Google Drive (requires browser/gdrive auth):
```
https://drive.google.com/file/d/1O_RU1iFh_sbItZCTkOHUrbVIQQ_89Djj/view?usp=sharing
```
Extract to `$REPO/image_video_text_understanding/datasets/`.

### 6.2 Video-Text ViT Features (VALUE benchmark)

Follow [VALUE DataRelease](https://github.com/VALUE-Leaderboard/DataRelease) to download.  
Extract to:
```
$REPO/image_video_text_understanding/datasets/video/
    ann/
    vis_features/
```

---

## 7. Data: LLaVA (Visual Instruction Tuning)

All paths relative to `$REPO/visual_instruction_tuning/playground/data/`.

### 7.1 Training Annotations (~983MB)

```bash
cd $REPO/visual_instruction_tuning/playground/data
# Download from HuggingFace
python -c "
import os; os.environ['HF_HOME']='$DATA_ROOT/hf_cache'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='liuhaotian/LLaVA-Instruct-150K',
    filename='llava_v1_5_mix665k.json',
    repo_type='dataset',
    local_dir='.'
)
"
```

### 7.2 Training Images (~75GB total)

```bash
cd $REPO/visual_instruction_tuning/playground/data

# COCO train2017 (~18GB)
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d coco/

# GQA (~20GB)
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O gqa_images.zip
unzip gqa_images.zip -d gqa/

# TextVQA (~7GB)
wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -O textvqa_images.zip
unzip textvqa_images.zip -d textvqa/

# VisualGenome (~15GB, two parts)
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O vg_images1.zip
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O vg_images2.zip
unzip vg_images1.zip -d vg/VG_100K
unzip vg_images2.zip -d vg/VG_100K_2

# OCR-VQA (~12GB) — requires their download script (saves all images as .jpg)
# See: https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_
# Download loadDataset.py from that folder and run:
# python loadDataset.py --out_dir ./ocr_vqa/images --imsize 480
```

### 7.3 Pretrained Projector (41MB)

```bash
mkdir -p $REPO/visual_instruction_tuning/checkpoints/llava-v1.5-7b-pretrain
python -c "
import os; os.environ['HF_HOME']='$DATA_ROOT/hf_cache'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5',
    filename='mm_projector.bin',
    local_dir='$REPO/visual_instruction_tuning/checkpoints/llava-v1.5-7b-pretrain'
)
"
```

### 7.4 Eval Datasets (~48GB)

Download `eval.zip` from Google Drive (requires browser/gdrive auth):
```
https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing
```
Extract to `$REPO/visual_instruction_tuning/playground/data/eval/`.

Then download additional eval images:
```bash
cd $REPO/visual_instruction_tuning/playground/data

# VQAv2 test images (~6GB)
wget http://images.cocodataset.org/zips/test2015.zip
unzip test2015.zip -d eval/vqav2/

# TextVQA val annotations
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json \
     -O eval/textvqa/TextVQA_0.5.1_val.json

# MMBench
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv \
     -O eval/mmbench/mmbench_dev_20230712.tsv
```

---

## 8. Training

### 8.1 VL-BART Image-Text (GPU 0, ~20 epochs, eval included)

```bash
conda activate vlt5
cd $REPO/image_video_text_understanding/VL-T5
bash scripts/image/dora.sh 1
# Script hardcodes CUDA_VISIBLE_DEVICES=1; override if needed:
# CUDA_VISIBLE_DEVICES=0 bash scripts/image/dora.sh 1
```

Key hyperparameters (paper Table 9):
- backbone: `facebook/bart-base`, lora_dim: 128, lr: 1e-3, epochs: 20, batch_size: 300
- Checkpoint: `snap/VLBart_multitask/tune+lr1e-3_plzplz2/LAST.pth`

### 8.2 VL-BART Video-Text (GPU 1, ~7 epochs)

```bash
conda activate vlt5
cd $REPO/image_video_text_understanding/VL-T5
CUDA_VISIBLE_DEVICES=1 bash scripts/video/dora.sh 1
```

Key hyperparameters: lora_dim: 128, lr: 2.4e-4, epochs: 7, batch_size: 40  
Checkpoint: `snap/VLBart_multitask_video/dora_lora_setting_2.4e-4_128/LAST.pth`  
Auto-generates VALUE benchmark submission file after training.

### 8.3 LLaVA DoRA (4 GPUs, DeepSpeed ZeRO-3, ~1 epoch, ~20hrs)

```bash
conda activate llava
cd $REPO/visual_instruction_tuning
HF_HOME=$DATA_ROOT/hf_cache CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./Dora_7b.sh
```

Key hyperparameters: rank 128, alpha 256, lr 2e-4, cosine decay, warmup 0.03  
Applied to: Q, K, V, O, Up, Down, Gate  
Checkpoint: `checkpoints/llava-v1.5-7b-dora-r128-alpha-256/`

---

## 9. Evaluation

### 9.1 VL-BART Image-Text

Evaluation runs automatically at end of training. Results printed to stdout.

### 9.2 VL-BART Video-Text

Submit the generated submission file to [VALUE benchmark](https://value-benchmark.github.io/).

### 9.3 LLaVA (local + server)

```bash
BASE=$REPO/visual_instruction_tuning
conda activate llava
cd $BASE
CUDA_VISIBLE_DEVICES=0,1,2,3 bash 7B_eval_dora.sh \
    llava-v1.5-7b-dora-r128-alpha-256 \
    $BASE/checkpoints/llava-v1.5-7b-dora-r128-alpha-256 \
    $BASE/playground/data/eval \
    $BASE/eval_result/llava-v1.5-7b-dora-r128-alpha-256 \
    $BASE
```

**Local results (automatic):** GQA, ScienceQA, TextVQA, POPE  
**Server submission required:**
- VQAv2 → `eval_result/.../vqav2/answers_upload` → [eval.ai](https://eval.ai/web/challenges/challenge-page/830/my-submission)
- VisWiz → `eval_result/.../vizwiz/answers_upload` → [eval.ai](https://eval.ai/web/challenges/challenge-page/1911/my-submission)
- MMBench → `eval_result/.../mmbench/answers_upload` → [OpenCompass](https://opencompass.org.cn/leaderboard-multimodal)

---

## 10. Expected Results

### VL-BART (Table 2 & 3)

| Task | DoRA Avg |
|---|---|
| Image-text (VQA/GQA/NLVR2/COCO Cap) | 77.4 |
| Video-text (TVQA/How2QA/TVC/YC2C) | 85.4 |

### LLaVA (Table 4 & 12)

| VQAv2 | GQA | VisWiz | SQA | VQAT | POPE | MMBench | Avg |
|---|---|---|---|---|---|---|---|
| 78.6 | 62.9 | 52.2 | 69.9 | 57.0 | 87.2 | 66.1 | **67.6** |

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `flash-attn` build fails with CUDA version mismatch | Make sure `torch` is installed with a CUDA version matching the toolkit (e.g. `torch==2.1.2+cu121` for CUDA 12.x). Use `CUDA_HOME=/usr/local/cuda`. |
| `pkg_resources` not found during flash-attn metadata | `pip install "setuptools<70"` |
| `numpy` import error with torch | `pip install "numpy<2"` |
| Cross-device link error when installing flash-attn | Add `--no-cache-dir` flag |
| `/home` disk full | Store all data and envs on large disk; set `HF_HOME` and `CONDA_PKGS_DIRS` accordingly |
| gdown can't download GDrive file | Use browser to download manually; large GDrive files hit rate limits |
