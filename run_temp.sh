# bash run_temp.sh >/dev/null 2>&1 &

# DEVICE=2 LR=2e-4 TRAIN_MODE=blocktt DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen CFG_SUFFIX="--enable-save-ckpt --save-grads-steps=0,10,30" bash run_sft.sh
# DEVICE=2 LR=2e-4 TRAIN_MODE=blocktt DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=frozen CFG_SUFFIX="--enable-save-ckpt --save-grads-steps=0,10,30" bash run_sft.sh
# DEVICE=2 LR=2e-4 TRAIN_MODE=blocktt DECOMP_MODE='{qkv:input,o:output,mlp_upgate:output,mlp_down:output}' TRAIN_POSITION=small S_MERGED_TO=frozen CFG_SUFFIX="--enable-save-ckpt --save-grads-steps=0,10,30" bash run_sft.sh

# DEVICE=2 LR=1e-4 TRAIN_MODE=blocktt DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=frozen CFG_SUFFIX="--enable-save-ckpt --save-grads-steps=0,10,30" bash run_rl.sh
# DEVICE=2 LR=1e-4 TRAIN_MODE=blocktt DECOMP_MODE='{qkv:input,o:output,mlp_upgate:output,mlp_down:output}' TRAIN_POSITION=small S_MERGED_TO=frozen CFG_SUFFIX="--enable-save-ckpt --save-grads-steps=0,10,30" bash run_rl.sh

### KD
### KD_LOSS_TYPE can be sft, kl, kl_online
# DEVICE=2 LR=1e-4 TRAIN_MODE=blocktt KD_LOSS_TYPE=sft DECOMP_MODE=output_one_block TRAIN_POSITION=both S_MERGED_TO=input CFG_SUFFIX="--enable-save-ckpt --save-grads-steps=0,10,30" bash run_kd.sh
# DEVICE=2 LR=1e-4 TRAIN_MODE=blocktt KD_LOSS_TYPE=kl DECOMP_MODE=output_one_block TRAIN_POSITION=both S_MERGED_TO=input CFG_SUFFIX="--enable-save-ckpt --save-grads-steps=0,10,30" bash run_kd.sh
# DEVICE=2 LR=1e-4 TRAIN_MODE=blocktt KD_LOSS_TYPE=kl_online DECOMP_MODE=output_one_block TRAIN_POSITION=both S_MERGED_TO=input CFG_SUFFIX="--teacher-model-id=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --enable-save-ckpt --save-grads-steps=0,10,30" bash run_kd.sh

### batch size
# DEVICE=2 LR=4e-4 CFG_SUFFIX="--gradient-accumulation-steps=32" NAME_SUFFIX="_bz32" TRAIN_MODE=blocktt DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen bash run_sft.sh
DEVICE=4 LR=4e-4 CFG_SUFFIX="--gradient-accumulation-steps=32" NAME_SUFFIX="_bz32" TRAIN_MODE=lora LORA_RANK=64 bash run_sft.sh >/dev/null 2>&1 &
DEVICE=5 LR=6e-5 CFG_SUFFIX="--gradient-accumulation-steps=32" NAME_SUFFIX="_bz32" TRAIN_MODE=full bash run_sft.sh >/dev/null 2>&1 &

DEVICE=6 LR=8e-4 CFG_SUFFIX="--gradient-accumulation-steps=64" NAME_SUFFIX="_bz64" TRAIN_MODE=blocktt DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen bash run_sft.sh >/dev/null 2>&1 &
# DEVICE=2 LR=8e-4 CFG_SUFFIX="--gradient-accumulation-steps=64" NAME_SUFFIX="_bz64" TRAIN_MODE=lora LORA_RANK=64 bash run_sft.sh
DEVICE=7 LR=1e-4 CFG_SUFFIX="--gradient-accumulation-steps=64" NAME_SUFFIX="_bz64" TRAIN_MODE=full bash run_sft.sh >/dev/null 2>&1 &