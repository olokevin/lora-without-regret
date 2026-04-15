# bash run_temp.sh >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=spectral --lr=1e-5
# CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=full --lr=1e-5 --batch-size=1 --gradient-accumulation-steps=8
# CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=blocktt --lr=1e-5 --decomp-mode=input_one_block --train-position=small --s-merged-to=frozen
# CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=blocktt --lr=1e-5 --decomp-mode=output_one_block --train-position=small --s-merged-to=frozen
# CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=lora --lr=1e-5 --lora-rank=64

# CUDA_VISIBLE_DEVICES=3 python llama_tune.py --model=blocktt --lr=1e-4 --decomp-mode=input_one_block --train-position=small --s-merged-to=frozen
# CUDA_VISIBLE_DEVICES=3 python llama_tune.py --model=spectral --lr=1e-5
# CUDA_VISIBLE_DEVICES=3 python llama_tune.py --model=lora --lr=5e-5 --lora-rank=64
# CUDA_VISIBLE_DEVICES=3 python llama_tune.py --model=blocktt --lr=5e-5 --decomp-mode=output_one_block --train-position=small --s-merged-to=frozen


# CUDA_VISIBLE_DEVICES=7 python llama_tune.py --model=blocktt --lr=1e-4 --decomp-mode=input_one_block --train-position=small --s-merged-to=trainable
# CUDA_VISIBLE_DEVICES=7 python llama_tune.py --model=blocktt --lr=1e-4 --decomp-mode=output_one_block --train-position=small --s-merged-to=trainable
# CUDA_VISIBLE_DEVICES=7 python llama_tune.py --model=blocktt --lr=2e-4 --decomp-mode=input_one_block --train-position=small --s-merged-to=trainable
# CUDA_VISIBLE_DEVICES=7 python llama_tune.py --model=blocktt --lr=2e-4 --decomp-mode=output_one_block --train-position=small --s-merged-to=trainable
# CUDA_VISIBLE_DEVICES=7 python llama_tune.py --model=blocktt --lr=7e-5 --decomp-mode=input_one_block --train-position=small --s-merged-to=trainable
# CUDA_VISIBLE_DEVICES=7 python llama_tune.py --model=blocktt --lr=7e-5 --decomp-mode=output_one_block --train-position=small --s-merged-to=trainable

CUDA_VISIBLE_DEVICES=1 python llama_tune.py --model=blocktt --lr=1e-4 --decomp-mode=input_one_block --train-position=small --s-merged-to=keep_trainable
CUDA_VISIBLE_DEVICES=7 python llama_tune.py --model=blocktt --lr=1e-4 --decomp-mode=output_one_block --train-position=small --s-merged-to=keep_trainable