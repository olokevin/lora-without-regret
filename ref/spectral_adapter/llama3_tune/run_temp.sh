CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=spectral --lr=1e-5
CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=full --lr=1e-5
CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=blocktt --lr=1e-5 --blocktt-type=all --decomp-mode=input_one_block --train-position=small --s-merged-to=frozen
CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=blocktt --lr=1e-5 --blocktt-type=all --decomp-mode=output_one_block --train-position=small --s-merged-to=frozen
CUDA_VISIBLE_DEVICES=0 python llama_tune.py --model=lora --lr=1e-5 --lora-rank=64