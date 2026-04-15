### train commonsense
# bash run.sh >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 lr=2e-4 bash bash_scripts/finetune_commonsense_lora.sh
# CUDA_VISIBLE_DEVICES=1 lr=5e-5 bash bash_scripts/finetune_commonsense_full.sh

# CUDA_VISIBLE_DEVICES=5 lr=2e-4 decomp_mode=input_one_block s_merged_to=frozen bash bash_scripts/finetune_commonsense_blocktt.sh
# CUDA_VISIBLE_DEVICES=4 lr=2e-4 decomp_mode=output_one_block s_merged_to=frozen bash bash_scripts/finetune_commonsense_blocktt.sh

# CUDA_VISIBLE_DEVICES=7 lr=2e-4 decomp_mode=input_one_block s_merged_to=trainable bash bash_scripts/finetune_commonsense_blocktt.sh
# CUDA_VISIBLE_DEVICES=7 lr=2e-4 decomp_mode=output_one_block s_merged_to=trainable bash bash_scripts/finetune_commonsense_blocktt.sh

# CUDA_VISIBLE_DEVICES=3 lr=2e-4 decomp_mode=output_one_block s_merged_to=keep_trainable bash bash_scripts/finetune_commonsense_blocktt.sh
# CUDA_VISIBLE_DEVICES=3 lr=2e-4 decomp_mode=input_one_block s_merged_to=keep_trainable bash bash_scripts/finetune_commonsense_blocktt.sh

CUDA_VISIBLE_DEVICES=7 lr=2e-4 decomp_mode=output_one_block calib_mode=v2_bp bash bash_scripts/finetune_commonsense_blocktt.sh


### eval

# /data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_input_one_block_pos_small_rank_full_smerge_frozen-seed_43
# /data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_output_one_block_pos_small_rank_full_smerge_frozen-seed_43
# /data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_input_one_block_pos_small_smerge_trainable-seed_43
# /data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_output_one_block_pos_small_smerge_keep_trainable-seed_43

# lora: eval_commonsense_lora.sh  adapter_name, base_model, model (ckpt)
# full, blocktt: eval_commonsense.sh  model (ckpt)
# CUDA_VISIBLE_DEVICES=0 bash bash_scripts/eval_commonsense_lora.sh adapter_name=lora base_model=meta-llama/Meta-Llama-3-8B CKPT=/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/lora-lr_2e-4-rank_128-seed_43 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=2 bash bash_scripts/eval_commonsense.sh base_model=meta-llama/Meta-Llama-3-8B CKPT=/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_output_one_block_pos_small_smerge_keep_trainable-seed_43 >/dev/null 2>&1 &


### train math
# bash run.sh >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=2 lr=2e-4 bash bash_scripts/finetune_math_lora.sh
# CUDA_VISIBLE_DEVICES=3 lr=5e-5 bash bash_scripts/finetune_math_full.sh

# CUDA_VISIBLE_DEVICES=5 lr=2e-4 decomp_mode=input_one_block s_merged_to=frozen bash bash_scripts/finetune_math_blocktt.sh
# CUDA_VISIBLE_DEVICES=7 lr=2e-4 decomp_mode=input_one_block s_merged_to=trainable bash bash_scripts/finetune_math_blocktt.sh

# CUDA_VISIBLE_DEVICES=4 lr=2e-4 decomp_mode=output_one_block s_merged_to=frozen bash bash_scripts/finetune_math_blocktt.sh
# CUDA_VISIBLE_DEVICES=4 lr=2e-4 decomp_mode=output_one_block s_merged_to=trainable bash bash_scripts/finetune_math_blocktt.sh

# CUDA_VISIBLE_DEVICES=5 lr=2e-4 decomp_mode=output_one_block s_merged_to=keep_trainable bash bash_scripts/finetune_math_blocktt.sh
# CUDA_VISIBLE_DEVICES=5 lr=2e-4 decomp_mode=input_one_block s_merged_to=keep_trainable bash bash_scripts/finetune_math_blocktt.sh

### lr search
# CUDA_VISIBLE_DEVICES=1 lr=1e-4 decomp_mode=output_one_block s_merged_to=trainable bash bash_scripts/finetune_math_blocktt.sh >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 lr=3e-4 decomp_mode=output_one_block s_merged_to=trainable bash bash_scripts/finetune_math_blocktt.sh >/dev/null 2>&1 &


### eval
# /data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43
# /data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-seed_43

# lora: eval_math_lora.sh  adapter_name, base_model, model (ckpt)
# full, blocktt: eval_math.sh  model (ckpt)
# CUDA_VISIBLE_DEVICES=0 bash bash_scripts/eval_math_lora.sh adapter_name=lora base_model=meta-llama/Meta-Llama-3-8B CKPT=/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/lora-lr_2e-4-rank_128-seed_43 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 bash bash_scripts/eval_math.sh base_model=meta-llama/Meta-Llama-3-8B CKPT=/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/blocktt-lr_3e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43 >/dev/null 2>&1 &
