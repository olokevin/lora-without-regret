# FURA: Full-Rank Parameter-Efficient Fine-Tuning that Outperforms Full Model Tuning

## Installation

```bash
# install dependencies
uv sync 

# install LLM-Adapters repository to obtain SFT datasets
cd ref/LIFT
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
```

## SFT Experiments

Model: Llama3-8B

Dataset: Commonsense-170K

```bash
bash ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh 
```

## RL Experiments

Model: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

Dataset: We use the first 7500 examples from [qwedsacf/competition_math](https://huggingface.co/datasets/qwedsacf/competition_math) for training and examples 7501 to 8500 for validation.

Reward function: We use the utilities from [hendrycks/math repo](https://github.com/hendrycks/math/tree/main/modeling/dataset) to extract boxed answers and compare mathematical equivalence to ground truth answers from the dataset.

```bash
TRAIN_MODE=blocktt bash run_rl.sh
```
