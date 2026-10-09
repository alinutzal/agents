#!/bin/bash
#SBATCH -A PLS0xxx
#SBATCH -p batch
#SBATCH --gpus-per-node=1
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out

cd ~/agents
source .venv/bin/activate
export HF_HOME=/fs/ess/pub_data/vllm/.cache/huggingface
export HF_HUB_OFFLINE=1

METHOD=${METHOD:-all_at_once}   # all_at_once | step_by_step | binary_search
DATA=data/Who_and_When/Who_When/Algorithm-Generated

python src/inference.py --method $METHOD \
  --model llama-3B --model_path meta-llama/Llama-3.2-3B-Instruct \
  --directory_path $DATA --device cuda:0

python src/evaluate.py --data_path $DATA \
  --eval_file outputs/${METHOD}_llama-3B_alg_generated.txt
