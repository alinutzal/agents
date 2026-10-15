#!/bin/bash
#SBATCH -A PLS0150
#SBATCH -p batch
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out

# LLM baseline on Who&When Pro through a local vLLM server (the Hugging Face generate path is far
# too slow for gpt-oss). From the repository root:
#   sbatch scripts/run_llm_vllm.sh                          # the 650-trace `llm` subset
#   LIMIT=5 OUT_DIR=outputs/pro_test sbatch scripts/run_llm_vllm.sh   # smoke test
# Resumes from the output file if cut off.

cd ~/agents
MODEL=${MODEL:-openai/gpt-oss-20b}
WORKERS=${WORKERS:-4}
OUT_DIR=${OUT_DIR:-outputs/pro}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}

module load vllm/0.23.0          # sets HF_HOME to the shared cache and puts vllm on PATH
export HF_HUB_OFFLINE=1
PORT=$(shuf -i 49152-65535 -n 1)
vllm serve $MODEL --port $PORT --max-model-len 32768 --gpu-memory-utilization 0.90 > logs/vllm-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

until curl -sf localhost:$PORT/v1/models > /dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
echo "vllm ready on port $PORT"

source .venv/bin/activate
python scripts/llm_whowhen_pro.py --subset llm --backend openai --base_url http://localhost:$PORT/v1 \
  --model $MODEL --workers $WORKERS --out_dir $OUT_DIR $LIMIT_ARG
