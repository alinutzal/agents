#!/bin/bash
#SBATCH -A PLS0150
#SBATCH -p batch
#SBATCH --gpus-per-node=1
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out

# Who&When Pro attribution experiments (claim -> experiment map). Run from the repository root:
#   STAGE=laya sbatch scripts/run_pro.sh      # P0, E1, E1b, E2, E3, E8: Laya on every trace
#   STAGE=windows sbatch scripts/run_pro.sh   # E9a/E9b: Laya on windows
#   STAGE=shuffled sbatch scripts/run_pro.sh  # position-bias check: Laya with options in random order
#   STAGE=llm sbatch scripts/run_pro.sh       # LLM baseline on the llm subset (E1 paired, E4)
# Then, anywhere: python scripts/analyze_claims.py --llm outputs/pro/llm_gpt-oss-20b_llm.jsonl
# Every stage resumes from its output file if the job is cut off.

cd ~/agents
source .venv/bin/activate
# Laya and the Pro tokenizer are in the default ~/.cache/huggingface, not the shared vLLM cache.
export HF_HUB_OFFLINE=1

STAGE=${STAGE:-laya}

# data/whowhen_pro/text_ctx8192.jsonl and text_over8192.jsonl come from scripts/filter_whowhen_pro.py
python scripts/make_manifest.py                       # idempotent; seeded from experiments/prereg.json

case $STAGE in
  laya)
    python scripts/laya_whowhen_pro.py --subset all ;;
  shuffled)
    for seed in 1 2; do python scripts/laya_whowhen_pro.py --subset all --order_seed $seed; done ;;
  windows)
    python scripts/laya_whowhen_pro.py --subset window --windows 1024,2048
    python scripts/laya_whowhen_pro.py --subset overlong --windows 1024,2048 ;;
  llm)
    # gpt-oss-20b is in the shared cache only.
    export HF_HOME=/fs/ess/pub_data/vllm/.cache/huggingface
    python scripts/llm_whowhen_pro.py --subset llm --model openai/gpt-oss-20b ;;
esac
