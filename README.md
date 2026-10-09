# Agents

This repository is being used to test Git and establish a connection between the OSC environment and GitHub.

It runs failure attribution on the [Who&When](https://huggingface.co/datasets/Kevin355/Who_and_When) benchmark: given a failed multi-agent conversation, an LLM predicts which agent made the mistake and at which step.

## Purpose

- Test Git configuration and basic Git operations.
- Connect the OSC project repository to my GitHub repository.

## Layout

- `src/inference.py` – runs an attribution method over a directory of chat logs and writes predictions to `outputs/`.
- `src/evaluate.py` – compares a predictions file against the ground-truth labels and reports agent and step accuracy.
- `scripts/run_infer.sh` – Slurm job that runs inference followed by evaluation.
- `data/Who_and_When/Who_When/` – the dataset (`Algorithm-Generated`: 126 logs, `Hand-Crafted`: 58 logs). Not tracked by Git.

## Setup on OSC

The Python environment lives in `.venv` (transformers + torch; vLLM is not installed, so local models run through the Hugging Face pipeline).

Model weights are read from OSC's shared cache, so nothing needs to be downloaded:

```bash
source .venv/bin/activate
export HF_HOME=/fs/ess/pub_data/vllm/.cache/huggingface
export HF_HUB_OFFLINE=1
```

Models available in that cache include `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-30B-A3B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` and `openai/gpt-oss-20b`. Run `ls $HF_HOME/hub` for the full list.

Notes:

- Always run from the repository root; outputs are written to `outputs/` relative to the current directory.
- Do not run inference on a login node. Use a GPU job (below).
- Pass `--device cuda:0`; the default `cuda:1` does not exist in a single-GPU job.
- Use an **Instruct** model. `meta-llama/Llama-3.2-3B` (the default for the `llama-3B` alias) is a base model with no chat template and fails on every file.
- Only the Llama aliases currently work without vLLM. The Qwen code path expects a `(model, tokenizer)` pair rather than a pipeline and fails with "Unsupported model_family".

## Running inference

### Interactive test (debug queue, 1 hour max)

```bash
salloc -A <your_project_account> -p debug --gpus-per-node=1 -t 1:00:00
cd ~/agents
source .venv/bin/activate
export HF_HOME=/fs/ess/pub_data/vllm/.cache/huggingface
export HF_HUB_OFFLINE=1

python src/inference.py --method all_at_once \
  --model llama-3B --model_path meta-llama/Llama-3.2-3B-Instruct \
  --directory_path data/Who_and_When/Who_When/Algorithm-Generated \
  --device cuda:0
```

### Batch job

First set your project account in the `#SBATCH -A` line of `scripts/run_infer.sh`. Then, from the repository root:

```bash
mkdir -p logs
sbatch scripts/run_infer.sh                         # all_at_once (default)
METHOD=binary_search sbatch scripts/run_infer.sh    # all_at_once | step_by_step | binary_search
```

The job runs inference and then evaluation; the log is written to `logs/<job-name>-<job-id>.out`.

`step_by_step` makes one model call per conversation step, so it is much slower than `all_at_once` or `binary_search`.

## Evaluation

Evaluation only parses text files, so it can run on a login node:

```bash
source .venv/bin/activate
python src/evaluate.py \
  --data_path data/Who_and_When/Who_When/Algorithm-Generated \
  --eval_file outputs/all_at_once_llama-3B_alg_generated.txt
```

It prints agent-level and step-level accuracy over all files in `--data_path`.

## Hand-Crafted subset

For the Hand-Crafted logs:

- add `--is_handcrafted True` to the inference command,
- point `--directory_path` and `--data_path` at `data/Who_and_When/Who_When/Hand-Crafted`,
- evaluate the output file ending in `_handcrafted.txt`.
