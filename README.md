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
- `scripts/build_telemetry_ablations.py` – builds matched telemetry views and original-step mappings for the first ablation experiment.
- `scripts/run_telemetry_experiment.py` – runs all-at-once attribution across the matched views and writes resumable JSONL.
- `scripts/analyze_telemetry_experiment.py` – reports accuracy/cost summaries and paired bootstrap intervals.
- `docs/telemetry_experiment.md` – complete design, commands, metrics, and experimental cautions.
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

## Telemetry-ablation experiment

The first poster experiment asks which portions of an agent trace are needed to attribute a failure. It builds matched `full`, no-system, no-tool-observation, tool-only, communication-only, and responsible-agent oracle views; runs the same all-at-once auditor over each view; maps local predictions back to original Who&When step numbers; and computes paired bootstrap intervals against the full-trace baseline.

See [`docs/telemetry_experiment.md`](docs/telemetry_experiment.md) for the study design and runnable commands.

## Hand-Crafted subset

For the Hand-Crafted logs:

- add `--is_handcrafted True` to the inference command,
- point `--directory_path` and `--data_path` at `data/Who_and_When/Who_When/Hand-Crafted`,
- evaluate the output file ending in `_handcrafted.txt`.

## Who&When Pro: calibration and selective attribution (Laya)

Implements `claim-experiment-map.md` for gaps G1, G3 and G4 on the Who&When Pro text subset. Thresholds are frozen in `experiments/prereg.json`; change them only through its `amendments` list, before looking at test numbers.

| Step | Command | Output |
|---|---|---|
| Filter to ≤ 8,192 Laya tokens (the > 8K traces become the E9b set) | `python scripts/filter_whowhen_pro.py` | `data/whowhen_pro/text_ctx8192.jsonl`, `text_over8192.jsonl` |
| Freeze dev/test split and subsets | `python scripts/make_manifest.py` | `experiments/manifest.json` |
| Laya WHO/WHEN/WHAT on every trace, with raw logits | `STAGE=laya sbatch scripts/run_pro.sh` | `outputs/pro/laya_full_all.jsonl` |
| Laya on 1K/2K windows (E9a, E9b) | `STAGE=windows sbatch scripts/run_pro.sh` | `outputs/pro/laya_windows_*.jsonl` |
| LLM baseline with verbalized confidence | `STAGE=llm sbatch scripts/run_pro.sh` | `outputs/pro/llm_*_llm.jsonl` |
| Verdict per claim | `python scripts/analyze_claims.py --llm outputs/pro/llm_gpt-oss-20b_llm.jsonl` | `outputs/pro/claims_report.{json,md}` |

Code lives in `src/attrib/` (`whowhen_pro.py`: rendering and label resolution for each framework; `stats.py`: ECE, Brier, NLL, AUROC, DeLong, bootstrap).

Notes:

- Laya is scored on all usable traces; paired Laya-vs-LLM comparisons (C1.1, C1.4) use the `llm` subset, which oversamples long traces.
- Perception modes (`P.*`) are dropped, so WHAT has 15 options. WHEN has up to 30; `--head_max_len 384` keeps the option texts distinct.
- Single-agent frameworks (alfagent, smolagents) have no WHO question.
- The local LLM is `gpt-oss-20b`, not the paper's `gpt-oss-120b`. Check the `llm_range_check` block of the report before reading anything into the C1.4 cost ratio. For 120b, serve it with vLLM and use `--backend openai --base_url`.
- 4 overlong traces whose label points past the end of the trajectory are excluded (listed in the manifest).
