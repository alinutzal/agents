# Experiment 1: What Agent Telemetry Is Needed for Failure Attribution?

This experiment measures how failure-attribution accuracy changes when an auditor receives different parts of a Who&When execution trace. It is a **matched ablation**: the model, prompt, decoding path, and test traces remain fixed while only the available telemetry changes.

## Research questions

1. Which telemetry channels matter for identifying the responsible agent (WHO)?
2. Which channels matter for locating the first decisive error (WHEN)?
3. How much trace volume can be removed before attribution deteriorates?

## Initial conditions

The default pilot builds these six trace views:

| Condition | Available information | Interpretation |
|---|---|---|
| `full` | Entire history, top-level `system_prompt`, and reference answer | Primary baseline |
| `no_system_messages` | Full history except explicit `role=system` messages, with top-level `system_prompt` removed | Value of system/instruction telemetry |
| `no_tool_observations` | Full history except passive tool/terminal results | Value of observations returned by tools |
| `tool_observations_only` | Only passive tool/terminal results | Whether outputs alone suffice |
| `communication_only` | Non-system, non-tool messages | Value of inter-agent communication without execution evidence |
| `oracle_responsible_agent` | Only messages produced by the gold responsible agent | Diagnostic upper bound for perfect agent retrieval, **not deployable** |

Additional controls can be requested explicitly: `no_reference_answer`, `first_half`, `recent_half`, and seeded `random_half`.

### Tool-observation rule

The classic Who&When files do not have one uniform event schema. The code therefore uses a conservative deterministic classifier:

- `role` is `tool` or `function`;
- a `tool_call_id` is present;
- the speaker is a terminal-like executor such as `Computer_terminal`; or
- content begins with a common result marker such as `exitcode:`, `Code output:`, `Tool result:`, or `Observation:`.

`WebSurfer` and `FileSurfer` are intentionally treated as agents, because they make decisions in the hand-crafted systems. The manifest makes every classification auditable through the retained original indices.

## 1. Build matched views

From the repository root:

```bash
python scripts/build_telemetry_ablations.py \
  --data data/Who_and_When/Who_When/Algorithm-Generated \
  --output data/telemetry_ablations/Algorithm-Generated
```

For a cheap smoke test, add `--limit 5`. Existing files are protected unless `--overwrite` is supplied.

The builder writes:

- `<output>/<condition>/<trace>.json`;
- `<output>/manifest.json`, with complete mappings and metadata;
- `<output>/manifest.csv`, for spreadsheet analysis.

Each manifest row records retained steps and characters as cost proxies, whether the gold decisive step remains visible, whether the gold agent appears in retained telemetry, and a `local_to_original` map. The builder validates that each gold step is an in-range raw-history index and flags annotation anomalies with `gold_step_valid=false`; do not include those records in localization denominators without a documented reconciliation. It also reports how often a condition is identical to the full telemetry view.

Filtered traces are numbered locally from zero, then predictions are translated back to original Who&When step numbers. Every condition receives the same candidate-agent roster derived from the full trace, so an evidence ablation does not mechanically change the WHO answer space. Hand-Crafted labels such as `Orchestrator (thought)` and `Orchestrator (-> WebSurfer)` are canonicalized to `Orchestrator` consistently in filtering, prompting, parsing, and scoring.

## 2. Run attribution

On a GPU node using the existing OSC model cache:

```bash
source .venv/bin/activate
export HF_HOME=/fs/ess/pub_data/vllm/.cache/huggingface
export HF_HUB_OFFLINE=1

python scripts/run_telemetry_experiment.py \
  --ablation-dir data/telemetry_ablations/Algorithm-Generated \
  --model llama-3B \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --device cuda:0
```

The default result name includes the subset, model alias, and a run-configuration hash, for example `outputs/telemetry/Algorithm-Generated_llama-3B_<run-id>.jsonl`. Every row stores the manifest SHA-256, resolved backend, model/revision, and decoding configuration. Resume is allowed only when the stored run ID matches; a conflicting explicit `--output` fails rather than silently mixing runs. Use `--overwrite` to start over and `--limit 5` for a per-condition pilot.

For final runs, pass an immutable model commit with `--model-revision`, keep decoding configuration identical across conditions, and set `--max-input-tokens` to an explicit context preflight limit. Token counts are recorded. Over-limit prompts and generation failures are reported but are **not** written as completed predictions, so they can be retried.

## 3. Analyze results

```bash
python scripts/analyze_telemetry_experiment.py \
  outputs/telemetry/Algorithm-Generated_llama-3B_<run-id>.jsonl \
  --baseline full \
  --output-dir outputs/telemetry/analysis
```

Outputs:

- `condition_summary.csv`: accuracy, cost proxies, latency, and 95% bootstrap intervals;
- `paired_differences.csv`: paired accuracy differences against `full` with 95% bootstrap intervals;
- `analysis.json`: the complete machine-readable analysis.

Metrics include agent accuracy, exact original-step accuracy, step accuracy within ±1, joint WHO+WHEN accuracy, parse rate, annotation validity, agent/step visibility, retained trace fraction, and latency. Primary step and joint metrics exclude invalid gold-step annotations. The analysis also reports agent accuracy conditional on the gold agent being visible and localization accuracy conditional on a valid visible gold step. Mean absolute step error is explicitly labeled as conditional on a valid gold label and mapped prediction and includes its missing-count denominator.

## Experimental cautions

- **Do not interpret `oracle_responsible_agent` as a deployable method.** It uses the label to select telemetry and is only an upper-bound diagnostic.
- A removed gold step makes exact localization impossible. Report both end-to-end localization and the predeclared visible-only secondary metric.
- Some public Hand-Crafted annotations have out-of-range `mistake_step` values. Report `gold_step_validity`; reconcile or exclude these from localization claims while retaining them for WHO if appropriate.
- On classic Who&When, `no_system_messages` may be identical to `full` because there are no `role=system` history entries and instruction-like text can be embedded in assistant content. Treat the builder's identical-view diagnostic as a manipulation check; do not claim an instruction-telemetry effect when the intervention is a no-op.
- Character and whitespace-token counts are dependency-free cost proxies. For final cost claims, also record tokens from the selected model tokenizer.
- Run algorithm-generated and hand-crafted subsets separately before pooling them; their speaker conventions and architectures differ.
- The experiment runner defaults to deterministic greedy decoding (`--temperature 0`) to isolate the telemetry intervention. If sampling is enabled, repeat each condition under matched documented seeds rather than treating one stochastic generation as definitive.
- The reference answer is shown in the primary baseline to match the repository's existing all-at-once method. Include `no_reference_answer` if the deployment setting does not have a trusted answer.
