#!/usr/bin/env python3
"""Score telemetry experiment JSONL and compute paired bootstrap intervals.

The primary step metric maps each condition-local model prediction back to its
original Who&When step before comparison.  This prevents filtered views from
being penalized merely because their displayed steps were renumbered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Sequence


def load_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("condition")), str(row.get("trace_id")))
                if key in seen:
                    raise ValueError(f"duplicate condition/trace pair {key} in {path}:{line_number}")
                seen.add(key)
                rows.append(row)
    return rows


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def enrich(row: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    predicted_agent = str(row.get("predicted_agent") or "").strip().casefold()
    gold_agent = str(row.get("gold_agent") or "").strip().casefold()
    predicted_step = as_int(row.get("predicted_original_step"))
    gold_step = as_int(row.get("gold_original_step"))
    result["agent_correct"] = bool(predicted_agent and predicted_agent == gold_agent)
    result["step_correct"] = predicted_step is not None and predicted_step == gold_step
    result["joint_correct"] = result["agent_correct"] and result["step_correct"]
    result["step_within_1"] = (
        predicted_step is not None and gold_step is not None and abs(predicted_step - gold_step) <= 1
    )
    result["step_absolute_error"] = (
        abs(predicted_step - gold_step)
        if predicted_step is not None and gold_step is not None else None
    )
    result["parse_success"] = bool(row.get("parse_success"))
    return result


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_interval(
    values: Sequence[float], *, samples: int, rng: random.Random
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    estimates = []
    n = len(values)
    for _ in range(samples):
        estimates.append(mean(values[rng.randrange(n)] for _ in range(n)))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize_condition(
    condition: str, rows: Sequence[Dict[str, Any]], samples: int, seed: int
) -> Dict[str, Any]:
    metrics: Dict[str, Callable[[Dict[str, Any]], float]] = {
        "agent_accuracy": lambda row: float(row["agent_correct"]),
        "step_accuracy": lambda row: float(row["step_correct"]),
        "joint_accuracy": lambda row: float(row["joint_correct"]),
        "step_within_1_accuracy": lambda row: float(row["step_within_1"]),
        "parse_rate": lambda row: float(row["parse_success"]),
        "gold_step_visibility": lambda row: float(bool(row.get("gold_step_visible"))),
        "retained_step_fraction": lambda row: (
            float(row.get("retained_steps", 0)) / float(row.get("original_steps", 1))
        ),
        "retained_character_fraction": lambda row: (
            float(row.get("retained_characters", 0)) / float(row.get("original_characters", 1))
        ),
        "latency_seconds": lambda row: float(row.get("latency_seconds", 0.0)),
    }
    summary: Dict[str, Any] = {"condition": condition, "n": len(rows)}
    for metric_name, getter in metrics.items():
        values = [getter(row) for row in rows]
        rng = random.Random(f"{seed}:{condition}:{metric_name}")
        low, high = bootstrap_interval(values, samples=samples, rng=rng)
        summary[metric_name] = mean(values) if values else float("nan")
        summary[f"{metric_name}_ci_low"] = low
        summary[f"{metric_name}_ci_high"] = high

    absolute_errors = [row["step_absolute_error"] for row in rows if row["step_absolute_error"] is not None]
    summary["mean_absolute_step_error"] = mean(absolute_errors) if absolute_errors else None
    summary["step_error_n"] = len(absolute_errors)
    return summary


def paired_differences(
    rows: Sequence[Dict[str, Any]], baseline: str, samples: int, seed: int
) -> List[Dict[str, Any]]:
    by_condition: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][str(row["trace_id"])] = row
    baseline_rows = by_condition.get(baseline, {})
    comparisons: List[Dict[str, Any]] = []
    for condition in sorted(by_condition):
        if condition == baseline:
            continue
        common = sorted(set(baseline_rows) & set(by_condition[condition]))
        for metric, key in (
            ("agent_accuracy", "agent_correct"),
            ("step_accuracy", "step_correct"),
            ("joint_accuracy", "joint_correct"),
            ("step_within_1_accuracy", "step_within_1"),
        ):
            differences = [
                float(by_condition[condition][trace][key]) - float(baseline_rows[trace][key])
                for trace in common
            ]
            rng = random.Random(f"{seed}:{baseline}:{condition}:{metric}")
            low, high = bootstrap_interval(differences, samples=samples, rng=rng)
            comparisons.append({
                "baseline": baseline,
                "condition": condition,
                "metric": metric,
                "n_paired": len(common),
                "mean_difference": mean(differences) if differences else float("nan"),
                "ci_low": low,
                "ci_high": high,
            })
    return comparisons


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="One or more telemetry result JSONL files")
    parser.add_argument("--baseline", default="full")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/telemetry/analysis"))
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    rows = [enrich(row) for row in load_jsonl(args.results)]
    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    if args.baseline not in by_condition:
        parser.error(f"baseline condition {args.baseline!r} is absent from results")

    summaries = [
        summarize_condition(condition, by_condition[condition], args.bootstrap_samples, args.seed)
        for condition in sorted(by_condition)
    ]
    comparisons = paired_differences(
        rows, args.baseline, args.bootstrap_samples, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "condition_summary.csv", summaries)
    write_csv(args.output_dir / "paired_differences.csv", comparisons)
    with (args.output_dir / "analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "baseline": args.baseline,
                "bootstrap_samples": args.bootstrap_samples,
                "summaries": summaries,
                "paired_differences": comparisons,
            },
            stream, indent=2, allow_nan=False,
        )
        stream.write("\n")

    print("condition                    n   agent    step   within1   joint   chars")
    for row in summaries:
        print(
            f"{row['condition']:28s} {row['n']:4d} "
            f"{row['agent_accuracy']:7.1%} {row['step_accuracy']:7.1%} "
            f"{row['step_within_1_accuracy']:9.1%} {row['joint_accuracy']:7.1%} "
            f"{row['retained_character_fraction']:7.1%}"
        )
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
