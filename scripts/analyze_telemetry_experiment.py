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
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Sequence

_ROLE_SUFFIX = re.compile(r"\s*\((?:thought|->\s*[^)]*)\)\s*$", re.IGNORECASE)
_MARKDOWN_WRAPPER = re.compile(r"^[*_`\"']+|[*_`\"']+$")
_MARKDOWN_CLOSER = re.compile(r"[*_`\"']+\s*$")


def canonical_agent_name(value: Any) -> str:
    name = str(value or "").strip()
    previous = None
    while previous != name:
        previous = name
        name = _MARKDOWN_WRAPPER.sub("", name).strip()
        name = _MARKDOWN_CLOSER.sub("", name).strip()
        name = _ROLE_SUFFIX.sub("", name).strip()
    return name


def load_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    run_ids = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                run_id = row.get("run_id")
                if not run_id:
                    raise ValueError(f"{path}:{line_number} has no run_id; regenerate with schema v2")
                run_ids.add(str(run_id))
                key = (str(row.get("condition")), str(row.get("trace_id")))
                if key in seen:
                    raise ValueError(f"duplicate condition/trace pair {key} in {path}:{line_number}")
                seen.add(key)
                rows.append(row)
    if len(run_ids) != 1:
        raise ValueError(f"results contain multiple run IDs: {sorted(run_ids)}")
    return rows


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def enrich(row: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    predicted_agent = canonical_agent_name(row.get("predicted_agent")).casefold()
    gold_agent = canonical_agent_name(row.get("gold_agent")).casefold()
    predicted_step = as_int(row.get("predicted_original_step"))
    gold_step = as_int(row.get("gold_original_step"))
    result["agent_correct"] = bool(predicted_agent and predicted_agent == gold_agent)
    gold_step_valid = bool(row.get("gold_step_valid", True))
    result["step_correct"] = (
        gold_step_valid and predicted_step is not None and predicted_step == gold_step
    )
    result["joint_correct"] = result["agent_correct"] and result["step_correct"]
    result["step_within_1"] = (
        gold_step_valid and predicted_step is not None and gold_step is not None
        and abs(predicted_step - gold_step) <= 1
    )
    result["step_absolute_error"] = (
        abs(predicted_step - gold_step)
        if gold_step_valid and predicted_step is not None and gold_step is not None else None
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
    metrics: Dict[str, tuple[Callable[[Dict[str, Any]], float], bool]] = {
        "agent_accuracy": (lambda row: float(row["agent_correct"]), False),
        "step_accuracy": (lambda row: float(row["step_correct"]), True),
        "joint_accuracy": (lambda row: float(row["joint_correct"]), True),
        "step_within_1_accuracy": (lambda row: float(row["step_within_1"]), True),
        "parse_rate": (lambda row: float(row["parse_success"]), False),
        "gold_step_visibility": (lambda row: float(bool(row.get("gold_step_visible"))), False),
        "gold_step_validity": (lambda row: float(bool(row.get("gold_step_valid", True))), False),
        "gold_agent_roster_coverage": (
            lambda row: float(bool(row.get("gold_agent_in_roster", True))), False
        ),
        "gold_agent_telemetry_visibility": (
            lambda row: float(bool(row.get("gold_agent_visible_in_telemetry", True))), False
        ),
        "retained_step_fraction": (
            lambda row: float(row.get("retained_steps", 0)) / float(row.get("original_steps", 1)),
            False,
        ),
        "retained_character_fraction": (
            lambda row: float(row.get("retained_characters", 0))
            / float(row.get("original_characters", 1)),
            False,
        ),
        "latency_seconds": (lambda row: float(row.get("latency_seconds", 0.0)), False),
    }
    summary: Dict[str, Any] = {"condition": condition, "n": len(rows)}
    for metric_name, (getter, valid_step_only) in metrics.items():
        metric_rows = (
            [row for row in rows if bool(row.get("gold_step_valid", True))]
            if valid_step_only else rows
        )
        values = [getter(row) for row in metric_rows]
        rng = random.Random(f"{seed}:{condition}:{metric_name}")
        low, high = bootstrap_interval(values, samples=samples, rng=rng)
        summary[metric_name] = mean(values) if values else float("nan")
        summary[f"{metric_name}_ci_low"] = low
        summary[f"{metric_name}_ci_high"] = high

    agent_visible_rows = [row for row in rows if bool(row.get("gold_agent_visible_in_telemetry", True))]
    summary["agent_visible_n"] = len(agent_visible_rows)
    summary["agent_accuracy_when_visible"] = (
        mean(float(row["agent_correct"]) for row in agent_visible_rows)
        if agent_visible_rows else None
    )
    valid_rows = [row for row in rows if bool(row.get("gold_step_valid", True))]
    summary["valid_step_n"] = len(valid_rows)
    summary["step_accuracy_on_valid_labels"] = (
        mean(float(row["step_correct"]) for row in valid_rows) if valid_rows else None
    )
    summary["step_within_1_on_valid_labels"] = (
        mean(float(row["step_within_1"]) for row in valid_rows) if valid_rows else None
    )
    visible_rows = [
        row for row in rows
        if bool(row.get("gold_step_visible")) and bool(row.get("gold_step_valid", True))
    ]
    summary["visible_step_n"] = len(visible_rows)
    summary["step_accuracy_when_visible"] = (
        mean(float(row["step_correct"]) for row in visible_rows) if visible_rows else None
    )
    summary["step_within_1_when_visible"] = (
        mean(float(row["step_within_1"]) for row in visible_rows) if visible_rows else None
    )
    absolute_errors = [row["step_absolute_error"] for row in rows if row["step_absolute_error"] is not None]
    summary["mean_absolute_step_error_conditional_on_mapped_prediction"] = (
        mean(absolute_errors) if absolute_errors else None
    )
    summary["step_error_n"] = len(absolute_errors)
    summary["step_error_missing_n"] = len(rows) - len(absolute_errors)
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
            metric_common = common
            if key in {"step_correct", "joint_correct", "step_within_1"}:
                metric_common = [
                    trace for trace in common
                    if bool(baseline_rows[trace].get("gold_step_valid", True))
                    and bool(by_condition[condition][trace].get("gold_step_valid", True))
                ]
            differences = [
                float(by_condition[condition][trace][key]) - float(baseline_rows[trace][key])
                for trace in metric_common
            ]
            rng = random.Random(f"{seed}:{baseline}:{condition}:{metric}")
            low, high = bootstrap_interval(differences, samples=samples, rng=rng)
            comparisons.append({
                "baseline": baseline,
                "condition": condition,
                "metric": metric,
                "n_paired": len(metric_common),
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
