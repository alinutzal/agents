#!/usr/bin/env python3
"""Build matched Who&When trace views for the first telemetry experiment.

The command reads the original per-trace JSON files and writes one directory per
condition, plus a manifest that records which original steps remain.  Generated
files retain the original labels only for offline scoring; inference code must
use only question, ground_truth, and history.

Example:
    python scripts/build_telemetry_ablations.py \
      --data data/Who_and_When/Who_When/Algorithm-Generated \
      --output data/telemetry_ablations/Algorithm-Generated
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib.telemetry_ablation import (  # noqa: E402
    DEFAULT_CONDITIONS,
    apply_condition,
    serialized_history_size,
)

ALL_CONDITIONS = DEFAULT_CONDITIONS + (
    "no_reference_answer",
    "first_half",
    "recent_half",
    "random_half",
)


def numeric_sort_key(path: Path):
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else float("inf"), path.name)


def parse_conditions(value: str) -> List[str]:
    conditions = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(ALL_CONDITIONS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown condition(s): {', '.join(unknown)}; choose from {', '.join(ALL_CONDITIONS)}"
        )
    if len(set(conditions)) != len(conditions):
        raise argparse.ArgumentTypeError("conditions must not contain duplicates")
    return conditions


def load_record(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if not isinstance(record.get("history"), list):
        raise ValueError(f"{path}: missing list-valued 'history'")
    return record


def build_condition_record(
    source: Dict[str, Any], condition: str, trace_id: str, seed: int
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    result = apply_condition(source, condition, trace_id=trace_id, seed=seed)
    output = dict(source)
    output["history"] = result.history
    # Avoid giving the reference answer to this condition while leaving the
    # original file untouched.  The label fields remain for offline scoring.
    if condition == "no_reference_answer":
        output.pop("ground_truth", None)
        output.pop("groundtruth", None)
    if condition == "no_system_messages":
        output.pop("system_prompt", None)
        output.pop("system_prompts", None)

    original_system_prompt = source.get("system_prompt", source.get("system_prompts"))
    retained_system_prompt = output.get("system_prompt", output.get("system_prompts"))
    original_size = serialized_history_size(source["history"], original_system_prompt)
    retained_size = serialized_history_size(result.history, retained_system_prompt)
    oracle_local_step = result.original_to_local(source.get("mistake_step"))
    row = {
        "trace_id": trace_id,
        "condition": condition,
        "original_steps": result.original_step_count,
        "retained_steps": len(result.history),
        "retained_step_fraction": (
            len(result.history) / result.original_step_count
            if result.original_step_count else 0.0
        ),
        "original_characters": original_size["characters"],
        "retained_characters": retained_size["characters"],
        "retained_character_fraction": (
            retained_size["characters"] / original_size["characters"]
            if original_size["characters"] else 0.0
        ),
        "original_word_tokens": original_size["word_tokens"],
        "retained_word_tokens": retained_size["word_tokens"],
        "retained_indices": result.retained_indices,
        "local_to_original": result.local_to_original,
        "gold_original_step": source.get("mistake_step"),
        "gold_local_step": oracle_local_step,
        "gold_step_visible": oracle_local_step is not None,
        "gold_agent": source.get("mistake_agent"),
    }
    return output, row


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_manifest_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = [
        "trace_id", "condition", "original_steps", "retained_steps",
        "retained_step_fraction", "original_characters", "retained_characters",
        "retained_character_fraction", "original_word_tokens", "retained_word_tokens",
        "gold_original_step", "gold_local_step", "gold_step_visible", "gold_agent",
        "retained_indices", "local_to_original",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["retained_indices"] = json.dumps(row["retained_indices"])
            csv_row["local_to_original"] = json.dumps(row["local_to_original"], sort_keys=True)
            writer.writerow(csv_row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Directory of original Who&When JSON files")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for condition folders and manifests")
    parser.add_argument(
        "--conditions", type=parse_conditions, default=list(DEFAULT_CONDITIONS),
        help=f"Comma-separated conditions (default: {','.join(DEFAULT_CONDITIONS)})",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Seed for deterministic random_half controls")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of sorted traces for a pilot run")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing condition files")
    args = parser.parse_args()

    if not args.data.is_dir():
        parser.error(f"data directory does not exist: {args.data}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    files = sorted(args.data.glob("*.json"), key=numeric_sort_key)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        parser.error(f"no JSON files found in {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    for source_path in files:
        source = load_record(source_path)
        for condition in args.conditions:
            destination = args.output / condition / source_path.name
            if destination.exists() and not args.overwrite:
                raise FileExistsError(
                    f"refusing to overwrite {destination}; use --overwrite or choose a new --output"
                )
            output_record, row = build_condition_record(
                source, condition, source_path.name, args.seed
            )
            write_json(destination, output_record)
            manifest_rows.append(row)

    manifest = {
        "schema_version": 1,
        "source_directory": str(args.data),
        "seed": args.seed,
        "conditions": args.conditions,
        "trace_count": len(files),
        "rows": manifest_rows,
    }
    write_json(args.output / "manifest.json", manifest)
    write_manifest_csv(args.output / "manifest.csv", manifest_rows)

    print(f"Built {len(manifest_rows)} views from {len(files)} traces.")
    for condition in args.conditions:
        subset = [row for row in manifest_rows if row["condition"] == condition]
        avg_steps = sum(row["retained_step_fraction"] for row in subset) / len(subset)
        avg_chars = sum(row["retained_character_fraction"] for row in subset) / len(subset)
        visible = sum(bool(row["gold_step_visible"]) for row in subset)
        print(
            f"  {condition:25s} steps={avg_steps:6.1%} chars={avg_chars:6.1%} "
            f"gold-step-visible={visible}/{len(subset)}"
        )
    print(f"Manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
