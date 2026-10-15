#!/usr/bin/env python3
"""Run matched all-at-once attribution over telemetry-ablation views.

This runner uses the repository's local model loader/generator but writes
machine-readable JSONL rather than parsing redirected console output.  Run the
builder first so every condition has an auditable original-step mapping.

Example (GPU node):
    python scripts/run_telemetry_experiment.py \
      --ablation-dir data/telemetry_ablations/Algorithm-Generated \
      --model llama-3B \
      --model-path meta-llama/Llama-3.2-3B-Instruct \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError as exc:  # pragma: no cover - only exercised on inference hosts
    raise SystemExit("torch and transformers are required for inference") from exc

from lib.local_model import _run_local_generation  # noqa: E402
from lib.telemetry_ablation import speaker_name  # noqa: E402

try:  # noqa: E402
    from vllm import LLM as VLLMClient
except ImportError:  # pragma: no cover - optional backend
    VLLMClient = None


MODEL_ALIASES = {
    "llama-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-30b": "Qwen/Qwen3-30B-A3B",
}


def numeric_sort_key(path: Path):
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else float("inf"), path.name)


def parse_conditions(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_prompt(record: Dict[str, Any]) -> str:
    history = record.get("history", [])
    lines = [
        "You are diagnosing a failed multi-agent execution.",
        f"Task: {record.get('question', '')}",
    ]
    reference_answer = record.get("ground_truth", record.get("groundtruth"))
    if reference_answer not in (None, ""):
        lines.append(f"Reference answer: {reference_answer}")
    system_prompt = record.get("system_prompt", record.get("system_prompts"))
    if system_prompt not in (None, "", {}, []):
        if isinstance(system_prompt, (dict, list)):
            system_prompt = json.dumps(system_prompt, ensure_ascii=False)
        lines.extend(["", "Agent/system instructions:", str(system_prompt)])
    lines.extend(["", "Available telemetry:"])
    for local_step, entry in enumerate(history):
        content = entry.get("content", "")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"Step {local_step} - {speaker_name(entry)}: {content}")

    lines.extend([
        "",
        "Using only the telemetry above, identify the single agent most directly responsible ",
        "for the failed result and the first decisive mistake by that agent. Step numbers refer ",
        "to the displayed local Step labels. If evidence is incomplete, make the best-supported choice.",
        "Return exactly these three lines:",
        "Agent Name: <agent label exactly as displayed>",
        "Step Number: <non-negative integer>",
        "Reason for Mistake: <brief explanation>",
    ])
    return "\n".join(lines)


def parse_prediction(response: str | None) -> Tuple[str | None, int | None]:
    if not response:
        return None, None
    agent_match = re.search(
        r"^\s*(?:\d+[.)]\s*)?Agent(?:\s+Name)?\s*:\s*[*\"']?(.+?)[*\"']?\s*$",
        response, re.IGNORECASE | re.MULTILINE,
    )
    step_match = re.search(
        r"^\s*(?:\d+[.)]\s*)?Step(?:\s+Number)?\s*:\s*[*\"']?(\d+)",
        response, re.IGNORECASE | re.MULTILINE,
    )
    agent = agent_match.group(1).strip() if agent_match else None
    step = int(step_match.group(1)) if step_match else None
    return agent, step


def load_model(alias: str, model_path: str | None, device: str, backend: str):
    model_id = model_path or MODEL_ALIASES.get(alias, alias)
    family = "qwen" if "qwen" in alias.casefold() or "qwen" in model_id.casefold() else "llama"

    if backend == "vllm" and VLLMClient is not None:
        return VLLMClient(model=model_id, trust_remote_code=True), family, model_id, "vllm"
    if backend == "vllm":
        raise RuntimeError("--backend vllm requested but vllm is not installed")

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    if family == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        )
        return (model, tokenizer), family, model_id, "transformers"

    model_object = pipeline(
        "text-generation", model=model_id,
        model_kwargs={"torch_dtype": dtype}, device=device,
    )
    return model_object, family, model_id, "transformers"


def existing_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            keys.add((str(record["condition"]), str(record["trace_id"])))
    return keys


def iter_items(ablation_dir: Path, conditions: Iterable[str], limit: int | None):
    manifest = load_json(ablation_dir / "manifest.json")
    rows_by_key = {
        (row["condition"], row["trace_id"]): row for row in manifest["rows"]
    }
    for condition in conditions:
        files = sorted((ablation_dir / condition).glob("*.json"), key=numeric_sort_key)
        if limit is not None:
            files = files[:limit]
        for path in files:
            yield condition, path, rows_by_key[(condition, path.name)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation-dir", required=True, type=Path)
    parser.add_argument("--conditions", type=parse_conditions, default=None)
    parser.add_argument("--model", default="llama-3B")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--backend", choices=("auto", "vllm", "transformers"), default="auto",
        help="Inference backend; auto uses transformers, while vllm must be requested explicitly",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Decoding temperature; 0 uses deterministic greedy decoding (default)",
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum traces per condition")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.ablation_dir / "manifest.json"
    if not manifest_path.exists():
        parser.error(f"manifest not found: {manifest_path}; run build_telemetry_ablations.py first")
    manifest = load_json(manifest_path)
    conditions = args.conditions or manifest["conditions"]
    unknown = sorted(set(conditions) - set(manifest["conditions"]))
    if unknown:
        parser.error(f"conditions absent from manifest: {', '.join(unknown)}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")

    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    output = args.output or REPO_ROOT / "outputs" / "telemetry" / f"{safe_model}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()
    done = existing_keys(output)

    model_object, family, model_id, backend = load_model(
        args.model, args.model_path, args.device, args.backend
    )
    print(f"Model: {model_id} | backend={backend} | device={args.device}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Output: {output}")

    attempted = written = skipped = 0
    with output.open("a", encoding="utf-8") as stream:
        for condition, path, manifest_row in iter_items(args.ablation_dir, conditions, args.limit):
            key = (condition, path.name)
            if key in done:
                skipped += 1
                continue
            attempted += 1
            record = load_json(path)
            prompt = build_prompt(record)
            start = time.perf_counter()
            response = _run_local_generation(
                model_object,
                [
                    {"role": "system", "content": "You are a careful agent-system failure auditor."},
                    {"role": "user", "content": prompt},
                ],
                family,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            elapsed = time.perf_counter() - start
            predicted_agent, predicted_local_step = parse_prediction(response)
            mapping = manifest_row["local_to_original"]
            predicted_original_step = (
                mapping.get(str(predicted_local_step))
                if predicted_local_step is not None else None
            )
            result = {
                "trace_id": path.name,
                "condition": condition,
                "model": model_id,
                "backend": backend,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "predicted_agent": predicted_agent,
                "predicted_local_step": predicted_local_step,
                "predicted_original_step": predicted_original_step,
                "gold_agent": manifest_row.get("gold_agent"),
                "gold_original_step": manifest_row.get("gold_original_step"),
                "gold_local_step": manifest_row.get("gold_local_step"),
                "gold_step_visible": manifest_row.get("gold_step_visible"),
                "retained_steps": manifest_row.get("retained_steps"),
                "original_steps": manifest_row.get("original_steps"),
                "retained_characters": manifest_row.get("retained_characters"),
                "original_characters": manifest_row.get("original_characters"),
                "latency_seconds": elapsed,
                "parse_success": predicted_agent is not None and predicted_local_step is not None,
                "raw_response": response,
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            done.add(key)
            written += 1
            print(
                f"[{written}] {condition}/{path.name}: "
                f"agent={predicted_agent!r}, local_step={predicted_local_step}, "
                f"original_step={predicted_original_step}"
            )

    print(f"Finished: attempted={attempted}, written={written}, skipped_existing={skipped}")


if __name__ == "__main__":
    main()
