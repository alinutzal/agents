"""Build a Who&When Pro text eval set that fits in Laya's context window.

Downloads the text subset of Who&When Pro, renders each trace to plain text,
tokenizes it with the laya-multilingual tokenizer, and keeps the traces whose
token count is at most --max_len (Laya ships with max_len=1024; 8192 is the
longest it supports). Logs the exclusion rate overall and per framework/benchmark.

Usage (from the repository root):
    python scripts/filter_whowhen_pro.py
    python scripts/filter_whowhen_pro.py --max_len 8192 --include_answer
"""
import os
import sys
import json
import argparse
import logging
from collections import Counter

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from attrib.whowhen_pro import render_trace  # noqa: E402

DATASET_REPO = "Leoxx/whowhen_pro"
DATASET_FILE = "data/text.jsonl"
TOKENIZER_REPO = "convaiinnovations/laya-multilingual"
TOKENIZER_SUBFOLDER = "tokenizer"

logger = logging.getLogger("filter_whowhen_pro")


def _rate(excluded, total):
    return 100.0 * excluded / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max_len", type=int, default=8192,
                        help="Keep traces with at most this many tokens (Laya default is 1024).")
    parser.add_argument("--out_dir", type=str, default="data/whowhen_pro",
                        help="Where the downloaded subset, filtered set and log are written.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Dataset revision (commit hash) to pin; default is the latest.")
    parser.add_argument("--include_answer", action="store_true",
                        help="Include the task's reference answer in the rendered trace.")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"filter_ctx{args.max_len}.log")
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    )
    logger.setLevel(logging.INFO)  # keep HTTP request logs from other libraries out

    src_path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset",
                               revision=args.revision, local_dir=args.out_dir)
    logger.info("Dataset: %s (%s) -> %s", DATASET_REPO, args.revision or "latest", src_path)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder=TOKENIZER_SUBFOLDER)
    logger.info("Tokenizer: %s/%s (vocab %d)", TOKENIZER_REPO, TOKENIZER_SUBFOLDER, len(tokenizer))
    logger.info("Context length: %d tokens (special tokens included)", args.max_len)

    with open(src_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d traces", len(rows))

    texts = [render_trace(r, include_answer=args.include_answer) for r in rows]
    lengths = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    kept_path = os.path.join(args.out_dir, f"text_ctx{args.max_len}.jsonl")
    # The excluded traces are the G4 set (E9b), scored by windows instead of full context.
    over_path = os.path.join(args.out_dir, f"text_over{args.max_len}.jsonl")
    total_by_group, excluded_by_group = Counter(), Counter()
    kept = 0
    with open(kept_path, "w", encoding="utf-8") as out, open(over_path, "w", encoding="utf-8") as over:
        for row, text, n_tokens in zip(rows, texts, lengths):
            group = (row.get("framework"), row.get("benchmark"))
            total_by_group[group] += 1
            record = json.dumps({**row, "rendered_text": text, "n_tokens": n_tokens}, ensure_ascii=False) + "\n"
            if n_tokens > args.max_len:
                excluded_by_group[group] += 1
                over.write(record)
                continue
            kept += 1
            out.write(record)

    total = len(rows)
    excluded = total - kept
    sorted_lengths = sorted(lengths)
    pct = lambda p: sorted_lengths[min(len(sorted_lengths) - 1, int(p / 100 * len(sorted_lengths)))]
    logger.info("Token length: min %d, median %d, p90 %d, p99 %d, max %d",
                sorted_lengths[0], pct(50), pct(90), pct(99), sorted_lengths[-1])
    logger.info("Would be excluded at the default 1024: %d/%d (%.2f%%)",
                sum(n > 1024 for n in lengths), total, _rate(sum(n > 1024 for n in lengths), total))

    logger.info("Exclusion by framework/benchmark (max_len=%d):", args.max_len)
    logger.info("  %-14s %-18s %7s %9s %8s", "framework", "benchmark", "total", "excluded", "rate")
    for group in sorted(total_by_group, key=lambda g: -_rate(excluded_by_group[g], total_by_group[g])):
        logger.info("  %-14s %-18s %7d %9d %7.2f%%", group[0], group[1], total_by_group[group],
                    excluded_by_group[group], _rate(excluded_by_group[group], total_by_group[group]))

    logger.info("Kept %d/%d traces; excluded %d (exclusion rate %.2f%%)",
                kept, total, excluded, _rate(excluded, total))
    logger.info("Filtered set: %s", kept_path)
    logger.info("Excluded (overlong) set: %s", over_path)
    logger.info("Log: %s", log_path)


if __name__ == "__main__":
    main()
