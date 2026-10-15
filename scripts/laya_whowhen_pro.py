"""Zero-shot Laya WHO/WHEN/WHAT attribution on Who&When Pro, with raw logits for calibration.

Full-context mode (default): one `Router.predict` per trace answers three `choice` questions:
  who   - agent responsible (multi-agent frameworks only)
  when  - trajectory step of the decisive mistake
  what  - error mode, over the 15 text modes of taxonomy.yaml
Laya rounds its probabilities to 4 decimals, which makes NLL infinite whenever the true option
rounds to 0, so the raw option logits are captured as well (`laya.agent._option_logits` is
wrapped) and checked against the returned probabilities.

Window mode (`--windows 1024,2048`, E9a/E9b): the trace's steps are packed into consecutive
windows of about that many tokens (whole steps, task header repeated); each window answers
WHEN over its own steps plus a "none of these steps" option. Aggregation is in analyze_claims.py.

Output is appended per trace, so an interrupted job resumes where it stopped.

Usage (from the repository root, on a GPU node):
    python scripts/laya_whowhen_pro.py                         # every trace in the manifest
    python scripts/laya_whowhen_pro.py --windows 1024,2048 --subset window
    python scripts/laya_whowhen_pro.py --windows 1024,2048 --subset overlong
"""
import os
import sys
import json
import time
import zlib
import argparse
import logging

import numpy as np
import torch
import laya.agent
from laya import Router
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from attrib.whowhen_pro import (  # noqa: E402
    WHO_INSTRUCTIONS, WHEN_INSTRUCTIONS, WHAT_INSTRUCTIONS, NONE_OPTION, NONE_TEXT,
    load_items, render_parts, default_pool_path,
)
from attrib.stats import softmax  # noqa: E402

TOKENIZER_REPO = "convaiinnovations/laya-multilingual"  # the one filter_whowhen_pro.py counts with
TOKENIZER_SUBFOLDER = "tokenizer"

logger = logging.getLogger("laya_whowhen_pro")

_captured = []
_option_logits = laya.agent._option_logits


def _capturing_option_logits(logits, items, offset):
    rows = _option_logits(logits, items, offset)
    _captured.append([r.detach().float().cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r, float)
                      for r in rows])
    return rows


laya.agent._option_logits = _capturing_option_logits


def choice(instructions, options):
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def shuffled(questions, seed, trace_id, salt):
    """Copy of `questions` with a seeded random display order per question (Laya `option_order`).

    Slot s shows option order[s]; Laya maps its probabilities back to the original option order.
    The permutation depends only on (seed, trace, salt, question), so reruns reproduce it.
    """
    out = {}
    for qid, qdef in questions.items():
        rng = np.random.default_rng([seed, zlib.crc32(f"{trace_id}|{salt}|{qid}".encode())])
        out[qid] = {**qdef, "option_order": [int(i) for i in rng.permutation(len(qdef["criteria"]))]}
    return out


def run(router, state, questions, args, trace_id="", salt=""):
    """Predict, and attach unrounded probabilities from the captured logits to each answer."""
    if args.order_seed is not None:
        questions = shuffled(questions, args.order_seed, trace_id, salt)
    _captured.clear()
    start = time.perf_counter()
    result = router.predict(state, questions, model=args.model, max_len=args.max_len,
                            head_max_len=args.head_max_len)
    elapsed = time.perf_counter() - start
    if len(_captured) != 1 or len(_captured[0]) != len(questions):
        raise RuntimeError("expected one logit row per question, got %s" % [len(c) for c in _captured])
    out = {}
    for qid, raw in zip(questions, _captured[0]):
        ans = result["answers"][qid]
        keys = list(questions[qid]["criteria"])
        order = questions[qid].get("option_order")
        if order is not None:
            # The model's row is in slot order; put it back in the question's option order.
            canonical = np.empty_like(raw)
            canonical[np.asarray(order)] = raw
            raw = canonical
        probs = softmax(raw)
        reported = np.array([ans["probabilities"][k] for k in keys])
        if np.abs(probs - reported).max() > 2e-3:
            # Laya applied a temperature or reordered options; the raw logits would then not be
            # the distribution being evaluated.
            raise RuntimeError(f"{qid}: softmax(raw logits) differs from Laya's probabilities")
        out[qid] = {"keys": keys, "logits": [round(float(z), 5) for z in raw],
                    "probs": [float(p) for p in probs], "pred": ans["choice"],
                    "answer_confidence": ans.get("answer_confidence"),
                    **({"option_order": order} if order is not None else {})}
    usage = result.get("usage", {})
    meta = {"seconds": elapsed, "input_tokens": usage.get("input_tokens"),
            "truncated": bool(usage.get("truncated", False)),
            "state_tokens_dropped": usage.get("state_tokens_dropped", 0),
            "options_collapsed": usage.get("options") or {}}
    return out, meta


def full_context(router, row, item, args):
    questions = {}
    if item["who_options"]:
        questions["who"] = choice(WHO_INSTRUCTIONS, item["who_options"])
    questions["when"] = choice(WHEN_INSTRUCTIONS, item["when_options"])
    questions["what"] = choice(WHAT_INSTRUCTIONS, item["what_options"])
    answers, meta = run(router, row["rendered_text"], questions, args, trace_id=item["id"])
    return {"id": item["id"], **meta, "answers": answers}


def make_windows(row, item, tokenizer, window_tokens, task_tokens):
    """Consecutive windows of whole candidate steps, each about `window_tokens` long."""
    header, blocks = render_parts(row)
    header_text = "\n".join(header)
    ids = tokenizer(header_text, add_special_tokens=False)["input_ids"]
    if len(ids) > task_tokens:
        header_text = tokenizer.decode(ids[:task_tokens]) + " ..."
    windows, current, used = [], [], 0
    for key in item["when_options"]:
        text = "\n".join(blocks[int(key)])
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if current and used + n > window_tokens:
            windows.append(current)
            current, used = [], 0
        current.append((key, text))
        used += n
    if current:
        windows.append(current)
    return [(header_text + "\n" + "\n".join(t for _, t in w), [k for k, _ in w]) for w in windows]


def windowed(router, row, item, tokenizer, window_tokens, args):
    out = []
    for w_idx, (state, keys) in enumerate(make_windows(row, item, tokenizer, window_tokens, args.task_tokens)):
        options = {k: item["when_options"][k] for k in keys}
        options[NONE_OPTION] = NONE_TEXT
        answers, meta = run(router, state, {"when": choice(WHEN_INSTRUCTIONS, options)}, args,
                            trace_id=item["id"], salt=f"{window_tokens}|{w_idx}")
        out.append({"window": w_idx, "steps": keys, **meta, "when": answers["when"]})
    return {"id": item["id"], "window_tokens": window_tokens, "windows": out}


def done_ids(path, key=lambda r: r["id"]):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {key(json.loads(line)) for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="experiments/manifest.json")
    parser.add_argument("--subset", default="all", choices=["all", "llm", "window", "overlong"])
    parser.add_argument("--windows", default=None, help="Comma-separated window sizes in tokens (E9).")
    parser.add_argument("--task_tokens", type=int, default=256, help="Task header budget per window.")
    parser.add_argument("--model", default="multilingual", help="Only 'multilingual' supports max_len above 1024.")
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--head_max_len", type=int, default=384,
                        help="Tokens for the question and its options; 30 WHEN options need more than "
                             "Laya's default 192 to keep each option distinct.")
    parser.add_argument("--order_seed", type=int, default=None,
                        help="Show each question's options in a seeded random order instead of trace order "
                             "(sensitivity check for position bias); the seed is added to the output tag.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N traces (smoke test).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="outputs/pro")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"laya_windows_{args.subset}" if args.windows else f"laya_full_{args.subset}"
    if args.order_seed is not None:
        tag += f"_shuf{args.order_seed}"
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(args.out_dir, f"{tag}.log"), mode="a")])
    logger.setLevel(logging.INFO)

    with open(args.manifest) as f:
        manifest = json.load(f)
    if args.subset == "overlong":
        pool = default_pool_path("overlong", os.path.dirname(manifest["pool"]))
        wanted = set(manifest["overlong"])
    else:
        pool = manifest["pool"]
        wanted = set(manifest["split"]) if args.subset == "all" else set(manifest["subsets"][args.subset])
    pairs = [(row, it) for row, it in load_items(pool) if it["id"] in wanted]
    if args.limit:
        pairs = pairs[:args.limit]

    out_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    sizes = [int(w) for w in args.windows.split(",")] if args.windows else [None]
    done = done_ids(out_path, key=lambda r: (r["id"], r.get("window_tokens")))
    logger.info("%s: %d traces from %s, %d records already in %s", tag, len(pairs), pool, len(done), out_path)
    logger.info("Laya '%s', max_len %d, head_max_len %d, device %s",
                args.model, args.max_len, args.head_max_len, args.device)

    router = Router(device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder=TOKENIZER_SUBFOLDER) if args.windows else None

    n_new = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for row, item in pairs:
            for size in sizes:
                if (item["id"], size) in done:
                    continue
                if size is None:
                    rec = full_context(router, row, item, args)
                else:
                    rec = windowed(router, row, item, tokenizer, size, args)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_new += 1
                if n_new % 100 == 0:
                    logger.info("  %d records written", n_new)
    logger.info("Wrote %d new records to %s", n_new, out_path)


if __name__ == "__main__":
    main()
