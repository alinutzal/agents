"""Zero-shot WHO/WHEN failure attribution on Who&When with Laya, compared with chance.

For every trace, Laya (laya-multilingual, max_len 8192) answers two `choice` questions in one
forward pass:
  WHO  - which agent is responsible for the failure (options: the agents in the trace)
  WHEN - at which step the decisive mistake happened (options: the trace's steps, 0-indexed)

Random baseline per trace is 1/#agents (WHO) and 1/#steps (WHEN); these are averaged over traces
to give the expected accuracy of uniform guessing. The one-sided p-value is P(X >= #correct) where
X is the number correct under that guesser, i.e. a sum of Bernoulli(1/n_i) (Poisson-binomial).

Usage (from the repository root, on a GPU node):
    uv run python scripts/laya_zero_shot_whowhen.py
    uv run python scripts/laya_zero_shot_whowhen.py --no_answer --device cpu
"""
import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from laya import Router
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

TOKENIZER_REPO = "convaiinnovations/laya"  # the bundle repo the Router loads
TOKENIZER_SUBFOLDER = "multilingual/tokenizer"

logger = logging.getLogger("laya_zero_shot_whowhen")

WHO_INSTRUCTIONS = ("Which agent made the mistake that is directly responsible for the wrong "
                    "solution to the task?")
WHEN_INSTRUCTIONS = ("At which step did the first decisive mistake that led to the wrong solution "
                     "happen?")


def render_state(row, agent_key, include_answer):
    lines = [f"Task: {row['question']}"]
    if include_answer:
        lines.append(f"Correct answer: {row['ground_truth']}")
    lines.append("")
    lines.append("Conversation:")
    for idx, entry in enumerate(row["history"]):
        lines.append(f"Step {idx} - {entry.get(agent_key) or 'Unknown Agent'}: {entry.get('content', '')}")
    return "\n".join(lines)


def build_questions(history, agent_key):
    speakers = [entry.get(agent_key) or "Unknown Agent" for entry in history]
    agents = list(dict.fromkeys(speakers))  # unique, in order of first appearance
    who = {"type": "choice", "instructions": WHO_INSTRUCTIONS,
           "criteria": {name: f"the agent {name}" for name in agents}}
    when = {"type": "choice", "instructions": WHEN_INSTRUCTIONS,
            "criteria": {str(i): f"step {i}, by {speaker}" for i, speaker in enumerate(speakers)}}
    return {"who": who, "when": when}, agents


def poisson_binomial_sf(probs, k):
    """P(X >= k) for X = sum of independent Bernoulli(p_i)."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum())


def summarize(name, correct, chance, true_prob):
    n = len(correct)
    k = int(sum(correct))
    acc = k / n
    base = float(np.mean(chance))
    p_value = poisson_binomial_sf(chance, k)
    logger.info("%-4s accuracy %6.2f%% (%d/%d) | random baseline %6.2f%% | lift %+6.2f pts | "
                "mean p(true) %.4f | one-sided p = %.3g",
                name, 100 * acc, k, n, 100 * base, 100 * (acc - base), float(np.mean(true_prob)), p_value)
    return {"accuracy": acc, "correct": k, "n": n, "random_baseline": base,
            "lift": acc - base, "mean_true_prob": float(np.mean(true_prob)), "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, default="data/Who_and_When/Algorithm-Generated.parquet")
    parser.add_argument("--is_handcrafted", action="store_true",
                        help="Hand-Crafted traces name agents by 'role' instead of 'name'.")
    parser.add_argument("--no_answer", action="store_true",
                        help="Do not show the task's correct answer in the state.")
    parser.add_argument("--model", type=str, default="multilingual",
                        help="Laya checkpoint; only 'multilingual' supports max_len above 1024.")
    parser.add_argument("--max_len", type=int, default=8192,
                        help="Laya context length in tokens (Laya's default is 1024).")
    parser.add_argument("--drop_overlong", action="store_true",
                        help="Skip traces whose state exceeds max_len tokens instead of letting Laya truncate them.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    split = os.path.splitext(os.path.basename(args.data))[0]
    tag = f"laya_zero_shot_{split}{'_no_answer' if args.no_answer else ''}"
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(args.out_dir, f"{tag}.log"), mode="w")])
    logger.setLevel(logging.INFO)

    agent_key = "role" if args.is_handcrafted else "name"
    df = pd.read_parquet(args.data)
    logger.info("Data: %s (%d traces), agent key '%s', answer shown: %s",
                args.data, len(df), agent_key, not args.no_answer)
    logger.info("Laya checkpoint '%s', max_len %d, device %s", args.model, args.max_len, args.device)

    # Tokenize every trace with Laya's tokenizer to see which fit in the context window.
    # Load from the local snapshot: with HF_HUB_OFFLINE=1, AutoTokenizer on the hub id fails
    # looking up a config.json this repo does not have.
    snapshot = snapshot_download(TOKENIZER_REPO, allow_patterns=[f"{TOKENIZER_SUBFOLDER}/*"])
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(snapshot, TOKENIZER_SUBFOLDER))
    states = [render_state(row, agent_key, include_answer=not args.no_answer) for _, row in df.iterrows()]
    n_tokens = [len(ids) for ids in tokenizer(states, add_special_tokens=True, truncation=False)["input_ids"]]
    overlong = [n > args.max_len for n in n_tokens]
    logger.info("State tokens (%s): min %d, median %d, max %d; over %d: %d/%d (%.2f%%); over 1024: %d/%d",
                TOKENIZER_REPO, min(n_tokens), int(np.median(n_tokens)), max(n_tokens), args.max_len,
                sum(overlong), len(df), 100 * sum(overlong) / len(df), sum(n > 1024 for n in n_tokens), len(df))
    logger.info("Laya also spends up to head_max_len tokens on each question, so a state near "
                "max_len can still be cut; per-trace truncation is recorded below.")
    if args.drop_overlong:
        logger.info("Excluding %d overlong traces (exclusion rate %.2f%%)",
                    sum(overlong), 100 * sum(overlong) / len(df))

    router = Router(device=args.device)

    records = []
    for (i, row), state, n_tok, too_long in zip(df.iterrows(), states, n_tokens, overlong):
        if args.drop_overlong and too_long:
            continue
        history = list(row["history"])
        questions, agents = build_questions(history, agent_key)
        result = router.predict(state, questions, model=args.model, max_len=args.max_len)

        who, when = result["answers"]["who"], result["answers"]["when"]
        usage = result.get("usage", {})
        true_agent, true_step = str(row["mistake_agent"]), str(row["mistake_step"])
        records.append({
            "question_ID": row["question_ID"],
            "n_agents": len(agents),
            "n_steps": len(history),
            "true_agent": true_agent,
            "true_step": true_step,
            "pred_agent": who["choice"],
            "pred_step": str(when["choice"]),
            "agent_correct": who["choice"] == true_agent,
            "step_correct": str(when["choice"]) == true_step,
            "p_true_agent": who["probabilities"].get(true_agent, 0.0),
            "p_true_step": when["probabilities"].get(true_step, 0.0),
            "who_probabilities": who["probabilities"],
            "when_probabilities": when["probabilities"],
            "state_tokens_laya_tokenizer": n_tok,
            "input_tokens": usage.get("input_tokens"),
            "state_tokens_dropped": usage.get("state_tokens_dropped", 0),
            "truncated": bool(usage.get("truncated", False)),
        })
        if (len(records)) % 20 == 0:
            logger.info("  %d traces done", len(records))

    preds_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    missing_agent = sum(r["true_agent"] not in r["who_probabilities"] for r in records)
    truncated = sum(r["truncated"] for r in records)
    logger.info("Truncated states: %d/%d (these lose the end of the conversation)", truncated, len(records))
    if missing_agent:
        logger.info("Traces whose labelled agent never speaks (WHO cannot be right): %d", missing_agent)

    logger.info("Results (%d traces):", len(records))
    summary = {
        "data": args.data, "model": args.model, "max_len": args.max_len,
        "answer_shown": not args.no_answer, "truncated": truncated,
        "n_total": len(df), "n_overlong": sum(overlong), "dropped_overlong": args.drop_overlong,
        "who": summarize("WHO", [r["agent_correct"] for r in records],
                         [1 / r["n_agents"] for r in records], [r["p_true_agent"] for r in records]),
        "when": summarize("WHEN", [r["step_correct"] for r in records],
                          [1 / r["n_steps"] for r in records], [r["p_true_step"] for r in records]),
    }
    summary_path = os.path.join(args.out_dir, f"{tag}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Per-trace predictions: %s", preds_path)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
