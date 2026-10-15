"""All-at-once LLM baseline on Who&When Pro with verbalized confidence (shared setup, C1.1-C1.4, C3.1).

The model sees the rendered trace (no reference answer) plus the option lists Laya gets, and
returns JSON with an agent, a step, an error mode and a 0-100 confidence for each. Input/output
tokens and wall-clock seconds are recorded per trace for the cascade's cost (E4).

Backends:
  hf      transformers on a local GPU (default openai/gpt-oss-20b from the OSC cache)
  openai  any OpenAI-compatible endpoint (e.g. a vLLM server for gpt-oss-120b): --base_url, --model

Usage (from the repository root):
    python scripts/llm_whowhen_pro.py --subset llm
    python scripts/llm_whowhen_pro.py --backend openai --base_url http://host:8000/v1 --model openai/gpt-oss-120b
"""
import os
import re
import sys
import json
import time
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from attrib.whowhen_pro import load_items, load_taxonomy  # noqa: E402

logger = logging.getLogger("llm_whowhen_pro")

SYSTEM = ("You are an expert at debugging AI agent systems. You read the full log of a failed "
          "agent run and identify the decisive mistake that caused the failure.")


def build_prompt(row, item, taxonomy):
    parts = [
        "Below is the log of an agent run that failed to solve its task. Exactly one decisive "
        "mistake caused the failure.",
        "", "=== LOG ===", row["rendered_text"], "=== END LOG ===", "",
    ]
    if item["who_options"]:
        parts.append("Agents: " + ", ".join(item["who_options"]))
    parts.append("Steps you may choose from: " + ", ".join(item["when_options"]))
    parts.append("Error modes:")
    for code in item["what_options"]:
        parts.append(f"- {code} {taxonomy[code]['name']}: {' '.join(taxonomy[code]['description'].split())}")
    parts += [
        "",
        "Identify " + ("which agent made the decisive mistake, " if item["who_options"] else "")
        + "at which step it happened (the number in '[Step N ...]'), and which error mode it is. "
        "For each answer give your confidence from 0 to 100 that it is correct.",
        "Reply with only a JSON object:",
        '{"agent": "<agent>", "agent_confidence": <0-100>, "step": <N>, "step_confidence": <0-100>, '
        '"mode": "<code>", "mode_confidence": <0-100>}' if item["who_options"] else
        '{"step": <N>, "step_confidence": <0-100>, "mode": "<code>", "mode_confidence": <0-100>}',
    ]
    return "\n".join(parts)


def parse_reply(text, item):
    """Predictions and confidences in [0, 1]; a field that fails to parse is None."""
    # gpt-oss puts reasoning before the final channel; the answer is the last JSON object.
    objs = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    data = None
    for raw in reversed(objs):
        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError:
            continue
    out = {"parsed": data is not None}
    data = data or {}

    def conf(key):
        try:
            v = float(data.get(key))
            return min(max(v / 100.0, 0.0), 1.0)
        except (TypeError, ValueError):
            return None

    if item["who_options"]:
        agent = str(data.get("agent", "")).strip()
        match = {a.lower(): a for a in item["who_options"]}.get(agent.lower())
        out["who"] = {"pred": match, "confidence": conf("agent_confidence")}
    step = re.search(r"\d+", str(data.get("step", "")))
    step = step.group(0) if step else None
    out["when"] = {"pred": step if step in item["when_options"] else None, "confidence": conf("step_confidence")}
    mode = str(data.get("mode", "")).strip().upper()
    mode = next((c for c in item["what_options"] if mode == c.upper() or mode.startswith(c.upper() + " ")), None)
    out["what"] = {"pred": mode, "confidence": conf("mode_confidence")}
    return out


class HFBackend:
    def __init__(self, model, device, max_new_tokens, reasoning_effort):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map=device)
        self.max_new_tokens = max_new_tokens
        self.template_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        self.torch = torch

    def __call__(self, prompt):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                              return_dict=True, **self.template_kwargs).to(self.model.device)
        n_in = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0, n_in:]
        return self.tok.decode(new, skip_special_tokens=False), n_in, len(new)


class OpenAIBackend:
    def __init__(self, model, base_url, max_new_tokens, reasoning_effort):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        self.model, self.max_new_tokens = model, max_new_tokens
        self.extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def __call__(self, prompt):
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=self.max_new_tokens,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], **self.extra)
        return resp.choices[0].message.content or "", resp.usage.prompt_tokens, resp.usage.completion_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="experiments/manifest.json")
    parser.add_argument("--subset", default="llm", choices=["llm", "all"])
    parser.add_argument("--backend", default="hf", choices=["hf", "openai"])
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--device", default="auto", help="hf device_map")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--reasoning_effort", default="medium",
                        help="gpt-oss reasoning effort; pass '' for models without it.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent requests (openai backend only). With more than one, `seconds` is "
                             "latency under shared load, not the cost of one request; token counts stay exact.")
    parser.add_argument("--taxonomy", default="data/whowhen_pro/taxonomy.yaml")
    parser.add_argument("--out_dir", default="outputs/pro")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"llm_{args.model.split('/')[-1]}_{args.subset}"
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(args.out_dir, f"{tag}.log"), mode="a")])
    logger.setLevel(logging.INFO)

    with open(args.manifest) as f:
        manifest = json.load(f)
    wanted = set(manifest["split"]) if args.subset == "all" else set(manifest["subsets"]["llm"])
    pairs = [(row, it) for row, it in load_items(manifest["pool"]) if it["id"] in wanted]
    if args.limit:
        pairs = pairs[:args.limit]
    taxonomy = load_taxonomy(args.taxonomy)

    out_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    logger.info("%s: %d traces, %d already done; backend %s", tag, len(pairs), len(done), args.backend)

    if args.backend == "hf":
        llm = HFBackend(args.model, args.device, args.max_new_tokens, args.reasoning_effort)
    else:
        llm = OpenAIBackend(args.model, args.base_url, args.max_new_tokens, args.reasoning_effort)

    todo = [(row, item) for row, item in pairs if item["id"] not in done]
    lock = threading.Lock()
    n_new = n_unparsed = 0

    def score(row, item):
        start = time.perf_counter()
        text, n_in, n_out = llm(build_prompt(row, item, taxonomy))
        seconds = time.perf_counter() - start
        parsed = parse_reply(text, item)
        return parsed, {"id": item["id"], "model": args.model, "input_tokens": n_in, "output_tokens": n_out,
                        "seconds": seconds, **parsed, "reply_tail": text[-1500:]}

    with open(out_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score, row, item) for row, item in todo]
        for fut in as_completed(futures):
            parsed, rec = fut.result()
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_new += 1
                n_unparsed += not parsed["parsed"]
                if n_new % 20 == 0:
                    logger.info("  %d done (%d unparsed)", n_new, n_unparsed)
    logger.info("Wrote %d records (%d unparsed) to %s", n_new, n_unparsed, out_path)


if __name__ == "__main__":
    main()
