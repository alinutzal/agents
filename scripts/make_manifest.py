"""Freeze the Who&When Pro splits and subsets every experiment reads (shared setup).

  split          dev / test over the traces that fit in 8K, stratified by framework x benchmark x
                 length stratum (< 3K / 3-8K). Temperature, cascade threshold, combiners and the
                 length-only baseline are fitted on dev; nothing is fitted on test.
  llm            stratified subset scored by the LLM baseline, with long traces oversampled to
                 `long_share` so C1.3/C1.4 have enough of them.
  window         subset scored window-by-window for E9a.
  overlong       every trace over 8K (E9b), if the filter wrote them.

Laya is cheap enough to score every trace, so P0/C1.1-C1.3 use the whole test split; paired
Laya-vs-LLM comparisons use the `llm` subset.

Usage (from the repository root):
    python scripts/make_manifest.py
"""
import os
import sys
import json
import argparse
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from attrib.whowhen_pro import load_items, check_item, default_pool_path  # noqa: E402


def stratified_split(items, dev_fraction, rng):
    groups = defaultdict(list)
    for it in items:
        groups[(it["framework"], it["benchmark"], it["stratum"])].append(it["id"])
    split = {}
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        n_dev = int(round(dev_fraction * len(ids)))
        for i, tid in enumerate(ids):
            split[tid] = "dev" if i < n_dev else "test"
    return split


def stratified_sample(items, n, long_share, rng):
    """n traces, `long_share` of them long, each half spread over frameworks proportionally."""
    chosen = []
    for stratum, share in (("long", long_share), ("short", 1 - long_share)):
        pool = [it for it in items if it["stratum"] == stratum]
        want = min(len(pool), int(round(n * share)))
        by_fw = defaultdict(list)
        for it in pool:
            by_fw[it["framework"]].append(it["id"])
        # Largest-remainder allocation, at least one per framework while budget lasts.
        quotas = {fw: want * len(ids) / len(pool) for fw, ids in by_fw.items()}
        alloc = {fw: min(len(by_fw[fw]), max(1, int(q))) for fw, q in quotas.items()}
        for fw in sorted(quotas, key=lambda f: -(quotas[f] - int(quotas[f]))):
            if sum(alloc.values()) >= want:
                break
            if alloc[fw] < len(by_fw[fw]):
                alloc[fw] += 1
        for fw in sorted(by_fw):
            ids = sorted(by_fw[fw])
            rng.shuffle(ids)
            chosen.extend(ids[:alloc[fw]])
    # Top up from the other stratum if one ran short.
    if len(chosen) < n:
        rest = sorted(set(it["id"] for it in items) - set(chosen))
        rng.shuffle(rest)
        chosen.extend(rest[:n - len(chosen)])
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prereg", default="experiments/prereg.json")
    parser.add_argument("--data_dir", default="data/whowhen_pro")
    parser.add_argument("--out", default="experiments/manifest.json")
    args = parser.parse_args()

    with open(args.prereg) as f:
        prereg = json.load(f)
    cfg = prereg["data"]
    rng = np.random.default_rng(prereg["seed"])

    items = [it for _, it in load_items(default_pool_path("fit", args.data_dir, cfg["max_len"]))]
    bad = {it["id"]: check_item(it) for it in items if check_item(it)}
    items = [it for it in items if it["id"] not in bad]
    split = stratified_split(items, cfg["dev_fraction"], rng)

    subsets = {}
    for name, sub_cfg in (("llm", cfg["llm_subset"]), ("window", cfg["window_subset"])):
        ids = []
        for part in ("dev", "test"):
            part_items = [it for it in items if split[it["id"]] == part]
            ids += stratified_sample(part_items, sub_cfg[part], sub_cfg.get("long_share", 0.5), rng)
        subsets[name] = ids

    over_path = default_pool_path("overlong", args.data_dir, cfg["max_len"])
    overlong = []
    if os.path.exists(over_path):
        over_items = [it for _, it in load_items(over_path)]
        bad.update({it["id"]: check_item(it) for it in over_items if check_item(it)})
        overlong = [it["id"] for it in over_items if not check_item(it)]

    manifest = {
        "prereg": args.prereg, "seed": prereg["seed"],
        "pool": default_pool_path("fit", args.data_dir, cfg["max_len"]),
        "excluded_unusable_labels": bad,
        "split": split,
        "subsets": subsets,
        "overlong": overlong,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)

    by = Counter((it["framework"], it["stratum"], split[it["id"]]) for it in items)
    print(f"{len(items)} usable traces ({len(bad)} with unusable labels); overlong: {len(overlong)}")
    for fw in sorted({k[0] for k in by}):
        print(f"  {fw:13s} " + "  ".join(f"{s}/{p}={by[(fw, s, p)]:4d}"
                                          for s in ("short", "long") for p in ("dev", "test")))
    lookup = {it["id"]: it for it in items}
    for name, ids in subsets.items():
        c = Counter((split[i], lookup[i]["stratum"]) for i in ids)
        print(f"  subset {name}: {len(ids)} " + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(c.items())))
    if not overlong:
        print(f"  no {over_path}; re-run scripts/filter_whowhen_pro.py to enable E9b")
    print(f"Manifest: {args.out}")


if __name__ == "__main__":
    main()
