"""Decide the G1/G3/G4 claims of the claim -> experiment map from saved predictions.

Reads the manifest, experiments/prereg.json, Laya full-context records, and (when present) the
LLM baseline and Laya window records. Writes a JSON with every number and a Markdown table with
one verdict per claim: SUPPORTED, FALSIFIED or INCONCLUSIVE (neither rule fired), or SKIPPED when
its inputs are missing. Thresholds come only from prereg.json.

  P0     accuracy vs per-trace random expectation (test)
  E1     C1.1  ECE / Brier / NLL / reliability, Laya vs LLM verbalized confidence
  E1b    C1.1b leave-one-framework-out temperature transfer
  E2     C1.2  error-detection AUROC, AURC, selective accuracy, length-only baseline (DeLong)
  E3     C1.3  confidence - accuracy gap by length stratum
  E4     C1.4  confidence-gated cascade vs length / random / oracle routers
  E8     C3.1  WHAT: macro-F1, confusion, error-detection AUROC, per-family ECE
  E9     C4.1  window aggregation (E9a on windowed test traces, E9b on traces over 8K)

An LLM reply that cannot be parsed counts as a wrong answer with confidence 0.

Usage (from the repository root):
    python scripts/analyze_claims.py --llm outputs/pro/llm_gpt-oss-20b_llm.jsonl
"""
import os
import sys
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from attrib import stats as S  # noqa: E402
from attrib.whowhen_pro import load_items, default_pool_path, mode_family, NONE_OPTION  # noqa: E402

QTYPES = ("who", "when", "what")


def read_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verdict(supported, falsified):
    if supported and not falsified:
        return "SUPPORTED"
    if falsified:
        return "FALSIFIED"
    return "INCONCLUSIVE"


class Table:
    """Per-trace rows for one question type, with column access by name."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def col(self, key, dtype=float):
        return np.array([r[key] for r in self.rows], dtype=dtype)

    def where(self, pred):
        return Table([r for r in self.rows if pred(r)])

    def strata(self):
        return np.array([f"{r['framework']}|{r['stratum']}" for r in self.rows])


def build_tables(items, split, laya, llm):
    tables = {q: [] for q in QTYPES}
    for rec in laya:
        it = items.get(rec["id"])
        if it is None or rec["id"] not in split:
            continue
        for q in QTYPES:
            ans = rec["answers"].get(q)
            if ans is None:
                continue
            accepted = set(it[f"{q}_true"])
            probs = np.array(ans["probs"])
            pred = ans["keys"][int(probs.argmax())]
            row = {
                "id": it["id"], "split": split[it["id"]], "framework": it["framework"],
                "benchmark": it["benchmark"], "stratum": it["stratum"], "n_tokens": it["n_tokens"],
                "n_steps": it["n_steps"], "n_agents": it["n_agents"],
                "keys": ans["keys"], "logits": ans["logits"], "probs": probs,
                "true_mask": [k in accepted for k in ans["keys"]],
                "random": len(accepted & set(ans["keys"])) / len(ans["keys"]),
                "pred": pred, "conf": float(probs.max()), "correct": pred in accepted,
                "accepted": accepted, "laya_seconds": rec["seconds"],
                "truncated": rec["truncated"],
            }
            l_rec = llm.get(it["id"])
            if l_rec is not None and q in l_rec:
                lp = l_rec[q]
                ok = l_rec["parsed"] and lp["pred"] is not None and lp["confidence"] is not None
                row.update({"llm_pred": lp["pred"], "llm_conf": lp["confidence"] if ok else 0.0,
                            "llm_correct": bool(ok and lp["pred"] in accepted), "llm_parsed": ok,
                            "llm_seconds": l_rec["seconds"], "llm_in": l_rec["input_tokens"],
                            "llm_out": l_rec["output_tokens"]})
            tables[q].append(row)
    return {q: Table(rows) for q, rows in tables.items()}


def boot(fn, t, cfg, seed_offset=0):
    return S.bootstrap(fn, len(t), strata=t.strata(), n_boot=cfg["n_boot"], seed=cfg["seed"] + seed_offset,
                       alpha=cfg["alpha"])


def acc_fn(c):
    return lambda i: float(c[i].mean())


# ---------------------------------------------------------------- P0

def majority_correct(tables_q):
    """Per-test-trace 0/1 for "always guess the most common accepted label of the trace's framework".

    The label is the key (agent name or step index) that is accepted most often among the
    framework's dev traces; frameworks with no dev traces fall back to the pooled dev majority.
    """
    dev = tables_q.where(lambda r: r["split"] == "dev")
    counts = defaultdict(Counter)
    for r in dev.rows:
        counts[r["framework"]].update(r["accepted"])
    pooled = sum(counts.values(), Counter())
    guess = {fw: c.most_common(1)[0][0] for fw, c in counts.items() if c}
    fallback = pooled.most_common(1)[0][0] if pooled else None
    test = tables_q.where(lambda r: r["split"] == "test")
    return np.array([guess.get(r["framework"], fallback) in r["accepted"] for r in test.rows], float)


def p0(tables, cfg):
    out = {}
    for q in ("who", "when"):
        t = tables[q].where(lambda r: r["split"] == "test")
        if not len(t):
            continue
        c, rnd = t.col("correct"), t.col("random")
        maj = majority_correct(tables[q])
        a = boot(acc_fn(c), t, cfg)
        # Paired: Laya minus the majority-label guesser on the same test traces.
        vs_maj = boot(lambda i: float(c[i].mean() - maj[i].mean()), t, cfg, 14)
        per_fw = {}
        for fw in sorted({r["framework"] for r in t.rows}):
            m = np.array([r["framework"] == fw for r in t.rows])
            per_fw[fw] = {"n": int(m.sum()), "accuracy": float(c[m].mean()), "random": float(rnd[m].mean()),
                          "majority": float(maj[m].mean())}
        out[q] = {"n": len(t), "accuracy": a, "random": float(rnd.mean()),
                  "majority_accuracy": boot(acc_fn(maj), t, cfg, 15),
                  "laya_minus_majority": vs_maj,
                  "p_value": S.poisson_binomial_sf(rnd, int(c.sum())), "per_framework": per_fw,
                  "beats_random": a["lo"] > rnd.mean(), "beats_majority": vs_maj["lo"] > 0}
    sup = all(v["beats_random"] for v in out.values()) and len(out) == 2
    fal = "when" in out and out["when"]["accuracy"]["lo"] <= out["when"]["random"]
    # Added 2026-10-15 (amendment 2): the same rule against the majority-label baseline.
    sup_maj = sup and all(v["beats_majority"] for v in out.values())
    return {"verdict": verdict(sup, fal), "verdict_vs_majority": verdict(sup_maj, not sup_maj), **out}


# ---------------------------------------------------------------- E1 / C1.1

def calib_block(conf, correct):
    return {"ece_width": S.ece(conf, correct, 10, "width"), "ece_mass": S.ece(conf, correct, 10, "mass"),
            "brier_top": S.brier_top(conf, correct), "binary_nll": S.binary_nll(conf, correct),
            "accuracy": float(np.mean(correct)), "mean_conf": float(np.mean(conf))}


def e1(tables, pre, cfg):
    rule = pre["C1.1"]
    scheme = rule["ece_scheme_for_verdict"]
    out = {}
    for q in ("who", "when"):
        t = tables[q].where(lambda r: r["split"] == "test")
        if not len(t):
            continue
        conf, c = t.col("conf"), t.col("correct")
        masks = [r["true_mask"] for r in t.rows]
        res = {"n": len(t), "laya": calib_block(conf, c),
               "laya_full_nll": S.nll_full([r["logits"] for r in t.rows], masks),
               "laya_multiclass_brier": S.multiclass_brier([r["probs"] for r in t.rows], masks),
               "reliability": S.reliability_bins(conf, c, rule["ece_bins"])}
        res["laya_ece_ci"] = boot(lambda i: S.ece(conf[i], c[i], rule["ece_bins"], scheme), t, cfg)
        p = t.where(lambda r: "llm_conf" in r)
        if len(p):
            lc, lk, yc, yk = p.col("llm_conf"), p.col("llm_correct"), p.col("conf"), p.col("correct")
            res["paired_n"] = len(p)
            res["paired_llm"] = calib_block(lc, lk)
            res["paired_laya"] = calib_block(yc, yk)
            res["llm_unparsed"] = int((~p.col("llm_parsed", bool)).sum())
            res["llm_reliability"] = S.reliability_bins(lc, lk, rule["ece_bins"])
            res["ece_diff_llm_minus_laya"] = boot(
                lambda i: S.ece(lc[i], lk[i], rule["ece_bins"], scheme) - S.ece(yc[i], yk[i], rule["ece_bins"], scheme),
                p, cfg, 1)
            res["brier_diff_llm_minus_laya"] = boot(
                lambda i: S.brier_top(lc[i], lk[i]) - S.brier_top(yc[i], yk[i]), p, cfg, 2)
        sup = res["laya_ece_ci"]["hi"] < rule["ece_max"] and res.get("ece_diff_llm_minus_laya", {}).get("lo", -1) > 0
        fal = res["laya_ece_ci"]["lo"] > rule["ece_max"] or res.get("ece_diff_llm_minus_laya", {}).get("hi", 1) < 0
        res["verdict"] = verdict(sup, fal)
        out[q] = res
    return out


# ---------------------------------------------------------------- E1b / C1.1b

def e1b(tables, cfg):
    out = {}
    for q in ("who", "when"):
        t = tables[q]
        dev, test = t.where(lambda r: r["split"] == "dev"), t.where(lambda r: r["split"] == "test")
        per_fw, before, after, fw_of = {}, [], [], []
        for fw in sorted({r["framework"] for r in test.rows}):
            train = dev.where(lambda r, fw=fw: r["framework"] != fw)
            held = test.where(lambda r, fw=fw: r["framework"] == fw)
            if len(train) < 10 or not len(held):
                continue
            temp = S.fit_temperature([r["logits"] for r in train.rows], [r["true_mask"] for r in train.rows])
            for r in held.rows:
                before.append(S.nll_full([r["logits"]], [r["true_mask"]], 1.0))
                after.append(S.nll_full([r["logits"]], [r["true_mask"]], temp))
                fw_of.append(fw)
            b, a = np.mean(before[-len(held):]), np.mean(after[-len(held):])
            per_fw[fw] = {"n": len(held), "temperature": temp, "nll_before": float(b), "nll_after": float(a),
                          "improves": bool(a < b)}
        if not per_fw:
            continue
        diff = np.array(after) - np.array(before)
        ci = S.bootstrap(lambda i: float(diff[i].mean()), len(diff), strata=np.array(fw_of),
                         n_boot=cfg["n_boot"], seed=cfg["seed"] + 3)
        n_imp = sum(v["improves"] for v in per_fw.values())
        sup = ci["hi"] < 0 and n_imp > len(per_fw) / 2
        fal = (len(per_fw) - n_imp) > len(per_fw) / 2
        out[q] = {"per_framework": per_fw, "pooled_nll_diff_after_minus_before": ci,
                  "frameworks_improved": f"{n_imp}/{len(per_fw)}", "verdict": verdict(sup, fal)}
    return out


# ---------------------------------------------------------------- E2 / C1.2

def length_features(t):
    return np.column_stack([np.log(t.col("n_tokens")), t.col("n_steps"), t.col("n_agents")])


def e2(tables, pre, cfg):
    out = {}
    for q in ("who", "when"):
        dev = tables[q].where(lambda r: r["split"] == "dev")
        t = tables[q].where(lambda r: r["split"] == "test")
        if not len(t) or not len(dev):
            continue
        wrong, score = 1 - t.col("correct"), 1 - t.col("conf")
        res = {"n": len(t), "error_rate": float(wrong.mean())}
        res["laya_auroc"] = boot(lambda i: S.auroc(score[i], wrong[i]), t, cfg, 4)
        res["laya_aurc"] = S.aurc(t.col("conf"), t.col("correct"))
        res["laya_selective_accuracy"] = {str(c): S.selective_accuracy(t.col("conf"), t.col("correct"), c)
                                          for c in pre["C1.2"]["coverages"]}
        lr = LogisticRegression(max_iter=1000).fit(length_features(dev), 1 - dev.col("correct"))
        length_score = lr.predict_proba(length_features(t))[:, 1]
        res["length_only_auroc"] = S.auroc(length_score, wrong)
        res["delong_laya_vs_length"] = S.delong_paired(wrong, score, length_score)
        p = t.where(lambda r: "llm_conf" in r)
        if len(p):
            pw = 1 - p.col("llm_correct")
            res["paired_llm_auroc"] = S.auroc(1 - p.col("llm_conf"), pw)
            res["paired_laya_auroc"] = S.auroc(1 - p.col("conf"), 1 - p.col("correct"))
            res["llm_aurc"] = S.aurc(p.col("llm_conf"), p.col("llm_correct"))
        dl = res["delong_laya_vs_length"]
        sup = res["laya_auroc"]["lo"] > 0.5 and dl["diff"] > 0 and dl["p_one_sided_1_gt_2"] < cfg["alpha"]
        fal = res["laya_auroc"]["lo"] <= 0.5 or res["length_only_auroc"] >= res["laya_auroc"]["point"]
        res["verdict"] = verdict(sup, fal)
        out[q] = res
    return out


# ---------------------------------------------------------------- E3 / C1.3

def e3(tables, pre, cfg):
    tol = pre["C1.3"]["gap_tolerance"]
    out = {}
    for q in ("who", "when"):
        t = tables[q].where(lambda r: r["split"] == "test")
        if not len(t):
            continue
        res = {}
        for s in ("short", "long"):
            st = t.where(lambda r, s=s: r["stratum"] == s)
            if not len(st):
                continue
            conf, c = st.col("conf"), st.col("correct")
            res[s] = {"n": len(st), "accuracy": boot(acc_fn(c), st, cfg),
                      "mean_conf": float(conf.mean()),
                      "gap": boot(lambda i: float(conf[i].mean() - c[i].mean()), st, cfg, 5)}
        if len(res) == 2:
            conf, c, long_ = t.col("conf"), t.col("correct"), np.array([r["stratum"] == "long" for r in t.rows])

            def gap_diff(i):
                li, si = i[long_[i]], i[~long_[i]]
                return float((conf[li].mean() - c[li].mean()) - (conf[si].mean() - c[si].mean()))

            def acc_diff(i):
                li, si = i[long_[i]], i[~long_[i]]
                return float(c[li].mean() - c[si].mean())
            res["gap_long_minus_short"] = boot(gap_diff, t, cfg, 6)
            res["accuracy_long_minus_short"] = boot(acc_diff, t, cfg, 7)
            falls = res["accuracy_long_minus_short"]["point"] < 0
            flat = all(abs(res[s]["gap"]["point"]) <= tol for s in ("short", "long"))
            sup = falls and flat and res["gap_long_minus_short"]["lo"] <= 0
            fal = falls and res["gap_long_minus_short"]["lo"] > 0
            res["verdict"] = verdict(sup, fal)
        out[q] = res
    return out


# ---------------------------------------------------------------- E4 / C1.4

def cascade_acc(laya_c, llm_c, escalate):
    return float(np.where(escalate, llm_c, laya_c).mean())


def e4(tables, pre, cfg, prices):
    rule = pre["C1.4"]
    out = {}
    for q in ("who", "when"):
        t = tables[q].where(lambda r: "llm_conf" in r)
        dev, test = t.where(lambda r: r["split"] == "dev"), t.where(lambda r: r["split"] == "test")
        if not len(dev) or not len(test):
            continue
        dconf, dl, dm = dev.col("conf"), dev.col("correct"), dev.col("llm_correct")
        llm_dev = float(dm.mean())
        sweep = []
        for tau in np.unique(np.concatenate([[0.0, 1.01], dconf])):
            esc = dconf < tau
            sweep.append({"tau": float(tau), "escalation": float(esc.mean()), "accuracy": cascade_acc(dl, dm, esc)})
        ok = [s for s in sweep if s["accuracy"] >= llm_dev - rule["delta"]]
        best = min(ok, key=lambda s: (s["escalation"], s["tau"]))
        tau = best["tau"]

        conf, lc, mc = test.col("conf"), test.col("correct"), test.col("llm_correct")
        esc = conf < tau
        k, n = int(esc.sum()), len(test)
        laya_s, llm_s = test.col("laya_seconds"), test.col("llm_seconds")
        cost = {"seconds_ratio": float((laya_s.sum() + llm_s[esc].sum()) / llm_s.sum())}
        if prices:
            p_in, p_out, gpu_hour = prices
            llm_cost = (test.col("llm_in") * p_in + test.col("llm_out") * p_out) / 1e6
            laya_cost = laya_s * gpu_hour / 3600
            cost["price_ratio"] = float((laya_cost.sum() + llm_cost[esc].sum()) / llm_cost.sum())

        order_len = np.argsort(-test.col("n_tokens"), kind="stable")
        routers = {
            "confidence": esc,
            "length": np.isin(np.arange(n), order_len[:k]),
            # Escalate Laya's errors first, those the LLM gets right before those it does not.
            "oracle": np.isin(np.arange(n), np.lexsort((-mc, lc))[:k]),
        }
        router_acc = {name: cascade_acc(lc, mc, e) for name, e in routers.items()}
        router_acc["random_expected"] = float(lc.mean() + k / n * (mc - lc).mean())
        cas = np.where(esc, mc, lc)
        cas_len = np.where(routers["length"], mc, lc)
        noninf = boot(lambda i: float(cas[i].mean() - mc[i].mean()), test, cfg, 8)
        conf_vs_len = boot(lambda i: float(cas[i].mean() - cas_len[i].mean()), test, cfg, 9)
        res = {"dev_llm_accuracy": llm_dev, "tau_star": tau, "dev_at_tau": best,
               "test_n": n, "test_escalation": k / n, "test_llm_accuracy": float(mc.mean()),
               "test_laya_accuracy": float(lc.mean()), "test_cascade_accuracy": float(cas.mean()),
               "cascade_minus_llm": noninf, "cost": cost, "routers_at_same_escalation": router_acc,
               "confidence_minus_length_router": conf_vs_len, "dev_sweep": sweep}
        cost_ok = cost.get("price_ratio", cost["seconds_ratio"]) <= rule["max_cost_ratio"]
        sup = noninf["lo"] > -rule["delta"] and cost_ok and conf_vs_len["lo"] > 0
        fal = not (noninf["lo"] > -rule["delta"] and cost_ok) or conf_vs_len["hi"] <= 0
        res["verdict"] = verdict(sup, fal)
        out[q] = res
    return out


# ---------------------------------------------------------------- E8 / C3.1

def what_labels(t, pred_key):
    """Canonical true mode, except an accepted alternate predicted counts as itself."""
    y_true, y_pred = [], []
    for r in t.rows:
        p = r.get(pred_key)
        y_true.append(p if p in r["accepted"] else sorted(r["accepted"])[0])
        y_pred.append(p if p is not None else "unparsed")
    return y_true, y_pred


def e8(tables, cfg, pre):
    t = tables["what"].where(lambda r: r["split"] == "test")
    if not len(t):
        return {}
    conf, c = t.col("conf"), t.col("correct")
    y_true, y_pred = what_labels(t, "pred")
    labels = sorted(set(y_true))
    res = {"n": len(t), "accuracy": boot(acc_fn(c), t, cfg),
           "macro_f1": 100 * f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
           "reference_macro_f1": pre["C3.1"]["reference_macro_f1"],
           "true_mode_counts": dict(Counter(y_true)),
           "confusion": {f"{a}->{b}": n for (a, b), n in Counter(zip(y_true, y_pred)).items()}}
    wrong = 1 - c
    res["auroc"] = boot(lambda i: S.auroc(1 - conf[i], wrong[i]), t, cfg, 10)
    res["calibration"] = calib_block(conf, c)
    # Family level: sum the mode probabilities within each family.
    fam_true, fam_pred, fam_conf = [], [], []
    for r in t.rows:
        fams = defaultdict(float)
        for k, p in zip(r["keys"], r["probs"]):
            fams[mode_family(k)] += p
        best = max(fams, key=fams.get)
        fam_pred.append(best)
        fam_conf.append(fams[best])
        fam_true.append({mode_family(m) for m in r["accepted"]})
    fam_c = np.array([p in s for p, s in zip(fam_pred, fam_true)], float)
    fam_conf = np.array(fam_conf)
    res["family"] = {"accuracy": float(fam_c.mean()), **calib_block(fam_conf, fam_c),
                     "auroc": S.auroc(1 - fam_conf, 1 - fam_c)}
    res["per_true_family_ece"] = {}
    for fam in sorted({f for s in fam_true for f in s}):
        m = np.array([fam in s for s in fam_true])
        res["per_true_family_ece"][fam] = {"n": int(m.sum()), "ece": S.ece(conf[m], c[m]),
                                           "accuracy": float(c[m].mean())}
    root = np.array([bool(s & {"PL", "V", "C"}) for s in fam_true])
    if root.any():
        res["root_cause_predicted_as_reasoning"] = float(np.mean([fam_pred[i] == "R" for i in np.flatnonzero(root)]))
    p = t.where(lambda r: "llm_pred" in r)
    if len(p):
        lt, lp = what_labels(p, "llm_pred")
        yt, yp = what_labels(p, "pred")
        res["paired_n"] = len(p)
        res["paired_llm_macro_f1"] = 100 * f1_score(lt, lp, labels=sorted(set(lt)), average="macro", zero_division=0)
        res["paired_laya_macro_f1"] = 100 * f1_score(yt, yp, labels=sorted(set(yt)), average="macro", zero_division=0)
        res["paired_llm_auroc"] = S.auroc(1 - p.col("llm_conf"), 1 - p.col("llm_correct"))
    res["verdict"] = verdict(res["auroc"]["lo"] > 0.5, res["auroc"]["lo"] <= 0.5)
    return res


# ---------------------------------------------------------------- E9 / C4.1

def window_candidates(rec):
    """Per candidate step: (step, window, p(step), log p(none) of its window); and per-window log p(none)."""
    cands, none_logs = [], []
    for w in rec["windows"]:
        probs = dict(zip(w["when"]["keys"], w["when"]["probs"]))
        none_logs.append(np.log(max(probs[NONE_OPTION], S.EPS)))
        for s in w["steps"]:
            cands.append((s, w["window"], probs[s]))
    return cands, np.array(none_logs)


def stack_features(rec):
    cands, none_logs = window_candidates(rec)
    n_w = len(none_logs)
    best_in_w = {}
    for s, w, p in cands:
        best_in_w[w] = max(best_in_w.get(w, 0.0), p)
    feats = [[np.log(max(p, S.EPS)), none_logs[w], none_logs.sum() - none_logs[w],
              float(p == best_in_w[w]), w / max(1, n_w - 1), np.log(n_w)] for s, w, p in cands]
    return [s for s, _, _ in cands], np.array(feats)


def aggregate(rec, method, tau=None, stacker=None):
    """(predicted step, confidence) for one windowed trace."""
    cands, none_logs = window_candidates(rec)
    if method == "max":
        s, _, p = max(cands, key=lambda x: x[2])
        return s, p
    if method == "product":
        scores = np.array([np.log(max(p, S.EPS)) + none_logs.sum() - none_logs[w] for _, w, p in cands])
    elif method == "first":
        for w in rec["windows"]:
            probs = dict(zip(w["when"]["keys"], w["when"]["probs"]))
            if 1 - probs[NONE_OPTION] >= tau:
                s = max(w["steps"], key=probs.get)
                return s, probs[s]
        return aggregate(rec, "max")
    elif method == "stacker":
        _, feats = stack_features(rec)
        scores = stacker.decision_function(feats)
    else:
        raise ValueError(method)
    p = S.softmax(scores)
    j = int(p.argmax())
    return cands[j][0], float(p[j])


def eval_agg(recs, items, method, **kw):
    preds = [aggregate(r, method, **kw) for r in recs]
    c = np.array([p in items[r["id"]]["when_true"] for (p, _), r in zip(preds, recs)], float)
    conf = np.array([q for _, q in preds])
    return conf, c


def e9(window_recs, over_recs, items, over_items, split, laya_when, pre, cfg):
    rule = pre["C4.1"]
    out = {}
    for size in sorted({r["window_tokens"] for r in window_recs + over_recs}):
        recs = [r for r in window_recs if r["window_tokens"] == size and r["id"] in split]
        dev = [r for r in recs if split[r["id"]] == "dev"]
        test = [r for r in recs if split[r["id"]] == "test"]
        res = {"window_tokens": size, "n_dev": len(dev), "n_test": len(test)}
        if not dev or not test:
            out[str(size)] = res
            continue
        # Fit on dev: first-over-tau threshold (accuracy) and the stacker.
        taus = np.linspace(0.05, 0.95, 19)
        tau = max(taus, key=lambda t_: eval_agg(dev, items, "first", tau=t_)[1].mean())
        X, y = [], []
        for r in dev:
            steps, f = stack_features(r)
            X.append(f)
            y += [s in items[r["id"]]["when_true"] for s in steps]
        stacker = LogisticRegression(max_iter=1000).fit(np.vstack(X), np.array(y, int))
        kw = {"max": {}, "product": {}, "first": {"tau": tau}, "stacker": {"stacker": stacker}}
        dev_nll = {m: S.binary_nll(*eval_agg(dev, items, m, **kw[m])) for m in kw}
        chosen = min(("product", "first", "stacker"), key=dev_nll.get)
        res.update({"tau_first": float(tau), "dev_binary_nll": dev_nll, "chosen_combiner": chosen})

        per = {m: eval_agg(test, items, m, **kw[m]) for m in kw}
        res["test"] = {m: {"accuracy": float(c.mean()), **calib_block(conf, c)} for m, (conf, c) in per.items()}
        strata = np.array([items[r["id"]]["framework"] for r in test])

        def loss(conf, c):
            q = np.clip(conf, S.EPS, 1 - S.EPS)
            return -(c * np.log(q) + (1 - c) * np.log(1 - q))
        dl = loss(*per[chosen]) - loss(*per["max"])
        da = per[chosen][1] - per["max"][1]
        res["nll_diff_combiner_minus_max"] = S.bootstrap(lambda i: float(dl[i].mean()), len(test), strata,
                                                         cfg["n_boot"], cfg["seed"] + 11)
        res["acc_diff_combiner_minus_max"] = S.bootstrap(lambda i: float(da[i].mean()), len(test), strata,
                                                         cfg["n_boot"], cfg["seed"] + 12)
        full = [laya_when.get(r["id"]) for r in test]
        if all(f is not None for f in full):
            res["full_context_same_traces"] = {"accuracy": float(np.mean([f["correct"] for f in full])),
                                               **calib_block([f["conf"] for f in full], [f["correct"] for f in full])}
        sup = res["nll_diff_combiner_minus_max"]["hi"] < 0 and res["acc_diff_combiner_minus_max"]["lo"] > -rule["accuracy_margin"]
        fal = not sup

        o = [r for r in over_recs if r["window_tokens"] == size and r["id"] in over_items]
        if o:
            ores = {}
            rnd = np.array([len(over_items[r["id"]]["when_true"]) / len(over_items[r["id"]]["when_options"]) for r in o])
            ostrata = np.array([over_items[r["id"]]["framework"] for r in o])
            for m in ("max", chosen):
                conf, c = eval_agg(o, over_items, m, **kw[m])
                ores[m] = {**calib_block(conf, c),
                           "accuracy": S.bootstrap(lambda i, c=c: float(c[i].mean()), len(o), ostrata,
                                                   cfg["n_boot"], cfg["seed"] + 13)}
            ores["random"] = float(rnd.mean())
            ores["n"] = len(o)
            res["E9b_overlong"] = ores
            fal = fal or ores[chosen]["accuracy"]["lo"] <= ores["random"]
        res["verdict"] = verdict(sup, fal)
        out[str(size)] = res
    return out


# ---------------------------------------------------------------- report

def fmt_ci(d, pct=False):
    k = 100 if pct else 1
    f = "%.1f" if pct else "%.3f"
    return (f + " [" + f + ", " + f + "]") % (k * d["point"], k * d["lo"], k * d["hi"])


def markdown(report):
    lines = [f"# Claim verdicts ({report['generated_from']['laya']})", "",
             "| Claim | Q | Verdict | Key numbers |", "|---|---|---|---|"]
    p = report.get("P0", {})
    for q in ("who", "when"):
        if q in p:
            lines.append(f"| P0 | {q} | {p['verdict']} | acc {fmt_ci(p[q]['accuracy'], True)} vs random "
                         f"{100 * p[q]['random']:.1f} (n={p[q]['n']}) |")
            lines.append(f"| P0 vs majority | {q} | {p['verdict_vs_majority']} | majority-label "
                         f"{fmt_ci(p[q]['majority_accuracy'], True)}; Laya − majority "
                         f"{fmt_ci(p[q]['laya_minus_majority'], True)} pp |")
    for q, r in report.get("E1", {}).items():
        extra = f"; LLM−Laya ECE {fmt_ci(r['ece_diff_llm_minus_laya'])}" if "ece_diff_llm_minus_laya" in r else ""
        lines.append(f"| C1.1 | {q} | {r['verdict']} | ECE {fmt_ci(r['laya_ece_ci'])}, Brier "
                     f"{r['laya']['brier_top']:.3f}{extra} |")
    for q, r in report.get("E1b", {}).items():
        lines.append(f"| C1.1b | {q} | {r['verdict']} | NLL after−before {fmt_ci(r['pooled_nll_diff_after_minus_before'])}; "
                     f"improved {r['frameworks_improved']} |")
    for q, r in report.get("E2", {}).items():
        lines.append(f"| C1.2 | {q} | {r['verdict']} | AUROC {fmt_ci(r['laya_auroc'])} vs length-only "
                     f"{r['length_only_auroc']:.3f} (DeLong p={r['delong_laya_vs_length']['p_one_sided_1_gt_2']:.3g}) |")
    for q, r in report.get("E3", {}).items():
        if "verdict" in r:
            lines.append(f"| C1.3 | {q} | {r['verdict']} | gap short {fmt_ci(r['short']['gap'])}, long "
                         f"{fmt_ci(r['long']['gap'])}; acc long−short {fmt_ci(r['accuracy_long_minus_short'])} |")
    for q, r in report.get("E4", {}).items():
        lines.append(f"| C1.4 | {q} | {r['verdict']} | τ*={r['tau_star']:.3f}, escalate {100 * r['test_escalation']:.0f}%, "
                     f"cascade−LLM {fmt_ci(r['cascade_minus_llm'], True)} pp, cost {r['cost']}, "
                     f"conf−length {fmt_ci(r['confidence_minus_length_router'], True)} pp |")
    r = report.get("E8")
    if r:
        lines.append(f"| C3.1 | what | {r['verdict']} | AUROC {fmt_ci(r['auroc'])}, macro-F1 {r['macro_f1']:.1f} "
                     f"(ref {r['reference_macro_f1']}) |")
    for size, r in report.get("E9", {}).items():
        if "verdict" in r:
            lines.append(f"| C4.1 | when@{size} | {r['verdict']} | {r['chosen_combiner']} − max NLL "
                         f"{fmt_ci(r['nll_diff_combiner_minus_max'])}, acc {fmt_ci(r['acc_diff_combiner_minus_max'], True)} pp |")
    for name, why in report["skipped"].items():
        lines.append(f"| {name} | | SKIPPED | {why} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="experiments/manifest.json")
    parser.add_argument("--laya", default="outputs/pro/laya_full_all.jsonl")
    parser.add_argument("--llm", default=None)
    parser.add_argument("--windows", default="outputs/pro/laya_windows_window.jsonl")
    parser.add_argument("--windows_overlong", default="outputs/pro/laya_windows_overlong.jsonl")
    parser.add_argument("--price_in", type=float, default=None, help="LLM $ per 1M input tokens")
    parser.add_argument("--price_out", type=float, default=None, help="LLM $ per 1M output tokens")
    parser.add_argument("--gpu_hour", type=float, default=None, help="$ per GPU-hour for Laya")
    parser.add_argument("--out", default="outputs/pro/claims_report")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(manifest["prereg"]) as f:
        pre = json.load(f)
    cfg = {"n_boot": pre["n_boot"], "seed": pre["seed"], "alpha": pre["alpha"]}
    split = manifest["split"]
    items = {it["id"]: it for _, it in load_items(manifest["pool"])}
    laya = read_jsonl(args.laya)
    llm = {r["id"]: r for r in read_jsonl(args.llm)}
    if not laya:
        sys.exit(f"no Laya records in {args.laya}")
    tables = build_tables(items, split, laya, llm)
    prices = (args.price_in, args.price_out, args.gpu_hour) if None not in (args.price_in, args.price_out, args.gpu_hour) else None

    report = {"generated_from": {"laya": args.laya, "llm": args.llm, "windows": args.windows},
              "n_laya": len(laya), "n_llm": len(llm),
              "laya_truncated": int(sum(r["truncated"] for r in laya)),
              "skipped": {}}
    report["P0"] = p0(tables, cfg)
    report["E1"] = e1(tables, pre, cfg)
    report["E1b"] = e1b(tables, cfg)
    report["E2"] = e2(tables, pre, cfg)
    report["E3"] = e3(tables, pre, cfg)
    report["E8"] = e8(tables, cfg, pre)
    if llm:
        report["E4"] = e4(tables, pre, cfg, prices)
        for q in ("who", "when"):
            p = tables[q].where(lambda r: "llm_conf" in r and r["split"] == "test")
            if len(p):
                report.setdefault("llm_range_check", {})[q] = {
                    "n": len(p), "llm_accuracy": float(p.col("llm_correct").mean()),
                    "reference": pre["llm_reference"], "mcnemar_laya_vs_llm": S.mcnemar_exact(p.col("correct"), p.col("llm_correct"))}
    else:
        report["skipped"]["C1.4"] = "no LLM records (--llm)"
        report["skipped"]["C1.1 paired"] = "no LLM records (--llm)"

    window_recs, over_recs = read_jsonl(args.windows), read_jsonl(args.windows_overlong)
    if window_recs:
        over_items = {it["id"]: it for _, it in load_items(default_pool_path("overlong", os.path.dirname(manifest["pool"])))} \
            if over_recs else {}
        laya_when = {r["id"]: r for r in tables["when"].rows}
        report["E9"] = e9(window_recs, over_recs, items, over_items, split, laya_when, pre, cfg)
    else:
        report["skipped"]["C4.1"] = f"no window records ({args.windows})"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    md = markdown(report)
    with open(args.out + ".md", "w") as f:
        f.write(md)
    print(md)
    print(f"Report: {args.out}.json, {args.out}.md")


if __name__ == "__main__":
    main()
