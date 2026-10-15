"""Calibration, selective-prediction and paired-comparison statistics.

Conventions: `conf` is the top-label confidence in [0, 1], `correct` is 0/1 per trace, and
error-detection scores treat a *wrong* attribution as the positive class (score = 1 - conf).
Every bootstrap resamples whole traces within strata, so paired metrics computed on the same
index array stay paired.
"""
import numpy as np
from scipy import optimize, stats

EPS = 1e-6


def _arr(x, dtype=float):
    return np.asarray(x, dtype=dtype)


# ---------------------------------------------------------------- calibration

def ece(conf, correct, n_bins=10, scheme="width"):
    """Expected calibration error with equal-width or equal-mass ("mass") bins."""
    conf, correct = _arr(conf), _arr(correct)
    if len(conf) == 0:
        return float("nan")
    if scheme == "width":
        idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    elif scheme == "mass":
        order = np.argsort(conf, kind="stable")
        idx = np.empty(len(conf), dtype=int)
        idx[order] = np.arange(len(conf)) * n_bins // len(conf)
    else:
        raise ValueError(scheme)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(total / len(conf))


def reliability_bins(conf, correct, n_bins=10):
    conf, correct = _arr(conf), _arr(correct)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": int(m.sum()),
                    "mean_conf": float(conf[m].mean()) if m.any() else None,
                    "accuracy": float(correct[m].mean()) if m.any() else None})
    return out


def brier_top(conf, correct):
    """Top-label Brier score: mean (conf - correct)^2. Comparable between Laya and an LLM."""
    return float(np.mean((_arr(conf) - _arr(correct)) ** 2))


def binary_nll(conf, correct):
    """Log loss of the confidence as a prediction of correctness."""
    c = np.clip(_arr(conf), EPS, 1 - EPS)
    y = _arr(correct)
    return float(-np.mean(y * np.log(c) + (1 - y) * np.log(1 - c)))


# ---------------------------------------------------------------- distributions / temperature

def softmax(z, temperature=1.0):
    z = _arr(z) / temperature
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def true_mass(probs, keys, accepted):
    """Probability on the accepted set (several answers can be correct)."""
    acc = set(accepted)
    return float(sum(p for k, p in zip(keys, probs) if k in acc))


def nll_full(logits_list, true_masks, temperature=1.0):
    """Mean -log p(accepted set) over traces, from raw logits at a temperature."""
    total = 0.0
    for z, mask in zip(logits_list, true_masks):
        z = _arr(z) / temperature
        lse = z.max() + np.log(np.exp(z - z.max()).sum())
        zt = z[_arr(mask, bool)]
        total += lse - (zt.max() + np.log(np.exp(zt - zt.max()).sum()))
    return float(total / len(logits_list))


def multiclass_brier(probs_list, true_masks):
    total = 0.0
    for p, mask in zip(probs_list, true_masks):
        m = _arr(mask, bool)
        # With several accepted answers the target puts its mass on the predicted one if it is
        # accepted, else on the most probable accepted option.
        target = np.zeros(len(p))
        p = _arr(p)
        target[np.flatnonzero(m)[np.argmax(p[m])]] = 1.0
        total += float(((p - target) ** 2).sum())
    return total / len(probs_list)


def fit_temperature(logits_list, true_masks, lo=0.05, hi=20.0):
    """Scalar temperature minimising NLL (searched in log space)."""
    if not logits_list:
        return 1.0
    res = optimize.minimize_scalar(lambda lt: nll_full(logits_list, true_masks, np.exp(lt)),
                                   bounds=(np.log(lo), np.log(hi)), method="bounded")
    return float(np.exp(res.x))


# ---------------------------------------------------------------- discrimination / selective

def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties counted half."""
    scores, labels = _arr(scores), _arr(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _placements(pos, neg):
    """DeLong structural components V10 (per positive) and V01 (per negative)."""
    allv = np.concatenate([pos, neg])
    r_all = stats.rankdata(allv)
    r_pos, r_neg = stats.rankdata(pos), stats.rankdata(neg)
    m, n = len(pos), len(neg)
    v10 = (r_all[:m] - r_pos) / n
    v01 = 1.0 - (r_all[m:] - r_neg) / m
    return v10, v01


def delong_paired(labels, s1, s2):
    """Paired DeLong test of AUROC(s1) - AUROC(s2) on the same traces."""
    labels = _arr(labels, bool)
    s1, s2 = _arr(s1), _arr(s2)
    a1, a2 = auroc(s1, labels), auroc(s2, labels)
    v10_1, v01_1 = _placements(s1[labels], s1[~labels])
    v10_2, v01_2 = _placements(s2[labels], s2[~labels])
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / labels.sum() + \
          (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / (~labels).sum()
    diff = a1 - a2
    se = float(np.sqrt(max(var, 1e-12)))
    z = diff / se
    return {"auroc_1": a1, "auroc_2": a2, "diff": diff, "se": se, "z": float(z),
            "p_two_sided": float(2 * stats.norm.sf(abs(z))), "p_one_sided_1_gt_2": float(stats.norm.sf(z)),
            "ci": [diff - 1.96 * se, diff + 1.96 * se]}


def risk_coverage(conf, correct):
    """Selective risk at every coverage, accepting the most confident traces first."""
    conf, correct = _arr(conf), _arr(correct)
    order = np.argsort(-conf, kind="stable")
    errs = 1 - correct[order]
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), np.cumsum(errs) / k


def aurc(conf, correct):
    _, risk = risk_coverage(conf, correct)
    return float(risk.mean())


def selective_accuracy(conf, correct, coverage):
    conf, correct = _arr(conf), _arr(correct)
    k = max(1, int(round(coverage * len(conf))))
    order = np.argsort(-conf, kind="stable")[:k]
    return float(correct[order].mean())


# ---------------------------------------------------------------- tests / bootstrap

def mcnemar_exact(a_correct, b_correct):
    a, b = _arr(a_correct, bool), _arr(b_correct, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    return {"a_only": n01, "b_only": n10, "p_value": p}


def stratified_indices(strata, rng):
    """One bootstrap resample, drawn with replacement within each stratum."""
    strata = np.asarray(strata)
    out = []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def bootstrap(fn, n, strata=None, n_boot=2000, seed=0, alpha=0.05):
    """Point estimate and percentile CI of fn(idx) under (stratified) trace resampling.

    `fn` receives an index array and returns a float; resamples where it is undefined (NaN) are
    dropped and counted.
    """
    rng = np.random.default_rng(seed)
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    point = fn(np.arange(n))
    vals = []
    for _ in range(n_boot):
        v = fn(stratified_indices(strata, rng))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": len(vals)}


def poisson_binomial_sf(probs, k):
    """P(X >= k) for X = sum of independent Bernoulli(p_i)."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum())
