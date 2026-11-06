"""Agreement between the LLM classifier and Leo engineers.

- Cohen's kappa (classifier vs adjudicated gold, and engineer vs engineer)
- Krippendorff's alpha (nominal) for any number of raters, missing allowed
- bootstrap CIs over items
- an annotation sheet generator (stratified, blind to the model label)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def kappa_ci(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    k = cohen_kappa_score(a, b)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i]) | set(b[i])) > 1:
            boots.append(cohen_kappa_score(a[i], b[i]))
    return {"kappa": k, "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
            "accuracy": float((a == b).mean()), "n": len(a)}


def krippendorff_alpha_nominal(ratings: pd.DataFrame) -> float:
    """ratings: rows = items, columns = raters, values = category or NaN."""
    data = [r.dropna().tolist() for _, r in ratings.iterrows()]
    data = [d for d in data if len(d) >= 2]
    cats = sorted({c for d in data for c in d})
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for d in data:
        m = len(d)
        for i, x in enumerate(d):
            for j, y in enumerate(d):
                if i != j:
                    o[idx[x], idx[y]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    d_o = (o.sum() - np.trace(o)) / n
    d_e = (n * n - (n_c ** 2).sum()) / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def annotation_sample(facets: pd.DataFrame, n: int = 600, strat_col: str = "task_l1",
                      min_per_stratum: int = 15, seed: int = 0) -> pd.DataFrame:
    """Stratified sample: at least `min_per_stratum` per category, rest proportional.
    The sheet hides the model label so annotators are blind."""
    rng = np.random.default_rng(seed)
    parts = []
    for k, g in facets.groupby(strat_col):
        parts.append(g.sample(min(len(g), min_per_stratum), random_state=int(rng.integers(1e9))))
    base = pd.concat(parts)
    rest = facets.drop(base.index)
    extra = rest.sample(max(0, min(n - len(base), len(rest))), random_state=seed)
    s = pd.concat([base, extra]).sample(frac=1, random_state=seed)
    sheet = s[["conversation_id", "summary"]].copy()
    for c in ["ann_task_l1", "ann_task_l2", "ann_complexity", "ann_failed", "ann_benchmark_ok", "notes"]:
        sheet[c] = ""
    return sheet
