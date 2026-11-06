"""Flag candidate items for the paper 3 benchmark (hard, real, expert-verifiable ME problems).

A conversation scores higher when it is:
  hard (C3/C4), shows a failure signal, has a checkable answer, is self-contained
  (no internal data, no geometry), and is mechanical (not OOS).
Candidates then go to Leo engineers who write a clean, IP-free version and a reference answer.
"""
from __future__ import annotations

import pandas as pd

WEIGHTS = {"complexity": {"C1": 0, "C2": 1, "C3": 2.5, "C4": 3}, "failure_any": 2.0, "verifiable_answer": 2.0,
           "self_contained": 1.5}


def score(df: pd.DataFrame) -> pd.Series:
    s = df["complexity"].map(WEIGHTS["complexity"]).fillna(0)
    s += WEIGHTS["failure_any"] * df.get("failure_any", 0)
    s += WEIGHTS["verifiable_answer"] * (df["verifiable_answer"] == "yes")
    self_contained = (df["needs_internal_data"] == "no") & (df["needs_geometry"] == "no")
    s += WEIGHTS["self_contained"] * self_contained
    s[df["task_l1"] == "OOS"] = -1
    return s


def candidates(df: pd.DataFrame, top_n: int = 500, per_category_cap: float = 0.2) -> pd.DataFrame:
    d = df.copy()
    d["bench_score"] = score(d)
    d = d[d["bench_score"] >= 5].sort_values("bench_score", ascending=False)
    cap = max(1, int(top_n * per_category_cap))
    d = d.groupby("task_l1", group_keys=False).head(cap)  # keep the set diverse
    return d.head(top_n)[["conversation_id", "task_l1", "task_l2", "complexity", "failure_any",
                          "verifiable_answer", "bench_score", "summary"]]
