"""Aggregation thresholds. Any reported cell must cover at least `min_users` distinct users
AND `min_orgs` distinct organizations, otherwise it is suppressed (Clio-style k-thresholds)."""
from __future__ import annotations

import pandas as pd


def safe_counts(df: pd.DataFrame, by: list[str] | str, min_users: int = 10, min_orgs: int = 3,
                user_col: str = "user_id", org_col: str = "org_id") -> pd.DataFrame:
    by = [by] if isinstance(by, str) else by
    g = df.groupby(by, dropna=False).agg(
        n=(user_col, "size"), users=(user_col, "nunique"), orgs=(org_col, "nunique")
    ).reset_index()
    g["suppressed"] = (g["users"] < min_users) | (g["orgs"] < min_orgs)
    total = g["n"].sum()
    g["share"] = g["n"] / total
    g.loc[g["suppressed"], ["n", "share"]] = pd.NA
    return g


def suppress_small(table: pd.DataFrame, n_col: str = "n", k: int = 5) -> pd.DataFrame:
    """Survey tables: blank any cell with fewer than k respondents."""
    t = table.copy()
    mask = t[n_col] < k
    for c in t.columns:
        if c != n_col and t[c].dtype.kind in "fi":
            t.loc[mask, c] = pd.NA
    t["suppressed"] = mask
    return t
