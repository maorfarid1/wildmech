"""Link query categories to survey-reported time and pain (contribution 4).

Each 2024 survey task is mapped to one or more taxonomy L1 codes. "Answer an engineering
question" is a bucket that covers many L1s, so its hours are split across them by the
within-bucket shares observed in the query data (or, before Leo data arrives, by the
LLM-labeled survey free text).

hours_at_stake[L1] = sum_tasks  relevance_rate * mean_hours_per_project(winsorized) * alloc[task, L1]
unmet_hours[L1]    = hours_at_stake[L1] * failure_rate[L1]
"""
from __future__ import annotations

import pandas as pd

SURVEY_TO_L1 = {
    "part_search_internal": {"SRCH": 1.0},
    "component_search_catalog": {"CMP": 0.8, "SUP": 0.2},
    "model_catalog_part": {"CAD": 1.0},
    "convert_mfg_method": {"DFM": 1.0},
    "answer_engineering_question": "SPLIT",
}
KNOWLEDGE_L1 = ["MAT", "CAL", "JNT", "GDT", "CMP", "STD", "DFM", "SIM", "FAIL", "TST", "MEC"]


def allocation(within_shares: pd.Series) -> dict[str, dict[str, float]]:
    """within_shares: index = L1 code, values = share among engineering-question queries."""
    s = within_shares.reindex(KNOWLEDGE_L1).fillna(0)
    s = s / s.sum() if s.sum() > 0 else s
    alloc = {}
    for task, m in SURVEY_TO_L1.items():
        alloc[task] = s.to_dict() if m == "SPLIT" else m
    return alloc


def hours_at_stake(task_time: pd.DataFrame, within_shares: pd.Series, hours_col: str = "mean_hours_per_project_winsor") -> pd.DataFrame:
    alloc = allocation(within_shares)
    rows = []
    for _, r in task_time.iterrows():
        for l1, w in alloc.get(r["task"], {}).items():
            rows.append({"task_l1": l1, "survey_task": r["task"], "hours": r["relevance_rate"] * r[hours_col] * w,
                         "pain_mean": r["pain_mean"]})
    d = pd.DataFrame(rows)
    out = d.groupby("task_l1").agg(hours_at_stake=("hours", "sum")).reset_index()
    pain = d.assign(wp=d["hours"] * d["pain_mean"]).groupby("task_l1")[["wp", "hours"]].sum()
    out["pain_weighted"] = (pain["wp"] / pain["hours"]).reindex(out["task_l1"]).to_numpy()
    return out.sort_values("hours_at_stake", ascending=False)


def opportunity(hours: pd.DataFrame, query_dist: pd.DataFrame, fail: pd.DataFrame) -> pd.DataFrame:
    """query_dist: task_l1, share. fail: task_l1, failure_rate."""
    t = hours.merge(query_dist[["task_l1", "share"]].rename(columns={"share": "query_share"}), on="task_l1", how="outer")
    t = t.merge(fail[["task_l1", "failure_rate"]], on="task_l1", how="left")
    t["unmet_hours"] = t["hours_at_stake"] * t["failure_rate"]
    return t.sort_values("unmet_hours", ascending=False)
