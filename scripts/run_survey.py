"""Survey analysis: aggregates and figures only. Usage:
    python scripts/run_survey.py --s2024 PATH --s2025 PATH [--s2025-copy PATH] [--labels PATH] --out outputs/survey
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wildmech import survey as S  # noqa: E402
from wildmech.taxonomy import load_taxonomy  # noqa: E402
from wildmech import figures as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2024", required=True)
    ap.add_argument("--s2025", required=True)
    ap.add_argument("--s2025-copy")
    ap.add_argument("--labels", help="LLM labels for 2024 hard-question free text (row,primary,...)")
    ap.add_argument("--out", default="outputs/survey")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cmap = S.load_map()
    tax = load_taxonomy()
    log = {}

    d24, r24 = S.load_survey(a.s2024, "2024", cmap)
    d25, r25 = S.load_survey(a.s2025, "2025", cmap)
    h24, h25 = r24.pop("respondent_hashes"), r25.pop("respondent_hashes")
    log["2024"], log["2025"] = r24, r25
    log["panel_overlap_2024_2025"] = len(h24 & h25)
    if a.s2025_copy:
        dc, rc = S.load_survey(a.s2025_copy, "2025", cmap)
        hc = rc.pop("respondent_hashes")
        log["2025_copy_file"] = {**rc, "respondents_also_in_master": len(hc & h25),
                                 "respondents_not_in_master": len(hc - h25),
                                 "decision": "duplicate snapshot; master file used"}

    T = {}
    # ---------- 2024: who answered
    for col in ["profession", "experience_band", "industry", "org_size", "project_duration", "region"]:
        s = S.translate(d24[col], cmap)
        order = {"experience_band": ["0-2", "3-5", "6-10", "11-20", "20+"],
                 "org_size": ["1-9", "10-49", "50-249", "250-999", "1000+"]}.get(col)
        T[f"2024_{col}"] = S.share_table(s, order=order)
    # ---------- 2024: time and pain per task
    T["2024_task_time"] = S.task_time_table(d24, cmap, "2024")
    T["2024_binary_tasks"] = S.binary_task_table(d24, cmap, "2024")

    # ---------- 2024: hard engineering questions, LLM-labeled with the taxonomy
    if a.labels:
        lab = pd.read_csv(a.labels)
        n_lab_raw = len(lab)
        lab = lab[lab["row"].isin(d24["raw_row"])]  # drop labels of removed duplicate submissions
        lab = lab[lab["primary"].notna() & (lab["primary"] != "NA")]
        lab["phase"] = lab["primary"].map(tax.phase_of)
        t = lab["primary"].value_counts().rename_axis("task_l1").reset_index(name="n")
        t["share"] = t["n"] / len(lab)
        t["name"] = t["task_l1"].map(tax.name_of)
        t["phase"] = t["task_l1"].map(tax.phase_of)
        T["2024_hardq_l1"] = t
        any_mention = pd.concat([lab["primary"], lab["secondary"].dropna()]).value_counts()
        T["2024_hardq_l1_any"] = (any_mention / len(lab)).rename("share_any").rename_axis("task_l1").reset_index()
        T["2024_hardq_l2_top"] = lab.groupby(["primary", "l2"]).size().rename("n").reset_index().sort_values("n", ascending=False).head(15)
        T["2024_hardq_complexity"] = S.share_table(lab["complexity"], order=["C1", "C2", "C3", "C4"])
        T["2024_hardq_intent"] = S.share_table(lab["intent"])
        T["2024_hardq_verifiable"] = S.share_table(lab["verifiable"])
        log["hardq"] = {"responses_with_text": int(d24["hard_question_examples"].notna().sum()),
                        "label_rows_in_file": n_lab_raw, "substantive_labeled": len(lab)}
        F.workflow_strip(t, tax, out / "figures" / "fig1_pilot_workflow_strip.png",
                         title=f"Hard engineering questions named by engineers (pilot, n={len(lab)})")

    # ---------- 2025: who answered
    for col in ["seniority", "industry", "specialization", "org_size", "cad", "plm", "education"]:
        order = {"org_size": ["1-9", "10-49", "50-99", "100-249", "250-499", "500-999", "1000+"]}.get(col)
        T[f"2025_{col}"] = S.share_table(S.translate(d25[col], cmap), order=order)
    log["2025_years_experience"] = pd.to_numeric(d25["years_experience"], errors="coerce").describe().round(2).to_dict()

    # ---------- 2025: generic AI use
    uses_ai = d25["uses_generic_ai"].astype(str).str.strip().eq("כן").astype(float)
    k, n = int(uses_ai.sum()), len(uses_ai)
    lo, hi = S.wilson(k, n)
    log["generic_ai_use"] = {"k": k, "n": n, "rate": k / n, "ci_lo": lo, "ci_hi": hi}
    users = d25[uses_ai == 1]
    ms = cmap["multi_select"]
    T["2025_ai_uses"] = S.multi_share(S.multi_select(users["ai_uses"], ms["ai_uses"]))
    T["2025_ai_barriers"] = S.multi_share(S.multi_select(users["ai_barriers"], ms["ai_barriers"]))
    T["2025_ai_reasons"] = S.multi_share(S.multi_select(users["ai_reasons"], ms["ai_reasons"]))
    T["2025_ai_tools"] = S.multi_share(S.multi_select(users["ai_tools"], ms["ai_tools"]))
    T["2025_ai_frequency"] = S.share_table(S.translate(users["ai_frequency"], cmap),
                                           order=["Several times a day", "Daily", "3-5x / week", "1-2x / week", "< 1x / week"])
    sat = pd.to_numeric(users["ai_satisfaction"], errors="coerce").dropna()
    log["generic_ai_satisfaction"] = {"n": len(sat), "mean": sat.mean(), "share_4_5": (sat >= 4).mean(), "share_1_2": (sat <= 2).mean()}
    pays = users["pays_for_ai"].astype(str).str.strip()
    log["pays_for_ai"] = {"answered": int(pays.isin(["כן", "לא"]).sum()), "yes": int(pays.eq("כן").sum())}

    d25t = d25.assign(seniority_en=S.translate(d25["seniority"], cmap), industry_en=d25["industry"],
                      org_size_en=d25["org_size"], spec_en=S.translate(d25["specialization"], cmap))
    T["2025_ai_use_by_seniority"] = S.crosstab_rate(d25t, "seniority_en", uses_ai)
    T["2025_ai_use_by_industry"] = S.crosstab_rate(d25t, "industry_en", uses_ai)
    T["2025_ai_use_by_org_size"] = S.crosstab_rate(d25t, "org_size_en", uses_ai)
    T["2025_ai_use_by_specialization"] = S.crosstab_rate(d25t, "spec_en", uses_ai)
    bar = S.multi_select(d25t.loc[uses_ai == 1, "ai_barriers"], ms["ai_barriers"])
    for b in ["Low trust in the results", "Data security / IP concerns", "Company restrictions"]:
        T[f"2025_barrier_{b.split()[0].lower()}_by_industry"] = S.crosstab_rate(
            d25t.loc[uses_ai == 1], "industry_en", bar[b])
    eng_q = S.multi_select(users["ai_uses"], ms["ai_uses"])["Technical / engineering questions"]
    T["2025_engq_use_by_seniority"] = S.crosstab_rate(d25t.loc[uses_ai == 1], "seniority_en", eng_q)

    # Leo-specific items: kept internal (conflict of interest; decide with Maor before any use)
    leo = S.translate(d25["leo_use"], cmap)
    T["INTERNAL_2025_leo_use"] = S.share_table(leo)
    ls = pd.to_numeric(d25["leo_satisfaction"], errors="coerce").dropna()
    log["INTERNAL_leo_satisfaction"] = {"n": len(ls), "mean": ls.mean()}

    # ---------- save
    for name, t in T.items():
        t.to_csv(out / "tables" / f"{name}.csv", index=False)
    (out / "survey_log.json").write_text(json.dumps(log, indent=2, default=float, ensure_ascii=False))

    F.task_time_pain(T["2024_task_time"], out / "figures" / "fig_survey_time_vs_pain.png")
    F.bar_share(T["2025_ai_uses"], "option", out / "figures" / "fig_ai_uses.png",
                f"What engineers who use generic AI use it for (n={T['2025_ai_uses'].attrs.get('N', len(users))})")
    F.bar_share(T["2025_ai_barriers"], "option", out / "figures" / "fig_ai_barriers.png",
                "What keeps them from using it more")
    print(json.dumps(log, indent=2, default=float, ensure_ascii=False))
    print("tables:", len(T))


if __name__ == "__main__":
    main()
