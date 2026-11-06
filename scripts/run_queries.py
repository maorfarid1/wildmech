"""Analysis of real Leo queries for the WildMech paper.

Inputs (private, never in the repo):
  --mixpanel    folder of Mixpanel pulls (msgs_*.json)
  --sample      label_sample.pkl (opening requests drawn for annotation)
  --labels      labels_claude.csv (sid,l1,c,ctx,v)
  --aggregates  mixpanel_aggregates.json (monthly totals, feedback counts, country counts)
  --survey      outputs/survey/tables (aggregates from run_survey.py)
Outputs (public, aggregates only): tables/*.csv, figures/*.pdf|png, numbers.tex, query_log.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wildmech import leo_logs as L  # noqa: E402
from wildmech import figures as F  # noqa: E402
from wildmech.agreement import kappa_ci  # noqa: E402
from wildmech.llm import RuleBackend  # noqa: E402
from wildmech.taxonomy import load_taxonomy  # noqa: E402

KNOWLEDGE = ["STD", "MEC", "CAL", "CMP", "MAT", "JNT", "SIM", "GDT", "CAD", "SRCH", "DOC", "DFM", "SUP", "TST", "FAIL"]
ERA_B_START = "2025-12"
FRUSTRATION = re.compile(
    r"\b(wrong|incorrect|not correct|not right|not what i|doesn'?t work|does not work|didn'?t work|failed|fail(?:ing)?|"
    r"you didn'?t|you did not|still not|still no|again|try again|nothing happen|no file|dead link|can'?t download|"
    r"cannot download|not working|error|mistake|useless|fuck|wtf|stop asking|not a|that'?s not|isn'?t|doesn'?t make sense|"
    r"not accurate|bad|broken|missing)\b|לא נכון|טעות|לא עובד|falsch|incorrecto|no funciona|faux|ne marche pas", re.I)


def wshare(df: pd.DataFrame, col: str, wcol: str = "w", n_boot: int = 2000, seed: int = 0, strata: str = "month") -> pd.DataFrame:
    """Weighted shares with stratified bootstrap CIs (resample items within month)."""
    rng = np.random.default_rng(seed)
    cats = sorted(df[col].dropna().unique())
    point = df.groupby(col)[wcol].sum() / df[wcol].sum()
    groups = [g for _, g in df.groupby(strata)]
    boots = []
    for _ in range(n_boot):
        b = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for g in groups])
        boots.append((b.groupby(col)[wcol].sum() / b[wcol].sum()).reindex(cats).fillna(0))
    B = pd.DataFrame(boots)
    out = pd.DataFrame({col: cats, "n": df[col].value_counts().reindex(cats).fillna(0).astype(int).to_numpy(),
                        "share": point.reindex(cats).fillna(0).to_numpy(),
                        "ci_lo": B.quantile(0.025).to_numpy(), "ci_hi": B.quantile(0.975).to_numpy()})
    return out.sort_values("share", ascending=False).reset_index(drop=True)


def rate_ci(x: np.ndarray, w: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    p = np.average(x, weights=w)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = (x[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return p, np.quantile(b, 0.025), np.quantile(b, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixpanel", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--aggregates", required=True)
    ap.add_argument("--survey", default="outputs/survey/tables")
    ap.add_argument("--out", default="outputs/queries")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tax = load_taxonomy()
    agg = json.load(open(a.aggregates))
    vol = pd.Series(agg["monthly_messages"])
    N = {}  # numbers for the paper

    # ---------------- corpus
    raw = L.load_mixpanel(a.mixpanel)
    d, clog = L.clean(raw)
    d = L.sessions(d)
    op = L.opening_requests(d)
    N.update(corpusMessages=agg["window_total"]["messages"], corpusUsers=agg["window_total"]["unique_users"],
             corpusCountries=agg["n_countries_with_messages"], corpusMonths=len(vol),
             sampleWindows=raw["window"].nunique(), sampleWindowsCapped=int(raw.groupby("window")["window_capped"].first().sum()),
             sampleRawEvents=clog["raw_events"], sampleDupes=clog["logging_duplicates"],
             sampleInternal=clog["internal_tenant_messages"], sampleMessages=clog["messages_clean"],
             sampleUsers=d["user_id"].nunique(), sampleSessions=d["session_id"].nunique(),
             sampleOpenings=len(op))
    N["dupeShare"] = clog["logging_duplicates"] / (clog["raw_events"] - clog["empty_text"])
    sess = d.groupby("session_id").agg(n=("text", "size"), month=("month", "first"), user=("user_id", "first"),
                                       ent=("enterprise", "first"), clar=("is_clarify_reply", "sum"))
    N["msgsPerSessionMedian"] = sess["n"].median()
    N["msgsPerSessionMean"] = sess["n"].mean()
    N["singleMsgSessionShare"] = (sess["n"] == 1).mean()
    sess["era"] = np.where(sess["month"] >= ERA_B_START, "B", "A")
    for e in "AB":
        s = sess[sess["era"] == e]
        N[f"msgsPerSessionMedian{e}"] = s["n"].median()
        N[f"agentFlowSessionShare{e}"] = (s["clar"] > 0).mean()
    N["enterpriseMsgShareSample"] = d["enterprise"].mean()
    N["enterpriseMsgShareFull"] = agg["enterprise_tenant_messages_full_window_total_named"] / agg["window_total"]["messages"]
    surf = d["surface"].value_counts(normalize=True)
    N.update(surfaceWeb=surf.get("desktop_web", 0), surfaceApp=surf.get("desktop_app", 0), surfaceMobile=surf.get("mobile_web", 0))
    for e, g in d.assign(era=np.where(d["month"] >= ERA_B_START, "B", "A")).groupby("era"):
        N[f"surfaceApp{e}"] = (g["surface"] == "desktop_app").mean()
    op = op.assign(lang=op["text"].map(L.language))
    lang = op["lang"].value_counts(normalize=True)
    lang.rename_axis("language").reset_index(name="share").to_csv(out / "tables" / "language_openings.csv", index=False)
    N["langEnglish"] = lang.get("english", 0)
    N["langCount"] = int((op["lang"].value_counts() > 0).sum())

    # usage over time figure
    users = pd.Series(agg["monthly_users"])
    F.usage_timeline(vol, users, out / "figures" / "fig_usage_timeline.png", era_b_start=ERA_B_START)
    N["peakMonthMessages"] = int(vol.max())
    N["usersNovTwentyFive"] = int(users["2025-11"])
    N["usersJanTwentySix"] = int(users["2026-01"])
    clar_months = sorted(d.loc[d["is_clarify_reply"], "month"].unique())
    N["firstAgentMonth"] = pd.Period(clar_months[0]).strftime("%B %Y") if clar_months else "none"
    N["meanMonthlyUsersA"] = users[users.index < ERA_B_START].mean()
    N["meanMonthlyUsersB"] = users[users.index >= ERA_B_START].mean()
    N["msgsPerUserMonthA"] = (vol[vol.index < ERA_B_START] / users[users.index < ERA_B_START]).mean()
    N["msgsPerUserMonthB"] = (vol[vol.index >= ERA_B_START] / users[users.index >= ERA_B_START]).mean()

    # countries (full corpus)
    cu = pd.Series(agg["country_users_top"])
    cm = pd.Series(agg["country_messages_top"])
    N["usersIndiaShare"] = cu["India"] / agg["window_total"]["unique_users"]
    N["usersUSShare"] = cu["United States"] / agg["window_total"]["unique_users"]
    N["msgsUSShare"] = cm["United States"] / agg["window_total"]["messages"]
    N["msgsIndiaShare"] = cm["India"] / agg["window_total"]["messages"]
    N["msgsIsraelShare"] = cm["Israel"] / agg["window_total"]["messages"]
    pd.DataFrame({"messages": cm, "users": cu}).sort_values("messages", ascending=False) \
        .to_csv(out / "tables" / "countries_top.csv")

    # ---------------- labeled sample
    s = pd.read_pickle(a.sample)
    lab = pd.read_csv(a.labels)
    s = s.merge(lab, on="sid", how="inner")
    s["month"] = s["window"].str[:7]
    n_m = s["month"].value_counts()
    s["w"] = s["month"].map(lambda m: (vol[m] / vol.sum()) / (n_m[m] / len(s)))
    s["era"] = np.where(s["month"] >= ERA_B_START, "B", "A")
    s["agent_flow"] = s["text"].str.match(r"^\s*Q:\s")
    N["labeled"] = len(s)
    N["labeledUNK"] = int((s["l1"] == "UNK").sum())
    N["labeledAgentFlow"] = int(s["agent_flow"].sum())
    c = s[s["l1"] != "UNK"].copy()
    N["classifiable"] = len(c)
    c["phase"] = c["l1"].map(tax.phase_of)
    dist = wshare(c, "l1")
    dist["name"] = dist["l1"].map(tax.name_of)
    dist["phase"] = dist["l1"].map(tax.phase_of)
    dist.to_csv(out / "tables" / "dist_l1.csv", index=False)
    pdist = wshare(c, "phase")
    pdist["name"] = pdist["phase"].map(lambda p: tax.phases[p]["name"])
    pdist.to_csv(out / "tables" / "dist_phase.csv", index=False)
    for _, r in dist.iterrows():
        N[f"share{r.l1}"] = r.share
        N[f"share{r.l1}Lo"] = r.ci_lo
        N[f"share{r.l1}Hi"] = r.ci_hi
    for _, r in pdist.iterrows():
        N[f"phase{r.phase}"] = r.share
    know = c["l1"].isin(KNOWLEDGE)
    N["shareKnowledge"] = np.average(know, weights=c["w"])
    N["shareMEcore"] = np.average(c["l1"].isin(KNOWLEDGE + ["GEN"]), weights=c["w"])

    # era and channel splits
    rows = []
    for key, g in [("Era A (Nov 2024 to Nov 2025)", c[c.era == "A"]), ("Era B (Dec 2025 to Oct 2026)", c[c.era == "B"])]:
        w = wshare(g, "l1", n_boot=500)
        w["group"] = key
        rows.append(w)
    era = pd.concat(rows)
    era.to_csv(out / "tables" / "dist_l1_by_era.csv", index=False)
    for e, lab_ in [("A", "Era A (Nov 2024 to Nov 2025)"), ("B", "Era B (Dec 2025 to Oct 2026)")]:
        t = era[era.group == lab_].set_index("l1")["share"]
        for k in ["GEN", "OOS", "EDU", "CMP", "CAL"]:
            N[f"share{k}Era{e}"] = t.get(k, 0)
        N[f"nEra{e}"] = int((c.era == e).sum())
    c["enterprise"] = c["tenant"].notna()
    N["nEnterpriseLabeled"] = int(c["enterprise"].sum())
    ent = c[c.enterprise]
    N["entShareGEN"] = (ent["l1"] == "GEN").mean() if len(ent) else np.nan
    N["entShareKnowledge"] = ent["l1"].isin(KNOWLEDGE).mean() if len(ent) else np.nan
    N["entShareOOS"] = (ent["l1"] == "OOS").mean() if len(ent) else np.nan

    # complexity, verifiable, context
    c["cx"] = "C" + c["c"].astype(str)
    cx = wshare(c, "cx")
    cx.to_csv(out / "tables" / "dist_complexity.csv", index=False)
    for _, r in cx.iterrows():
        N[f"share{r.cx}"] = r.share
    grp = c["l1"].where(c["l1"].isin(["GEN", "OOS", "EDU", "MEC"]), "KNOW")
    c["grp"] = np.where(c["l1"].isin(KNOWLEDGE), "Knowledge and analysis", grp.map(
        {"GEN": "Generate a design", "OOS": "Outside ME", "EDU": "Learning", "MEC": "Concept"}))
    cxg = (c.groupby(["grp", "cx"])["w"].sum() / c.groupby("grp")["w"].sum()).unstack(fill_value=0)
    cxg.to_csv(out / "tables" / "complexity_by_group.csv")
    F.complexity_stack(cxg, out / "figures" / "fig_complexity.png")
    hard = c["c"] >= 3
    N["shareHard"] = np.average(hard, weights=c["w"])
    N["shareHardGEN"] = np.average(hard[c.l1 == "GEN"], weights=c.loc[c.l1 == "GEN", "w"])
    N["shareHardKnow"] = np.average(hard[know], weights=c.loc[know, "w"])
    v = c["v"] == 1
    N["shareVerifiableKnow"] = np.average(v[know], weights=c.loc[know, "w"])
    N["shareVerifiableAll"] = np.average(v, weights=c["w"])
    ctx = c["ctx"].isin(["g", "b"])
    N["shareNeedsGeometry"] = np.average(ctx, weights=c["w"])
    N["shareNeedsGeometryGEN"] = np.average(ctx[c.l1 == "GEN"], weights=c.loc[c.l1 == "GEN", "w"])
    N["shareNeedsInternal"] = np.average(c["ctx"].isin(["i", "b"]), weights=c["w"])
    bench = c[know & (c["c"] >= 3) & v & (c["ctx"] == "n")]
    N["benchSeeds"] = len(bench)
    N["benchSeedsAnyHard"] = int((know & (c["c"] >= 3)).sum())
    bench[["sid", "l1", "c"]].to_csv(Path(a.labels).parent / "benchmark_seed_sids.csv", index=False)  # private

    # ---------------- failure signals inside sessions (labeled sessions only)
    rest = d.merge(s[["session_id", "sid", "l1", "w", "text"]].rename(columns={"text": "open_text"}), on="session_id")
    rest = rest[rest["turn"] > 0]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(pd.concat([s["text"], rest["text"]]).astype(str))
    sig = []
    for sid_, g in s.set_index("sid").iterrows():
        r = rest[rest["sid"] == sid_]
        follow = r[~r["is_clarify_reply"]]
        fr = follow["text"].str.contains(FRUSTRATION).any() if len(follow) else False
        reph = False
        if len(follow):
            sims = cosine_similarity(vec.transform([str(g["text"])]), vec.transform(follow["text"].astype(str)))[0]
            dt = (follow["time"] - g["time"]).dt.total_seconds().to_numpy() / 60
            reph = bool(((sims >= 0.6) & (dt <= 15)).any())
        sig.append({"sid": sid_, "n_follow": len(follow), "frustration": fr, "rephrase": reph,
                    "single": len(r) == 0})
    sig = pd.DataFrame(sig)
    c2 = c.merge(sig, on="sid")
    c2["failure"] = c2["frustration"] | c2["rephrase"]
    N["failAll"], N["failAllLo"], N["failAllHi"] = rate_ci(c2["failure"].to_numpy(float), c2["w"].to_numpy())
    multi = c2[c2["n_follow"] > 0]
    N["failMulti"], N["failMultiLo"], N["failMultiHi"] = rate_ci(multi["failure"].to_numpy(float), multi["w"].to_numpy())
    N["singleShare"] = np.average(c2["single"], weights=c2["w"])
    frows = []
    for gname, g in c2.groupby("grp"):
        p, lo, hi = rate_ci(g["failure"].to_numpy(float), g["w"].to_numpy())
        gm = g[g["n_follow"] > 0]
        pm, lom, him = rate_ci(gm["failure"].to_numpy(float), gm["w"].to_numpy())
        frows.append({"group": gname, "n": len(g), "fail": p, "lo": lo, "hi": hi, "n_multi": len(gm),
                      "fail_multi": pm, "lo_multi": lom, "hi_multi": him})
    ft = pd.DataFrame(frows).sort_values("fail_multi", ascending=False)
    ft.to_csv(out / "tables" / "failure_by_group.csv", index=False)
    for _, r in ft.iterrows():
        k = {"Knowledge and analysis": "Know", "Generate a design": "GEN", "Outside ME": "OOS", "Learning": "EDU", "Concept": "MEC"}[r.group]
        N[f"failMulti{k}"] = r.fail_multi
        N[f"failMulti{k}Lo"] = r.lo_multi
        N[f"failMulti{k}Hi"] = r.hi_multi
        N[f"nMulti{k}"] = r.n_multi
    F.failure_bars(ft, out / "figures" / "fig_failure.png")
    fb = agg["feedback"]
    N.update(fbTotal=fb["total"], fbUsers=fb["unique_users"], fbDislike=fb["dislike"]["total"],
             fbDislikeShare=fb["dislike"]["total"] / fb["total"],
             fbNotFactual=fb["dislike"]["Not factually correct"] / fb["dislike"]["total"],
             fbNotFollow=fb["dislike"]["Did not fully follow instructions"] / fb["dislike"]["total"],
             fbNoAnswer=fb["dislike"]["Didn't give me an answer"] / fb["dislike"]["total"],
             fbRate=fb["total"] / agg["window_total"]["messages"],
             fbPerThousand=1000 * fb["total"] / agg["window_total"]["messages"])

    # ---------------- keyword baseline vs LLM labels
    rb = RuleBackend()
    c["rule"] = c["text"].astype(str).map(lambda t: rb.classify_text(t)["task_l1"])
    kc = kappa_ci(c["l1"].to_numpy(), c["rule"].to_numpy(), n_boot=1000)
    N["kappaRules"], N["kappaRulesLo"], N["kappaRulesHi"], N["accRules"] = kc["kappa"], kc["ci_lo"], kc["ci_hi"], kc["accuracy"]
    kk = c[know]
    kc2 = kappa_ci(kk["l1"].to_numpy(), kk["rule"].to_numpy(), n_boot=1000)
    N["kappaRulesKnow"] = kc2["kappa"]

    # ---------------- stated (survey) vs revealed (queries) among knowledge categories
    sv = pd.read_csv(Path(a.survey) / "2024_hardq_l1.csv").set_index("task_l1")
    rv = dist.set_index("l1")
    cats = [k for k in KNOWLEDGE if k in sv.index or k in rv.index]
    comp = pd.DataFrame({"survey": sv["n"].reindex(cats).fillna(0), "queries": rv["n"].reindex(cats).fillna(0)})
    comp["survey_share"] = comp["survey"] / comp["survey"].sum()
    qk = c[know]
    qshare = qk.groupby("l1")["w"].sum() / qk["w"].sum()
    comp["query_share"] = qshare.reindex(cats).fillna(0)
    comp["name"] = [tax.name_of(k) for k in cats]
    comp = comp.sort_values("survey_share", ascending=False)
    comp.to_csv(out / "tables" / "stated_vs_revealed.csv")
    from scipy.stats import spearmanr
    rho, pval = spearmanr(comp["survey_share"], comp["query_share"])
    N["spearmanStatedRevealed"], N["spearmanP"] = rho, pval
    N["matStated"] = comp.loc["MAT", "survey_share"]
    N["matRevealed"] = comp.loc["MAT", "query_share"]
    N["cmpStated"] = comp.loc["CMP", "survey_share"]
    N["cmpRevealed"] = comp.loc["CMP", "query_share"]
    N["calStated"] = comp.loc["CAL", "survey_share"]
    N["calRevealed"] = comp.loc["CAL", "query_share"]
    N["gdtStated"] = comp.loc["GDT", "survey_share"]
    N["gdtRevealed"] = comp.loc["GDT", "query_share"]

    # page-1 figure: revealed shares (all categories) with stated overlay where comparable
    F.page1(dist, comp, tax, out / "figures" / "fig1_wildmech.png")

    # hours at stake: survey engineering-question hours split by revealed knowledge shares
    tt = pd.read_csv(Path(a.survey) / "2024_task_time.csv").set_index("task")
    eq = tt.loc["answer_engineering_question"]
    N["eqHoursMedian"] = eq["median_hours_per_project"]
    stake = (qshare * eq["relevance_rate"] * eq["median_hours_per_project"]).rename("hours").sort_values(ascending=False)
    stake.to_csv(out / "tables" / "eq_hours_by_category.csv")

    # ---------------- write numbers
    log = {k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v) for k, v in N.items()}
    (out / "query_log.json").write_text(json.dumps(log, indent=1))
    write_macros(N, out / "numbers.tex")
    print(json.dumps(log, indent=1))


def write_macros(N: dict, path: Path):
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    not_pct = ("kappa", "spearman", "msgsPer", "mean", "eqHours", "peak")
    lines = ["% Auto-generated by scripts/run_queries.py. Do not edit by hand."]
    seen = set()
    for k, v in N.items():
        name = "".join(digits.get(ch, ch) for ch in k)
        name = re.sub(r"[^A-Za-z]", "", name)
        assert name not in seen, name
        seen.add(name)
        is_float = isinstance(v, (float, np.floating))
        if is_float and 0 <= v <= 1 and not k.startswith(not_pct):
            txt = f"{100 * v:.0f}\\%"
        elif is_float and float(v).is_integer():
            txt = f"{int(v):,}"
        elif is_float:
            txt = f"{v:.2f}" if abs(v) < 10 else f"{v:,.0f}"
        elif isinstance(v, (int, np.integer)):
            txt = f"{v:,}"
        else:
            txt = str(v)
        lines.append(f"\\newcommand{{\\N{name}}}{{{txt}}}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
