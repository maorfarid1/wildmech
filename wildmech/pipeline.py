"""End-to-end query pipeline.

    python -m wildmech.pipeline --synthetic --out runs/synth            # dry run, offline
    python -m wildmech.pipeline --export $WILDMECH_EXPORT_DIR --out runs/v1 --backend anthropic

Stages: load+validate -> scrub -> facets -> signals -> clusters -> safe aggregates ->
        annotation sheet -> benchmark candidates -> survey link -> figures.
Only aggregates, de-identified summaries and IDs are written. Raw and scrubbed text are not.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from . import CONFIG_DIR
from . import agreement, benchmark, cluster, figures, privacy, signals, weighting
from .facets import extract_facets
from .llm import make_backend
from .schema import load_tables, validate
from .scrub import Scrubber
from .taxonomy import load_taxonomy


def load_config(path=None) -> dict:
    with open(path or CONFIG_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(os.path.expandvars(f.read()))


def _read_lines(p):
    if p and not p.startswith("$") and Path(p).exists():
        return [x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    return []


def run(tables: dict, cfg: dict, out: Path, backend_name: str | None = None, survey_task_time: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    tax = load_taxonomy()
    log = {"warnings": validate(tables)}
    conv, msgs, users, orgs = (tables[k] for k in ("conversations", "messages", "users", "orgs"))
    if cfg["pipeline"].get("max_conversations"):
        keep = conv["conversation_id"].head(cfg["pipeline"]["max_conversations"])
        conv, msgs = conv[conv["conversation_id"].isin(keep)], msgs[msgs["conversation_id"].isin(keep)]

    # 1. scrub
    scr = Scrubber(_read_lines(cfg["data"].get("deny_terms_file")), _read_lines(cfg["data"].get("part_number_patterns_file")))
    res = msgs["text"].map(scr.scrub)
    msgs = msgs.assign(text_scrubbed=res.map(lambda r: r.text))
    scrub_counts = pd.Series([r.counts for r in res]).apply(pd.Series).sum().to_dict() if len(res) else {}
    log["scrub_counts"] = {k: int(v) for k, v in scrub_counts.items()}

    # 2. facets
    llm_cfg = dict(cfg["llm"])
    if backend_name:
        llm_cfg["backend"] = backend_name
    backend = make_backend(llm_cfg)
    fac = extract_facets(msgs, backend, tax)
    log["backend"] = backend.name

    # 3. signals
    sig = signals.compute_signals(msgs, cfg["pipeline"]["rephrase_threshold"], cfg["pipeline"]["rephrase_window_min"])
    df = (conv[["conversation_id", "user_id", "org_id", "started_at"]]
          .merge(fac, on="conversation_id").merge(sig, on="conversation_id")
          .merge(users.drop(columns=["org_id"], errors="ignore"), on="user_id", how="left")
          .merge(orgs, on="org_id", how="left"))

    # 4. clusters on de-identified summaries (fallback: scrubbed first user turn, in memory only)
    first_user = msgs[msgs["role"] == "user"].sort_values("turn_index").groupby("conversation_id")["text_scrubbed"].first()
    texts = df["summary"].where(df["summary"].str.len() > 0, df["conversation_id"].map(first_user)).fillna("").tolist()
    labels, _, cinfo = cluster.cluster(texts, k=cfg["pipeline"].get("cluster_k"), method=cfg["pipeline"]["cluster_embedding"])
    df["cluster"] = labels
    log["clustering"] = cinfo
    terms = cluster.top_terms(texts, labels)
    cl = privacy.safe_counts(df, "cluster", cfg["privacy"]["min_users_per_cell"], cfg["privacy"]["min_orgs_per_cell"])
    cl["top_terms"] = cl["cluster"].map(lambda c: ", ".join(terms.get(int(c), [])))
    cl["majority_l1"] = cl["cluster"].map(df.groupby("cluster")["task_l1"].agg(lambda s: s.value_counts().index[0]))
    cl["l1_purity"] = cl["cluster"].map(df.groupby("cluster")["task_l1"].agg(lambda s: s.value_counts(normalize=True).iloc[0]))
    cl.loc[cl["suppressed"], "top_terms"] = ""
    cl.to_csv(out / "clusters.csv", index=False)

    # 5. safe aggregates
    mu, mo = cfg["privacy"]["min_users_per_cell"], cfg["privacy"]["min_orgs_per_cell"]
    tables_out = {
        "dist_l1": privacy.safe_counts(df, "task_l1", mu, mo),
        "dist_l1_l2": privacy.safe_counts(df, ["task_l1", "task_l2"], mu, mo),
        "dist_phase": privacy.safe_counts(df, "phase", mu, mo),
        "dist_complexity": privacy.safe_counts(df, "complexity", mu, mo),
        "dist_intent": privacy.safe_counts(df, "intent", mu, mo),
        "l1_by_industry": privacy.safe_counts(df, ["industry", "task_l1"], mu, mo),
        "l1_by_size": privacy.safe_counts(df, ["size_band", "task_l1"], mu, mo),
        "l1_by_seniority": privacy.safe_counts(df, ["seniority", "task_l1"], mu, mo),
        "complexity_by_l1": privacy.safe_counts(df, ["task_l1", "complexity"], mu, mo),
        "needs_internal_by_l1": privacy.safe_counts(df, ["task_l1", "needs_internal_data"], mu, mo),
        "failure_by_l1": signals.failure_rates(df, "task_l1"),
        "failure_by_complexity": signals.failure_rates(df, "complexity"),
    }
    for k, t in tables_out.items():
        t.to_csv(out / f"{k}.csv", index=False)
    d = tables_out["dist_l1"].dropna(subset=["share"]).rename(columns={"share": "share"})
    figures.workflow_strip(d, tax, out / "fig1_workflow_strip.png", title=f"What engineers ask AI (n={len(df)} conversations)")

    # 6. annotation sheet for Leo engineers (summaries only, model label hidden)
    agreement.annotation_sample(df, cfg["annotation"]["sample_size"], min_per_stratum=cfg["annotation"]["min_per_category"]) \
        .to_csv(out / "annotation_sheet.csv", index=False)

    # 7. benchmark candidates
    benchmark.candidates(df).to_csv(out / "benchmark_candidates.csv", index=False)

    # 8. survey link
    if survey_task_time and Path(survey_task_time).exists():
        tt = pd.read_csv(survey_task_time)
        eq_share = d.set_index("task_l1")["share"]
        hrs = weighting.hours_at_stake(tt, eq_share)
        opp = weighting.opportunity(hrs, d[["task_l1", "share"]], tables_out["failure_by_l1"])
        opp.to_csv(out / "opportunity.csv", index=False)
        figures.opportunity_map(opp.dropna(subset=["hours_at_stake", "query_share", "failure_rate"]),
                                out / "fig_opportunity.png", "Where AI could save the most engineer time")

    # facets table without any text, for internal analysis
    df.drop(columns=["summary"]).to_parquet(out / "facets_no_text.parquet") if _has_parquet() else \
        df.drop(columns=["summary"]).to_csv(out / "facets_no_text.csv", index=False)
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    return df, tables_out, log


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="runs/synth")
    ap.add_argument("--backend")
    ap.add_argument("--survey-task-time", default="outputs/survey/tables/2024_task_time.csv")
    a = ap.parse_args()
    cfg = load_config()
    if a.synthetic:
        from .synth import make
        tables = make(n_conv=a.n)
        backend = a.backend or "rules"
    else:
        tables = load_tables(a.export or cfg["data"]["export_dir"])
        backend = a.backend
    df, t, log = run(tables, cfg, Path(a.out), backend, a.survey_task_time)
    if a.synthetic and "_true_l1" in tables["conversations"]:
        truth = tables["conversations"].set_index("conversation_id")["_true_l1"]
        k = agreement.kappa_ci(truth.reindex(df["conversation_id"]).to_numpy(), df["task_l1"].to_numpy(), n_boot=300)
        log["synthetic_sanity_kappa_rules_vs_template"] = k
        (Path(a.out) / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
