"""Figures. Plain matplotlib, colorblind-safe palette, vector PDF next to every PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Okabe-Ito
PHASE_COLORS = {"P1": "#0072B2", "P2": "#E69F00", "P3": "#009E73", "P4": "#CC79A7", "P5": "#D55E00",
                "P6": "#56B4E9", "P0": "#999999"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def workflow_strip(dist: pd.DataFrame, tax, path, title: str = "", share_col: str = "share",
                   code_col: str = "task_l1"):
    """Page-1 figure: task shares laid out along the design workflow, one column per phase.
    Bar length = share of queries. Grouped and ordered by phase."""
    d = dist.copy()
    d["phase"] = d[code_col].map(tax.phase_of)
    d["porder"] = d["phase"].map(lambda p: tax.phases[p]["order"])
    d = d.sort_values(["porder", share_col], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(d) + 1.4))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d[share_col] * 100, color=[PHASE_COLORS[p] for p in d["phase"]], height=0.72)
    ax.set_yticks(y, [f"{tax.name_of(c)}" for c in d[code_col]])
    for yi, v in zip(y, d[share_col] * 100):
        ax.text(v + 0.4, yi, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_xlabel("Share of questions (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[p]) for p in tax.phases if p in set(d["phase"])]
    labels = [tax.phases[p]["name"] for p in tax.phases if p in set(d["phase"])]
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=7.5, title="Design phase", title_fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    _save(fig, path)


def task_time_pain(t: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = t["median_hours_per_project"]
    yv = t["pain_mean"]
    s = 700 * t["relevance_rate"]
    ax.scatter(x, yv, s=s, alpha=0.6, color="#0072B2", edgecolor="k", linewidth=0.5)
    ax.errorbar(x, yv, xerr=[x - t["median_hours_ci_lo"], t["median_hours_ci_hi"] - x], fmt="none",
                ecolor="#555", elinewidth=0.8, capsize=2)
    short = {"answer_engineering_question": "Answer an engineering question",
             "convert_mfg_method": "Convert part to another process",
             "component_search_catalog": "Find a catalog component",
             "model_catalog_part": "Model a catalog part",
             "part_search_internal": "Find an existing internal part"}
    offsets = {"model_catalog_part": (-10, 14), "component_search_catalog": (8, -16),
               "answer_engineering_question": (-60, 16), "part_search_internal": (16, 8)}
    for _, r in t.iterrows():
        ax.annotate(short.get(r["task"], r["label"]), (r["median_hours_per_project"], r["pain_mean"]),
                    xytext=offsets.get(r["task"], (10, 6)), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(0, max(t["median_hours_ci_hi"]) * 1.15)
    ax.set_xlabel("Median hours per typical project (95% bootstrap CI)")
    ax.set_ylabel("Mean pain (1 = enjoy, 5 = suffer)")
    ax.set_ylim(2.35, 3.25)
    _save(fig, path)


def bar_share(t: pd.DataFrame, label_col: str, path, title: str):
    d = t[t[label_col] != "Other text"].dropna(subset=["share"]).sort_values("share")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(d) + 1))
    y = np.arange(len(d))
    ax.barh(y, d["share"] * 100, xerr=[(d["share"] - d["ci_lo"]) * 100, (d["ci_hi"] - d["share"]) * 100],
            color="#009E73", ecolor="#444", capsize=2, height=0.7)
    ax.set_yticks(y, d[label_col])
    ax.set_xlabel("% of respondents (95% Wilson CI)")
    ax.set_title(title, loc="left", fontsize=9)
    _save(fig, path)


def opportunity_map(t: pd.DataFrame, path, title: str):
    """x = survey hours at stake, y = share of AI queries, color = failure rate."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sc = ax.scatter(t["hours_at_stake"], t["query_share"] * 100, c=t["failure_rate"] * 100, s=140,
                    cmap="viridis_r", edgecolor="k", linewidth=0.5)
    for _, r in t.iterrows():
        ax.annotate(r["task_l1"], (r["hours_at_stake"], r["query_share"] * 100), xytext=(5, 4),
                    textcoords="offset points", fontsize=8)
    fig.colorbar(sc, ax=ax, label="Failure-signal rate (%)")
    ax.set_xlabel("Engineer hours at stake per project (survey)")
    ax.set_ylabel("Share of AI queries (%)")
    ax.set_title(title, loc="left", fontsize=9)
    _save(fig, path)


GREY = "#9a9a9a"
ACCENT = "#0072B2"


def usage_timeline(vol: pd.Series, users: pd.Series, path, era_b_start: str = "2025-12"):
    x = np.arange(len(vol))
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.bar(x, vol.values / 1000, color=GREY, width=0.75, label="Messages (thousands)")
    ax.set_ylabel("Messages per month (thousands)")
    ax2 = ax.twinx()
    ax2.plot(x, users.values, color=ACCENT, marker="o", ms=3, lw=1.5, label="Active users")
    ax2.set_ylabel("Users sending messages", color=ACCENT)
    ax2.tick_params(axis="y", colors=ACCENT)
    ax2.spines["top"].set_visible(False)
    b = list(vol.index).index(era_b_start)
    ax.axvline(b - 0.5, color="k", lw=0.8, ls="--")
    ax.set_ylim(0, vol.max() / 1000 * 1.22)
    ax2.set_ylim(0, users.max() * 1.22)
    ax.text(b - 0.2, ax.get_ylim()[1] * 0.98, "Era B", fontsize=7.5, va="top",
            bbox=dict(fc="white", ec="none", pad=1))
    ax.text(-0.4, ax.get_ylim()[1] * 0.98, "Era A", fontsize=7.5, va="top",
            bbox=dict(fc="white", ec="none", pad=1))
    labels = [m if m.endswith(("-01", "-07")) else "" for m in vol.index]
    ax.set_xticks(x, labels, fontsize=7.5)
    fig.tight_layout()
    _save(fig, path)


def complexity_stack(cxg: pd.DataFrame, path):
    order = ["Generate a design", "Knowledge and analysis", "Concept", "Learning", "Outside ME"]
    cxg = cxg.reindex([o for o in order if o in cxg.index])
    cols = [c for c in ["C1", "C2", "C3", "C4"] if c in cxg.columns]
    shades = ["#d6e6f4", "#9ecae1", "#4292c6", "#08519c"]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    left = np.zeros(len(cxg))
    y = np.arange(len(cxg))[::-1]
    names = {"C1": "C1 lookup", "C2": "C2 one rule or calculation", "C3": "C3 multi-step or trade-off", "C4": "C4 open-ended design"}
    for col, sh in zip(cols, shades):
        v = cxg[col].to_numpy() * 100
        ax.barh(y, v, left=left, color=sh, height=0.65, label=names[col], edgecolor="white", lw=0.5)
        for yi, l, vi in zip(y, left, v):
            if vi >= 7:
                ax.text(l + vi / 2, yi, f"{vi:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if col in ("C3", "C4") else "black")
        left += v
    ax.set_yticks(y, cxg.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of requests in the group (%)")
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.22))
    fig.tight_layout()
    _save(fig, path)


def failure_bars(ft: pd.DataFrame, path):
    d = ft.dropna(subset=["fail_multi"]).sort_values("fail_multi")
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    y = np.arange(len(d))
    cols = [ACCENT if g == "Knowledge and analysis" else GREY for g in d["group"]]
    ax.barh(y, d["fail_multi"] * 100, color=cols, height=0.6,
            xerr=[(d["fail_multi"] - d["lo_multi"]) * 100, (d["hi_multi"] - d["fail_multi"]) * 100],
            ecolor="#333", capsize=2)
    ax.set_yticks(y, [f"{g} (n={n})" for g, n in zip(d["group"], d["n_multi"])])
    ax.set_xlabel("Sessions with a failure signal (%), multi-turn sessions, 95% CI")
    fig.tight_layout()
    _save(fig, path)


SHORT = {"STD": "Standards and codes", "MEC": "Concepts and mechanisms", "GEN": "Generate a design or CAD model",
         "CAL": "Calculations", "CMP": "Component selection", "MAT": "Materials and coatings", "JNT": "Joints and seals",
         "SIM": "Simulation", "GDT": "Tolerancing and GD&T", "CAD": "CAD and PLM how-to", "SRCH": "Internal search",
         "DOC": "Drawings and documents", "DFM": "Manufacturing and DFM", "SUP": "Sourcing and cost",
         "TST": "Testing", "FAIL": "Failure analysis", "EDU": "Learning and coursework", "OOS": "Outside ME", "UNK": "Unclassifiable"}


def page1(dist: pd.DataFrame, comp: pd.DataFrame, tax, path):
    """a: what engineers actually ask (all categories, grouped by design phase).
    b: inside knowledge questions, what engineers say is hard (survey) vs what they ask (logs)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    d = dist.copy()
    d["porder"] = d["phase"].map(lambda p: tax.phases[p]["order"])
    d = d.sort_values(["porder", "share"], ascending=[True, False])
    y = np.arange(len(d))[::-1]
    a1.barh(y, d["share"] * 100, color=[PHASE_COLORS.get(p, "#999") for p in d["phase"]], height=0.72,
            xerr=[(d["share"] - d["ci_lo"]) * 100, (d["ci_hi"] - d["share"]) * 100], ecolor="#444",
            error_kw={"lw": 0.7, "capsize": 1.5})
    for yi, v, hi in zip(y, d["share"] * 100, d["ci_hi"] * 100):
        a1.text(hi + 0.8, yi, f"{v:.0f}", va="center", fontsize=6.5, color="#333")
    a1.set_yticks(y, [SHORT[k] for k in d["l1"]], fontsize=7)
    a1.set_xlabel("Share of requests (%), 95% CI", fontsize=7.5)
    a1.set_title("a  What engineers ask (n = %d requests)" % d["n"].sum(), loc="left", fontsize=8.5, fontweight="bold")
    ph = [p for p in sorted(tax.phases, key=lambda p: tax.phases[p]["order"]) if p in set(d["phase"])]
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS.get(p, "#999")) for p in ph],
               [tax.phases[p]["name"] for p in ph], fontsize=6.5, frameon=False, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.01), title="Design phase", title_fontsize=7)
    c = comp.sort_values("survey_share")
    y2 = np.arange(len(c))
    a2.hlines(y2, c[["survey_share", "query_share"]].min(axis=1) * 100, c[["survey_share", "query_share"]].max(axis=1) * 100,
              color="#bbb", lw=1.5)
    a2.scatter(c["survey_share"] * 100, y2, color="#E69F00", s=24, zorder=3, label="Stated: named as hard (survey)")
    a2.scatter(c["query_share"] * 100, y2, color=ACCENT, s=24, zorder=3, label="Revealed: asked (logs)")
    a2.set_yticks(y2, [SHORT[k] for k in c.index], fontsize=7)
    a2.set_xlabel("Share within knowledge questions (%)", fontsize=7.5)
    a2.set_title("b  Stated versus revealed", loc="left", fontsize=8.5, fontweight="bold")
    a2.legend(fontsize=6.5, frameon=False, loc="lower right", bbox_to_anchor=(1.02, 0.17), handletextpad=0.2, borderaxespad=0.2)
    a2.set_ylim(-0.6, len(c) - 0.4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, path)


PHASE_COLORS["PL"] = "#8c6d31"


def survey_ai_panels(uses: pd.DataFrame, barriers: pd.DataFrame, path, n_users: int):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, t, title, hi in [(a1, uses, f"a  Uses (AI users, n = {n_users})", "Technical / engineering questions"),
                             (a2, barriers, "b  Barriers to using it more", "Low trust in the results")]:
        d = t[t["option"] != "Other text"].dropna(subset=["share"]).sort_values("share")
        y = np.arange(len(d))
        ax.barh(y, d["share"] * 100, color=[ACCENT if o == hi else GREY for o in d["option"]], height=0.68,
                xerr=[(d["share"] - d["ci_lo"]) * 100, (d["ci_hi"] - d["share"]) * 100], ecolor="#444",
                error_kw={"lw": 0.7, "capsize": 1.5})
        ax.set_yticks(y, d["option"], fontsize=7)
        ax.set_xlabel("Respondents (%), 95% Wilson CI", fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
