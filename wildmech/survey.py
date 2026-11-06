"""Load, clean and aggregate the Leo annual ME surveys (2024, 2025).

Rules: PII columns are dropped on load. Emails are hashed with a run-local salt only to
dedupe, then the hash is dropped. Only aggregates leave this module. Cells with fewer
than k respondents are suppressed.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pandas as pd
import yaml

from . import CONFIG_DIR


def load_map(path=None) -> dict:
    with open(path or CONFIG_DIR / "survey_columns.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("_x000a_", "\n")).strip()


def _check_prefix(header: str, prefix: str, idx: int):
    if not _norm(header).startswith(_norm(prefix)[: max(6, len(_norm(prefix)) // 2)]):
        raise ValueError(f"column {idx} header {header[:40]!r} does not match expected {prefix[:40]!r}")


def load_survey(path: str, year: str, cmap: dict | None = None, salt: str | None = None) -> tuple[pd.DataFrame, dict]:
    cmap = cmap or load_map()
    spec = cmap[year]
    raw = pd.read_excel(path, sheet_name="Form Responses 1")
    report = {"year": year, "rows_raw": len(raw)}
    cols = list(raw.columns)
    out = pd.DataFrame(index=raw.index)
    out["raw_row"] = raw.index  # lets external label files (keyed by raw row) be joined after dedupe
    for idx, f in spec["fields"].items():
        _check_prefix(cols[int(idx)], f["prefix"], idx)
        out[f["key"]] = raw.iloc[:, int(idx)]
    for key, t in spec.get("tasks", {}).items():
        _check_prefix(cols[t["dur"]], t["prefix_dur"], t["dur"])
        out[f"{key}__times"] = raw.iloc[:, t["times"]]
        out[f"{key}__pain_raw"] = raw.iloc[:, t["pain"]]
        out[f"{key}__dur"] = raw.iloc[:, t["dur"]]
        out[f"{key}__unit"] = raw.iloc[:, t["unit"]]
    for key, t in spec.get("binary_tasks", {}).items():
        out[f"{key}__applies"] = raw.iloc[:, t["applies"]]
        out[f"{key}__pain_raw"] = raw.iloc[:, t["pain"]]

    # Dedupe: keep the latest submission per respondent (salted email hash), then drop the hash.
    salt = salt or os.environ.get("WILDMECH_SALT", "local-run-salt")
    h = out["email"].astype(str).str.strip().str.lower().map(lambda e: hashlib.sha256((salt + e).encode()).hexdigest())
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.assign(_h=h).sort_values("timestamp")
    report["duplicate_respondents"] = int(out["_h"].duplicated(keep="last").sum())
    out = out.drop_duplicates("_h", keep="last")
    report["rows_clean"] = len(out)
    report["respondent_hashes"] = set(out["_h"])  # in-memory only, used for cross-year overlap
    out = out.drop(columns=["email", "_h"]).reset_index(drop=True)
    return out, report


def translate(series: pd.Series, cmap: dict) -> pd.Series:
    t = cmap["translate"]
    return series.map(lambda v: t.get(str(v).strip(), v) if pd.notna(v) else v)


def multi_select(series: pd.Series, mapping: dict) -> pd.DataFrame:
    """Return a 0/1 frame, one column per known option (English). Unknown free text -> 'Other text'."""
    rows = []
    for v in series:
        if pd.isna(v):
            rows.append(None)
            continue
        toks = [t.strip() for t in str(v).split(",") if t.strip()]
        r = {eng: 0 for eng in set(mapping.values())}
        other = 0
        for tok in toks:
            hit = next((eng for heb, eng in mapping.items() if tok.lower() == heb.lower()), None)
            if hit:
                r[hit] = 1
            else:
                other = 1
        r["Other text"] = other
        rows.append(r)
    idx = series.index
    df = pd.DataFrame([r if r is not None else {} for r in rows], index=idx)
    df[series.isna()] = np.nan
    return df


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def share_table(s: pd.Series, k_min: int = 5, order: list | None = None) -> pd.DataFrame:
    s = s.dropna()
    vc = s.value_counts()
    if order:
        vc = vc.reindex([o for o in order if o in vc.index] + [i for i in vc.index if i not in order])
    n = len(s)
    rows = []
    for cat, k in vc.items():
        lo, hi = wilson(int(k), n)
        rows.append({"category": cat, "n": int(k), "share": k / n, "ci_lo": lo, "ci_hi": hi})
    t = pd.DataFrame(rows)
    small = t["n"] < k_min
    t.loc[small, ["share", "ci_lo", "ci_hi"]] = np.nan
    t["suppressed"] = small
    t.attrs["N"] = n
    return t


def multi_share(df01: pd.DataFrame, k_min: int = 5) -> pd.DataFrame:
    d = df01.dropna(how="all")
    n = len(d)
    rows = []
    for c in d.columns:
        k = int(d[c].sum())
        lo, hi = wilson(k, n)
        rows.append({"option": c, "n": k, "share": k / n, "ci_lo": lo, "ci_hi": hi, "suppressed": k < k_min})
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    t.loc[t["suppressed"], ["share", "ci_lo", "ci_hi"]] = np.nan
    t.attrs["N"] = n
    return t


def _to_num(v, censored: str):
    if pd.isna(v):
        return np.nan, False
    if str(v).strip() == censored:
        return 50.0, True
    try:
        return float(v), False
    except ValueError:
        return np.nan, False


def task_time_table(df: pd.DataFrame, cmap: dict, year: str = "2024", n_boot: int = 2000,
                    winsor_q: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Per task: relevance, frequency, duration, minutes per project, pain. Medians with bootstrap CIs."""
    rng = np.random.default_rng(seed)
    units = cmap["time_units_minutes"]
    cens = cmap["censored_value"]
    rows = []
    for key, t in cmap[year]["tasks"].items():
        times = df[f"{key}__times"].map(lambda v: _to_num(v, cens))
        dur = df[f"{key}__dur"].map(lambda v: _to_num(v, cens))
        unit = df[f"{key}__unit"].map(lambda u: units.get(str(u).strip(), np.nan))
        tn = times.map(lambda x: x[0])
        dn = dur.map(lambda x: x[0])
        censored = times.map(lambda x: x[1]) | dur.map(lambda x: x[1])
        relevant = tn > 0
        valid = relevant & (dn > 0) & unit.notna()
        minutes_each = (dn * unit)[valid]
        per_project = (tn * dn * unit)[valid]
        pain = pd.to_numeric(df[f"{key}__pain_raw"], errors="coerce")  # Form anchors: 1 = enjoy very much, 5 = suffer very much
        pain_rel = pain[relevant]
        cap = per_project.quantile(winsor_q)
        pp_w = per_project.clip(upper=cap)

        def boot_median(x):
            x = x.to_numpy()
            if len(x) < 5:
                return (np.nan, np.nan)
            b = np.median(rng.choice(x, size=(n_boot, len(x))), axis=1)
            return tuple(np.quantile(b, [0.025, 0.975]))

        lo, hi = boot_median(per_project)
        rows.append({
            "task": key, "label": t["label"], "N": len(df),
            "relevance_rate": relevant.mean(),
            "n_valid_time": int(valid.sum()),
            "median_times_per_project": tn[relevant].median(),
            "median_minutes_each": minutes_each.median(),
            "median_hours_per_project": per_project.median() / 60,
            "median_hours_ci_lo": lo / 60, "median_hours_ci_hi": hi / 60,
            "mean_hours_per_project_winsor": pp_w.mean() / 60,
            "pain_mean": pain_rel.mean(),
            "pain_high_share": (pain_rel >= 4).mean(),     # 4-5 = suffer
            "censored_share": censored[relevant].mean(),
            "multi_unit_or_invalid_unit": int((relevant & (dn > 0) & unit.isna()).sum()),
        })
    return pd.DataFrame(rows).sort_values("median_hours_per_project", ascending=False)


def binary_task_table(df: pd.DataFrame, cmap: dict, year: str = "2024") -> pd.DataFrame:
    rows = []
    for key, t in cmap[year].get("binary_tasks", {}).items():
        a = df[f"{key}__applies"].astype(str).str.strip()
        applies = a.eq("כן")
        pain = pd.to_numeric(df[f"{key}__pain_raw"], errors="coerce")  # Form anchors: 1 = enjoy very much, 5 = suffer very much
        rows.append({"task": key, "label": t["label"], "N": len(df), "applies_rate": applies.mean(),
                     "ambiguous_answers": int(a.eq("כן, לא").sum()),
                     "pain_mean_if_applies": pain[applies].mean(), "pain_high_share_if_applies": (pain[applies] >= 4).mean()})
    return pd.DataFrame(rows)


def crosstab_rate(df: pd.DataFrame, group: str, flag: pd.Series, k_min: int = 10) -> pd.DataFrame:
    rows = []
    for g, idx in df.groupby(group).groups.items():
        f = flag.loc[idx].dropna()
        n, k = len(f), int(f.sum())
        lo, hi = wilson(k, n)
        rows.append({group: g, "n": n, "rate": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi, "suppressed": n < k_min})
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    t.loc[t["suppressed"], ["rate", "ci_lo", "ci_hi"]] = np.nan
    return t
