"""Implicit and explicit failure signals per conversation.

Signals (each 0/1 per conversation):
  thumbs_down   any assistant turn rated down
  regenerated   user regenerated an answer
  rephrase      a user turn closely restates the previous user turn after an answer
                (TF-IDF cosine >= threshold within `window_min` minutes)
  correction    a user turn after an answer contains correction language
  abandoned     single exchange, no positive signal, no follow-up (weak, reported separately)
  thumbs_up / copied   positive signals

failure_any = thumbs_down | regenerated | rephrase | correction   (abandonment excluded by default)
These are proxies. The paper validates them against expert judgment on a labeled sample.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORRECTION = re.compile(
    r"\b(?:that'?s (?:wrong|not right|incorrect)|incorrect|not correct|you(?:'re| are) wrong|wrong answer|"
    r"doesn'?t make sense|no,? (?:i meant|that|it)|not what i asked|try again|you misunderstood|"
    r"that is not|recheck|check again|mistake|error in your)\b|לא נכון|טעות|לא מה ששאלתי|תבדוק שוב|שגוי",
    re.I,
)


def _bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def compute_signals(messages: pd.DataFrame, rephrase_threshold: float = 0.6,
                    window_min: float = 15.0) -> pd.DataFrame:
    m = messages.sort_values(["conversation_id", "turn_index"]).copy()
    text = m["text_scrubbed"].fillna("").astype(str)
    m["is_correction"] = (m["role"] == "user") & text.str.contains(CORRECTION)
    fb = m.get("feedback", pd.Series("none", index=m.index)).fillna("none").astype(str).str.lower()
    m["is_down"] = fb.eq("down")
    m["is_up"] = fb.eq("up")
    m["is_regen"] = _bool(m["regenerated"]) if "regenerated" in m else False
    m["is_copy"] = _bool(m["copied"]) if "copied" in m else False

    users = m[m["role"] == "user"].copy()
    rephrase_ids = set()
    if len(users) > 1:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit(users["text_scrubbed"].fillna(""))
        for cid, g in users.groupby("conversation_id", sort=False):
            if len(g) < 2:
                continue
            X = vec.transform(g["text_scrubbed"].fillna(""))
            t = pd.to_datetime(g["created_at"], utc=True).to_numpy()
            for i in range(1, len(g)):
                sim = cosine_similarity(X[i - 1], X[i])[0, 0]
                dt = (t[i] - t[i - 1]) / np.timedelta64(1, "m")
                if sim >= rephrase_threshold and dt <= window_min:
                    rephrase_ids.add(cid)
                    break

    agg = m.groupby("conversation_id").agg(
        n_turns=("turn_index", "size"),
        n_user_turns=("role", lambda r: int((r == "user").sum())),
        thumbs_down=("is_down", "max"),
        thumbs_up=("is_up", "max"),
        regenerated=("is_regen", "max"),
        copied=("is_copy", "max"),
        correction=("is_correction", "max"),
    ).reset_index()
    agg["rephrase"] = agg["conversation_id"].isin(rephrase_ids)
    agg["abandoned"] = (agg["n_user_turns"] == 1) & ~agg["thumbs_up"] & ~agg["copied"]
    agg["failure_any"] = agg[["thumbs_down", "regenerated", "rephrase", "correction"]].any(axis=1)
    for c in ["thumbs_down", "thumbs_up", "regenerated", "copied", "correction", "rephrase", "abandoned", "failure_any"]:
        agg[c] = agg[c].astype(int)
    return agg


def failure_rates(facets_signals: pd.DataFrame, by: str = "task_l1", n_boot: int = 1000,
                  seed: int = 0) -> pd.DataFrame:
    """Failure rate per category with bootstrap 95% CI (resampling conversations within category)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, g in facets_signals.groupby(by):
        x = g["failure_any"].to_numpy()
        boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1) if len(x) else np.array([np.nan])
        rows.append({by: k, "n": len(x), "failure_rate": x.mean(), "ci_lo": np.quantile(boots, 0.025),
                     "ci_hi": np.quantile(boots, 0.975), "thumbs_down_rate": g["thumbs_down"].mean(),
                     "rephrase_rate": g["rephrase"].mean(), "correction_rate": g["correction"].mean()})
    return pd.DataFrame(rows).sort_values("failure_rate", ascending=False)
