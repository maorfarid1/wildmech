"""Load Leo chat logs exported from Mixpanel ("Chat Message Sent", expanded mode) and turn them
into deduplicated messages and sessions.

Design notes
- Sampling: one randomly drawn weekday per calendar month (seed 2027), Nov 2024 to Oct 2026.
  Low-volume months (Jan to Mar 2026) and some late-2026 months use a 2 to 7 day window so each
  draw holds a comparable number of messages. Each pull is capped at 1,000 events; capped windows
  are flagged. Estimates are post-stratified by each month's true total volume.
- Mixpanel stores at most 255 characters of each message.
- The client logs many messages twice (once per routing path, seconds apart). Same user + same
  text within 120 s counts once.
- A session is a run of one user's messages with gaps under 30 minutes.
- "Q: ... A: ..." messages are the user's answers to the assistant's clarifying questions inside an
  agentic task, not new requests. They are kept for session statistics but never treated as a
  session's opening request.
"""
from __future__ import annotations

import glob
import json
import re

import pandas as pd

import os
# Names of the company's own employee, development and test workspaces. Kept in a private file
# (one name per line) so that no workspace or customer name appears in the released code.
_INTERNAL_FILE = os.environ.get("WILDMECH_INTERNAL_TENANTS", "")
INTERNAL_TENANTS = (set(open(_INTERNAL_FILE).read().splitlines()) - {""}) if _INTERNAL_FILE and os.path.exists(_INTERNAL_FILE) else set()
CLARIFY = re.compile(r"^\s*Q:\s")
GREETING = re.compile(r"^\s*(hi+|hello|hey|hii+|shalom|שלום|test|testing|ok|thanks?|thank you|yes|no|continue|go|"
                      r"\?|\.|hola|bonjour)\s*[.!?]*\s*$", re.I)


def load_mixpanel(folder: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(f"{folder}/msgs_*.json")):
        d = json.load(open(f))
        h = d["headers"]
        trunc = any(r[0] is None for r in d["rows"])
        for r in d["rows"]:
            if r[0] is None:
                continue
            x = dict(zip(h, r))
            x["window"] = f.split("msgs_")[1][:10]
            x["window_capped"] = trunc
            rows.append(x)
    df = pd.DataFrame(rows).rename(columns={"$distinct_id": "user_id", "inputText": "text",
                                            "tenantName": "tenant", "mp_country_code": "country",
                                            "applicationType": "surface"})
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["window"].str[:7]
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"raw_events": len(df)}
    d = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()
    log["empty_text"] = len(df) - len(d)
    d = d.sort_values(["user_id", "time"])
    same = (d["user_id"].eq(d["user_id"].shift()) & d["text"].eq(d["text"].shift())
            & ((d["time"] - d["time"].shift()).dt.total_seconds() <= 120))
    log["logging_duplicates"] = int(same.sum())
    d = d[~same]
    internal = d["tenant"].isin(INTERNAL_TENANTS)
    log["internal_tenant_messages"] = int(internal.sum())
    d = d[~internal].copy()
    d["enterprise"] = d["tenant"].notna()
    d["is_clarify_reply"] = d["text"].str.contains(CLARIFY)
    d["is_greeting"] = d["text"].str.match(GREETING)
    d["n_words"] = d["text"].str.split().str.len()
    log["messages_clean"] = len(d)
    return d, log


def sessions(d: pd.DataFrame, gap_min: float = 30) -> pd.DataFrame:
    d = d.sort_values(["user_id", "time"]).copy()
    new = ~d["user_id"].eq(d["user_id"].shift()) | ((d["time"] - d["time"].shift()).dt.total_seconds() > gap_min * 60)
    d["session_id"] = new.cumsum()
    d["turn"] = d.groupby("session_id").cumcount()
    return d


def opening_requests(d: pd.DataFrame) -> pd.DataFrame:
    """First substantive request in each session: skip greetings and clarify replies."""
    cand = d[~d["is_clarify_reply"] & ~d["is_greeting"] & (d["n_words"] >= 2)]
    return cand.sort_values(["session_id", "turn"]).groupby("session_id").head(1)


SCRIPT = {"hebrew": r"[\u0590-\u05FF]", "arabic_persian": r"[\u0600-\u06FF]", "cyrillic": r"[\u0400-\u04FF]",
          "cjk": r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]", "thai": r"[\u0E00-\u0E7F]",
          "devanagari": r"[\u0900-\u097F]"}
LATIN_HINTS = {"spanish": r"\b(el|los|las|una|para|cómo|diseño|qué|necesito)\b",
               "french": r"\b(le|les|des|une|pour|est|je|quelle|vous)\b",
               "german": r"\b(der|die|das|und|ist|ich|eine|mit|für|wie)\b",
               "portuguese": r"\b(não|uma|para|como|você|projeto|peça)\b",
               "italian": r"\b(il|della|per|una|come|sono|che)\b"}


def language(text: str) -> str:
    t = str(text)
    for k, p in SCRIPT.items():
        if re.search(p, t):
            return k
    tl = t.lower()
    scores = {k: len(re.findall(p, tl)) for k, p in LATIN_HINTS.items()}
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best
    return "english"
