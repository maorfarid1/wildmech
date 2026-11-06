"""Expected export schema for the Leo query export, plus validation.

Four tables, joined by hashed IDs. Leo keeps the salt; we never see raw IDs.
See docs/data_request.md for the human-readable version.
"""
from __future__ import annotations

import pandas as pd

SCHEMA = {
    "conversations": {
        "conversation_id": "str",      # hashed
        "user_id": "str",              # hashed
        "org_id": "str",               # hashed
        "started_at": "datetime",
        "surface": "str",              # web_chat | cad_addin | api | other
        "model_version": "str",        # Leo model/backend version at the time
    },
    "messages": {
        "message_id": "str",
        "conversation_id": "str",
        "turn_index": "int",
        "role": "str",                 # user | assistant
        "text": "str",                 # raw text, stays inside Leo's environment
        "created_at": "datetime",
        "has_attachment": "bool",
        "attachment_types": "str",     # comma list: cad, pdf, image, drawing, none
        "feedback": "str",             # up | down | none (on assistant turns)
        "regenerated": "bool",         # user asked to regenerate this assistant turn
        "copied": "bool",              # user copied the answer (if logged)
    },
    "users": {
        "user_id": "str",
        "org_id": "str",
        "seniority": "str",            # optional: student|junior|engineer|senior|lead|manager|unknown
        "role": "str",                 # optional: design|analysis|manufacturing|other|unknown
        "signup_month": "str",         # YYYY-MM
        "plan": "str",                 # self_serve | enterprise | trial
        "country": "str",              # ISO-2
    },
    "orgs": {
        "org_id": "str",
        "industry": "str",             # coarse bucket, see data_request.md
        "size_band": "str",            # 1-49 | 50-249 | 250-999 | 1000-9999 | 10000+
        "cad_system": "str",
        "plm_system": "str",
    },
}

REQUIRED = {
    "conversations": ["conversation_id", "user_id", "org_id", "started_at"],
    "messages": ["message_id", "conversation_id", "turn_index", "role", "text", "created_at"],
    "users": ["user_id", "org_id"],
    "orgs": ["org_id"],
}


class SchemaError(ValueError):
    pass


def validate(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Raise on missing required columns. Return warnings for missing optional ones."""
    warnings = []
    for name, req in REQUIRED.items():
        if name not in tables:
            raise SchemaError(f"missing table: {name}")
        df = tables[name]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise SchemaError(f"{name}: missing required columns {missing}")
        opt_missing = [c for c in SCHEMA[name] if c not in df.columns]
        if opt_missing:
            warnings.append(f"{name}: optional columns absent {opt_missing}")
    roles = set(tables["messages"]["role"].unique())
    if not roles <= {"user", "assistant", "system", "tool"}:
        raise SchemaError(f"unexpected roles {roles}")
    orphan = ~tables["messages"]["conversation_id"].isin(tables["conversations"]["conversation_id"])
    if orphan.any():
        warnings.append(f"{int(orphan.sum())} messages point to unknown conversations")
    return warnings


def load_tables(path: str) -> dict[str, pd.DataFrame]:
    """Load parquet or jsonl files named <table>.parquet / <table>.jsonl from a folder."""
    from pathlib import Path

    p = Path(path)
    out = {}
    for name in SCHEMA:
        for ext, reader in ((".parquet", pd.read_parquet), (".jsonl", lambda f: pd.read_json(f, lines=True)),
                            (".csv", pd.read_csv)):
            f = p / f"{name}{ext}"
            if f.exists():
                out[name] = reader(f)
                break
    for t, col in (("conversations", "started_at"), ("messages", "created_at")):
        if t in out and col in out[t]:
            out[t][col] = pd.to_datetime(out[t][col], utc=True)
    return out
