"""Per-conversation facet extraction (Clio-style).

Input: scrubbed user turns (+ a short slice of assistant turns for context).
Output: categorical facets + a de-identified one-line summary. Raw text never leaves
this step; downstream stages see facets and summaries only.
"""
from __future__ import annotations

import pandas as pd

from .llm import LLMBackend
from .taxonomy import Taxonomy

SYSTEM = """You classify conversations between mechanical engineers and an AI assistant for a research study.
Privacy rules (strict):
- Never output company names, customer names, product names, project names, part numbers, people, places or any number that could identify a design.
- The summary must be generic enough that it could describe many companies' questions. Replace specifics with generic terms (e.g. "a bracket", "an aluminum housing").
Return ONE JSON object and nothing else."""

USER_TMPL = """Taxonomy of mechanical-engineering tasks (pick the best primary L1 code, an L2 option from that code, and an optional different secondary L1):
{taxonomy}

Facet definitions:
- intent: one of {intents}
- complexity: C1 single fact or lookup; C2 one rule or one calculation; C3 multi-step reasoning or trade-off; C4 open-ended design judgment that needs context the conversation does not hold
- needs_internal_data: "yes" if a correct answer needs the company's own parts, documents, standards or history
- needs_geometry: "yes" if a correct answer needs the actual CAD geometry or drawing
- verifiable_answer: "yes" if there is a checkable numeric or closed-form answer
- standards: list of standard FAMILIES mentioned (e.g. "ISO 2768", "ASME Y14.5"); [] if none
- summary: <= 20 words, de-identified, describing what the engineer wanted

Conversation (user turns marked U, assistant turns marked A and truncated):
<conversation>
{conversation}
</conversation>

JSON keys: task_l1, task_l2, secondary_l1, intent, complexity, needs_internal_data, needs_geometry, verifiable_answer, standards, summary"""


def build_transcript(msgs: pd.DataFrame, max_user_chars: int = 1500, max_asst_chars: int = 300,
                     max_turns: int = 8) -> str:
    msgs = msgs.sort_values("turn_index").head(max_turns)
    parts = []
    for _, m in msgs.iterrows():
        if m["role"] == "user":
            parts.append("U: " + str(m["text_scrubbed"])[:max_user_chars])
        elif m["role"] == "assistant":
            parts.append("A: " + str(m["text_scrubbed"])[:max_asst_chars])
    return "\n".join(parts)


def validate_facets(out: dict, tax: Taxonomy) -> dict:
    f = tax.facets
    clean = dict(out)
    if clean.get("task_l1") not in tax.l1_codes:
        clean["task_l1"] = "OOS"
        clean["_invalid"] = True
    if clean.get("task_l2") not in tax.l2_of(clean["task_l1"]):
        clean["task_l2"] = None
    if clean.get("secondary_l1") not in tax.l1_codes or clean.get("secondary_l1") == clean["task_l1"]:
        clean["secondary_l1"] = None
    if clean.get("intent") not in f["intent"]:
        clean["intent"] = "lookup"
    if clean.get("complexity") not in f["complexity"]:
        clean["complexity"] = "C2"
    for k in ("needs_internal_data", "needs_geometry", "verifiable_answer"):
        clean[k] = "yes" if str(clean.get(k, "no")).lower() in ("yes", "true", "1") else "no"
    if not isinstance(clean.get("standards"), list):
        clean["standards"] = []
    clean["summary"] = str(clean.get("summary") or "")[:200]
    clean["phase"] = tax.phase_of(clean["task_l1"])
    return clean


def extract_facets(messages: pd.DataFrame, backend: LLMBackend, tax: Taxonomy,
                   rule_input: str = "user_only") -> pd.DataFrame:
    """messages must hold text_scrubbed. Returns one row per conversation."""
    rows = []
    intents = ", ".join(tax.facets["intent"])
    for cid, g in messages.groupby("conversation_id", sort=False):
        transcript = build_transcript(g)
        if backend.name == "rules" and rule_input == "user_only":
            # the keyword baseline only sees user text
            payload = " ".join(g.loc[g["role"] == "user", "text_scrubbed"].astype(str))
        else:
            payload = USER_TMPL.format(taxonomy=tax.prompt_block(), intents=intents, conversation=transcript)
        try:
            out = backend.complete_json(SYSTEM, payload)
        except Exception as e:  # noqa: BLE001
            out = {"task_l1": "OOS", "_error": str(e)[:200]}
        out = validate_facets(out, tax)
        out["conversation_id"] = cid
        out["backend"] = backend.name
        rows.append(out)
    return pd.DataFrame(rows)
