"""LLM backends.

- AnthropicBackend: production classifier (needs `pip install anthropic` and ANTHROPIC_API_KEY,
  and legal sign-off that query text may be sent to the provider under Leo's DPA).
- RuleBackend: offline keyword baseline. Used for tests, synthetic runs, and as a
  reported baseline in the paper. It is NOT the classifier we report results with.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path


class LLMBackend:
    name = "base"

    def complete_json(self, system: str, user: str) -> dict:
        raise NotImplementedError


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str, max_tokens: int = 800, temperature: float = 0.0,
                 cache_dir: str | None = None, max_retries: int = 4):
        import anthropic  # imported lazily so offline runs need no SDK

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _key(self, system: str, user: str) -> str:
        return hashlib.sha256(f"{self.model}\n{system}\n{user}".encode()).hexdigest()

    def complete_json(self, system: str, user: str) -> dict:
        key = self._key(system, user)
        if self.cache and (self.cache / f"{key}.json").exists():
            return json.loads((self.cache / f"{key}.json").read_text())
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                    system=system, messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                out = parse_json(text)
                if self.cache:  # cache stores model OUTPUT only (de-identified facets), keyed by hash
                    (self.cache / f"{key}.json").write_text(json.dumps(out))
                return out
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


# Keyword baseline (English + Hebrew). Order matters: first match wins for primary.
RULES = [
    ("FAIL", r"\bwhy (?:does|did|is)\b|fail|crack|broke|root cause|defect|mismatch|discrepan|לא עובד|נשבר|כשל|סדק"),
    ("STD", r"\bstandard\b|\bISO\b|\bASME\b|\bMIL\b|\bDIN\b|\bASTM\b|\bSAE\b|\bcode\s+(?:requirement|compliance)|תקן"),
    ("GDT", r"toleranc|gd&t|\bfit\b|stack-?up|datum|position tolerance|טולרנס|סבולת|צבר"),
    ("JNT", r"\bbolt|screw|torque spec|preload|weld|o-?ring|gasket|\bseal|thread|snap|rivet|בורג|ברגים|אורינג|אטימ|ריתוך|הברג"),
    ("MAT", r"material|alloy|coating|anodi|plating|heat treat|corrosion|stainless|aluminum|aluminium|steel|polymer|חומר|ציפוי|אנודייז|קורוזיה|טיפול תרמי"),
    ("CMP", r"bearing|motor|gearbox|spring|actuator|sensor|heat ?sink|valve|linear guide|pump|מיסב|מנוע|קפיץ|חיישן|היטסינק|מסב"),
    ("SIM", r"\bfea\b|\bcfd\b|ansys|abaqus|simulation|mesh|boundary condition|סימולציה|אנליזה"),
    ("CAL", r"calculat|stress|deflection|load|pressure drop|heat transfer|thermal|fatigue|buckling|vibration|frequency|inertia|flow rate|חישוב|עומס|מאמץ|חום|לחץ|זרימה"),
    ("DFM", r"machin|mold|moulding|casting|sheet metal|bend radius|3d print|additive|cnc|manufactur|draft angle|עיבוד שבבי|הזרקה|יציקה|פח|כיפוף|הדפס|ייצור"),
    ("CAD", r"solidworks|creo|\bnx\b|catia|inventor|onshape|fusion|sketch|extrude|fillet|assembly mate|macro|pdm|plm|teamcenter|windchill|מידול"),
    ("SRCH", r"do we have|our (?:part|library|vault)|find (?:the|our) (?:part|drawing|document)|existing part|חלק קיים|במאגר"),
    ("DOC", r"drawing|\bbom\b|report|write (?:an?|the) (?:email|spec|summary)|\beco\b|documentation|שרטוט|מסמך|דוח"),
    ("SUP", r"supplier|vendor|lead time|cost|price|alternative part|obsolete|ספק|עלות|מחיר"),
    ("TST", r"\btest|qualification|inspection|validation|בדיק"),
    ("MEC", r"mechanism|linkage|concept|architecture|cam\b|ratchet|latch|מנגנון|קונספט"),
]
INTENT_RULES = [
    ("diagnose", r"\bwhy\b|fail|problem|issue|למה"),
    ("calculate", r"calculat|how much|how many|what (?:force|load|torque|stress)|size|חישוב|כמה"),
    ("select", r"which|choose|select|recommend|best|איזה|בחירת"),
    ("how_to_tool", r"how (?:do|can) i .* in (?:solidworks|creo|nx|catia|onshape)|how to model"),
    ("generate_document", r"write|draft|summari|כתוב"),
    ("design", r"design|concept|mechanism|תכנון|תכן"),
    ("explain", r"what is|explain|difference|מה ההבדל|הסבר"),
]


class RuleBackend(LLMBackend):
    """Deterministic keyword classifier. Returns the same JSON shape as the LLM prompt."""
    name = "rules"

    def classify_text(self, text: str) -> dict:
        t = text.lower()
        hits = [code for code, pat in RULES if re.search(pat, t, re.I)]
        primary = hits[0] if hits else "OOS"
        secondary = hits[1] if len(hits) > 1 else None
        intent = next((k for k, p in INTENT_RULES if re.search(p, t, re.I)), "lookup")
        n_words = len(t.split())
        complexity = "C1" if n_words < 12 else "C2" if n_words < 35 else "C3" if n_words < 90 else "C4"
        return {
            "task_l1": primary, "task_l2": None, "secondary_l1": secondary, "intent": intent,
            "complexity": complexity,
            "needs_internal_data": "yes" if re.search(r"\bour\b|\bwe\b|existing|internal|שלנו", t) else "no",
            "needs_geometry": "yes" if re.search(r"attached|this part|this assembly|model|drawing", t) else "no",
            "verifiable_answer": "yes" if intent == "calculate" else "no",
            "standards": re.findall(r"\b(ISO|ASME|DIN|MIL|ASTM|SAE|EN|API)\b", text),
            "summary": "",
        }

    def complete_json(self, system: str, user: str) -> dict:
        return self.classify_text(user)


def make_backend(cfg: dict) -> LLMBackend:
    kind = cfg.get("backend", "rules")
    if kind == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return AnthropicBackend(model=cfg["model"], cache_dir=cfg.get("cache_dir"))
    return RuleBackend()
