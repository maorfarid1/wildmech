"""Rule-based scrubbing that runs BEFORE any LLM sees the text.

Two layers protect customer data:
  1. this scrubber (deterministic, conservative), and
  2. the facet prompt, which tells the model to output no identifiers and
     only a de-identified summary (Clio-style).
Nothing scrubbed here is ever written to disk outside the secure run folder.

Engineering tokens that are public knowledge (standards, threads, alloy grades,
catalog bearing codes) are whitelisted so the scrubber does not destroy meaning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)\d{3}[\s-]?\d{3,4}(?!\w)")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\|/(?:home|Users|mnt|srv|data)/)[^\s\"']+")
CAD_FILE = re.compile(r"\b[\w-]+\.(?:sldprt|sldasm|slddrw|prt|asm|drw|step|stp|igs|iges|x_t|catpart|catproduct|ipt|iam|dwg|dxf|pdf|xlsx?)\b", re.I)

# Public engineering identifiers we keep.
WHITELIST = [
    re.compile(r"\b(?:ISO|ASME|ANSI|DIN|EN|BS|JIS|IEC|UL|NEMA|ASTM|SAE|AMS|MIL|NAS|API|IPC|AWS|ECSS|DO|RTCA|IS)[\s-]?(?:STD[\s-]?)?[A-Z]?\d[\w.:/-]*", re.I),
    re.compile(r"\bM\d{1,3}(?:[xX]\d+(?:\.\d+)?)?\b"),                    # metric threads
    re.compile(r"\b\d+/\d+[\s-]\d+\s?(?:UNC|UNF|UNEF)?\b"),               # 1/4-20 UNC
    re.compile(r"\b\d{4}-T\d+\b|\b(?:1[0-9]|[2-7]\d)\d{2}\b(?=\s?(?:aluminum|aluminium|al|steel|ss|stainless))", re.I),
    re.compile(r"\b17-4\s?PH\b|\b15-5\s?PH\b|\b3[01][0-9]L?\b|\b4[1-4][0-9]0\b|\bTi-?6Al-?4V\b|\bS[23]\d5\w*\b", re.I),
    re.compile(r"\b6[0-3]\d{2}(?:-2RS|-ZZ|-2Z|-RS)?\b"),                   # deep-groove bearing codes
    re.compile(r"\bAS568-?\d{3}\b", re.I),                                 # O-ring dash sizes
]

# Generic internal part / drawing / serial number heuristic.
PART_NO = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9]{2,}(?:-[A-Z0-9]{1,}){1,}\b|\b\d{3,}-\d{3,}(?:-\d+)*\b|\b[A-Z]{2,4}\d{5,}\b")


@dataclass
class ScrubResult:
    text: str
    counts: dict = field(default_factory=dict)


class Scrubber:
    def __init__(self, deny_terms: list[str] | None = None, extra_part_patterns: list[str] | None = None):
        """deny_terms: customer, org, product and project names (private list kept by Leo)."""
        terms = sorted({t.strip() for t in (deny_terms or []) if t and len(t.strip()) >= 2}, key=len, reverse=True)
        self.deny = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, terms)) + r")(?!\w)", re.I) if terms else None
        self.extra = [re.compile(p) for p in (extra_part_patterns or [])]

    def _protect(self, text: str):
        keep = {}
        def stash(m):
            k = f"\u0000{len(keep)}\u0000"
            keep[k] = m.group(0)
            return k
        for w in WHITELIST:
            text = w.sub(stash, text)
        return text, keep

    @staticmethod
    def _restore(text: str, keep: dict) -> str:
        for k, v in keep.items():
            text = text.replace(k, v)
        return text

    def scrub(self, text: str) -> ScrubResult:
        if not isinstance(text, str) or not text:
            return ScrubResult("" if text is None else str(text), {})
        counts: dict[str, int] = {}

        def sub(pattern, label, s):
            n = len(pattern.findall(s))
            if n:
                counts[label] = counts.get(label, 0) + n
            return pattern.sub(f"[{label}]", s)

        s = sub(EMAIL, "EMAIL", text)
        s = sub(URL, "URL", s)
        s = sub(PATH, "PATH", s)
        s = sub(CAD_FILE, "FILE", s)
        s = sub(IPV4, "IP", s)
        if self.deny:
            s = sub(self.deny, "ORG", s)
        s, keep = self._protect(s)
        for p in self.extra:
            s = sub(p, "PART_NO", s)
        s = sub(PART_NO, "PART_NO", s)
        s = sub(PHONE, "PHONE", s)
        s = self._restore(s, keep)
        return ScrubResult(s, counts)
