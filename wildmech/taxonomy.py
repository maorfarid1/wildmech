"""Load the taxonomy and expose lookups."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from . import CONFIG_DIR


@dataclass(frozen=True)
class Taxonomy:
    raw: dict

    @property
    def l1_codes(self) -> list[str]:
        return list(self.raw["l1"].keys())

    @property
    def phases(self) -> dict:
        return self.raw["phases"]

    def phase_of(self, l1: str) -> str:
        return self.raw["l1"][l1]["phase"]

    def name_of(self, l1: str) -> str:
        return self.raw["l1"][l1]["name"]

    def l2_of(self, l1: str) -> list[str]:
        return self.raw["l1"][l1].get("l2", [])

    @property
    def facets(self) -> dict:
        return self.raw["facets"]

    def prompt_block(self) -> str:
        """Compact text block of the taxonomy, used inside LLM prompts."""
        lines = []
        for code, d in self.raw["l1"].items():
            lines.append(f"{code}: {d['name']}. {d['description']} L2 options: {', '.join(d.get('l2', []))}")
        return "\n".join(lines)


@lru_cache(maxsize=4)
def load_taxonomy(path: str | None = None) -> Taxonomy:
    p = path or (CONFIG_DIR / "taxonomy.yaml")
    with open(p, encoding="utf-8") as f:
        return Taxonomy(yaml.safe_load(f))
