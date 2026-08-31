"""Persistent training-pair blacklist and high-signal alignment filters."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from translation_quality import extract_acronyms, extract_numbers


MILITARY_RE = re.compile(
    r"\b(?:mortar shell|rocket|artillery|rifle|grenade|military|soldier|attack|missile|ammunition)\b",
    re.IGNORECASE,
)
TECHNICAL_RE = re.compile(
    r"\b(?:API|GPU|CPU|model|algorithm|database|software|hardware|network|training|inference)\b",
    re.IGNORECASE,
)
LITERARY_RE = re.compile(
    r"\b(?:castle|moor|shroud|lineage|shadow|soul|whispered|ancient|poem|novel)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TrainingBlacklist:
    exact_hashes: frozenset[str]
    source_patterns: tuple[re.Pattern[str], ...]

    @classmethod
    def load(cls, path: Path | None) -> "TrainingBlacklist":
        if path is None or not path.is_file():
            return cls(frozenset(), ())
        raw = json.loads(path.read_text(encoding="utf-8"))
        hashes: set[str] = set(map(str, raw.get("pair_hashes", [])))
        for pair in raw.get("exact_pairs", []):
            source = str(pair.get("source", "")).strip()
            target = str(pair.get("target", "")).strip()
            if source and target:
                hashes.add(pair_hash(source, target))
        patterns = tuple(
            re.compile(str(value), re.IGNORECASE)
            for value in raw.get("source_patterns", [])
            if str(value)
        )
        return cls(frozenset(hashes), patterns)

    def reason(self, source: str, target: str) -> str | None:
        if pair_hash(source, target) in self.exact_hashes:
            return "blacklisted_pair"
        if any(pattern.search(source) for pattern in self.source_patterns):
            return "blacklisted_source"
        return None


def pair_hash(source: str, target: str) -> str:
    return hashlib.sha256(
        (source.strip().casefold() + "\0" + target.strip().casefold()).encode("utf-8")
    ).hexdigest()


def classify_domain(source: str, target: str = "") -> str:
    joined = source + " " + target
    if MILITARY_RE.search(joined):
        return "military"
    if TECHNICAL_RE.search(joined):
        return "technical"
    if LITERARY_RE.search(joined):
        return "literary"
    return "general"


def alignment_quality_reason(
    source: str,
    target: str,
    source_language: str,
    blacklist: TrainingBlacklist,
) -> str | None:
    blocked = blacklist.reason(source, target)
    if blocked:
        return blocked

    source_numbers = extract_numbers(source)
    if source_numbers:
        retained = sum(number in target for number in source_numbers) / len(source_numbers)
        if retained < 0.6:
            return "number_mismatch"
    source_acronyms = extract_acronyms(source)
    if source_acronyms and not any(value in target for value in source_acronyms):
        return "acronym_mismatch"

    english, chinese = (source, target) if source_language == "en" else (target, source)
    folded = english.casefold()
    if "shroud" in folded and not any(term in chinese for term in ("裹尸", "帷幕", "笼罩", "遮蔽", "面纱")):
        return "shroud_alignment"
    if "mortar" in folded:
        military = bool(MILITARY_RE.search(english))
        expected = ("迫击炮",) if military else ("灰浆", "灰泥", "水泥", "砂浆", "研钵")
        if not any(term in chinese for term in expected):
            return "mortar_alignment"
    return None


def domain_allowed(
    domain: str,
    counts: Mapping[str, int],
    target_total: int,
    maximum_fractions: Mapping[str, float],
) -> bool:
    maximum = maximum_fractions.get(domain)
    if maximum is None:
        return True
    return counts.get(domain, 0) < max(1, round(target_total * maximum))
