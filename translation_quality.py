"""Reference-free quality checks and document-level consistency helpers.

The checks in this module deliberately avoid loading a neural model.  They are
fast enough to run for every paragraph and decide whether the optional local
post-editor is worth invoking.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
NUMBER_RE = re.compile(
    r"(?<![\w])(?:\d{1,4}(?:[.,:/-]\d+)*|\d+)(?:\s?(?:%|°C|km|m|cm|mm|kg|g|GB|MB|TB|USD|EUR|CNY|美元|元|年|月|日))?",
    re.IGNORECASE,
)
NUMBER_VALUE_RE = re.compile(r"\d+(?:[.,]\d+)?")
ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9.-]{1,}(?![A-Za-z0-9])")
PROPER_NAME_RE = re.compile(
    r"(?<!\w)(?:[A-Z][a-z]{2,})(?:\s+(?:[A-Z][a-z]{2,})){1,3}(?!\w)"
)
CLAUSE_RE = re.compile(r"[,，;；:：]|\b(?:which|that|whose|while|although|because|whereas|from which)\b", re.IGNORECASE)
ZH_SEMANTIC_RISK_RE = re.compile(
    r"(?:理性|本能|混沌|谱系|血脉|荒原|帷幕|石板|隐喻|偏差|量化|显存|修辞|演化)"
)
PROMPT_LEAK_RE = re.compile(
    r"(?:<translation>|</translation>|原文[:：]|初译[:：]|system prompt|assistant:)",
    re.IGNORECASE,
)
IMMUTABLE_RE = re.compile(
    r"https?://[^\s)\]]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\{\{[^{}]+\}\}|\$\{[^{}]+\}|"
    r"(?<![A-Za-z0-9])\d{1,4}(?:[-/:]\d{1,4}){1,2}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?\s?(?:%|°C|km|cm|mm|kg|GB|MB|TB|bit|USD|EUR|CNY)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QualityIssue:
    code: str
    weight: int
    message: str
    critical: bool = False


@dataclass(frozen=True)
class QualityAssessment:
    score: int
    issues: tuple[QualityIssue, ...]
    trigger_threshold: int

    @property
    def should_review(self) -> bool:
        return self.score >= self.trigger_threshold or self.has_critical

    @property
    def has_critical(self) -> bool:
        return any(issue.critical for issue in self.issues)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


@dataclass(frozen=True)
class DocumentContext:
    entities: tuple[str, ...] = ()
    acronyms: tuple[str, ...] = ()
    glossary: tuple[tuple[str, str], ...] = ()
    fingerprint: str = ""

    def prompt_block(self) -> str:
        values: list[str] = []
        if self.entities:
            values.append("专名：" + "；".join(self.entities[:30]))
        if self.acronyms:
            values.append("缩写：" + "；".join(self.acronyms[:30]))
        if self.glossary:
            values.append(
                "术语：" + "；".join(f"{source} → {target}" for source, target in self.glossary[:40])
            )
        return "\n".join(values) or "无"


@dataclass
class ReviewStats:
    examined: int = 0
    triggered: int = 0
    reviewed: int = 0
    fallback: int = 0
    cache_hits: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    fallback_reasons: dict[str, int] = field(default_factory=dict)

    def record_assessment(self, assessment: QualityAssessment) -> None:
        self.examined += 1
        if assessment.should_review:
            self.triggered += 1
        for code in assessment.reason_codes:
            self.reasons[code] = self.reasons.get(code, 0) + 1

    def summary(self) -> str:
        if not self.examined:
            return "未检测到可翻译段落"
        return (
            f"质量检测 {self.examined} 段，触发审校 {self.triggered} 段，"
            f"成功 {self.reviewed} 段，回退 {self.fallback} 段"
        )

    def record_fallback(self, reason: str) -> None:
        self.fallback += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def extract_numbers(text: str) -> tuple[str, ...]:
    return _unique(NUMBER_VALUE_RE.findall(text))


def extract_acronyms(text: str) -> tuple[str, ...]:
    return _unique(value.rstrip(".") for value in ACRONYM_RE.findall(text))


def extract_entities(text: str) -> tuple[str, ...]:
    return _unique(PROPER_NAME_RE.findall(text))


def detect_translation_direction(text: str) -> str:
    cjk_count = len(CJK_RE.findall(text))
    latin_count = len(LATIN_RE.findall(text))
    if cjk_count == 0 and latin_count == 0:
        return "zh-en"
    return "zh-en" if cjk_count * 2 >= latin_count else "en-zh"


def protect_immutable_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Replace URLs, variables, dates and number-unit facts with stable tokens."""

    replacements: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"ZXQPH{len(replacements):03d}QXZ"
        replacements[token] = match.group(0)
        return token

    return IMMUTABLE_RE.sub(replace, text), replacements


def restore_immutable_tokens(text: str, replacements: Mapping[str, str]) -> str:
    restored = text
    for token, value in replacements.items():
        flexible = r"\s*".join(re.escape(part) for part in re.findall(r"[A-Za-z]+|\d+", token))
        restored, count = re.subn(flexible, lambda _match: value, restored, flags=re.IGNORECASE)
        if count == 0 and token in restored:
            restored = restored.replace(token, value)
    return restored


def build_document_context(
    texts: Sequence[str],
    glossary: Mapping[str, Mapping[str, object]] | None,
    direction: str,
) -> DocumentContext:
    joined = "\n".join(texts)
    acronyms = extract_acronyms(joined)
    entities = extract_entities(joined) if direction == "en-zh" else ()
    glossary_pairs: list[tuple[str, str]] = []
    for source, rule in (glossary or {}).items():
        if source.casefold() not in joined.casefold():
            continue
        target = str(rule.get("target", "")).strip()
        if target:
            glossary_pairs.append((str(source), target))
    payload = {
        "entities": entities,
        "acronyms": acronyms,
        "glossary": glossary_pairs,
        "version": 2,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return DocumentContext(
        entities=entities,
        acronyms=acronyms,
        glossary=tuple(glossary_pairs),
        fingerprint=fingerprint,
    )


def _repeated_ngram(text: str, size: int = 3) -> bool:
    del size  # retained for compatibility with earlier callers
    if re.search(r"\b([A-Za-z]{2,})\b(?:[\s,;:-]+\1\b){2,}", text, re.IGNORECASE):
        return True
    return bool(
        re.search(
            r"(?P<phrase>[\u3400-\u4dbf\u4e00-\u9fff]{2,8})"
            r"(?:的?\s*(?P=phrase)){2,}",
            text,
        )
    )


def _target_language_leakage(translation: str, direction: str) -> bool:
    cjk = len(CJK_RE.findall(translation))
    latin = len(LATIN_RE.findall(translation))
    if direction == "en-zh":
        return len(translation) >= 30 and cjk < max(3, latin // 8)
    return len(translation) >= 20 and latin < max(3, cjk // 3)


def assess_translation(
    source: str,
    translation: str,
    direction: str,
    ambiguous_terms: Sequence[str] = (),
    trigger_threshold: int = 35,
    forbidden_terms: Sequence[str] = (),
) -> QualityAssessment:
    issues: list[QualityIssue] = []
    source_folded = source.casefold()

    if not translation.strip():
        issues.append(QualityIssue("empty", 100, "译文为空", True))
    long_limit = 100 if direction == "zh-en" else 180
    if len(source) >= long_limit:
        issues.append(QualityIssue("long_paragraph", 35, "长段落需要语义复核"))
    clause_limit = 2 if direction == "zh-en" else 4
    if len(CLAUSE_RE.findall(source)) >= clause_limit:
        issues.append(QualityIssue("complex_syntax", 35, "包含多个从句或长定语"))
    if any(term.casefold() in source_folded for term in ambiguous_terms if term):
        issues.append(QualityIssue("ambiguous_term", 35, "命中多义词或高风险表达"))
    if len(extract_entities(source)) >= 2:
        issues.append(QualityIssue("entity_consistency", 35, "包含多个专名，需要文档一致性复核"))
    if direction == "zh-en" and ZH_SEMANTIC_RISK_RE.search(source):
        issues.append(QualityIssue("semantic_domain", 35, "抽象、文学或技术语义需要复核"))
    acronyms = extract_acronyms(source)
    if len(acronyms) >= 2:
        issues.append(QualityIssue("technical_density", 35, "包含多个技术缩写或型号"))

    missing_numbers = [value for value in extract_numbers(source) if value not in translation]
    if missing_numbers:
        issues.append(
            QualityIssue(
                "missing_number",
                55,
                "数字或单位缺失：" + ", ".join(missing_numbers),
                True,
            )
        )
    missing_acronyms = [value for value in extract_acronyms(source) if value not in translation]
    if missing_acronyms:
        issues.append(
            QualityIssue(
                "missing_acronym",
                35,
                "缩写缺失：" + ", ".join(missing_acronyms),
                True,
            )
        )
    if _target_language_leakage(translation, direction):
        issues.append(QualityIssue("language_leakage", 45, "译文目标语言比例异常", True))
    if _repeated_ngram(translation):
        issues.append(QualityIssue("repetition", 45, "译文出现异常重复", True))
    if PROMPT_LEAK_RE.search(translation):
        issues.append(QualityIssue("prompt_leakage", 70, "译文包含提示词残留", True))
    forbidden_hits = [
        term for term in forbidden_terms if term and term.casefold() in translation.casefold()
    ]
    if forbidden_hits:
        issues.append(
            QualityIssue(
                "forbidden_literalism",
                60,
                "命中被禁止的机械直译：" + ", ".join(forbidden_hits),
                True,
            )
        )

    source_visible = len(re.sub(r"\s+", "", source))
    target_visible = len(re.sub(r"\s+", "", translation))
    ratio = target_visible / max(source_visible, 1)
    low, high = ((0.25, 2.2) if direction == "en-zh" else (0.35, 3.2))
    if translation and not low <= ratio <= high:
        issues.append(QualityIssue("length_ratio", 30, f"长度比例异常：{ratio:.2f}"))

    return QualityAssessment(
        score=min(100, sum(issue.weight for issue in issues)),
        issues=tuple(issues),
        trigger_threshold=max(1, trigger_threshold),
    )


def validate_review_candidate(
    source: str,
    draft: str,
    candidate: str,
    direction: str,
    required_terms: Sequence[str] = (),
    forbidden_terms: Sequence[str] = (),
) -> tuple[bool, str]:
    if not candidate.strip():
        return False, "审校器返回空文本"
    if PROMPT_LEAK_RE.search(candidate):
        return False, "审校器泄漏了提示词"
    missing_numbers = [value for value in extract_numbers(source) if value not in candidate]
    if missing_numbers:
        return False, "审校结果遗漏数字或单位"
    missing_acronyms = [value for value in extract_acronyms(source) if value not in candidate]
    if missing_acronyms:
        return False, "审校结果遗漏缩写"
    if any(term and term.casefold() not in candidate.casefold() for term in required_terms):
        return False, "审校结果违反文档术语或语境译法"
    if any(term and term.casefold() in candidate.casefold() for term in forbidden_terms):
        return False, "审校结果仍包含被禁止的机械直译"
    if _target_language_leakage(candidate, direction):
        return False, "审校结果目标语言异常"
    if _repeated_ngram(candidate):
        return False, "审校结果存在异常重复"
    if len(candidate) < max(2, int(len(draft) * 0.35)):
        return False, "审校结果疑似大量删减"
    if len(candidate) > max(80, int(len(draft) * 2.5)):
        return False, "审校结果疑似扩写"
    return True, "ok"
