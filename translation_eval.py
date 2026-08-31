"""Reusable machine-translation evaluation and promotion gates."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from translation_quality import assess_translation, extract_numbers


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    split: str
    direction: str
    category: str
    source: str
    reference: str
    required_terms: tuple[str, ...] = ()
    terminology: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    preserve: tuple[str, ...] = ()


def load_eval_cases(
    path: Path,
    *,
    direction: str | None = None,
    split: str | None = None,
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            case = EvalCase(
                case_id=str(raw["id"]),
                split=str(raw.get("split", "test")),
                direction=str(raw["direction"]),
                category=str(raw.get("category", "general")),
                source=str(raw["source"]),
                reference=str(raw["reference"]),
                required_terms=tuple(map(str, raw.get("required_terms", []))),
                terminology=tuple(map(str, raw.get("terminology", []))),
                forbidden_terms=tuple(map(str, raw.get("forbidden_terms", []))),
                preserve=tuple(map(str, raw.get("preserve", []))),
            )
            if direction and case.direction != direction:
                continue
            if split and case.split != split:
                continue
            if not case.source or not case.reference:
                raise ValueError(f"评测数据第 {line_number} 行缺少原文或参考译文")
            cases.append(case)
    if not cases:
        raise ValueError("没有匹配的评测样例")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("评测样例 id 重复")
    return cases


def _character_fscore(hypothesis: str, reference: str, order: int = 6) -> float:
    def ngrams(value: str, size: int) -> dict[str, int]:
        compact = re.sub(r"\s+", " ", value.casefold()).strip()
        result: dict[str, int] = {}
        for index in range(max(0, len(compact) - size + 1)):
            gram = compact[index : index + size]
            result[gram] = result.get(gram, 0) + 1
        return result

    scores: list[float] = []
    for size in range(1, order + 1):
        hyp = ngrams(hypothesis, size)
        ref = ngrams(reference, size)
        overlap = sum(min(count, ref.get(gram, 0)) for gram, count in hyp.items())
        precision = overlap / max(sum(hyp.values()), 1)
        recall = overlap / max(sum(ref.values()), 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return 100.0 * sum(scores) / max(len(scores), 1)


def segment_chrf(hypothesis: str, reference: str) -> float:
    try:
        from sacrebleu.metrics import CHRF

        return float(CHRF(word_order=2).sentence_score(hypothesis, [reference]).score)
    except ImportError:
        return _character_fscore(hypothesis, reference)


def bootstrap_difference(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 2026,
) -> dict[str, float]:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("bootstrap 输入长度必须一致且非空")
    rng = random.Random(seed)
    differences: list[float] = []
    size = len(candidate)
    for _ in range(samples):
        indices = [rng.randrange(size) for _ in range(size)]
        differences.append(
            sum(candidate[index] - baseline[index] for index in indices) / size
        )
    differences.sort()
    return {
        "mean": sum(c - b for c, b in zip(candidate, baseline)) / size,
        "ci95_low": differences[math.floor(samples * 0.025)],
        "ci95_high": differences[min(samples - 1, math.floor(samples * 0.975))],
    }


def evaluate_predictions(
    cases: Sequence[EvalCase],
    predictions: Sequence[str],
) -> dict[str, object]:
    if len(cases) != len(predictions):
        raise ValueError("评测样例和预测数量不一致")
    rows: list[dict[str, object]] = []
    total_required = 0
    matched_required = 0
    total_terminology = 0
    matched_terminology = 0
    total_preserve = 0
    matched_preserve = 0
    critical_errors = 0
    number_total = 0
    number_matched = 0
    simple_scores: list[float] = []
    all_scores: list[float] = []

    for case, prediction in zip(cases, predictions):
        chrf = segment_chrf(prediction, case.reference)
        all_scores.append(chrf)
        if case.category == "general":
            simple_scores.append(chrf)
        required_hits = [term for term in case.required_terms if term.casefold() in prediction.casefold()]
        terminology_hits = [term for term in case.terminology if term.casefold() in prediction.casefold()]
        forbidden_hits = [term for term in case.forbidden_terms if term.casefold() in prediction.casefold()]
        preserve_hits = [term for term in case.preserve if term in prediction]
        numbers = extract_numbers(case.source)
        number_hits = [number for number in numbers if number in prediction]
        assessment = assess_translation(case.source, prediction, case.direction)
        row_critical = bool(forbidden_hits) or assessment.has_critical
        critical_errors += int(row_critical)
        total_required += len(case.required_terms)
        matched_required += len(required_hits)
        total_terminology += len(case.terminology)
        matched_terminology += len(terminology_hits)
        total_preserve += len(case.preserve)
        matched_preserve += len(preserve_hits)
        number_total += len(numbers)
        number_matched += len(number_hits)
        rows.append(
            {
                "id": case.case_id,
                "split": case.split,
                "direction": case.direction,
                "category": case.category,
                "source": case.source,
                "reference": case.reference,
                "prediction": prediction,
                "chrf": chrf,
                "missing_required_terms": sorted(set(case.required_terms) - set(required_hits)),
                "missing_terminology": sorted(set(case.terminology) - set(terminology_hits)),
                "forbidden_hits": forbidden_hits,
                "missing_preserve": sorted(set(case.preserve) - set(preserve_hits)),
                "missing_numbers": sorted(set(numbers) - set(number_hits)),
                "quality_issues": list(assessment.reason_codes),
                "critical": row_critical,
            }
        )

    def ratio(matched: int, total: int) -> float:
        return 1.0 if total == 0 else matched / total

    return {
        "case_count": len(cases),
        "chrf": sum(all_scores) / len(all_scores),
        "simple_chrf": sum(simple_scores) / len(simple_scores) if simple_scores else None,
        "required_accuracy": ratio(matched_required, total_required),
        "term_accuracy": ratio(matched_terminology, total_terminology),
        "preserve_accuracy": ratio(matched_preserve, total_preserve),
        "number_accuracy": ratio(number_matched, number_total),
        "critical_errors": critical_errors,
        "segment_chrf": all_scores,
        "rows": rows,
    }


def compare_reports(
    candidate: dict[str, object],
    baseline: dict[str, object],
    *,
    require_absolute_targets: bool = True,
) -> dict[str, object]:
    bootstrap = bootstrap_difference(
        list(map(float, candidate["segment_chrf"])),
        list(map(float, baseline["segment_chrf"])),
    )
    checks = {
        "critical_errors_not_worse": int(candidate["critical_errors"]) <= int(baseline["critical_errors"]),
        "chrf_improved": float(candidate["chrf"]) > float(baseline["chrf"]),
        "chrf_statistically_significant": bootstrap["ci95_low"] > 0.0,
        "simple_not_regressed": (
            candidate.get("simple_chrf") is None
            or baseline.get("simple_chrf") is None
            or float(candidate["simple_chrf"]) >= float(baseline["simple_chrf"]) - 0.5
        ),
        "terms_not_regressed": float(candidate["term_accuracy"]) >= float(baseline["term_accuracy"]),
        "content_anchors_not_regressed": float(candidate["required_accuracy"]) >= float(baseline["required_accuracy"]),
        "numbers_complete": float(candidate["number_accuracy"]) >= 1.0,
        "preserved_tokens_complete": float(candidate["preserve_accuracy"]) >= 1.0,
    }
    if require_absolute_targets:
        checks["critical_errors_zero"] = int(candidate["critical_errors"]) == 0
        checks["terms_at_least_98_percent"] = float(candidate["term_accuracy"]) >= 0.98
    comet_difference = None
    if isinstance(candidate.get("comet"), dict) and isinstance(
        baseline.get("comet"), dict
    ):
        candidate_scores = list(map(float, candidate["comet"].get("scores", [])))
        baseline_scores = list(map(float, baseline["comet"].get("scores", [])))
        if candidate_scores and len(candidate_scores) == len(baseline_scores):
            comet_difference = bootstrap_difference(candidate_scores, baseline_scores)
            checks["comet_improved"] = comet_difference["mean"] > 0.0
            checks["comet_statistically_significant"] = comet_difference["ci95_low"] > 0.0
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "chrf_difference": bootstrap,
        "comet_difference": comet_difference,
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def predictions_from_rows(rows: Iterable[dict[str, object]]) -> list[str]:
    return [str(row["prediction"]) for row in rows]


def generate_seq2seq_predictions(
    model,
    tokenizer,
    cases: Sequence[EvalCase],
    device,
    *,
    target_prefix: str = "",
    max_source_tokens: int = 384,
    batch_size: int = 8,
) -> list[str]:
    """Generate deterministic predictions for a directional seq2seq model."""

    import torch

    model.eval()
    predictions: list[str] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        encoded = tokenizer(
            [target_prefix + case.source for case in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_tokens,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                num_beams=5,
                max_new_tokens=512,
                no_repeat_ngram_size=3,
                repetition_penalty=1.08,
                early_stopping=True,
            )
        predictions.extend(
            value.strip()
            for value in tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
    return predictions
