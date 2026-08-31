from translation_eval import EvalCase, compare_reports, evaluate_predictions


CASES = [
    EvalCase(
        case_id="one",
        split="test",
        direction="en-zh",
        category="general",
        source="The API returned 42 records.",
        reference="API 返回了 42 条记录。",
        required_terms=("返回",),
        preserve=("API",),
    )
]


def test_improved_prediction_passes_quality_gate() -> None:
    report = evaluate_predictions(CASES, [CASES[0].reference])
    baseline = evaluate_predictions(CASES, ["API 返回 42。"])
    comparison = compare_reports(report, baseline)
    assert report["number_accuracy"] == 1.0
    assert report["preserve_accuracy"] == 1.0
    assert comparison["accepted"]


def test_dropped_number_fails_quality_gate() -> None:
    baseline = evaluate_predictions(CASES, [CASES[0].reference])
    candidate = evaluate_predictions(CASES, ["API 返回了记录。"])
    comparison = compare_reports(candidate, baseline)
    assert not comparison["accepted"]
    assert not comparison["checks"]["numbers_complete"]
