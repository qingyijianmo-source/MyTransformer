from pathlib import Path

from translation_eval import load_eval_cases


ROOT = Path(__file__).resolve().parents[1]


def test_eval_splits_cover_both_directions_and_unique_ids() -> None:
    path = ROOT / "eval" / "translation_eval.jsonl"
    cases = load_eval_cases(path)
    assert {case.direction for case in cases} == {"zh-en", "en-zh"}
    assert {case.split for case in cases} >= {"dev", "test"}
    assert len(cases) == len({case.case_id for case in cases})

