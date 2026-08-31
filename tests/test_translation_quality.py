from translation_quality import (
    assess_translation,
    build_document_context,
    protect_immutable_tokens,
    restore_immutable_tokens,
    validate_review_candidate,
    extract_acronyms,
)


def test_immutable_tokens_round_trip() -> None:
    source = "Version 3.2 shipped on 2026-08-31; see https://example.com/a and {{NAME}}."
    protected, replacements = protect_immutable_tokens(source)
    assert source != protected
    assert restore_immutable_tokens(protected, replacements) == source


def test_missing_number_is_critical_and_triggers_review() -> None:
    result = assess_translation(
        "The API processed 12,480 records.",
        "该接口处理了记录。",
        "en-zh",
    )
    assert result.should_review
    assert result.has_critical
    assert "missing_number" in result.reason_codes


def test_document_context_is_stable_and_contains_terms() -> None:
    texts = ["Alice Chen joined ACME Corp.", "Alice Chen deployed the API."]
    glossary = {"API": {"target": "应用程序接口"}}
    first = build_document_context(texts, glossary, "en-zh")
    second = build_document_context(texts, glossary, "en-zh")
    assert first.fingerprint == second.fingerprint
    assert "Alice Chen" in first.entities
    assert "API" in first.prompt_block()


def test_reviewer_candidate_cannot_drop_facts() -> None:
    valid, reason = validate_review_candidate(
        "ACME delivered 42 files.",
        "ACME 交付了 42 个文件。",
        "交付了文件。",
        "en-zh",
    )
    assert not valid
    assert "遗漏" in reason


def test_chinese_adjacent_acronym_is_detected_without_false_repetition() -> None:
    assert extract_acronyms("系统随后命名为ARI。") == ("ARI",)
    assessment = assess_translation(
        "The archive held the records that the curator had reviewed.",
        "档案馆保存着馆员审查过的记录。",
        "en-zh",
    )
    assert "repetition" not in assessment.reason_codes
