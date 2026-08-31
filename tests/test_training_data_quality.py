import json

from training_data_quality import (
    TrainingBlacklist,
    alignment_quality_reason,
    classify_domain,
    domain_allowed,
)


def test_blacklist_and_sensitive_alignment(tmp_path) -> None:
    path = tmp_path / "blacklist.json"
    path.write_text(
        json.dumps({"exact_pairs": [{"source": "bad", "target": "坏"}]}),
        encoding="utf-8",
    )
    blacklist = TrainingBlacklist.load(path)
    assert blacklist.reason("bad", "坏") == "blacklisted_pair"
    assert (
        alignment_quality_reason("a heavy shroud", "沉重的阴唇", "en", blacklist)
        == "shroud_alignment"
    )


def test_domain_caps_military_examples() -> None:
    assert classify_domain("The mortar shell hit the trench.") == "military"
    assert domain_allowed("military", {"military": 2}, 100, {"military": 0.03})
    assert not domain_allowed("military", {"military": 3}, 100, {"military": 0.03})

