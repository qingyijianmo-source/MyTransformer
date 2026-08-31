from local_reviewer import LocalReviewer, ReviewerSettings, ReviewRequest


class FakeBackend:
    def __init__(self, output: str):
        self.output = output
        self.released = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        assert "<translation>" in system_prompt
        assert "当前初译" in user_prompt
        return self.output

    def release(self) -> None:
        self.released = True


def request() -> ReviewRequest:
    return ReviewRequest(
        direction="en-zh",
        source="The shroud covered 42 stones.",
        draft="裹尸布覆盖了 42 块石头。",
        previous_source="",
        next_source="",
        document_context="术语：shroud -> 裹尸布",
        reasons=("ambiguous_term",),
    )


def test_reviewer_extracts_wrapped_translation() -> None:
    backend = FakeBackend("<translation>一层裹尸布覆盖了 42 块石头。</translation>")
    reviewer = LocalReviewer(
        ReviewerSettings(enabled=True), backend_factory=lambda _settings: backend
    )
    result = reviewer.review(request())
    assert result.success
    assert result.translation == "一层裹尸布覆盖了 42 块石头。"


def test_reviewer_failure_falls_back_to_draft() -> None:
    class BrokenBackend(FakeBackend):
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("simulated OOM")

    reviewer = LocalReviewer(
        ReviewerSettings(enabled=True),
        backend_factory=lambda _settings: BrokenBackend(""),
    )
    result = reviewer.review(request())
    assert not result.success
    assert result.translation == request().draft
    assert "simulated OOM" in result.message
