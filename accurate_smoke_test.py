"""Integration check for the pretrained accurate translation path."""

from accurate_translator import AccurateTranslator


def main() -> None:
    translator = AccurateTranslator()
    sources = [
        "我们需要共同努力解决这个问题。",
        "人工智能正在改变我们的生活。",
        "如果明天天气很好，我们就去公园散步。",
    ]
    outputs = translator.translate_many_texts(sources)
    checks = [
        ("work together", "address this issue"),
        ("artificial intelligence", "our lives"),
        ("weather", "park"),
    ]
    for source, output, required in zip(sources, outputs, checks):
        lowered = output.lower()
        if not all(term in lowered for term in required):
            raise AssertionError(f"unexpected translation: {source!r} -> {output!r}")
        print(f"ZH: {source}\nEN: {output}")
    print(f"ACCURATE TRANSLATION TEST PASSED | device={translator.device}")


if __name__ == "__main__":
    main()
