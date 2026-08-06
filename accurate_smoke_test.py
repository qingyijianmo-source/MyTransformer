"""Integration check for both accurate translation directions."""

import gc

import torch

from accurate_translator import AccurateTranslator, detect_translation_direction


def release(translator: AccurateTranslator) -> None:
    translator.model.to("cpu")
    del translator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_zh_to_en() -> None:
    translator = AccurateTranslator(direction="zh-en")
    sources = [
        "我们需要共同努力解决这个问题。",
        "人工智能正在改变我们的生活。",
        "如果明天天气很好，我们就去公园散步。",
    ]
    outputs = translator.translate_many_texts(sources)
    checks = [
        (("work together",), ("address this issue", "solve this problem")),
        (("artificial intelligence", "our lives"),),
        (("weather", "park"),),
    ]
    for source, output, alternatives in zip(sources, outputs, checks):
        lowered = output.lower()
        if not any(all(term in lowered for term in required) for required in alternatives):
            raise AssertionError(f"unexpected zh-en translation: {source!r} -> {output!r}")
        print(f"ZH: {source}\nEN: {output}")
    model_kind = "fine-tuned" if translator.using_fine_tuned_model else "base"
    print(f"ZH-EN PASSED | device={translator.device} | model={model_kind}")
    release(translator)


def check_en_to_zh() -> None:
    translator = AccurateTranslator(direction="en-zh")
    sources = [
        "We need to work together to solve this problem.",
        "Artificial intelligence is changing our lives.",
        "If the weather is good tomorrow, we will go for a walk in the park.",
        "Machine translation helps people read long documents.",
        "Data cleansing and modelling training are complete.",
    ]
    outputs = translator.translate_many_texts(sources)
    checks = [
        (("共同努力", "问题"),),
        (("人工智能", "生活"),),
        (("天气", "公园"),),
        (("机器翻译", "文档"), ("机器翻译", "文件")),
        (("数据清洗", "模型训练"),),
    ]
    for source, output, alternatives in zip(sources, outputs, checks):
        if not any(all(term in output for term in required) for required in alternatives):
            raise AssertionError(f"unexpected en-zh translation: {source!r} -> {output!r}")
        print(f"EN: {source}\nZH: {output}")
    check_en_to_zh_long_context(translator)
    model_kind = "fine-tuned" if translator.using_fine_tuned_model else "base"
    print(f"EN-ZH PASSED | device={translator.device} | model={model_kind}")
    release(translator)


def check_en_to_zh_long_context(translator: AccurateTranslator) -> None:
    sources = [
        (
            "To insist that human consciousness operates purely through logical "
            "structures is to misunderstand the visceral, organic chaos from which "
            "reason originally emerged. Our biological lineage has left us "
            "flesh-bound creatures whose intuitions often precede deliberate thought."
        ),
        (
            "The castle did not merely stand upon the desolate moorland; it loomed "
            "like a heavy, suffocating shroud, swallowing the broken battlements in "
            "shadow. Generations had crossed its cold flagstones, believing that "
            "their lineage was secured by a structure of stone and mortar."
        ),
        (
            "The Neon District never slept. Rain hissed through its crowded, "
            "neon-drenched alleys while multinational megacorporations treated human "
            "beings as interchangeable hardware. People abandoned their physical "
            "bodies for borrowed identities, and anyone asking the wrong question "
            "might buy himself a quiet bullet before dawn."
        ),
    ]
    outputs = translator.translate_many_texts(sources)
    required = [
        ("本能", "混沌", "理性最初", "演化谱系", "血肉之躯"),
        ("帷幕", "石板", "血脉", "灰浆"),
        ("霓虹区", "霓虹浸染", "肉身", "子弹"),
    ]
    forbidden = [
        ("生物系", "肉质生物"),
        ("阴唇", "旗石", "迫击炮"),
        ("尼恩区", "尼昂区", "被烧焦的", "物理体", "买一颗", "光线照亮了它", "肉身和血肉"),
    ]
    for source, output, expected, rejected in zip(
        sources, outputs, required, forbidden
    ):
        if len(translator.split_for_translation(source)) != 1:
            raise AssertionError("short paragraph was split and lost its context")
        if not all(term in output for term in expected):
            raise AssertionError(f"missing contextual terms: {output!r}")
        if any(term in output for term in rejected):
            raise AssertionError(f"dangerous contextual mistranslation: {output!r}")
        print(f"LONG EN: {source}\nLONG ZH: {output}")

    architectural, _ = translator._apply_context_rules(
        "The old wall was built from stone and mortar."
    )
    military, _ = translator._apply_context_rules(
        "The soldiers fired a mortar shell during the attack."
    )
    if "building cement" not in architectural:
        raise AssertionError("architectural mortar was not disambiguated")
    if "mortar" not in military or "building cement" in military:
        raise AssertionError("military mortar was incorrectly rewritten")

    architecture_output, military_output = translator.translate_many_texts(
        [
            "The old wall was built from stone and mortar.",
            "The soldiers fired a mortar shell during the attack.",
        ]
    )
    if not any(term in architecture_output for term in ("灰浆", "灰泥", "水泥")):
        raise AssertionError(f"architectural mortar mistranslated: {architecture_output!r}")
    if "迫击炮" not in military_output:
        raise AssertionError(f"military mortar mistranslated: {military_output!r}")

    oversized = " ".join(
        f"Sentence {index} contains enough words to exercise contextual grouping."
        for index in range(1, 121)
    )
    chunks = translator.split_for_translation(oversized)
    if len(chunks) <= 1 or len(chunks) >= 120:
        raise AssertionError("oversized paragraph was not grouped into contextual chunks")
    if any(
        translator._token_count(chunk) > translator.settings.max_source_tokens
        for chunk in chunks
    ):
        raise AssertionError("contextual chunk exceeded the source token limit")


def check_auto_detection() -> None:
    cases = {
        "这是中文句子，包含 AI 产品名。": "zh-en",
        "This is an English sentence about 人工智能.": "en-zh",
    }
    for text, expected in cases.items():
        actual = detect_translation_direction(text)
        if actual != expected:
            raise AssertionError(f"direction detection failed: {text!r} -> {actual}")
    print("AUTO DIRECTION DETECTION PASSED")


def main() -> None:
    check_auto_detection()
    check_zh_to_en()
    check_en_to_zh()
    print("BIDIRECTIONAL ACCURATE TRANSLATION TEST PASSED")


if __name__ == "__main__":
    main()
