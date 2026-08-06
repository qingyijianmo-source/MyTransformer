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
    model_kind = "fine-tuned" if translator.using_fine_tuned_model else "base"
    print(f"EN-ZH PASSED | device={translator.device} | model={model_kind}")
    release(translator)


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
