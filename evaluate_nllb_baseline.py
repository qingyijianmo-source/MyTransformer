"""Optional research-only NLLB-600M comparison; never used for production translation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from translation_eval import evaluate_predictions, load_eval_cases, write_report
from cache_config import configure_huggingface_cache


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANGUAGES = {"zh-en": ("zho_Hans", "eng_Latn"), "en-zh": ("eng_Latn", "zho_Hans")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "eval" / "reports" / "nllb.json"
    )
    args = parser.parse_args()
    configure_huggingface_cache()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    source_language, target_language = LANGUAGES[args.direction]
    cases = load_eval_cases(
        PROJECT_DIR / "eval" / "translation_eval.jsonl",
        direction=args.direction,
        split=args.split,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        src_lang=source_language,
        local_files_only=not args.allow_download,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        local_files_only=not args.allow_download,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    predictions: list[str] = []
    for start in range(0, len(cases), 4):
        batch = cases[start : start + 4]
        encoded = tokenizer(
            [case.source for case in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_language),
                num_beams=5,
                max_new_tokens=512,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    report = evaluate_predictions(cases, predictions)
    write_report(
        args.output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "research_only": True,
            "license_note": "CC BY-NC; not the default document translator",
            "direction": args.direction,
            "split": args.split,
            "metrics": report,
        },
    )
    print(f"NLLB 对照报告：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
