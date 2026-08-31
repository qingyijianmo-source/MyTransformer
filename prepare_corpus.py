"""Build a broader Chinese-English corpus while preserving the original data."""

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from cache_config import configure_huggingface_cache

configure_huggingface_cache()

from datasets import Dataset, load_dataset


PROJECT_DIR = Path(__file__).resolve().parent
WHITESPACE_RE = re.compile(r"\s+")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_text(value) -> str:
    if not isinstance(value, str):
        return ""
    text = html.unescape(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return WHITESPACE_RE.sub(" ", text).strip()


def validate_pair(zh: str, en: str) -> str | None:
    if not zh or not en:
        return "empty"
    if "\ufffd" in zh or "\ufffd" in en or "\x00" in zh or "\x00" in en:
        return "invalid_character"
    if not (2 <= len(zh) <= 500 and 2 <= len(en) <= 500):
        return "length"
    if not CJK_RE.search(zh) or not LATIN_RE.search(en):
        return "language"
    if zh.casefold() == en.casefold():
        return "identical"

    ratio = len(en) / len(zh)
    if not 0.3 <= ratio <= 8.0:
        return "length_ratio"

    # Filter obvious untranslated leakage while allowing names and product terms.
    zh_latin_count = len(LATIN_RE.findall(zh))
    en_cjk_count = len(CJK_RE.findall(en))
    if zh_latin_count > max(12, int(len(zh) * 0.35)):
        return "english_leakage"
    if en_cjk_count > max(6, int(len(en) * 0.20)):
        return "chinese_leakage"
    return None


def pair_key(zh: str, en: str) -> tuple[str, str]:
    return zh.casefold(), en.casefold()


def load_original(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"original corpus not found: {path}")
    frame = pd.read_parquet(path)
    if "translation" not in frame.columns:
        raise ValueError(f"missing 'translation' column in {path}")

    rows = []
    seen = set()
    skipped = 0
    for translation in frame["translation"]:
        if not isinstance(translation, dict):
            skipped += 1
            continue
        zh = normalize_text(translation.get("zh_CN"))
        en = normalize_text(translation.get("en_GB"))
        if not zh or not en:
            skipped += 1
            continue
        key = pair_key(zh, en)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        rows.append({"translation": {"zh_CN": zh, "en_GB": en}})
    return rows, seen, skipped


def collect_opus100(target_size: int, seed: int, shuffle_buffer: int, seen: set):
    stream = load_dataset(
        "Helsinki-NLP/opus-100",
        "en-zh",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)

    rows = []
    rejection_counts = Counter()
    scanned = 0
    for item in stream:
        scanned += 1
        translation = item.get("translation") or {}
        zh = normalize_text(translation.get("zh"))
        en = normalize_text(translation.get("en"))
        reason = validate_pair(zh, en)
        if reason is not None:
            rejection_counts[reason] += 1
            continue
        key = pair_key(zh, en)
        if key in seen:
            rejection_counts["duplicate"] += 1
            continue

        seen.add(key)
        rows.append({"translation": {"zh_CN": zh, "en_GB": en}})
        if len(rows) % 10_000 == 0:
            print(
                f"Accepted {len(rows):,}/{target_size:,} OPUS-100 pairs "
                f"after scanning {scanned:,}."
            )
        if len(rows) >= target_size:
            break

    if len(rows) < target_size:
        raise RuntimeError(
            f"OPUS-100 stream ended after only {len(rows):,} acceptable pairs"
        )
    return rows, scanned, rejection_counts


def build_corpus(
    original_path: Path,
    output_path: Path,
    metadata_path: Path,
    additional_size: int,
    seed: int,
    shuffle_buffer: int,
):
    original_rows, seen, original_skipped = load_original(original_path)
    print(
        f"Loaded {len(original_rows):,} unique original pairs "
        f"({original_skipped:,} empty/duplicate rows skipped)."
    )
    additional_rows, scanned, rejection_counts = collect_opus100(
        additional_size, seed, shuffle_buffer, seen
    )
    combined_rows = original_rows + additional_rows

    output_path.parent.mkdir(exist_ok=True, parents=True)
    Dataset.from_list(combined_rows).to_parquet(str(output_path))
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "name": "OPUS OpenOffice en-zh",
                "path": str(original_path),
                "accepted_pairs": len(original_rows),
                "skipped_pairs": original_skipped,
            },
            {
                "name": "Helsinki-NLP/opus-100",
                "config": "en-zh",
                "split": "train",
                "accepted_pairs": len(additional_rows),
                "scanned_pairs": scanned,
                "rejections": dict(sorted(rejection_counts.items())),
                "dataset_card": "https://huggingface.co/datasets/Helsinki-NLP/opus-100",
                "license_note": "Dataset card does not specify a unified corpus license.",
            },
        ],
        "total_pairs": len(combined_rows),
        "seed": seed,
    }
    metadata_path.parent.mkdir(exist_ok=True, parents=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(combined_rows):,} pairs to {output_path}")
    print(f"Saved provenance and filter statistics to {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge cleaned OPUS-100 zh-en data with the original corpus."
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=PROJECT_DIR / "data" / "opus_openoffice_en_zh.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data" / "combined_general_en_zh.parquet",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_DIR / "data" / "combined_general_en_zh.metadata.json",
    )
    parser.add_argument("--additional-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=20_000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.additional_size < 1:
        raise ValueError("additional-size must be positive")
    build_corpus(
        args.original.resolve(),
        args.output.resolve(),
        args.metadata.resolve(),
        args.additional_size,
        args.seed,
        args.shuffle_buffer,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
