"""Prepare a clean, compact IWSLT 2017 zh-en subset for fast training."""

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download
import pandas as pd

from prepare_corpus import normalize_text, pair_key, validate_pair


PROJECT_DIR = Path(__file__).resolve().parent
REVISION = "1063e8e4b8da33ceb9d2538293fcf587cc8c2fa7"
FILENAME = "iwslt2017-zh-en/iwslt2017-train.parquet"


def prepare(output: Path, metadata: Path, target_size: int, seed: int):
    cached_path = hf_hub_download(
        repo_id="IWSLT/iwslt2017",
        repo_type="dataset",
        revision=REVISION,
        filename=FILENAME,
    )
    frame = pd.read_parquet(cached_path)
    indices = list(range(len(frame)))
    random.Random(seed).shuffle(indices)

    rows = []
    seen = set()
    rejections = Counter()
    scanned = 0
    for index in indices:
        scanned += 1
        translation = frame.iloc[index]["translation"]
        zh = normalize_text(translation.get("zh"))
        en = normalize_text(translation.get("en"))
        reason = validate_pair(zh, en)
        if reason is not None:
            rejections[reason] += 1
            continue
        # Keep the fast corpus focused on sentence-sized examples.
        if len(zh) > 240 or len(en) > 360:
            rejections["fast_length"] += 1
            continue
        key = pair_key(zh, en)
        if key in seen:
            rejections["duplicate"] += 1
            continue
        seen.add(key)
        rows.append({"translation": {"zh_CN": zh, "en_GB": en}})
        if len(rows) % 10_000 == 0:
            print(f"Accepted {len(rows):,}/{target_size:,} IWSLT pairs.")
        if len(rows) >= target_size:
            break

    if len(rows) < target_size:
        raise RuntimeError(f"Only {len(rows):,} acceptable IWSLT pairs were found")

    output.parent.mkdir(exist_ok=True, parents=True)
    Dataset.from_list(rows).to_parquet(str(output))
    details = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "IWSLT/iwslt2017",
        "config": "iwslt2017-zh-en",
        "revision": REVISION,
        "license": "CC BY-NC-ND 4.0",
        "dataset_card": "https://huggingface.co/datasets/IWSLT/iwslt2017",
        "accepted_pairs": len(rows),
        "scanned_pairs": scanned,
        "rejections": dict(sorted(rejections.items())),
        "seed": seed,
    }
    metadata.parent.mkdir(exist_ok=True, parents=True)
    metadata.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(rows):,} clean pairs to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data" / "iwslt2017_zh_en_fast.parquet",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_DIR / "data" / "iwslt2017_zh_en_fast.metadata.json",
    )
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.metadata.resolve(), args.size, args.seed)


if __name__ == "__main__":
    main()
