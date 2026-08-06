"""Run fast supervised training followed by SCST reward fine-tuning."""

import argparse
import json
from pathlib import Path

from prepare_iwslt_fast import prepare
from rl_finetune import finetune
from train_test import PROJECT_DIR, resolve_config_path, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_DIR / "config.fast_rl.json"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--retrain-tokenizers", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    data_path = resolve_config_path(config["data"]["dataset_path"], config_path.parent)
    metadata_path = data_path.with_suffix(".metadata.json")
    if args.rebuild_data or not data_path.is_file():
        prepare(data_path, metadata_path, 100_000, int(config["data"]["seed"]))

    tokenizer_dir = resolve_config_path(
        config["tokenizer"]["tokenizers_dir"], config_path.parent
    )
    tokenizer_missing = not all(
        (tokenizer_dir / config["run"][name]).is_file()
        for name in ("zh_tokenizer_file_name", "en_tokenizer_file_name")
    )
    status = train(
        config,
        config_path.parent,
        args.resume,
        args.retrain_tokenizers or tokenizer_missing,
    )
    if status == "paused":
        print("Pipeline paused before reward fine-tuning.")
        return
    if not args.skip_rl:
        finetune(config, config_path.parent)
    print("Fast translation pipeline completed.")


if __name__ == "__main__":
    main()
