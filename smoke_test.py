"""Fast end-to-end check for data, tokenizers, model, loss, and decoding."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from build import build_transformer
from run import Controller
from train_test import (
    PROJECT_DIR,
    UsableDataset,
    deal_raw_dataset,
    load_and_get_tokenizer,
    require_token_id,
    resolve_config_path,
    train_and_save_tokenizer,
)


def main():
    config_path = PROJECT_DIR / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    data_config = config["data"]
    tokenizer_config = config["tokenizer"]
    dataset_path = resolve_config_path(data_config["dataset_path"], PROJECT_DIR)
    tokenizer_dir = resolve_config_path(
        tokenizer_config["tokenizers_dir"], PROJECT_DIR
    )
    train_dataset, _ = deal_raw_dataset(
        dataset_path,
        data_config["train_size"],
        data_config["test_size"],
        int(data_config["seed"]),
    )
    with TemporaryDirectory(prefix="mytransformer-smoke-") as temporary_dir:
        source_path = tokenizer_dir / config["run"]["zh_tokenizer_file_name"]
        target_path = tokenizer_dir / config["run"]["en_tokenizer_file_name"]
        if not source_path.is_file() or not target_path.is_file():
            temporary_path = Path(temporary_dir)
            tokenizer_dataset = train_dataset.select(
                range(min(512, len(train_dataset)))
            )
            source_path = train_and_save_tokenizer(
                tokenizer_dataset,
                data_config["source_language"],
                vocab_size=1000,
                min_frequency=1,
                tokenizer_path=temporary_path / "source_tokenizer.json",
            )
            target_path = train_and_save_tokenizer(
                tokenizer_dataset,
                data_config["target_language"],
                vocab_size=1000,
                min_frequency=1,
                tokenizer_path=temporary_path / "target_tokenizer.json",
            )

        source_tokenizer = load_and_get_tokenizer(source_path)
        target_tokenizer = load_and_get_tokenizer(target_path)
        run_check(config, train_dataset, source_tokenizer, target_tokenizer)


def run_check(config, train_dataset, source_tokenizer, target_tokenizer):
    data_config = config["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_count = min(4, len(train_dataset))
    train_dataset = train_dataset.select(range(sample_count))

    seq_len = 32
    dataset = UsableDataset(
        train_dataset,
        data_config["source_language"],
        data_config["target_language"],
        source_tokenizer,
        target_tokenizer,
        seq_len,
        seq_len,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    model = build_transformer(
        d_model=32,
        d_hidden=64,
        num_heads=4,
        drop_prob=0.0,
        num_encode_layers=1,
        num_decode_layers=1,
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
        source_seq_len=seq_len,
        target_seq_len=seq_len,
    ).to(device)
    encoder_input = batch["encoder_input"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    source_mask = batch["source_mask"].to(device)
    target_mask = batch["target_mask"].to(device)
    label = batch["label"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(
        ignore_index=require_token_id(target_tokenizer, "[PAD]")
    )

    model.train()
    encoder_output = model.encode(encoder_input, source_mask)
    decoder_output = model.decode(
        decoder_input,
        encoder_output,
        target_mask,
        source_mask,
    )
    logits = model.project(decoder_output)
    loss = criterion(logits.flatten(0, 1), label.flatten())
    if not torch.isfinite(loss):
        raise RuntimeError(f"smoke-test loss is not finite: {loss.item()}")
    loss.backward()
    optimizer.step()

    model.eval()
    controller = Controller(
        model,
        source_tokenizer,
        target_tokenizer,
        seq_len,
        target_seq_len=8,
        device=device,
        max_new_tokens=8,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )
    decoded = controller.talk("你好")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"SMOKE TEST PASSED | device={device} | loss={loss.item():.4f} | "
        f"parameters={parameter_count:,} | decoded={decoded!r}"
    )


if __name__ == "__main__":
    main()
