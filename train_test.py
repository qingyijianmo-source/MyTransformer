import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.amp as amp
import torch.nn as nn
from cache_config import configure_huggingface_cache

configure_huggingface_cache()

from datasets import Dataset as HFDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset as TorchDataset

from build import Transformer, build_transformer


PROJECT_DIR = Path(__file__).resolve().parent


def require_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required special token {token!r}")
    return token_id


class UsableDataset(TorchDataset):
    def __init__(
        self,
        used_dataset: HFDataset,
        source_language: str,
        target_language: str,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        source_seq_len: int,
        target_seq_len: int,
    ):
        super().__init__()
        if source_seq_len < 2 or target_seq_len < 2:
            raise ValueError("source_seq_len and target_seq_len must both be at least 2")

        self.used_dataset = used_dataset
        self.source_language = source_language
        self.target_language = target_language
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.source_sos = require_token_id(source_tokenizer, "[SOS]")
        self.source_eos = require_token_id(source_tokenizer, "[EOS]")
        self.source_pad = require_token_id(source_tokenizer, "[PAD]")
        self.target_sos = require_token_id(target_tokenizer, "[SOS]")
        self.target_eos = require_token_id(target_tokenizer, "[EOS]")
        self.target_pad = require_token_id(target_tokenizer, "[PAD]")

        self.target_causal_mask = torch.triu(
            torch.ones(target_seq_len, target_seq_len, dtype=torch.bool), diagonal=1
        )

    def __len__(self):
        return len(self.used_dataset)

    def __getitem__(self, index: int):
        single = self.used_dataset[index]
        source_text = single["translation"][self.source_language]
        target_text = single["translation"][self.target_language]

        # Truncation keeps a single unexpectedly long record from terminating a run.
        source_ids = self.source_tokenizer.encode(source_text).ids[: self.source_seq_len - 2]
        target_ids = self.target_tokenizer.encode(target_text).ids[: self.target_seq_len - 1]

        num_source_pads = self.source_seq_len - len(source_ids) - 2
        num_target_pads = self.target_seq_len - len(target_ids) - 1

        encoder_input = torch.tensor(
            [self.source_sos]
            + source_ids
            + [self.source_eos]
            + [self.source_pad] * num_source_pads,
            dtype=torch.int64,
        )
        decoder_input = torch.tensor(
            [self.target_sos] + target_ids + [self.target_pad] * num_target_pads,
            dtype=torch.int64,
        )
        label = torch.tensor(
            target_ids + [self.target_eos] + [self.target_pad] * num_target_pads,
            dtype=torch.int64,
        )

        # True values denote key positions that attention must ignore.
        source_mask = (encoder_input == self.source_pad).unsqueeze(0).unsqueeze(0)
        target_pad_mask = (decoder_input == self.target_pad).unsqueeze(0).unsqueeze(0)
        target_mask = target_pad_mask | self.target_causal_mask

        return {
            "source_text": source_text,
            "target_text": target_text,
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "source_mask": source_mask,
            "target_mask": target_mask,
        }


class DynamicPaddingCollator:
    """Trim each batch to its longest sequence instead of the global maximum."""

    def __init__(self, source_pad: int, target_pad: int):
        self.source_pad = source_pad
        self.target_pad = target_pad

    def __call__(self, samples):
        max_source_len = max(
            int((sample["encoder_input"] != self.source_pad).sum())
            for sample in samples
        )
        max_target_len = max(
            int((sample["label"] != self.target_pad).sum()) for sample in samples
        )
        encoder_input = torch.stack(
            [sample["encoder_input"][:max_source_len] for sample in samples]
        )
        decoder_input = torch.stack(
            [sample["decoder_input"][:max_target_len] for sample in samples]
        )
        label = torch.stack([sample["label"][:max_target_len] for sample in samples])
        source_mask = (encoder_input == self.source_pad).unsqueeze(1).unsqueeze(2)
        target_pad_mask = (decoder_input == self.target_pad).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.triu(
            torch.ones(max_target_len, max_target_len, dtype=torch.bool), diagonal=1
        )
        target_mask = target_pad_mask | causal_mask
        return {
            "source_text": [sample["source_text"] for sample in samples],
            "target_text": [sample["target_text"] for sample in samples],
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "source_mask": source_mask,
            "target_mask": target_mask,
        }


class Checker:
    def __init__(
        self,
        model: Transformer,
        data_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        patience: int,
        value: float,
        device: torch.device,
        save_dir: Path,
        use_amp: bool = False,
    ):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.patience = patience
        self.value = value
        self.min_loss = float("inf")
        self.count = 0
        self.device = device
        self.newest_loss: Optional[float] = None
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.use_amp = use_amp

    def __call__(self) -> bool:
        self.model.eval()
        losses = []
        with torch.inference_mode():
            for batch in self.data_loader:
                encoder_input = batch["encoder_input"].to(self.device)
                decoder_input = batch["decoder_input"].to(self.device)
                label = batch["label"].to(self.device)
                source_mask = batch["source_mask"].to(self.device)
                target_mask = batch["target_mask"].to(self.device)

                with amp.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_amp,
                ):
                    encoder_output = self.model.encode(encoder_input, source_mask)
                    decoder_output = self.model.decode(
                        decoder_input, encoder_output, target_mask, source_mask
                    )
                    logits = self.model.project(decoder_output)
                    loss = self.criterion(logits.flatten(0, 1), label.flatten())
                losses.append(loss.item())

        if not losses:
            raise ValueError("validation dataset is empty")

        avg_loss = sum(losses) / len(losses)
        self.newest_loss = avg_loss
        meaningful_improvement = avg_loss < self.min_loss - self.value
        print(
            f"Validation loss: {avg_loss:.4f} "
            f"(best: {min(self.min_loss, avg_loss):.4f})"
        )

        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            torch.save(self.model.state_dict(), self.save_dir / "best_state_from_checker.pt")

        self.count = 0 if meaningful_improvement else self.count + 1
        self.model.train()
        return self.count >= self.patience


def get_every_sentence(dataset: HFDataset, language: str):
    for item in dataset:
        yield item["translation"][language]


def train_and_save_tokenizer(
    dataset: HFDataset,
    language: str,
    vocab_size: int,
    min_frequency: int,
    tokenizer_path: Path,
) -> Path:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
    )

    tokenizer.train_from_iterator(get_every_sentence(dataset, language), trainer=trainer)
    tokenizer_path.parent.mkdir(exist_ok=True, parents=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer_path


def load_and_get_tokenizer(tokenizer_path: Path) -> Tokenizer:
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def load_or_train_tokenizer(
    dataset: HFDataset,
    language: str,
    vocab_size: int,
    min_frequency: int,
    tokenizer_path: Path,
    retrain: bool,
) -> Tokenizer:
    if retrain or not tokenizer_path.is_file():
        print(f"Training tokenizer: {tokenizer_path.name}")
        train_and_save_tokenizer(
            dataset, language, vocab_size, min_frequency, tokenizer_path
        )
    else:
        print(f"Reusing tokenizer: {tokenizer_path.name}")
    return load_and_get_tokenizer(tokenizer_path)


def deal_raw_dataset(dataset_path: Path, train_size, test_size, seed: int):
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    dataset = load_dataset(path="parquet", data_files=str(dataset_path), split="train")
    if len(dataset) < 2:
        raise ValueError("dataset must contain at least two records")
    split_dataset = dataset.train_test_split(
        train_size=train_size, test_size=test_size, seed=seed
    )
    return split_dataset["train"], split_dataset["test"]


def get_dynamic_rate(step: int, d_model: int, warmup: int, factor: float) -> float:
    if d_model <= 0 or warmup <= 0 or factor <= 0:
        raise ValueError("d_model, warmup, and factor must be positive")
    step = max(step, 1)
    return factor * (d_model**-0.5 * min(step**-0.5, step * warmup**-1.5))


def resolve_config_path(path_value: str, config_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        return torch.load(path, map_location=device)


def load_state_dict(path: Path, device: torch.device):
    checkpoint = load_checkpoint(path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def save_training_checkpoint(
    path: Path,
    model: Transformer,
    optimizer,
    scheduler,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    epoch_generator_state,
    checker: Checker,
):
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "epoch_generator_state": epoch_generator_state,
            "checker_min_loss": checker.min_loss,
            "checker_count": checker.count,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def train(config: dict, config_dir: Path, resume: Optional[Path], retrain_tokenizers: bool):
    model_config = config["model"]
    tokenizer_config = config["tokenizer"]
    data_config = config["data"]
    train_config = config["train"]
    run_config = config["run"]

    seed = int(data_config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_path = resolve_config_path(data_config["dataset_path"], config_dir)
    tokenizers_dir = resolve_config_path(tokenizer_config["tokenizers_dir"], config_dir)
    checkpoints_dir = resolve_config_path(train_config["checkpoints_dir"], config_dir)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    train_dataset, test_dataset = deal_raw_dataset(
        dataset_path,
        data_config["train_size"],
        data_config["test_size"],
        seed,
    )
    max_train_samples = data_config.get("max_train_samples")
    max_test_samples = data_config.get("max_test_samples")
    if max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(int(max_train_samples), len(train_dataset)))
        )
    if max_test_samples is not None:
        test_dataset = test_dataset.select(
            range(min(int(max_test_samples), len(test_dataset)))
        )

    zh_tokenizer = load_or_train_tokenizer(
        train_dataset,
        data_config["source_language"],
        tokenizer_config["vocab_size"],
        tokenizer_config["min_frequency"],
        tokenizers_dir / run_config["zh_tokenizer_file_name"],
        retrain_tokenizers,
    )
    en_tokenizer = load_or_train_tokenizer(
        train_dataset,
        data_config["target_language"],
        tokenizer_config["vocab_size"],
        tokenizer_config["min_frequency"],
        tokenizers_dir / run_config["en_tokenizer_file_name"],
        retrain_tokenizers,
    )

    source_seq_len = int(model_config["source_seq_len"])
    target_seq_len = int(model_config["target_seq_len"])
    train_usable_dataset = UsableDataset(
        train_dataset,
        data_config["source_language"],
        data_config["target_language"],
        zh_tokenizer,
        en_tokenizer,
        source_seq_len,
        target_seq_len,
    )
    test_usable_dataset = UsableDataset(
        test_dataset,
        data_config["source_language"],
        data_config["target_language"],
        zh_tokenizer,
        en_tokenizer,
        source_seq_len,
        target_seq_len,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    batch_size = int(train_config["batch_size"])
    num_workers = int(train_config.get("num_workers", 0))
    source_pad_id = require_token_id(zh_tokenizer, "[PAD]")
    target_pad_id = require_token_id(en_tokenizer, "[PAD]")
    collate_fn = None
    if bool(train_config.get("dynamic_padding", True)):
        collate_fn = DynamicPaddingCollator(source_pad_id, target_pad_id)
    data_generator = torch.Generator().manual_seed(seed)
    train_data_loader = DataLoader(
        train_usable_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        generator=data_generator,
    )
    test_data_loader = DataLoader(
        test_usable_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    d_model = int(model_config["d_model"])
    target_vocab_size = en_tokenizer.get_vocab_size()
    model = build_transformer(
        d_model,
        int(model_config["d_hidden"]),
        int(model_config["num_heads"]),
        float(model_config["drop_prob"]),
        int(model_config["num_encode_layers"]),
        int(model_config["num_decode_layers"]),
        zh_tokenizer.get_vocab_size(),
        target_vocab_size,
        source_seq_len,
        target_seq_len,
        bool(model_config.get("tie_target_embedding", False)),
    ).to(device)

    resume_checkpoint = None
    resume_path = None
    if resume is not None:
        resume_path = resume if resume.is_absolute() else (config_dir / resume).resolve()
        resume_checkpoint = load_checkpoint(resume_path, device)
        if (
            isinstance(resume_checkpoint, dict)
            and "model_state_dict" in resume_checkpoint
        ):
            model.load_state_dict(resume_checkpoint["model_state_dict"])
        else:
            model.load_state_dict(resume_checkpoint)
        print(f"Resumed model weights from {resume_path}")

    warmup = int(train_config["warmup"])
    warmup_factor = float(train_config["warmup_factor"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_dynamic_rate(step, d_model, warmup, warmup_factor),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=target_pad_id,
        label_smoothing=float(train_config["label_smoothing"]),
    )

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    checker = Checker(
        model,
        test_data_loader,
        criterion,
        int(train_config["patience_times"]),
        float(train_config["min_progress_value"]),
        device,
        checkpoints_dir,
        use_amp,
    )

    num_epochs = int(train_config["num_epochs"])
    check_steps = int(train_config["check_steps"])
    save_steps = int(train_config.get("save_steps", 250))
    accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(train_config.get("max_grad_norm", 0.0))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    start_epoch = 0
    resume_batch = 0
    global_step = 0
    resume_generator_state = None
    if (
        isinstance(resume_checkpoint, dict)
        and "optimizer_state_dict" in resume_checkpoint
    ):
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        resume_batch = int(resume_checkpoint.get("batch_in_epoch", 0))
        global_step = int(resume_checkpoint.get("global_step", 0))
        resume_generator_state = resume_checkpoint.get("epoch_generator_state")
        if torch.is_tensor(resume_generator_state):
            resume_generator_state = resume_generator_state.cpu()
        checker.min_loss = float(resume_checkpoint.get("checker_min_loss", float("inf")))
        checker.count = int(resume_checkpoint.get("checker_count", 0))
        print(
            f"Restored optimizer/scheduler at epoch {start_epoch + 1}, "
            f"batch {resume_batch}, optimizer step {global_step}."
        )
    elif resume_checkpoint is not None:
        print("The checkpoint contains weights only; optimizer state starts fresh.")

    last_training_state = checkpoints_dir / "last_training_state.pt"
    pause_request = checkpoints_dir / "pause.request"
    if pause_request.exists():
        pause_request.unlink()
    control_dir = config_dir / "output"
    control_dir.mkdir(exist_ok=True, parents=True)
    (control_dir / "training.current.pause").write_text(
        str(pause_request), encoding="utf-8"
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Training on {device} ({'bf16 autocast' if use_amp else 'float32'}), "
        f"{len(train_usable_dataset)} train / {len(test_usable_dataset)} validation records, "
        f"{parameter_count:,} trainable parameters."
    )
    print(
        f"Micro-batch {batch_size}, gradient accumulation {accumulation_steps}, "
        f"effective batch {batch_size * accumulation_steps}."
    )
    print("=" * 72)
    early_stopped = False
    paused = False

    for epoch in range(start_epoch, num_epochs):
        if epoch == start_epoch and resume_generator_state is not None:
            data_generator.set_state(resume_generator_state)
        epoch_generator_state = data_generator.get_state()
        model.train()
        running_loss = 0.0
        observed_batches = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        optimizer.zero_grad(set_to_none=True)

        for batch_count, batch in enumerate(train_data_loader, start=1):
            if epoch == start_epoch and batch_count <= resume_batch:
                continue
            encoder_input = batch["encoder_input"].to(device, non_blocking=pin_memory)
            decoder_input = batch["decoder_input"].to(device, non_blocking=pin_memory)
            label = batch["label"].to(device, non_blocking=pin_memory)
            source_mask = batch["source_mask"].to(device, non_blocking=pin_memory)
            target_mask = batch["target_mask"].to(device, non_blocking=pin_memory)

            with amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                encoder_output = model.encode(encoder_input, source_mask)
                decoder_output = model.decode(
                    decoder_input, encoder_output, target_mask, source_mask
                )
                logits = model.project(decoder_output)
                loss = criterion(logits.flatten(0, 1), label.flatten())

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss encountered: {loss.item()}")
            (loss / accumulation_steps).backward()
            running_loss += loss.item()
            observed_batches += 1

            should_step = (
                batch_count % accumulation_steps == 0
                or batch_count == len(train_data_loader)
            )
            if should_step:
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if batch_count % 50 == 0 or batch_count == len(train_data_loader):
                print(
                    f"  batch {batch_count}/{len(train_data_loader)}, "
                    f"step {global_step}, loss {loss.item():.4f}, "
                    f"lr {scheduler.get_last_lr()[0]:.3e}, "
                    f"shape {encoder_input.shape[1]}x{decoder_input.shape[1]}"
                )

            if should_step and save_steps > 0 and global_step % save_steps == 0:
                save_training_checkpoint(
                    last_training_state,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    batch_count,
                    global_step,
                    epoch_generator_state,
                    checker,
                )

            if should_step and pause_request.exists():
                save_training_checkpoint(
                    last_training_state,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    batch_count,
                    global_step,
                    epoch_generator_state,
                    checker,
                )
                pause_request.unlink()
                paused = True
                print(
                    f"Pause requested; saved exact training state at epoch "
                    f"{epoch + 1}, batch {batch_count}, step {global_step}: "
                    f"{last_training_state}"
                )
                break

            if should_step and check_steps > 0 and global_step % check_steps == 0:
                if checker():
                    early_stopped = True
                    print(
                        f"Early stopping after validation loss {checker.newest_loss:.4f}; "
                        f"best weights: {checkpoints_dir / 'best_state_from_checker.pt'}"
                    )
                    save_training_checkpoint(
                        last_training_state,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        batch_count,
                        global_step,
                        epoch_generator_state,
                        checker,
                    )
                    break

        if early_stopped or paused:
            break

        epoch_loss = running_loss / max(observed_batches, 1)
        save_path = checkpoints_dir / f"epoch_{epoch + 1}_of_{num_epochs}_state.pt"
        torch.save(model.state_dict(), save_path)
        save_training_checkpoint(
            last_training_state,
            model,
            optimizer,
            scheduler,
            epoch + 1,
            0,
            global_step,
            data_generator.get_state(),
            checker,
        )
        print(f"Epoch {epoch + 1} complete; mean loss {epoch_loss:.4f}; saved {save_path}")
        resume_batch = 0

    if paused:
        print("Training paused safely. Resume with --resume last_training_state.pt")
        return "paused"

    if checker.newest_loss is None or not early_stopped:
        checker()
    print("=" * 72)
    print(f"Training finished. Final validation loss: {checker.newest_loss:.4f}")
    print(f"Best weights: {checkpoints_dir / 'best_state_from_checker.pt'}")
    return "complete"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the hand-written Transformer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "config.json",
        help="Path to a JSON config file (default: project config.json).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Optional full training state or weights-only checkpoint to resume from.",
    )
    parser.add_argument(
        "--retrain-tokenizers",
        action="store_true",
        help="Replace existing tokenizers instead of reusing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open(mode="r", encoding="utf-8") as file:
        config = json.load(file)
    train(config, config_path.parent, args.resume, args.retrain_tokenizers)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
