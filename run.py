import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from build import Transformer, build_transformer
from train_test import (
    PROJECT_DIR,
    load_and_get_tokenizer,
    load_state_dict,
    require_token_id,
    resolve_config_path,
)


class Controller:
    def __init__(
        self,
        model: Transformer,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        source_seq_len: int,
        target_seq_len: int,
        device: torch.device,
        max_new_tokens: int | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ):
        self.model = model
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
        self.target_unk = require_token_id(target_tokenizer, "[UNK]")
        self.device = device
        self.max_new_tokens = min(
            max_new_tokens or target_seq_len, target_seq_len
        )
        self.repetition_penalty = max(float(repetition_penalty), 1.0)
        self.no_repeat_ngram_size = max(int(no_repeat_ngram_size), 0)

    def _banned_tokens(self, generated_ids: list[int]) -> set[int]:
        n = self.no_repeat_ngram_size
        if n <= 0 or len(generated_ids) < n - 1:
            return set()
        if n == 1:
            return set(generated_ids)
        prefix = tuple(generated_ids[-(n - 1):])
        banned = set()
        for index in range(len(generated_ids) - n + 1):
            if tuple(generated_ids[index:index + n - 1]) == prefix:
                banned.add(generated_ids[index + n - 1])
        return banned

    @torch.inference_mode()
    def talk(self, raw_text: str) -> str:
        source_ids = self.source_tokenizer.encode(raw_text).ids[: self.source_seq_len - 2]
        encoder_input = torch.tensor(
            [
                [self.source_sos]
                + source_ids
                + [self.source_eos]
            ],
            dtype=torch.int64,
            device=self.device,
        )
        source_mask = (encoder_input == self.source_pad).unsqueeze(1).unsqueeze(2)
        use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.amp.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            encoder_output = self.model.encode(encoder_input, source_mask)
            decoder_input = torch.tensor(
                [[self.target_sos]], dtype=torch.int64, device=self.device
            )
            generated_ids = []

            for _ in range(self.max_new_tokens):
                now_length = decoder_input.shape[1]
                target_mask = torch.triu(
                    torch.ones(
                        now_length,
                        now_length,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
                decoder_output = self.model.decode(
                    decoder_input, encoder_output, target_mask, source_mask
                )
                logits = self.model.project(decoder_output[:, -1])[0].float()
                logits[[self.target_pad, self.target_sos, self.target_unk]] = -torch.inf

                if self.repetition_penalty > 1.0:
                    for token_id in set(generated_ids):
                        if logits[token_id] < 0:
                            logits[token_id] *= self.repetition_penalty
                        else:
                            logits[token_id] /= self.repetition_penalty
                banned_tokens = self._banned_tokens(generated_ids)
                if banned_tokens:
                    logits[list(banned_tokens)] = -torch.inf

                next_token_id = logits.argmax().item()
                if next_token_id == self.target_eos:
                    break
                generated_ids.append(next_token_id)
                decoder_input = torch.cat(
                    [
                        decoder_input,
                        torch.tensor(
                            [[next_token_id]], dtype=torch.int64, device=self.device
                        ),
                    ],
                    dim=-1,
                )

        return self.target_tokenizer.decode(generated_ids, skip_special_tokens=True)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation has no CUDA support")
    return torch.device(requested)


def build_controller(
    config: dict,
    config_dir: Path,
    checkpoint_override: Path | None,
    device: torch.device,
) -> Controller:
    model_config = config["model"]
    tokenizer_config = config["tokenizer"]
    train_config = config["train"]
    run_config = config["run"]

    tokenizers_dir = resolve_config_path(tokenizer_config["tokenizers_dir"], config_dir)
    zh_tokenizer_path = tokenizers_dir / run_config["zh_tokenizer_file_name"]
    en_tokenizer_path = tokenizers_dir / run_config["en_tokenizer_file_name"]
    checkpoints_dir = resolve_config_path(train_config["checkpoints_dir"], config_dir)
    if checkpoint_override is None:
        checkpoint_path = checkpoints_dir / run_config["selected_checkpoint"]
    else:
        checkpoint_path = (
            checkpoint_override
            if checkpoint_override.is_absolute()
            else (config_dir / checkpoint_override).resolve()
        )

    # Validate artifacts before allocating the full base model.
    for artifact, description in (
        (zh_tokenizer_path, "source tokenizer"),
        (en_tokenizer_path, "target tokenizer"),
        (checkpoint_path, "model checkpoint"),
    ):
        if not artifact.is_file():
            raise FileNotFoundError(
                f"{description} not found: {artifact}\n"
                "Run `python train_test.py` first, or pass the correct --config/--checkpoint."
            )

    zh_tokenizer = load_and_get_tokenizer(zh_tokenizer_path)
    en_tokenizer = load_and_get_tokenizer(en_tokenizer_path)
    model = build_transformer(
        int(model_config["d_model"]),
        int(model_config["d_hidden"]),
        int(model_config["num_heads"]),
        float(model_config["drop_prob"]),
        int(model_config["num_encode_layers"]),
        int(model_config["num_decode_layers"]),
        zh_tokenizer.get_vocab_size(),
        en_tokenizer.get_vocab_size(),
        int(model_config["source_seq_len"]),
        int(model_config["target_seq_len"]),
        bool(model_config.get("tie_target_embedding", False)),
    ).to(device)
    try:
        model.load_state_dict(load_state_dict(checkpoint_path, device))
    except RuntimeError as error:
        raise RuntimeError(
            f"checkpoint is incompatible with the selected config: {checkpoint_path}"
        ) from error
    model.eval()
    print(f"Loaded {checkpoint_path.name} on {device}.")
    return Controller(
        model,
        zh_tokenizer,
        en_tokenizer,
        int(model_config["source_seq_len"]),
        int(model_config["target_seq_len"]),
        device,
        run_config.get("max_new_tokens"),
        float(run_config.get("repetition_penalty", 1.0)),
        int(run_config.get("no_repeat_ngram_size", 0)),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Chinese-to-English Transformer inference.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "config.json",
        help="Path to a JSON config file (default: project config.json).",
    )
    parser.add_argument("--checkpoint", type=Path, help="Override the configured checkpoint.")
    parser.add_argument("--text", help="Translate one sentence and exit instead of prompting.")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open(mode="r", encoding="utf-8") as file:
        config = json.load(file)

    controller = build_controller(
        config, config_path.parent, args.checkpoint, select_device(args.device)
    )
    if args.text is not None:
        print(controller.talk(args.text))
        return

    print("===== 中译英开始（输入 quit 退出）=====")
    while True:
        try:
            text = input("中文 >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if text.lower() == "quit":
            print("Bye!")
            break
        if text:
            print("英文:", controller.talk(text))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
