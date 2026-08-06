"""Fast SCST-style reward fine-tuning for the hand-written Transformer."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from build import build_transformer
from train_test import (
    PROJECT_DIR,
    DynamicPaddingCollator,
    UsableDataset,
    deal_raw_dataset,
    load_and_get_tokenizer,
    load_state_dict,
    require_token_id,
    resolve_config_path,
)


def causal_mask(length: int, device: torch.device):
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)


def trim_tokens(tokens, eos_id: int, pad_id: int):
    result = []
    for token in tokens:
        token = int(token)
        if token == eos_id:
            break
        if token != pad_id:
            result.append(token)
    return result


def ngram_f1(hypothesis, reference, n: int) -> float:
    if len(hypothesis) < n or len(reference) < n:
        return 0.0
    hypothesis_counts = Counter(
        tuple(hypothesis[index:index + n])
        for index in range(len(hypothesis) - n + 1)
    )
    reference_counts = Counter(
        tuple(reference[index:index + n])
        for index in range(len(reference) - n + 1)
    )
    overlap = sum(
        min(count, reference_counts[gram])
        for gram, count in hypothesis_counts.items()
    )
    precision = overlap / max(sum(hypothesis_counts.values()), 1)
    recall = overlap / max(sum(reference_counts.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def translation_reward(hypothesis, reference) -> float:
    if not hypothesis or not reference:
        return 0.0
    unigram = ngram_f1(hypothesis, reference, 1)
    bigram = ngram_f1(hypothesis, reference, 2)
    brevity_penalty = math.exp(min(0.0, 1.0 - len(reference) / len(hypothesis)))
    return brevity_penalty * (0.4 * unigram + 0.6 * bigram)


@torch.no_grad()
def rollout(
    model,
    encoder_input,
    source_mask,
    target_sos: int,
    target_eos: int,
    target_pad: int,
    target_unk: int,
    max_new_tokens: int,
    temperature: float,
    sample: bool,
    use_amp: bool,
):
    model.eval()
    device = encoder_input.device
    batch_size = encoder_input.shape[0]
    with torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
    ):
        encoder_output = model.encode(encoder_input, source_mask)
        decoder_input = torch.full(
            (batch_size, 1), target_sos, dtype=torch.int64, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated = []
        for _ in range(max_new_tokens):
            length = decoder_input.shape[1]
            decoder_output = model.decode(
                decoder_input,
                encoder_output,
                causal_mask(length, device),
                source_mask,
            )
            logits = model.project(decoder_output[:, -1]).float()
            logits[:, [target_pad, target_sos, target_unk]] = -torch.inf
            if sample:
                probabilities = F.softmax(logits / max(temperature, 1e-4), dim=-1)
                next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            else:
                next_tokens = logits.argmax(dim=-1)
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, target_pad), next_tokens
            )
            generated.append(next_tokens)
            finished = finished | (next_tokens == target_eos)
            decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
            if finished.all():
                break
    return torch.stack(generated, dim=1)


@torch.no_grad()
def evaluate_greedy_reward(
    model,
    loader,
    device: torch.device,
    target_sos: int,
    target_eos: int,
    target_pad: int,
    target_unk: int,
    max_new_tokens: int,
    use_amp: bool,
):
    """Measure the deployment-time greedy policy on held-out sentence pairs."""
    scores = []
    for batch in loader:
        encoder_input = batch["encoder_input"].to(device)
        source_mask = batch["source_mask"].to(device)
        generated = rollout(
            model,
            encoder_input,
            source_mask,
            target_sos,
            target_eos,
            target_pad,
            target_unk,
            max_new_tokens,
            1.0,
            False,
            use_amp,
        ).cpu()
        for hypothesis, reference in zip(generated.tolist(), batch["label"].tolist()):
            scores.append(
                translation_reward(
                    trim_tokens(hypothesis, target_eos, target_pad),
                    trim_tokens(reference, target_eos, target_pad),
                )
            )
    return sum(scores) / max(len(scores), 1)


def finetune(config: dict, config_dir: Path):
    model_config = config["model"]
    tokenizer_config = config["tokenizer"]
    data_config = config["data"]
    train_config = config["train"]
    run_config = config["run"]
    rl_config = config["rl"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    tokenizers_dir = resolve_config_path(tokenizer_config["tokenizers_dir"], config_dir)
    checkpoints_dir = resolve_config_path(train_config["checkpoints_dir"], config_dir)
    source_tokenizer = load_and_get_tokenizer(
        tokenizers_dir / run_config["zh_tokenizer_file_name"]
    )
    target_tokenizer = load_and_get_tokenizer(
        tokenizers_dir / run_config["en_tokenizer_file_name"]
    )
    input_checkpoint = checkpoints_dir / rl_config["input_checkpoint"]
    output_checkpoint = checkpoints_dir / rl_config["output_checkpoint"]

    model = build_transformer(
        int(model_config["d_model"]),
        int(model_config["d_hidden"]),
        int(model_config["num_heads"]),
        float(model_config["drop_prob"]),
        int(model_config["num_encode_layers"]),
        int(model_config["num_decode_layers"]),
        source_tokenizer.get_vocab_size(),
        target_tokenizer.get_vocab_size(),
        int(model_config["source_seq_len"]),
        int(model_config["target_seq_len"]),
        bool(model_config.get("tie_target_embedding", False)),
    ).to(device)
    model.load_state_dict(load_state_dict(input_checkpoint, device))

    train_dataset, validation_dataset = deal_raw_dataset(
        resolve_config_path(data_config["dataset_path"], config_dir),
        data_config["train_size"],
        data_config["test_size"],
        int(data_config["seed"]),
    )
    sample_count = min(int(rl_config["samples"]), len(train_dataset))
    train_dataset = train_dataset.shuffle(seed=int(data_config["seed"]) + 1).select(
        range(sample_count)
    )
    usable_dataset = UsableDataset(
        train_dataset,
        data_config["source_language"],
        data_config["target_language"],
        source_tokenizer,
        target_tokenizer,
        int(model_config["source_seq_len"]),
        int(model_config["target_seq_len"]),
    )
    source_pad = require_token_id(source_tokenizer, "[PAD]")
    target_pad = require_token_id(target_tokenizer, "[PAD]")
    target_sos = require_token_id(target_tokenizer, "[SOS]")
    target_eos = require_token_id(target_tokenizer, "[EOS]")
    target_unk = require_token_id(target_tokenizer, "[UNK]")
    loader = DataLoader(
        usable_dataset,
        batch_size=int(rl_config["batch_size"]),
        shuffle=True,
        collate_fn=DynamicPaddingCollator(source_pad, target_pad),
    )
    validation_count = min(
        int(rl_config.get("validation_samples", 256)), len(validation_dataset)
    )
    validation_dataset = validation_dataset.shuffle(
        seed=int(data_config["seed"]) + 2
    ).select(range(validation_count))
    validation_loader = DataLoader(
        UsableDataset(
            validation_dataset,
            data_config["source_language"],
            data_config["target_language"],
            source_tokenizer,
            target_tokenizer,
            int(model_config["source_seq_len"]),
            int(model_config["target_seq_len"]),
        ),
        batch_size=int(rl_config.get("validation_batch_size", 32)),
        shuffle=False,
        collate_fn=DynamicPaddingCollator(source_pad, target_pad),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(rl_config["learning_rate"])
    )
    criterion = nn.CrossEntropyLoss(ignore_index=target_pad)
    supervised_weight = float(rl_config["supervised_weight"])
    max_grad_norm = float(rl_config["max_grad_norm"])
    max_new_tokens = int(rl_config["max_new_tokens"])
    temperature = float(rl_config["temperature"])
    validation_interval = max(int(rl_config.get("validation_interval", 16)), 1)
    validation_min_delta = float(rl_config.get("validation_min_delta", 1e-4))
    print(
        f"SCST reward fine-tuning on {device}: {sample_count} samples, "
        f"batch {rl_config['batch_size']}, {rl_config['epochs']} epoch(s)."
    )
    baseline_validation_reward = evaluate_greedy_reward(
        model,
        validation_loader,
        device,
        target_sos,
        target_eos,
        target_pad,
        target_unk,
        max_new_tokens,
        use_amp,
    )
    best_validation_reward = baseline_validation_reward
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    print(
        f"Held-out greedy reward before SCST: {baseline_validation_reward:.6f} "
        f"({validation_count} samples)."
    )

    for epoch in range(int(rl_config["epochs"])):
        reward_total = 0.0
        baseline_total = 0.0
        loss_total = 0.0
        for batch_index, batch in enumerate(loader, start=1):
            encoder_input = batch["encoder_input"].to(device)
            source_mask = batch["source_mask"].to(device)
            references = batch["label"].to(device)
            sampled = rollout(
                model,
                encoder_input,
                source_mask,
                target_sos,
                target_eos,
                target_pad,
                target_unk,
                max_new_tokens,
                temperature,
                True,
                use_amp,
            )
            greedy = rollout(
                model,
                encoder_input,
                source_mask,
                target_sos,
                target_eos,
                target_pad,
                target_unk,
                max_new_tokens,
                temperature,
                False,
                use_amp,
            )

            sampled_rewards = []
            greedy_rewards = []
            for sample_tokens, greedy_tokens, reference_tokens in zip(
                sampled.tolist(), greedy.tolist(), references.tolist()
            ):
                reference = trim_tokens(reference_tokens, target_eos, target_pad)
                sampled_rewards.append(
                    translation_reward(
                        trim_tokens(sample_tokens, target_eos, target_pad), reference
                    )
                )
                greedy_rewards.append(
                    translation_reward(
                        trim_tokens(greedy_tokens, target_eos, target_pad), reference
                    )
                )
            sampled_reward = torch.tensor(sampled_rewards, device=device)
            greedy_reward = torch.tensor(greedy_rewards, device=device)
            advantage = (sampled_reward - greedy_reward).detach()
            if bool(rl_config.get("scale_advantage", True)):
                advantage = advantage / advantage.std().clamp_min(0.05)

            sampled_decoder_input = torch.cat(
                [
                    torch.full(
                        (sampled.shape[0], 1),
                        target_sos,
                        dtype=torch.int64,
                        device=device,
                    ),
                    sampled[:, :-1],
                ],
                dim=1,
            )
            sampled_mask = (
                (sampled_decoder_input == target_pad).unsqueeze(1).unsqueeze(2)
                | causal_mask(sampled.shape[1], device)
            )
            gold_decoder_input = batch["decoder_input"].to(device)
            gold_target_mask = batch["target_mask"].to(device)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                encoder_output = model.encode(encoder_input, source_mask)
                sampled_output = model.decode(
                    sampled_decoder_input,
                    encoder_output,
                    sampled_mask,
                    source_mask,
                )
                sampled_logits = model.project(sampled_output)
                log_probabilities = F.log_softmax(sampled_logits.float(), dim=-1)
                token_log_probabilities = log_probabilities.gather(
                    -1, sampled.unsqueeze(-1)
                ).squeeze(-1)
                sampled_token_mask = sampled != target_pad
                sequence_log_probability = (
                    (token_log_probabilities * sampled_token_mask).sum(dim=1)
                    / sampled_token_mask.sum(dim=1).clamp_min(1)
                )
                policy_loss = -(advantage * sequence_log_probability).mean()

                gold_output = model.decode(
                    gold_decoder_input,
                    encoder_output,
                    gold_target_mask,
                    source_mask,
                )
                gold_logits = model.project(gold_output)
                supervised_loss = criterion(
                    gold_logits.flatten(0, 1), references.flatten()
                )
                loss = policy_loss + supervised_weight * supervised_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            reward_total += sampled_reward.mean().item()
            baseline_total += greedy_reward.mean().item()
            loss_total += loss.item()
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"  RL batch {batch_index}/{len(loader)}, "
                    f"reward {reward_total / batch_index:.4f}, "
                    f"greedy {baseline_total / batch_index:.4f}, "
                    f"loss {loss_total / batch_index:.4f}"
                )

            if batch_index % validation_interval == 0 or batch_index == len(loader):
                validation_reward = evaluate_greedy_reward(
                    model,
                    validation_loader,
                    device,
                    target_sos,
                    target_eos,
                    target_pad,
                    target_unk,
                    max_new_tokens,
                    use_amp,
                )
                improved = validation_reward > (
                    best_validation_reward + validation_min_delta
                )
                print(
                    f"    held-out reward {validation_reward:.6f} "
                    f"(best {best_validation_reward:.6f})"
                    + (" [selected]" if improved else "")
                )
                if improved:
                    best_validation_reward = validation_reward
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }

    model.load_state_dict(best_state)
    # Save from the model so tied embedding/projection weights keep shared
    # storage instead of being serialized twice from the CPU snapshot.
    torch.save(model.state_dict(), output_checkpoint)
    metadata_path = output_checkpoint.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "method": "self-critical sequence training",
                "baseline_validation_reward": baseline_validation_reward,
                "selected_validation_reward": best_validation_reward,
                "validation_samples": validation_count,
                "rl_update_selected": (
                    best_validation_reward > baseline_validation_reward
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"Selected held-out reward: {best_validation_reward:.6f} "
        f"(baseline {baseline_validation_reward:.6f})."
    )
    print(f"Validated SCST checkpoint saved: {output_checkpoint}")
    return output_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_DIR / "config.fast_rl.json"
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    finetune(config, config_path.parent)


if __name__ == "__main__":
    main()
