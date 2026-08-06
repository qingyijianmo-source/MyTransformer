"""Quality-gated GPU fine-tuning for the accurate OPUS-MT translator.

The script fine-tunes the same Marian model used by ``accurate_translator.py``.
It evaluates held-out loss before and after training and only promotes the new
checkpoint when it does not regress.  Progress is printed live and mirrored to
``output/accurate_finetuned/training.log``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config.finetune.json"
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class DatasetSource:
    path: Path
    weight: float


@dataclass(frozen=True)
class FineTuneSettings:
    base_model: str
    revision: str | None
    source_language: str
    target_language: str
    target_prefix: str
    output_dir: Path
    datasets: tuple[DatasetSource, ...]
    max_train_samples: int
    validation_samples: int
    max_source_tokens: int
    max_target_tokens: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    checkpoint_steps: int
    log_steps: int
    seed: int
    max_allowed_validation_loss_regression: float
    config_hash: str

    @classmethod
    def from_file(cls, path: Path) -> "FineTuneSettings":
        path = path.resolve()
        raw_text = path.read_text(encoding="utf-8")
        raw = json.loads(raw_text)

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (path.parent / candidate).resolve()

        datasets = tuple(
            DatasetSource(resolve(str(item["path"])), max(0.0, float(item.get("weight", 1.0))))
            for item in raw.get("datasets", [])
        )
        if not datasets:
            raise ValueError("config.finetune.json 至少需要配置一份训练语料")
        source_language = str(raw.get("source_language", "zh")).lower()
        target_language = str(raw.get("target_language", "en")).lower()
        if (source_language, target_language) not in {("zh", "en"), ("en", "zh")}:
            raise ValueError("当前微调仅支持 zh→en 或 en→zh")
        return cls(
            base_model=str(raw["base_model"]),
            revision=raw.get("revision") or None,
            source_language=source_language,
            target_language=target_language,
            target_prefix=str(raw.get("target_prefix", "")),
            output_dir=resolve(str(raw.get("output_dir", "./output/accurate_finetuned"))),
            datasets=datasets,
            max_train_samples=max(1, int(raw.get("max_train_samples", 12000))),
            validation_samples=max(32, int(raw.get("validation_samples", 256))),
            max_source_tokens=max(32, int(raw.get("max_source_tokens", 192))),
            max_target_tokens=max(32, int(raw.get("max_target_tokens", 192))),
            micro_batch_size=max(1, int(raw.get("micro_batch_size", 8))),
            gradient_accumulation_steps=max(1, int(raw.get("gradient_accumulation_steps", 4))),
            max_steps=max(1, int(raw.get("max_steps", 200))),
            learning_rate=float(raw.get("learning_rate", 1e-5)),
            weight_decay=max(0.0, float(raw.get("weight_decay", 0.01))),
            warmup_ratio=min(1.0, max(0.0, float(raw.get("warmup_ratio", 0.06)))),
            max_grad_norm=max(0.0, float(raw.get("max_grad_norm", 1.0))),
            checkpoint_steps=max(1, int(raw.get("checkpoint_steps", 100))),
            log_steps=max(1, int(raw.get("log_steps", 5))),
            seed=int(raw.get("seed", 42)),
            max_allowed_validation_loss_regression=float(
                raw.get("max_allowed_validation_loss_regression", 0.0)
            ),
            config_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        )


class TeeLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")

    def close(self) -> None:
        self.file.close()


class PairDataset(Dataset):
    def __init__(self, pairs: Sequence[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.pairs[index]


class TranslationCollator:
    def __init__(
        self,
        tokenizer,
        max_source_tokens: int,
        max_target_tokens: int,
        target_prefix: str = "",
    ):
        self.tokenizer = tokenizer
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens
        self.target_prefix = target_prefix

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> dict[str, torch.Tensor]:
        sources, targets = zip(*pairs)
        model_sources = [self.target_prefix + source for source in sources]
        inputs = self.tokenizer(
            model_sources,
            padding=True,
            truncation=True,
            max_length=self.max_source_tokens,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=list(targets),
            padding=True,
            truncation=True,
            max_length=self.max_target_tokens,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def normalize_text(value: object) -> str:
    return SPACE_PATTERN.sub(" ", str(value or "")).strip()


def quality_reason(source: str, target: str, source_language: str = "zh") -> str | None:
    if not source or not target:
        return "empty"
    if len(source) < 2 or len(target) < 2 or len(source) > 600 or len(target) > 900:
        return "length"
    source_cjk = len(CJK_PATTERN.findall(source))
    target_cjk = len(CJK_PATTERN.findall(target))
    if source_language == "zh":
        if source_cjk < 2:
            return "source_language"
        if not LATIN_PATTERN.search(target):
            return "target_language"
        if target_cjk > max(2, len(target) // 10):
            return "target_cjk"
        ratio = len(target) / max(source_cjk, 1)
    else:
        if len(LATIN_PATTERN.findall(source)) < 2:
            return "source_language"
        if target_cjk < 2:
            return "target_language"
        if source_cjk > max(2, len(source) // 10):
            return "source_cjk"
        ratio = len(source) / max(target_cjk, 1)
    if source.casefold() == target.casefold():
        return "identical"
    if ratio < 0.45 or ratio > 18.0:
        return "length_ratio"
    if source.startswith(("/", "http://", "https://")):
        return "path_or_url"
    return None


def translation_pair(
    value: object, source_language: str = "zh"
) -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None
    zh = normalize_text(value.get("zh_CN") or value.get("zh") or value.get("zh-CN"))
    en = normalize_text(value.get("en_GB") or value.get("en") or value.get("en-US"))
    source, target = (zh, en) if source_language == "zh" else (en, zh)
    if quality_reason(source, target, source_language) is not None:
        return None
    return source, target


def select_pairs(
    settings: FineTuneSettings, logger: TeeLogger
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, object]]:
    rng = random.Random(settings.seed)
    pools: list[tuple[DatasetSource, list[tuple[str, str]], int]] = []
    source_stats: list[dict[str, object]] = []
    for source in settings.datasets:
        if not source.path.is_file():
            logger.write(f"跳过不存在的语料：{source.path}")
            continue
        frame = pd.read_parquet(source.path, columns=["translation"])
        accepted: list[tuple[str, str]] = []
        seen_local: set[str] = set()
        rejected = 0
        for value in frame["translation"]:
            pair = translation_pair(value, settings.source_language)
            if pair is None:
                rejected += 1
                continue
            key = hashlib.sha1((pair[0] + "\0" + pair[1]).encode("utf-8")).hexdigest()
            if key in seen_local:
                rejected += 1
                continue
            seen_local.add(key)
            accepted.append(pair)
        rng.shuffle(accepted)
        pools.append((source, accepted, 0))
        source_stats.append(
            {
                "path": str(source.path),
                "rows": len(frame),
                "accepted_after_filter": len(accepted),
                "rejected_or_duplicate": rejected,
            }
        )
        logger.write(
            f"语料 {source.path.name}：读取 {len(frame):,}，过滤后 {len(accepted):,}"
        )

    if not pools:
        raise FileNotFoundError("没有找到可用于微调的 Parquet 语料")
    target_total = settings.max_train_samples + settings.validation_samples
    total_weight = sum(source.weight for source, _, _ in pools) or float(len(pools))
    selected: list[tuple[str, str]] = []
    selected_keys: set[str] = set()
    positions: list[int] = [0 for _ in pools]

    def append_unique(pair: tuple[str, str]) -> bool:
        key = hashlib.sha1((pair[0] + "\0" + pair[1]).encode("utf-8")).hexdigest()
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected.append(pair)
        return True

    for pool_index, (source, pool, _) in enumerate(pools):
        weight = source.weight if total_weight else 1.0
        quota = max(1, round(target_total * weight / total_weight))
        while positions[pool_index] < len(pool) and quota > 0:
            pair = pool[positions[pool_index]]
            positions[pool_index] += 1
            if append_unique(pair):
                quota -= 1

    while len(selected) < target_total:
        progressed = False
        for pool_index, (_, pool, _) in enumerate(pools):
            while positions[pool_index] < len(pool):
                pair = pool[positions[pool_index]]
                positions[pool_index] += 1
                if append_unique(pair):
                    progressed = True
                    break
            if len(selected) >= target_total:
                break
        if not progressed:
            break

    if len(selected) <= settings.validation_samples:
        raise RuntimeError(
            f"清洗后只有 {len(selected):,} 对语料，不足以划分验证集"
        )
    rng.shuffle(selected)
    validation = selected[: settings.validation_samples]
    training = selected[settings.validation_samples : target_total]
    logger.write(
        f"最终使用训练集 {len(training):,} 对，独立验证集 {len(validation):,} 对"
    )
    stats = {
        "sources": source_stats,
        "training_pairs": len(training),
        "validation_pairs": len(validation),
    }
    return training, validation, stats


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def evaluate_loss(
    model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    logger: TeeLogger,
    label: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(**batch).loss
        total_loss += float(loss.detach())
        total_batches += 1
        if index % 10 == 0 or index == len(loader):
            logger.write(f"{label}：{index}/{len(loader)} 批")
    if not total_batches:
        raise RuntimeError("验证集为空")
    return total_loss / total_batches


@torch.inference_mode()
def generate_samples(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    device: torch.device,
    target_prefix: str = "",
    max_source_tokens: int = 192,
) -> list[dict[str, str]]:
    model.eval()
    rows: list[dict[str, str]] = []
    for source, reference in pairs[:8]:
        encoded = tokenizer(
            target_prefix + source,
            return_tensors="pt",
            truncation=True,
            max_length=max_source_tokens,
        ).to(device)
        output = model.generate(**encoded, num_beams=4, max_new_tokens=192)
        rows.append(
            {
                "source": source,
                "reference": reference,
                "translation": tokenizer.decode(output[0], skip_special_tokens=True),
            }
        )
    return rows


def save_resume_checkpoint(
    model,
    tokenizer,
    resume_dir: Path,
    trainer_state: dict[str, object],
    logger: TeeLogger,
) -> None:
    resume_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(resume_dir, safe_serialization=True)
    tokenizer.save_pretrained(resume_dir)
    # transformers 4.50 does not consistently copy Marian's SentencePiece
    # assets from a Hub cache when save_pretrained() targets a new directory.
    tokenizer.save_vocabulary(resume_dir)
    state_path = resume_dir / "trainer_state.pt"
    temporary = resume_dir / "trainer_state.pt.tmp"
    torch.save(trainer_state, temporary)
    os.replace(temporary, state_path)
    logger.write(f"已保存断点：优化步骤 {trainer_state['step']}")


def safe_reset_directory(path: Path, root: Path) -> None:
    path = path.resolve()
    root = root.resolve()
    if path.parent != root:
        raise RuntimeError(f"拒绝清理输出目录之外的路径：{path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def promote_candidate(candidate: Path, best: Path, output_root: Path) -> None:
    candidate = candidate.resolve()
    best = best.resolve()
    output_root = output_root.resolve()
    if candidate.parent != output_root or best.parent != output_root:
        raise RuntimeError("模型提升路径超出微调输出目录")
    backup = output_root / "best.previous"
    if backup.exists():
        shutil.rmtree(backup)
    if best.exists():
        best.replace(backup)
    try:
        candidate.replace(best)
    except Exception:
        if backup.exists() and not best.exists():
            backup.replace(best)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def train(settings: FineTuneSettings, fresh: bool = False) -> int:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(settings.output_dir / "training.log")
    started_at = utc_now()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("当前微调配置要求 NVIDIA CUDA GPU，但 PyTorch 未检测到 CUDA")
        device = torch.device("cuda")
        torch.manual_seed(settings.seed)
        torch.cuda.manual_seed_all(settings.seed)
        random.seed(settings.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        direction_label = (
            "中译英" if settings.source_language == "zh" else "英译中"
        )
        logger.write(f"开始增强当前 OPUS-MT {direction_label}模型")
        logger.write(f"GPU：{torch.cuda.get_device_name(0)}")
        logger.write(
            f"有效批量：{settings.micro_batch_size * settings.gradient_accumulation_steps}；"
            f"优化步骤：{settings.max_steps}；学习率：{settings.learning_rate:g}"
        )
        training_pairs, validation_pairs, data_stats = select_pairs(settings, logger)

        resume_dir = settings.output_dir / "resume"
        candidate_dir = settings.output_dir / "candidate"
        best_dir = settings.output_dir / "best"
        if fresh and resume_dir.exists():
            safe_reset_directory(resume_dir, settings.output_dir)
        resume_state_path = resume_dir / "trainer_state.pt"
        resume_state: dict[str, object] | None = None
        if resume_state_path.is_file() and (resume_dir / "config.json").is_file():
            loaded = torch.load(resume_state_path, map_location="cpu", weights_only=True)
            if loaded.get("config_hash") == settings.config_hash:
                resume_state = loaded
                model_source = str(resume_dir)
                revision_args: dict[str, str] = {}
                logger.write(f"发现兼容断点，将从步骤 {loaded.get('step', 0)} 继续")
            else:
                model_source = settings.base_model
                revision_args = {"revision": settings.revision} if settings.revision else {}
                logger.write("已有断点与当前配置不兼容，将从官方基础模型重新开始")
        else:
            model_source = settings.base_model
            revision_args = {"revision": settings.revision} if settings.revision else {}

        # Fine-tuning never changes Marian's vocabulary.  Keep using the base
        # tokenizer cache so resume also works when the project path contains
        # non-ASCII characters that SentencePiece cannot open on Windows.
        tokenizer_revision_args = (
            {"revision": settings.revision} if settings.revision else {}
        )
        tokenizer = AutoTokenizer.from_pretrained(
            settings.base_model, **tokenizer_revision_args
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_source, **revision_args).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        logger.write(f"模型参数：{parameter_count:,}")
        collator = TranslationCollator(
            tokenizer,
            settings.max_source_tokens,
            settings.max_target_tokens,
            settings.target_prefix,
        )
        generator = torch.Generator().manual_seed(settings.seed)
        train_loader = DataLoader(
            PairDataset(training_pairs),
            batch_size=settings.micro_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            PairDataset(validation_pairs),
            batch_size=settings.micro_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        use_amp = True

        if resume_state and "base_validation_loss" in resume_state:
            base_validation_loss = float(resume_state["base_validation_loss"])
            base_samples = list(resume_state.get("base_samples", []))
            logger.write(f"沿用断点中的基线验证损失：{base_validation_loss:.5f}")
        else:
            logger.write("正在测量官方基础模型的验证损失…")
            base_validation_loss = evaluate_loss(
                model, validation_loader, device, use_amp, logger, "基线验证"
            )
            base_samples = generate_samples(
                model,
                tokenizer,
                validation_pairs,
                device,
                settings.target_prefix,
                settings.max_source_tokens,
            )
            logger.write(f"官方基础模型验证损失：{base_validation_loss:.5f}")

        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        warmup_steps = round(settings.max_steps * settings.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=settings.max_steps
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        start_step = 0
        if resume_state:
            start_step = int(resume_state.get("step", 0))
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            if "scaler" in resume_state:
                scaler.load_state_dict(resume_state["scaler"])

        optimizer.zero_grad(set_to_none=True)
        train_iterator = iter(train_loader)
        recent_loss = 0.0
        recent_micro_steps = 0
        training_started = time.monotonic()
        logger.write(f"进入训练循环，从 {start_step}/{settings.max_steps} 开始")
        for step in range(start_step + 1, settings.max_steps + 1):
            for _ in range(settings.gradient_accumulation_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                batch = move_batch(batch, device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(**batch).loss
                    scaled_loss = loss / settings.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                recent_loss += float(loss.detach())
                recent_micro_steps += 1

            scaler.unscale_(optimizer)
            if settings.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % settings.log_steps == 0 or step == settings.max_steps:
                elapsed = time.monotonic() - training_started
                completed = step - start_step
                seconds_per_step = elapsed / max(completed, 1)
                eta_seconds = seconds_per_step * (settings.max_steps - step)
                average_loss = recent_loss / max(recent_micro_steps, 1)
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                logger.write(
                    f"进度 {step}/{settings.max_steps}（{step / settings.max_steps:.0%}） | "
                    f"loss {average_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"显存 {allocated_gb:.2f} GB | ETA {eta_seconds / 60:.1f} 分钟"
                )
                recent_loss = 0.0
                recent_micro_steps = 0

            if step % settings.checkpoint_steps == 0 and step < settings.max_steps:
                save_resume_checkpoint(
                    model,
                    tokenizer,
                    resume_dir,
                    {
                        "step": step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config_hash": settings.config_hash,
                        "base_validation_loss": base_validation_loss,
                        "base_samples": base_samples,
                    },
                    logger,
                )
                model.train()
                model.config.use_cache = False

        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        logger.write("训练完成，正在执行独立验证集质量门控…")
        candidate_validation_loss = evaluate_loss(
            model, validation_loader, device, use_amp, logger, "增强后验证"
        )
        candidate_samples = generate_samples(
            model,
            tokenizer,
            validation_pairs,
            device,
            settings.target_prefix,
            settings.max_source_tokens,
        )
        loss_change = candidate_validation_loss - base_validation_loss
        accepted = loss_change <= settings.max_allowed_validation_loss_regression
        status = "accepted" if accepted else "rejected"
        logger.write(
            f"验证结果：基础 {base_validation_loss:.5f}，增强后 {candidate_validation_loss:.5f}，"
            f"变化 {loss_change:+.5f}"
        )

        safe_reset_directory(candidate_dir, settings.output_dir)
        model.save_pretrained(candidate_dir, safe_serialization=True)
        tokenizer.save_pretrained(candidate_dir)
        tokenizer.save_vocabulary(candidate_dir)
        manifest = {
            "status": status,
            "started_at": started_at,
            "finished_at": utc_now(),
            "base_model": settings.base_model,
            "revision": settings.revision,
            "parameter_count": parameter_count,
            "base_validation_loss": base_validation_loss,
            "candidate_validation_loss": candidate_validation_loss,
            "validation_loss_change": loss_change,
            "quality_gate": {
                "max_allowed_validation_loss_regression": settings.max_allowed_validation_loss_regression
            },
            "training": {
                key: value
                for key, value in asdict(settings).items()
                if key not in {"datasets", "output_dir", "config_hash"}
            },
            "data": data_stats,
            "examples": [
                {
                    "source": base_row["source"],
                    "reference": base_row["reference"],
                    "before": base_row["translation"],
                    "after": candidate_row["translation"],
                }
                for base_row, candidate_row in zip(base_samples, candidate_samples)
            ],
        }
        (candidate_dir / "training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if accepted:
            promote_candidate(candidate_dir, best_dir, settings.output_dir)
            if resume_dir.exists():
                shutil.rmtree(resume_dir)
            logger.write(f"质量门控通过，增强模型已启用：{best_dir}")
            logger.write("请在翻译界面点击“重新加载模型”，或重启翻译界面。")
            return 0

        logger.write("质量门控未通过：候选权重已保留，但当前翻译器继续使用原模型。")
        return 4
    except KeyboardInterrupt:
        logger.write("收到停止信号；最近一次定期断点可用于下次继续训练。")
        return 130
    except Exception as error:
        logger.write(f"训练失败：{type(error).__name__}: {error}")
        raise
    finally:
        logger.close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--fresh", action="store_true", help="忽略已有训练断点，从官方基础模型重新开始"
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    settings = FineTuneSettings.from_file(args.config)
    return train(settings, fresh=args.fresh)


if __name__ == "__main__":
    sys.exit(main())
