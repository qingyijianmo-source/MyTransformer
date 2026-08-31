"""Lazy, fully local Qwen post-editor with fail-safe fallback behavior."""

from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol


@dataclass(frozen=True)
class ReviewerSettings:
    enabled: bool = True
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    backend: str = "transformers"
    quantization: str = "4bit-nf4"
    trigger_threshold: int = 35
    max_context_tokens: int = 4096
    max_new_tokens: int = 1024
    timeout_seconds: int = 120
    allow_download: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "ReviewerSettings":
        raw = value or {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            model=str(raw.get("model", cls.model)),
            backend=str(raw.get("backend", cls.backend)),
            quantization=str(raw.get("quantization", cls.quantization)),
            trigger_threshold=max(1, int(raw.get("trigger_threshold", cls.trigger_threshold))),
            max_context_tokens=max(512, int(raw.get("max_context_tokens", cls.max_context_tokens))),
            max_new_tokens=max(64, int(raw.get("max_new_tokens", cls.max_new_tokens))),
            timeout_seconds=max(10, int(raw.get("timeout_seconds", cls.timeout_seconds))),
            allow_download=bool(raw.get("allow_download", False)),
        )

    def signature(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "backend": self.backend,
            "quantization": self.quantization,
            "trigger_threshold": self.trigger_threshold,
            "max_context_tokens": self.max_context_tokens,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class ReviewRequest:
    direction: str
    source: str
    draft: str
    previous_source: str = ""
    next_source: str = ""
    document_context: str = "无"
    reasons: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewResult:
    translation: str
    success: bool
    message: str
    elapsed_seconds: float = 0.0


class ReviewerBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
    def release(self) -> None: ...


class TransformersQwenBackend:
    """Load the official Qwen checkpoint only when a paragraph needs review."""

    def __init__(self, settings: ReviewerSettings):
        self.settings = settings
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "本地审校需要 transformers>=4.51、accelerate 和 bitsandbytes"
            ) from error

        kwargs: dict[str, object] = {
            "device_map": "auto",
            "local_files_only": not self.settings.allow_download,
            "low_cpu_mem_usage": True,
        }
        if self.settings.quantization.startswith("4bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            kwargs["dtype"] = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model,
            local_files_only=not self.settings.allow_download,
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.settings.model, **kwargs)
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_loaded()
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        deadline = time.monotonic() + self.settings.timeout_seconds

        class DeadlineStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return time.monotonic() >= deadline

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        encoded = encoded[:, -self.settings.max_context_tokens :].to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=encoded,
                attention_mask=torch.ones_like(encoded),
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([DeadlineStoppingCriteria()]),
            )
        return self.tokenizer.decode(output[0, encoded.shape[-1] :], skip_special_tokens=True)

    def release(self) -> None:
        if self.model is not None:
            try:
                self.model.to("cpu")
            except (RuntimeError, ValueError):
                pass
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class LocalReviewer:
    def __init__(
        self,
        settings: ReviewerSettings,
        backend_factory: Callable[[ReviewerSettings], ReviewerBackend] | None = None,
    ):
        self.settings = settings
        self.enabled = settings.enabled
        self._backend_factory = backend_factory or TransformersQwenBackend
        self._backend: ReviewerBackend | None = None
        self.unavailable_reason = ""

    @property
    def available(self) -> bool:
        return self.enabled and not self.unavailable_reason

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def _ensure_backend(self) -> ReviewerBackend:
        if self._backend is None:
            if self.settings.backend != "transformers":
                raise RuntimeError(f"不支持的本地审校后端：{self.settings.backend}")
            self._backend = self._backend_factory(self.settings)
        return self._backend

    @staticmethod
    def _prompts(request: ReviewRequest) -> tuple[str, str]:
        target = "自然、准确的英文" if request.direction == "zh-en" else "自然、准确的简体中文"
        system = (
            "你是严谨的专业翻译审校器。只纠正初译，不增添原文没有的信息。"
            "必须保留全部数字、日期、单位、缩写、专名、引号与占位符。"
            f"输出必须是{target}。不要解释，不要列出修改过程，"
            "只在 <translation> 与 </translation> 之间返回最终译文。"
        )
        user = (
            f"审校触发原因：{', '.join(request.reasons) or '质量复核'}\n"
            f"文档词汇与专名：\n{request.document_context}\n\n"
            f"上一段原文：{request.previous_source or '无'}\n"
            f"当前原文：{request.source}\n"
            f"下一段原文：{request.next_source or '无'}\n\n"
            f"当前初译：{request.draft}\n\n"
            f"必须采用的译法：{', '.join(request.required_terms) or '无'}\n"
            f"禁止出现的机械直译：{', '.join(request.forbidden_terms) or '无'}\n"
            "请结合上下文消除歧义、重组长句并保持文体，不得遗漏任何事实。"
        )
        return system, user

    @staticmethod
    def _extract_translation(value: str) -> str:
        match = re.search(r"<translation>\s*(.*?)\s*</translation>", value, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        cleaned = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
        return cleaned.strip()

    def review(self, request: ReviewRequest) -> ReviewResult:
        if not self.enabled:
            return ReviewResult(request.draft, False, "自动深度审校已关闭")
        if self.unavailable_reason:
            return ReviewResult(request.draft, False, self.unavailable_reason)
        started = time.monotonic()
        try:
            backend = self._ensure_backend()
            system, user = self._prompts(request)
            raw = backend.generate(system, user)
            elapsed = time.monotonic() - started
            if elapsed > self.settings.timeout_seconds:
                return ReviewResult(request.draft, False, "本地审校超时，已回退初译", elapsed)
            candidate = self._extract_translation(raw)
            if not candidate:
                return ReviewResult(request.draft, False, "本地审校返回空文本", elapsed)
            return ReviewResult(candidate, True, "本地审校完成", elapsed)
        except Exception as error:
            self.unavailable_reason = f"本地审校不可用，已回退初译：{type(error).__name__}: {error}"
            return ReviewResult(
                request.draft,
                False,
                self.unavailable_reason,
                time.monotonic() - started,
            )

    def release(self) -> None:
        if self._backend is not None:
            self._backend.release()
        self._backend = None
