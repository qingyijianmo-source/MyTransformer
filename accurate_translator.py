"""Accurate Chinese-to-English translation for long text and documents.

The original hand-written Transformer remains available in ``run.py``.  This
module uses a pretrained OPUS-MT checkpoint, sentence-aware chunking, batched
beam search, and a durable translation cache for practical document work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config.accurate.json"
ProgressCallback = Callable[[int, int, str], None]
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
MARKDOWN_PREFIX = re.compile(r"^(\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+))(.*)$")
MARKDOWN_PROTECTED = re.compile(r"(`[^`]*`|https?://\S+|!?\[[^\]]*\]\([^)]*\))")


@dataclass(frozen=True)
class TranslatorSettings:
    model_name: str
    revision: Optional[str]
    glossary_file: Optional[str]
    device: str
    batch_size: int
    num_beams: int
    length_penalty: float
    no_repeat_ngram_size: int
    max_source_tokens: int
    max_new_tokens: int

    @classmethod
    def from_file(cls, path: Path) -> "TranslatorSettings":
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return cls(
            model_name=str(raw["model_name"]),
            revision=raw.get("revision") or None,
            glossary_file=raw.get("glossary_file") or None,
            device=str(raw.get("device", "auto")),
            batch_size=max(1, int(raw.get("batch_size", 16))),
            num_beams=max(1, int(raw.get("num_beams", 5))),
            length_penalty=float(raw.get("length_penalty", 1.0)),
            no_repeat_ngram_size=max(0, int(raw.get("no_repeat_ngram_size", 3))),
            max_source_tokens=max(32, int(raw.get("max_source_tokens", 384))),
            max_new_tokens=max(32, int(raw.get("max_new_tokens", 512))),
        )


class TranslationCache:
    """Append-only JSONL cache; an interrupted document run can be resumed."""

    def __init__(self, path: Optional[Path], signature: str):
        self.path = path
        self.signature = signature
        self.values: dict[str, str] = {}
        if path is not None and path.is_file():
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if record.get("signature") == signature:
                        self.values[str(record.get("key", ""))] = str(
                            record.get("translation", "")
                        )

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[str]:
        return self.values.get(self._key(text))

    def put(self, text: str, translation: str) -> None:
        key = self._key(text)
        self.values[key] = translation
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "signature": self.signature,
            "key": key,
            "translation": translation,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())


class AccurateTranslator:
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        cache_path: Optional[Path] = None,
        progress: Optional[ProgressCallback] = None,
    ):
        self.config_path = config_path.resolve()
        self.settings = TranslatorSettings.from_file(self.config_path)
        self.progress = progress
        self.glossary = self._load_glossary()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.device = self._select_device(self.settings.device)
        self._notify(0, 1, f"正在加载 {self.settings.model_name}…")
        revision_args = (
            {"revision": self.settings.revision} if self.settings.revision else {}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_name, **revision_args
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.settings.model_name, torch_dtype=dtype, **revision_args
        ).to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        signature_source = json.dumps(
            {
                "pipeline_version": 3,
                "model": self.settings.model_name,
                "revision": self.settings.revision,
                "beams": self.settings.num_beams,
                "length_penalty": self.settings.length_penalty,
                "no_repeat_ngram_size": self.settings.no_repeat_ngram_size,
                "glossary": self.glossary,
            },
            sort_keys=True,
        )
        signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()
        self.cache = TranslationCache(cache_path, signature)
        self._notify(1, 1, f"模型已加载到 {self.device}")

    def _load_glossary(self) -> dict[str, dict[str, object]]:
        configured = self.settings.glossary_file
        if not configured:
            return {}
        path = Path(configured)
        if not path.is_absolute():
            path = (self.config_path.parent / path).resolve()
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as file:
            values = json.load(file)
        if not isinstance(values, dict):
            raise ValueError(f"术语表必须是 JSON 对象：{path}")
        glossary: dict[str, dict[str, object]] = {}
        for source, raw_rule in values.items():
            source = str(source)
            if isinstance(raw_rule, str):
                target = raw_rule
                aliases: list[str] = []
            elif isinstance(raw_rule, dict):
                target = str(raw_rule.get("target", ""))
                aliases = [
                    str(alias)
                    for alias in raw_rule.get("aliases", [])
                    if str(alias)
                ]
            else:
                raise ValueError(f"术语 {source!r} 的规则必须是字符串或对象")
            if source and target:
                glossary[source] = {"target": target, "aliases": aliases}
        return glossary

    def _protect_glossary(self, text: str) -> tuple[str, dict[str, str]]:
        protected = text
        replacements: dict[str, str] = {}
        terms = sorted(self.glossary.items(), key=lambda item: len(item[0]), reverse=True)
        # Give the model the approved English term in the source so it can
        # arrange fluent grammar; known alternative outputs are normalized
        # back to the approved term after generation.
        for source, rule in terms:
            if source not in protected:
                continue
            target = str(rule["target"])
            protected = protected.replace(source, target)
            for alias in rule["aliases"]:
                replacements[str(alias)] = target
        return protected, replacements

    @staticmethod
    def _restore_glossary(text: str, replacements: dict[str, str]) -> str:
        restored = text
        for alias, target in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            restored = re.sub(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                target,
                restored,
                flags=re.IGNORECASE,
            )
        return restored

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前 PyTorch 无法使用 CUDA")
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda" or torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _notify(self, completed: int, total: int, message: str) -> None:
        if self.progress is not None:
            self.progress(completed, max(total, 1), message)

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        if not text:
            return []
        endings = set("。！？!?；;\n")
        closers = set("”’」』】）》)]\"")
        pieces: list[str] = []
        start = 0
        index = 0
        while index < len(text):
            if text[index] in endings:
                end = index + 1
                while end < len(text) and text[end] in closers:
                    end += 1
                piece = text[start:end].strip()
                if piece:
                    pieces.append(piece)
                start = end
                index = end
                continue
            index += 1
        remainder = text[start:].strip()
        if remainder:
            pieces.append(remainder)
        return pieces

    def _hard_token_split(self, text: str) -> list[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        size = self.settings.max_source_tokens
        return [
            self.tokenizer.decode(token_ids[index : index + size]).strip()
            for index in range(0, len(token_ids), size)
            if token_ids[index : index + size]
        ]

    def split_for_translation(self, text: str) -> list[str]:
        """Split at sentence/clause boundaries, then enforce the model token limit."""
        result: list[str] = []
        for sentence in self._sentence_split(text):
            if self._token_count(sentence) <= self.settings.max_source_tokens:
                result.append(sentence)
                continue
            clauses = re.split(r"(?<=[，,、：:])", sentence)
            current = ""
            for clause in clauses:
                candidate = current + clause
                if current and self._token_count(candidate) > self.settings.max_source_tokens:
                    if self._token_count(current) <= self.settings.max_source_tokens:
                        result.append(current.strip())
                    else:
                        result.extend(self._hard_token_split(current))
                    current = clause
                else:
                    current = candidate
            if current.strip():
                if self._token_count(current) <= self.settings.max_source_tokens:
                    result.append(current.strip())
                else:
                    result.extend(self._hard_token_split(current))
        return [piece for piece in result if piece]

    def _generate_batch(self, texts: Sequence[str]) -> list[str]:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.settings.max_source_tokens + 8,
        ).to(self.device)
        generation_options = {
            "num_beams": self.settings.num_beams,
            "max_new_tokens": self.settings.max_new_tokens,
            "length_penalty": self.settings.length_penalty,
            "early_stopping": True,
            "no_repeat_ngram_size": self.settings.no_repeat_ngram_size,
            "renormalize_logits": True,
        }
        with torch.inference_mode():
            generated = self.model.generate(**encoded, **generation_options)
        return [
            value.strip()
            for value in self.tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        ]

    def translate_segments(self, segments: Sequence[str]) -> list[str]:
        if not segments:
            return []
        translated: dict[str, str] = {}
        missing: list[str] = []
        seen: set[str] = set()
        for segment in segments:
            cached = self.cache.get(segment)
            if cached is not None:
                translated[segment] = cached
            elif segment not in seen:
                seen.add(segment)
                missing.append(segment)

        total = len(missing)
        cached_count = len(segments) - sum(1 for segment in segments if segment in missing)
        if total == 0:
            self._notify(len(segments), len(segments), "全部内容已从断点缓存恢复")
            return [translated[segment] for segment in segments]

        batch_size = self.settings.batch_size
        for start in range(0, total, batch_size):
            batch = missing[start : start + batch_size]
            outputs = self._generate_batch(batch)
            for source, output in zip(batch, outputs):
                translated[source] = output
                self.cache.put(source, output)
            completed = min(start + len(batch), total)
            self._notify(
                completed,
                total,
                f"正在翻译 {completed}/{total} 个片段（另有 {cached_count} 个缓存命中）",
            )
        return [translated[segment] for segment in segments]

    def translate_many_texts(self, texts: Sequence[str]) -> list[str]:
        plans: list[tuple[str, str, list[str], str, dict[str, str]]] = []
        all_segments: list[str] = []
        for text in texts:
            if not text or not text.strip():
                plans.append(("", text, [], "", {}))
                continue
            leading_match = re.match(r"^\s*", text)
            trailing_match = re.search(r"\s*$", text)
            leading = leading_match.group(0) if leading_match else ""
            trailing = trailing_match.group(0) if trailing_match else ""
            end = len(text) - len(trailing) if trailing else len(text)
            core = text[len(leading) : end]
            if not CJK_PATTERN.search(core):
                segments: list[str] = []
                replacements: dict[str, str] = {}
            else:
                protected_core, replacements = self._protect_glossary(core)
                segments = self.split_for_translation(protected_core)
                all_segments.extend(segments)
            plans.append((leading, core, segments, trailing, replacements))

        outputs = iter(self.translate_segments(all_segments))
        results: list[str] = []
        for leading, core, segments, trailing, replacements in plans:
            if not segments:
                results.append(leading + core + trailing)
                continue
            translated_segments = [next(outputs) for _ in segments]
            translated_core = self._restore_glossary(
                " ".join(translated_segments), replacements
            )
            results.append(leading + translated_core + trailing)
        return results

    def translate_text(self, text: str) -> str:
        lines = text.splitlines(keepends=True)
        if not lines:
            return ""
        bodies: list[str] = []
        endings: list[str] = []
        for line in lines:
            body = line.rstrip("\r\n")
            bodies.append(body)
            endings.append(line[len(body) :])
        translated = self.translate_many_texts(bodies)
        return "".join(value + ending for value, ending in zip(translated, endings))


def default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}.en{source.suffix}")


def default_cache_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.translation-cache.jsonl")


def _atomic_write_text(output: Path, text: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)


def _iter_table_paragraphs(table) -> Iterable:
    for row in table.rows:
        for cell in row.cells:
            yield from cell.paragraphs
            for nested in cell.tables:
                yield from _iter_table_paragraphs(nested)


def _iter_docx_paragraphs(document) -> Iterable:
    # Keep the OOXML element objects themselves alive.  Storing only id(...)
    # allows CPython to reuse an id after a temporary paragraph wrapper is
    # collected, which can incorrectly suppress an unrelated later paragraph.
    seen: set[object] = set()

    def emit(paragraphs: Iterable):
        for paragraph in paragraphs:
            identity = paragraph._p
            if identity not in seen:
                seen.add(identity)
                yield paragraph

    yield from emit(document.paragraphs)
    for table in document.tables:
        yield from emit(_iter_table_paragraphs(table))
    for section in document.sections:
        for container in (
            section.header,
            section.first_page_header,
            section.even_page_header,
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        ):
            yield from emit(container.paragraphs)
            for table in container.tables:
                yield from emit(_iter_table_paragraphs(table))


def _docx_paragraph_text(paragraph) -> str:
    return "".join(node.text or "" for node in paragraph._p.xpath(".//w:t"))


def _replace_docx_paragraph_text(paragraph, translation: str) -> None:
    text_nodes = paragraph._p.xpath(".//w:t")
    if text_nodes:
        text_nodes[0].text = translation
        for node in text_nodes[1:]:
            node.text = ""
    else:
        paragraph.add_run(translation)


def translate_docx(
    translator: AccurateTranslator,
    source: Path,
    output: Path,
) -> None:
    from docx import Document

    document = Document(source)
    targets = []
    source_texts = []
    for paragraph in _iter_docx_paragraphs(document):
        # Leave Word fields (TOC, page numbers, cross-references) intact.
        if paragraph._p.xpath(".//w:instrText"):
            continue
        text = _docx_paragraph_text(paragraph)
        if CJK_PATTERN.search(text):
            targets.append(paragraph)
            source_texts.append(text)
    translated = translator.translate_many_texts(source_texts)
    for paragraph, value in zip(targets, translated):
        _replace_docx_paragraph_text(paragraph, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    document.save(temporary)
    os.replace(temporary, output)


def translate_markdown(
    translator: AccurateTranslator,
    source_text: str,
) -> str:
    lines = source_text.splitlines(keepends=True)
    output_lines: list[str] = []
    in_fence = False
    for line in lines:
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        if body.lstrip().startswith("```"):
            in_fence = not in_fence
            output_lines.append(line)
            continue
        if in_fence or not CJK_PATTERN.search(body):
            output_lines.append(line)
            continue
        prefix_match = MARKDOWN_PREFIX.match(body)
        prefix, content = (
            (prefix_match.group(1), prefix_match.group(2))
            if prefix_match
            else ("", body)
        )
        pieces = MARKDOWN_PROTECTED.split(content)
        candidates = [
            piece
            for index, piece in enumerate(pieces)
            if index % 2 == 0 and CJK_PATTERN.search(piece)
        ]
        translations = iter(translator.translate_many_texts(candidates))
        rebuilt = []
        for index, piece in enumerate(pieces):
            if index % 2 == 0 and CJK_PATTERN.search(piece):
                rebuilt.append(next(translations))
            else:
                rebuilt.append(piece)
        output_lines.append(prefix + "".join(rebuilt) + ending)
    return "".join(output_lines)


def translate_document(
    source: Path,
    output: Optional[Path] = None,
    config_path: Path = DEFAULT_CONFIG,
    overwrite: bool = False,
    progress: Optional[ProgressCallback] = None,
    translator: Optional[AccurateTranslator] = None,
) -> Path:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    output = (output or default_output_path(source)).resolve()
    if source == output:
        raise ValueError("输出文件不能覆盖输入文件")
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出已存在：{output}；使用 --overwrite 才能替换")
    suffix = source.suffix.lower()
    if suffix not in {".txt", ".md", ".markdown", ".docx"}:
        raise ValueError("目前支持 .txt、.md、.markdown 和 .docx")
    cache_path = default_cache_path(output)
    if translator is None:
        translator = AccurateTranslator(config_path, cache_path, progress)
    else:
        translator.progress = progress
        translator.cache = TranslationCache(cache_path, translator.cache.signature)
    if suffix == ".docx":
        translate_docx(translator, source, output)
    else:
        source_text = source.read_text(encoding="utf-8-sig")
        translated = (
            translate_markdown(translator, source_text)
            if suffix in {".md", ".markdown"}
            else translator.translate_text(source_text)
        )
        _atomic_write_text(output, translated)
    if progress is not None:
        progress(1, 1, f"翻译完成：{output}")
    return output


def _console_progress(completed: int, total: int, message: str) -> None:
    percent = 100.0 * completed / max(total, 1)
    print(f"[{percent:6.2f}%] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="翻译一段文字并退出")
    group.add_argument("--input", type=Path, help="输入 TXT、Markdown 或 DOCX")
    parser.add_argument("--output", type=Path, help="文档输出路径")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is not None:
        output = translate_document(
            args.input,
            args.output,
            args.config,
            args.overwrite,
            _console_progress,
        )
        print(output)
        return
    translator = AccurateTranslator(args.config, progress=_console_progress)
    if args.text is not None:
        print(translator.translate_text(args.text))
        return

    print("===== 高精度中译英：可粘贴多行文字 =====")
    print("单独输入 END 开始翻译；输入 quit 退出。")
    while True:
        lines: list[str] = []
        print("中文 >>>", flush=True)
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                return
            if line.strip().lower() == "quit" and not lines:
                print("Bye!")
                return
            if line.strip() == "END":
                break
            lines.append(line)
        text = "\n".join(lines)
        if text.strip():
            print("英文 >>>")
            print(translator.translate_text(text))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
