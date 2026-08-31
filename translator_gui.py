"""Foreground GUI for accurate long-text and document translation."""

from __future__ import annotations

import ctypes
import gc
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch

from accurate_translator import (
    DEFAULT_CONFIG,
    PROJECT_DIR,
    AccurateTranslator,
    default_output_path,
    detect_document_direction,
    detect_translation_direction,
    translate_document,
)


DIRECTION_LABELS = {
    "自动识别": "auto",
    "中译英": "zh-en",
    "英译中": "en-zh",
}
DIRECTION_NAMES = {value: key for key, value in DIRECTION_LABELS.items()}


class TranslatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MyTransformer 中英文互译 / 文档翻译（上下文增强版）")
        self.root.geometry("1280x820")
        self.root.minsize(920, 620)
        self.events: queue.Queue[tuple] = queue.Queue()
        self.translator: AccurateTranslator | None = None
        self.busy = True
        self._build_ui()
        self.root.after(100, self._poll_events)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("TButton", padding=(10, 6))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Hint.TLabel", font=("Microsoft YaHei UI", 9))

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        title_row = ttk.Frame(outer)
        title_row.pack(fill=tk.X)
        ttk.Label(
            title_row,
            text="高精度中英文互译",
            style="Title.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Label(
            title_row,
            text="双向 OPUS-MT · GPU · 自动识别 · 文档缓存 · 双向术语表",
            style="Hint.TLabel",
        ).pack(side=tk.LEFT, padx=(14, 0), pady=(6, 0))

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill=tk.X, pady=(12, 10))
        ttk.Label(toolbar, text="方向：").pack(side=tk.LEFT)
        self.direction_var = tk.StringVar(value="自动识别")
        self.direction_box = ttk.Combobox(
            toolbar,
            textvariable=self.direction_var,
            values=tuple(DIRECTION_LABELS),
            width=9,
            state="readonly",
        )
        self.direction_box.pack(side=tk.LEFT, padx=(0, 8))
        self.direction_box.bind("<<ComboboxSelected>>", self._on_direction_changed)
        self.direction_box.configure(state="disabled")
        self.reviewer_var = tk.BooleanVar(value=True)
        self.reviewer_checkbox = ttk.Checkbutton(
            toolbar,
            text="自动深度审校",
            variable=self.reviewer_var,
            command=self._on_reviewer_changed,
        )
        self.reviewer_checkbox.pack(side=tk.LEFT, padx=(0, 10))
        self.swap_button = ttk.Button(
            toolbar, text="⇄ 交换", command=self._swap_languages, state=tk.DISABLED
        )
        self.swap_button.pack(side=tk.LEFT, padx=(0, 12))
        self.translate_button = ttk.Button(
            toolbar, text="翻译左侧文字", command=self._translate_text, state=tk.DISABLED
        )
        self.translate_button.pack(side=tk.LEFT)
        self.document_button = ttk.Button(
            toolbar, text="翻译文档…", command=self._translate_document, state=tk.DISABLED
        )
        self.document_button.pack(side=tk.LEFT, padx=(8, 0))
        self.save_button = ttk.Button(
            toolbar, text="保存右侧译文…", command=self._save_output
        )
        self.save_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="清空", command=self._clear).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="复制译文", command=self._copy_output).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.train_button = ttk.Button(
            toolbar,
            text="训练增强模型…",
            command=self._launch_finetune,
            state=tk.DISABLED,
        )
        self.train_button.pack(side=tk.LEFT, padx=(22, 0))
        self.reload_button = ttk.Button(
            toolbar, text="重新加载模型", command=self._reload_model, state=tk.DISABLED
        )
        self.reload_button.pack(side=tk.LEFT, padx=(8, 0))

        pane = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(pane)
        right = ttk.Frame(pane)
        pane.add(left, weight=1)
        pane.add(right, weight=1)
        self.source_caption = tk.StringVar(value="原文（自动识别；可粘贴大段文字）")
        self.target_caption = tk.StringVar(value="译文")
        ttk.Label(left, textvariable=self.source_caption).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(right, textvariable=self.target_caption).pack(anchor=tk.W, pady=(0, 5))
        self.input_text = tk.Text(
            left,
            wrap=tk.WORD,
            undo=True,
            font=("Microsoft YaHei UI", 11),
            padx=10,
            pady=10,
        )
        self.output_text = tk.Text(
            right,
            wrap=tk.WORD,
            undo=True,
            font=("Microsoft YaHei UI", 11),
            padx=10,
            pady=10,
        )
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=(0, 6))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=(6, 0))

        status_row = ttk.Frame(outer)
        status_row.pack(fill=tk.X, pady=(10, 0))
        self.status = tk.StringVar(value="正在加载高精度翻译模型…")
        ttk.Label(status_row, textvariable=self.status).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(status_row, length=310, mode="indeterminate")
        self.progress.pack(side=tk.RIGHT)
        self.progress.start(12)

    def _emit_progress(self, completed: int, total: int, message: str) -> None:
        self.events.put(("progress", completed, total, message))

    def _load_model(self) -> None:
        try:
            translator = AccurateTranslator(
                DEFAULT_CONFIG,
                progress=self._emit_progress,
                direction="zh-en",
                reviewer_enabled=True,
            )
            self.events.put(("model_ready", translator))
        except Exception as error:  # surfaced in the foreground UI
            self.events.put(("error", f"模型加载失败：{error}"))

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = tk.DISABLED if busy or self.translator is None else tk.NORMAL
        self.translate_button.configure(state=state)
        self.document_button.configure(state=state)
        self.reload_button.configure(state=state)
        self.swap_button.configure(state=state)
        self.train_button.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.direction_box.configure(state="disabled" if busy else "readonly")
        self.reviewer_checkbox.configure(state=tk.DISABLED if busy else tk.NORMAL)

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, completed, total, message = event
                    self.status.set(message)
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=max(total, 1))
                    self.progress["value"] = completed
                elif kind == "model_ready":
                    self.translator = event[1]
                    self._set_busy(False)
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=1, value=1)
                    model_kind = (
                        "本地增强模型"
                        if self.translator.using_fine_tuned_model
                        else "官方基础模型"
                    )
                    self.status.set(
                        f"{self.translator.direction_label} {model_kind}已就绪："
                        f"{self.translator.device}；可自动识别或手动选择方向"
                    )
                    self._update_captions(self.translator.direction)
                elif kind == "text_done":
                    self.output_text.delete("1.0", tk.END)
                    self.output_text.insert("1.0", event[1])
                    self._set_busy(False)
                    self._update_captions(event[2])
                    self.status.set(
                        f"{DIRECTION_NAMES[event[2]]}文字翻译完成；{event[3]}"
                    )
                    self.progress["value"] = self.progress["maximum"]
                elif kind == "document_done":
                    self._set_busy(False)
                    self.status.set(f"文档翻译完成：{event[1]}；{event[2]}")
                    self.progress["value"] = self.progress["maximum"]
                    messagebox.showinfo("翻译完成", f"译文已保存到：\n{event[1]}")
                elif kind == "error":
                    self._set_busy(False)
                    self.progress.stop()
                    self.status.set(event[1])
                    messagebox.showerror("翻译失败", event[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _translate_text(self) -> None:
        text = self.input_text.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showwarning("没有内容", "请先在左侧粘贴或输入中文/英文。")
            return
        direction = self._selected_direction(text=text)
        reviewer_enabled = self.reviewer_var.get()
        self._update_captions(direction)
        self._set_busy(True)
        self.status.set(f"正在准备{DIRECTION_NAMES[direction]}并切分文字…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)

        def worker() -> None:
            try:
                translator = self._ensure_translator(direction, reviewer_enabled)
                translator.set_reviewer_enabled(reviewer_enabled)
                result = translator.translate_text(text)
                self.events.put(
                    ("text_done", result, direction, translator.review_summary)
                )
            except Exception as error:
                self.events.put(("error", f"文字翻译失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _translate_document(self) -> None:
        source_name = filedialog.askopenfilename(
            title="选择中文或英文文档",
            filetypes=[
                ("支持的文档", "*.docx *.txt *.md *.markdown"),
                ("Word 文档", "*.docx"),
                ("文本文件", "*.txt"),
                ("Markdown", "*.md *.markdown"),
                ("所有文件", "*.*"),
            ],
        )
        if not source_name:
            return
        source = Path(source_name)
        try:
            direction = self._selected_direction(source=source)
        except Exception as error:
            messagebox.showerror("无法识别文档", str(error))
            return
        self._update_captions(direction)
        suggested = default_output_path(source, direction)
        output_name = filedialog.asksaveasfilename(
            title=f"保存{('英文' if direction == 'zh-en' else '中文')}文档",
            initialdir=str(suggested.parent),
            initialfile=suggested.name,
            defaultextension=source.suffix,
            filetypes=[("同输入格式", f"*{source.suffix}"), ("所有文件", "*.*")],
        )
        if not output_name:
            return
        output = Path(output_name)
        reviewer_enabled = self.reviewer_var.get()
        self._set_busy(True)
        self.status.set("正在准备文档翻译…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)

        def worker() -> None:
            try:
                translator = self._ensure_translator(direction, reviewer_enabled)
                translator.set_reviewer_enabled(reviewer_enabled)
                result = translate_document(
                    source,
                    output,
                    DEFAULT_CONFIG,
                    overwrite=True,
                    progress=self._emit_progress,
                    translator=translator,
                    direction=direction,
                )
                self.events.put(
                    ("document_done", result, translator.review_summary)
                )
            except Exception as error:
                self.events.put(("error", f"文档翻译失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _save_output(self) -> None:
        text = self.output_text.get("1.0", "end-1c")
        if not text:
            messagebox.showwarning("没有译文", "右侧还没有可保存的译文。")
            return
        filename = filedialog.asksaveasfilename(
            title="保存译文",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
        )
        if filename:
            Path(filename).write_text(text, encoding="utf-8")
            self.status.set(f"译文已保存：{filename}")

    def _copy_output(self) -> None:
        text = self.output_text.get("1.0", "end-1c")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status.set("译文已复制到剪贴板")

    def _launch_finetune(self) -> None:
        direction = self._selected_direction(
            text=self.input_text.get("1.0", "end-1c")
        )
        script_name = (
            "start_accurate_finetune.cmd"
            if direction == "zh-en"
            else "start_accurate_finetune_en_zh.cmd"
        )
        script = PROJECT_DIR / script_name
        if not script.is_file():
            messagebox.showerror("无法启动", f"训练入口不存在：\n{script}")
            return
        proceed = messagebox.askokcancel(
            "训练增强模型",
            f"将训练{DIRECTION_NAMES[direction]}模型，并打开独立前台窗口，实时显示损失、进度、显存和预计剩余时间。\n\n"
            "训练期间请不要同时执行大文档翻译。质量门控通过后，回到此界面点击“重新加载模型”。",
        )
        if not proceed:
            return
        try:
            subprocess.Popen(
                ["cmd.exe", "/c", str(script)],
                cwd=str(PROJECT_DIR),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            self.status.set(f"{DIRECTION_NAMES[direction]}增强训练已在独立前台窗口启动")
        except OSError as error:
            messagebox.showerror("无法启动训练", str(error))

    def _reload_model(self) -> None:
        if self.busy:
            return
        direction = self._selected_direction(
            text=self.input_text.get("1.0", "end-1c")
        )
        previous = self.translator
        self.translator = None
        self._set_busy(True)
        self.reload_button.configure(state=tk.DISABLED)
        self.status.set("正在释放旧模型并加载最新的质量门控权重…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)

        def worker(old_translator=previous) -> None:
            try:
                self._release_translator(old_translator)
                translator = AccurateTranslator(
                    DEFAULT_CONFIG,
                    progress=self._emit_progress,
                    direction=direction,
                    reviewer_enabled=self.reviewer_var.get(),
                )
                self.events.put(("model_ready", translator))
            except Exception as error:
                self.events.put(("error", f"重新加载模型失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _selected_direction(
        self, text: str | None = None, source: Path | None = None
    ) -> str:
        selected = DIRECTION_LABELS[self.direction_var.get()]
        if selected != "auto":
            return selected
        if source is not None:
            return detect_document_direction(source)
        if text and text.strip():
            return detect_translation_direction(text)
        if self.translator is not None:
            return self.translator.direction
        return "zh-en"

    def _ensure_translator(
        self, direction: str, reviewer_enabled: bool
    ) -> AccurateTranslator:
        if self.translator is not None and self.translator.direction == direction:
            return self.translator
        previous = self.translator
        self.translator = None
        self._release_translator(previous)
        translator = AccurateTranslator(
            DEFAULT_CONFIG,
            progress=self._emit_progress,
            direction=direction,
            reviewer_enabled=reviewer_enabled,
        )
        self.translator = translator
        return translator

    @staticmethod
    def _release_translator(translator: AccurateTranslator | None) -> None:
        if translator is not None:
            translator.release()
        del translator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _on_reviewer_changed(self) -> None:
        if self.translator is not None:
            self.translator.set_reviewer_enabled(self.reviewer_var.get())
        state = "开启" if self.reviewer_var.get() else "关闭"
        self.status.set(f"自动深度审校已{state}；简单段落仍使用快速翻译")

    def _update_captions(self, direction: str) -> None:
        if direction == "zh-en":
            self.source_caption.set("中文原文（可直接粘贴大段文字）")
            self.target_caption.set("英文译文")
        else:
            self.source_caption.set("英文原文（可直接粘贴大段文字）")
            self.target_caption.set("中文译文")

    def _on_direction_changed(self, _event=None) -> None:
        direction = self._selected_direction(
            text=self.input_text.get("1.0", "end-1c")
        )
        self._update_captions(direction)
        selected = self.direction_var.get()
        if selected == "自动识别":
            self.status.set(f"自动识别当前内容为：{DIRECTION_NAMES[direction]}")
        else:
            self.status.set(f"已选择：{selected}；模型将在翻译时按需切换")

    def _swap_languages(self) -> None:
        if self.busy:
            return
        source = self.input_text.get("1.0", "end-1c")
        target = self.output_text.get("1.0", "end-1c")
        direction = self._selected_direction(text=source)
        opposite = "en-zh" if direction == "zh-en" else "zh-en"
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", target)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", source)
        self.direction_var.set(DIRECTION_NAMES[opposite])
        self._update_captions(opposite)
        self.status.set(f"已交换文本，方向切换为{DIRECTION_NAMES[opposite]}")

    def _clear(self) -> None:
        if self.busy:
            return
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.status.set("已清空")


def main() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    TranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
