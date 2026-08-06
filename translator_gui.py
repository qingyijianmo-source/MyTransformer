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
    translate_document,
)


class TranslatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MyTransformer 高精度长文本 / 文档翻译")
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
            text="高精度中译英",
            style="Title.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Label(
            title_row,
            text="OPUS-MT · GPU · 可质量门控微调 · 长文档断点缓存 · glossary.json",
            style="Hint.TLabel",
        ).pack(side=tk.LEFT, padx=(14, 0), pady=(6, 0))

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill=tk.X, pady=(12, 10))
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
            toolbar, text="训练增强模型…", command=self._launch_finetune
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
        ttk.Label(left, text="中文（可直接粘贴大段文字）").pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(right, text="英文译文").pack(anchor=tk.W, pady=(0, 5))
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
            font=("Segoe UI", 11),
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
            translator = AccurateTranslator(DEFAULT_CONFIG, progress=self._emit_progress)
            self.events.put(("model_ready", translator))
        except Exception as error:  # surfaced in the foreground UI
            self.events.put(("error", f"模型加载失败：{error}"))

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = tk.DISABLED if busy or self.translator is None else tk.NORMAL
        self.translate_button.configure(state=state)
        self.document_button.configure(state=state)
        self.reload_button.configure(state=state)

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
                        f"{model_kind}已就绪：{self.translator.device}；可粘贴文字或选择文档"
                    )
                elif kind == "text_done":
                    self.output_text.delete("1.0", tk.END)
                    self.output_text.insert("1.0", event[1])
                    self._set_busy(False)
                    self.status.set("文字翻译完成")
                    self.progress["value"] = self.progress["maximum"]
                elif kind == "document_done":
                    self._set_busy(False)
                    self.status.set(f"文档翻译完成：{event[1]}")
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
            messagebox.showwarning("没有内容", "请先在左侧粘贴或输入中文。")
            return
        if self.translator is None:
            return
        self._set_busy(True)
        self.status.set("正在切分并翻译文字…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)

        def worker() -> None:
            try:
                result = self.translator.translate_text(text)
                self.events.put(("text_done", result))
            except Exception as error:
                self.events.put(("error", f"文字翻译失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _translate_document(self) -> None:
        source_name = filedialog.askopenfilename(
            title="选择中文文档",
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
        suggested = default_output_path(source)
        output_name = filedialog.asksaveasfilename(
            title="保存英文文档",
            initialdir=str(suggested.parent),
            initialfile=suggested.name,
            defaultextension=source.suffix,
            filetypes=[("同输入格式", f"*{source.suffix}"), ("所有文件", "*.*")],
        )
        if not output_name:
            return
        output = Path(output_name)
        self._set_busy(True)
        self.status.set("正在准备文档翻译…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)

        def worker() -> None:
            try:
                result = translate_document(
                    source,
                    output,
                    DEFAULT_CONFIG,
                    overwrite=True,
                    progress=self._emit_progress,
                    translator=self.translator,
                )
                self.events.put(("document_done", result))
            except Exception as error:
                self.events.put(("error", f"文档翻译失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _save_output(self) -> None:
        text = self.output_text.get("1.0", "end-1c")
        if not text:
            messagebox.showwarning("没有译文", "右侧还没有可保存的译文。")
            return
        filename = filedialog.asksaveasfilename(
            title="保存英文译文",
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
        script = PROJECT_DIR / "start_accurate_finetune.cmd"
        if not script.is_file():
            messagebox.showerror("无法启动", f"训练入口不存在：\n{script}")
            return
        proceed = messagebox.askokcancel(
            "训练增强模型",
            "将打开独立的前台训练窗口，实时显示损失、进度、显存和预计剩余时间。\n\n"
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
            self.status.set("增强训练已在独立前台窗口启动")
        except OSError as error:
            messagebox.showerror("无法启动训练", str(error))

    def _reload_model(self) -> None:
        if self.busy:
            return
        previous = self.translator
        self.translator = None
        self._set_busy(True)
        self.reload_button.configure(state=tk.DISABLED)
        self.status.set("正在释放旧模型并加载最新的质量门控权重…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)

        def worker(old_translator=previous) -> None:
            try:
                if old_translator is not None:
                    old_translator.model.to("cpu")
                del old_translator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                translator = AccurateTranslator(DEFAULT_CONFIG, progress=self._emit_progress)
                self.events.put(("model_ready", translator))
            except Exception as error:
                self.events.put(("error", f"重新加载模型失败：{error}"))

        threading.Thread(target=worker, daemon=True).start()

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
