# 高精度中英文互译与文档翻译

[返回项目首页](README.md)

实际翻译使用两套独立的 OPUS-MT：中译英继续使用当前质量门控增强权重，英译中使用 `Helsinki-NLP/opus-mt-en-zh` 及其独立增强权重。原来的手写 Transformer 只用于学习和实验，说明见 [手写模型文档](README_CUSTOM_TRANSFORMER.md)。

## 图形界面

双击 `start_accurate_translator.cmd`，或运行：

```powershell
python translator_gui.py
```

界面提供三种方向：

- **自动识别**：按照输入中的中英文字符比例选择方向。
- **中译英**：强制使用中译英模型。
- **英译中**：强制使用英译中模型并输出简体中文。

“⇄ 交换”会交换左右文本并反转方向。切换方向时，程序先释放旧模型的 GPU 显存，再按需加载另一个模型。

大段文字可直接粘贴到左侧；文档按钮支持 `.docx`、`.txt`、`.md` 和 `.markdown`。自动模式会先读取文档文字样本，再生成 `.en` 或 `.zh` 输出文件名。

## 命令行

自动识别文字方向：

```powershell
.\run_gpu_test.cmd --text "需要翻译的大段中文"
.\run_gpu_test.cmd --text "A long English passage to translate."
```

手动指定方向：

```powershell
python accurate_translator.py --direction zh-en --text "需要翻译的中文"
python accurate_translator.py --direction en-zh --text "Translate this into Chinese."
```

自动识别并翻译文档：

```powershell
python accurate_translator.py --input "D:\资料\报告.docx"
```

指定方向和输出文件：

```powershell
python accurate_translator.py `
  --direction en-zh `
  --input "D:\资料\report.docx" `
  --output "D:\资料\报告.zh.docx" `
  --overwrite
```

每个输出文档旁会生成 `.translation-cache.jsonl`。缓存签名包含翻译方向、模型版本、增强权重、解码参数和术语表，因此中译英与英译中不会互相复用错误缓存。

## 双向专业术语

- `glossary.json`：中文源术语 → 批准的英文译法。
- `glossary.en_zh.json`：英文源术语 → 批准的中文译法。

普通条目可以写成字符串；需要统一常见误译时使用 `target` 和 `aliases`：

```json
{
  "machine translation": {
    "target": "机器翻译",
    "aliases": ["机械翻译"]
  }
}
```

中译英允许将批准的英文术语注入中文源句，再做译后统一。英译中不会把中文提前混入英文原文，而是在生成后统一已知误译，避免模型漏掉混合语言片段。

## 分别增强两个方向

中译英：

```powershell
.\start_accurate_finetune.cmd
```

英译中：

```powershell
.\start_accurate_finetune_en_zh.cmd
```

也可以先准备更丰富的本地语料：

```powershell
python prepare_iwslt_fast.py
python prepare_corpus.py
```

两个方向都采用：

- 12,000 对清洗、去重的训练句对；
- 256 对训练过程未见过的验证句对；
- FP16、微批量 8、梯度累积 4；
- 200 个优化步骤，每 100 步保存恢复断点；
- 训练前后验证损失质量门控。

当前开发机验收：

| 方向 | 基础验证损失 | 增强后 | 结果 |
|---|---:|---:|---|
| 中译英 | 1.60919 | 1.53671 | 通过并启用 |
| 英译中 | 2.39090 | 2.28609 | 通过并启用 |

权重分别写入 `output/accurate_finetuned/best` 和 `output/accurate_finetuned_en_zh/best`。训练中断后重新启动会从各自最近的 `resume` 断点继续；增加 `--fresh` 可从官方模型重新开始。

## 文档保留范围

- DOCX：保留段落、标题层级、列表、表格、分页、图片、页眉、页脚和 Word 字段。
- TXT：保留原换行和空行。
- Markdown：保留标题/列表前缀、代码块、行内代码、链接和 URL。
- DOCX 同一段落内的复杂局部字体、粗体或颜色可能统一继承第一个文本运行的样式。
- PDF、扫描 OCR、复杂文本框和脚注尚未接入当前入口。

## 长文档原理与边界

程序先按段落和句末标点切分，超长句再按从句和 Token 上限切分；随后在 GPU 上批量执行 5 路 Beam Search，最后合并并写回原文档结构。

这种方式可以处理很长的文件，但模型不会一次读取整篇文档，因此跨段代词、人名一致性和全文语境仍可能需要人工复核。法律、医疗、合同等高风险文档不应直接把机器译文当成最终版本。

## 验收

同时检查自动识别、中译英增强模型和英译中增强模型：

```powershell
python accurate_smoke_test.py
```

看到 `BIDIRECTIONAL ACCURATE TRANSLATION TEST PASSED` 即表示双向 CUDA 翻译链路正常。
