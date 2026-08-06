# 高精度长文本与文档翻译

原来的 `run.py` 使用本项目从零训练的 1146 万参数教学模型。实际翻译请使用新的 OPUS-MT 路径；它保留原模型用于学习和实验，但默认不再让小模型承担正式文档翻译。

## 图形界面

双击 `start_accurate_translator.cmd`（旧入口 `run_gpu_test.cmd` 也会打开同一界面）。模型加载完成后：

1. 大段文字：粘贴到左侧，点击“翻译左侧文字”，右侧得到按原换行合并的英文。
2. 文档：点击“翻译文档…”，选择 `.docx`、`.txt`、`.md` 或 `.markdown`，再选择英文输出位置。
3. 专业术语：编辑 `glossary.json`。普通条目可写成 `"源术语": "target term"`；需要校正常见误译时，使用 `target` 和 `aliases` 对象。

首次使用需要联网下载约 312 MB 模型；以后从 Hugging Face 本地缓存加载。CUDA 可用时自动使用 RTX GPU 和 FP16。

## 命令行

翻译大段文字：

```powershell
.\run_gpu_test.cmd --text "需要翻译的大段中文"
```

翻译文档并自动生成 `原名.en.扩展名`：

```powershell
.\run_gpu_test.cmd --input "D:\资料\报告.docx"
```

指定输出文件：

```powershell
python accurate_translator.py --input "D:\资料\报告.docx" --output "D:\资料\report.en.docx"
```

重新生成已有输出时增加 `--overwrite`。每个文档旁会生成 `.translation-cache.jsonl`，记录已经完成的片段；程序中断后重新运行即可从缓存继续。

## 文档保留范围

- DOCX：保留段落样式、标题层级、列表、表格、分页、图片、页眉、页脚和 Word 字段。
- TXT：保留原换行和空行。
- Markdown：保留标题/列表前缀、代码块、行内代码、链接和 URL。
- 为了让整段译文语法完整，DOCX 同一段落内的复杂混合字体、局部粗体或局部颜色会统一继承该段第一个文本运行的样式。
- 扫描 PDF、复杂文本框、脚注和需要像素级原版复刻的 PDF 尚不在当前入口内。

## 准确度机制

- 使用预训练 `Helsinki-NLP/opus-mt-zh-en`，不是从零开始的小模型。
- 按中文句末标点切分，超长句再按从句和 token 上限切分。
- GPU 批处理，使用 5-beam search、长度惩罚和重复抑制。
- `glossary.json` 只在原文含对应中文术语时生效，避免全局错误替换。
- 文档缓存与模型版本、解码参数和术语表绑定；配置变化后会自动重新翻译。

运行验收：

```powershell
python accurate_smoke_test.py
```
