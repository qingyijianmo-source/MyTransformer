# MyTransformer Enhanced

**简体中文 | [English](README_EN.md)**

面向实际使用的 GPU 中英文互译工具：支持自动识别方向、大段文字和文档翻译、双向专业术语表、断点缓存，以及带质量门控的 OPUS-MT 微调。同时保留一套从基础算子搭建的 Transformer，供学习网络结构和训练流程。

> 本仓库是基于 [K2etn/MyTransformer](https://github.com/K2etn/MyTransformer) 的二次开发，不是上游项目的官方版本。上游的手写 Transformer 代码与署名完整保留；本仓库新增了实际翻译、文档处理、GPU 微调、质量验证和图形界面。

## 两条功能路线

| 路线 | 适合场景 | 模型 | 主要入口 |
|---|---|---|---|
| **高精度文档互译（推荐）** | 中英大段文字、DOCX、TXT、Markdown | 两个方向各使用 7,794 万参数 OPUS-MT，可分别微调 | `start_accurate_translator.cmd` |
| **手写 Transformer 实验** | 学习注意力、编码器/解码器、训练与强化学习 | 从零训练的自定义 Transformer | `train_test.py` / `fast_rl_pipeline.py` |

两条路线相互独立：训练手写模型不会改变高精度翻译界面；增强 OPUS-MT 也不会覆盖手写模型检查点。

## 当前增强版功能

- 中文与英文大段文字双向翻译，保留原始换行。
- 自动识别、中译英、英译中三种方向模式，并可一键交换文本。
- 支持 `.docx`、`.txt`、`.md` 和 `.markdown`。
- DOCX 保留段落、标题、列表、表格、图片、页眉和页脚。
- NVIDIA GPU + FP16 批处理，默认使用 5 路 Beam Search。
- 长文本按句子和 Token 上限自动切分。
- `glossary.json` 与 `glossary.en_zh.json` 分别维护两个方向的术语。
- JSONL 断点缓存，中断后可继续文档翻译。
- OPUS-MT 低学习率微调，实时显示 loss、进度、显存和 ETA。
- 独立验证集质量门控：候选模型退化时不会替换当前模型。

## 快速开始

### 1. 安装准确翻译依赖

```powershell
python -m pip install -r requirements-accurate.txt
```

需要训练原始手写模型时，再安装：

```powershell
python -m pip install -r requirements.txt
```

Windows + NVIDIA GPU 可以根据本机 CUDA/PyTorch 环境使用 `requirements-gpu.txt`。项目会自动选择 CUDA；CUDA 不可用时，文档翻译回退到 CPU。

### 2. 启动翻译界面

```powershell
.\start_accurate_translator.cmd
```

也可以使用跨平台 Python 入口：

```bash
python translator_gui.py
```

首次使用某个方向时会下载对应的固定版本 OPUS-MT 模型，以后直接使用本地缓存。两个模型按需切换，不会同时长期占用 GPU 显存。

### 3. 翻译文档

在界面中点击“翻译文档…”，或者使用命令行：

```powershell
python accurate_translator.py --input "D:\资料\报告.docx"
```

程序默认自动判断文档语言，并生成 `.en` 或 `.zh` 输出。也可以手动指定方向：

```powershell
python accurate_translator.py --direction zh-en --text "需要翻译的中文"
python accurate_translator.py --direction en-zh --text "Text to translate into Chinese."
```

指定输出位置：

```powershell
python accurate_translator.py `
  --input "D:\资料\报告.docx" `
  --output "D:\资料\report.en.docx" `
  --overwrite
```

详细说明见 [高精度翻译使用手册](README_ACCURATE.md)。

## 增强当前 OPUS-MT

准备更丰富的本地中英语料（可选，但推荐）：

```powershell
python prepare_iwslt_fast.py
python prepare_corpus.py
```

随后在翻译界面选择方向并点击“训练增强模型…”，或分别运行：

```powershell
.\start_accurate_finetune.cmd
.\start_accurate_finetune_en_zh.cmd
```

默认配置针对 RTX 4060 Laptop 8GB：

- 从本地语料选择 12,000 对清洗、去重后的训练句对；
- 另外保留 256 对独立验证数据；
- FP16、微批量 8、梯度累积 4、200 个优化步骤；
- 每 100 步保存恢复断点；
- 只有验证损失不高于官方基础模型时才启用新权重。

当前开发机的留出集验收结果：中译英 `1.60919 → 1.53671`，英译中 `2.39090 → 2.28609`。这些是本地质量门控结果，不代表通用公开基准。两个方向的增强权重分别保存在 `output/accurate_finetuned/best` 和 `output/accurate_finetuned_en_zh/best`，不会提交到 Git。

## 与上游版本的区别

| 部分 | 上游 `K2etn/MyTransformer` | 本仓库增强版 |
|---|---|---|
| 主要定位 | 从基础算子学习 Transformer | 可实际使用的长文本/文档翻译，同时保留教学代码 |
| 默认翻译核心 | 本地从零训练的中译英模型 | 双向固定版本 OPUS-MT + 两套可选本地增强权重 |
| 文档输入 | 单句交互为主 | DOCX、TXT、Markdown 和大段粘贴文本 |
| 解码与稳定性 | 贪心解码 | Beam Search、重复抑制、术语校正、断点缓存 |
| 模型训练 | 监督训练 | 保留原训练，并新增 SCST 实验和 OPUS-MT 质量门控微调 |
| 使用界面 | 命令行 | 前台双栏 GUI + 文档进度显示 |

原始手写模型的结构、训练、续训、SCST 和配置说明已单独整理到 [手写 Transformer 实验说明](README_CUSTOM_TRANSFORMER.md)，不再与实际翻译入口混排。

## 项目结构

```text
MyTransformer/
├── 实际翻译
│   ├── accurate_translator.py       # 长文本/文档翻译核心
│   ├── translator_gui.py            # 前台双栏图形界面
│   ├── finetune_accurate.py         # OPUS-MT GPU 微调和质量门控
│   ├── config.accurate.json         # 翻译与解码配置
│   ├── config.finetune.json         # 快速增强训练配置
│   ├── config.finetune.en_zh.json   # 英译中增强训练配置
│   ├── glossary*.json               # 双向专业术语表
│   ├── start_accurate_translator.cmd
│   └── start_accurate_finetune*.cmd
├── 手写模型实验
│   ├── parts.py / build.py          # Transformer 基础组件与装配
│   ├── train_test.py / run.py       # 训练、验证与推理
│   ├── fast_rl_pipeline.py          # 监督训练 + SCST
│   ├── rl_finetune.py               # Self-Critical 策略梯度
│   └── config*.json                 # 各训练方案配置
├── data/                            # 示例或本地生成的平行语料
├── README_ACCURATE.md               # 实际翻译详细手册
├── README_CUSTOM_TRANSFORMER.md     # 手写模型实验说明
└── LICENSE
```

## 验证

检查高精度翻译及增强权重是否正确加载：

```powershell
python accurate_smoke_test.py
```

检查原始手写模型的前向、反向和解码链路：

```powershell
python smoke_test.py
```

## 已知边界

- 超长文档会分段翻译，无法完整记住全文所有跨段指代。
- PDF、扫描件 OCR、复杂文本框和脚注尚未接入当前入口。
- 通用模型不能保证法律、医疗、合同等高风险内容无需人工复核。
- GitHub 仓库不包含训练检查点；首次切换到某个方向时下载对应基础模型，本地微调后生成增强权重。

## 来源与许可

- 上游项目：[K2etn/MyTransformer](https://github.com/K2etn/MyTransformer)
- 预训练模型：[中译英 opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) / [英译中 opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
- Transformer 论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 训练语料来源包括 [OPUS](https://opus.nlpl.eu/) 与 IWSLT；请分别遵守相应数据集许可。

代码按 [MIT License](LICENSE) 发布。上游版权声明与本仓库修改者声明均保留在许可证中。
