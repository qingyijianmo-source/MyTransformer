# MyTransformer Enhanced

**简体中文 | [English](README_EN.md)**

面向实际使用的 GPU 中英文互译工具：使用 OPUS-MT 快速初译，只把低置信度段落交给可选的本地 4-bit Qwen 审校器；支持大段文字、文档级术语上下文、断点缓存，以及带综合质量门控的 OPUS-MT 微调。同时保留一套从基础算子搭建的 Transformer，供学习网络结构和训练流程。

> 本仓库是基于 [K2etn/MyTransformer](https://github.com/K2etn/MyTransformer) 的二次开发，不是上游项目的官方版本。上游的手写 Transformer 代码与署名完整保留；本仓库新增了实际翻译、文档处理、GPU 微调、质量验证和图形界面。

## 两条功能路线

| 路线 | 适合场景 | 模型 | 主要入口 |
|---|---|---|---|
| **高精度文档互译（推荐）** | 中英大段文字、DOCX、TXT、Markdown | 7,794 万参数 OPUS-MT + 可选 4-bit Qwen3-4B 审校 | `start_accurate_translator.cmd` |
| **手写 Transformer 实验** | 学习注意力、编码器/解码器、训练与强化学习 | 从零训练的自定义 Transformer | `train_test.py` / `fast_rl_pipeline.py` |

两条路线相互独立：训练手写模型不会改变高精度翻译界面；增强 OPUS-MT 也不会覆盖手写模型检查点。

## 当前增强版功能

- 中文与英文大段文字双向翻译，保留原始换行。
- 自动识别、中译英、英译中三种方向模式，并可一键交换文本。
- 支持 `.docx`、`.txt`、`.md` 和 `.markdown`。
- DOCX 保留段落、标题、列表、表格、图片、页眉和页脚。
- NVIDIA GPU + FP16 批处理，默认使用 5 路 Beam Search。
- 优先保留完整段落上下文，仅在超过 Token 上限时按句组合切分。
- 自动检测长难句、多义词、重复、未翻译残留、异常长度和数字/缩写丢失，只审校疑难段落。
- 审校失败、超时、返回格式异常或遗漏事实时自动退回 OPUS 初译，文档任务继续运行。
- 翻译前抽取文档专名、缩写和术语，并向审校器传递相邻段落上下文。
- `glossary.json` 与 `glossary.en_zh.json` 分别维护两个方向的术语。
- 英译中支持带语境条件的多义词消歧，避免建筑 `mortar` 被机械译成“迫击炮”。
- JSONL 断点缓存，中断后可继续文档翻译。
- OPUS-MT 低学习率微调，实时显示 loss、进度、显存和 ETA。
- 独立验证集综合质量门控：chrF++、普通句、术语、数字、严重错误与 loss 任一不合格都不会替换当前模型。

## 快速开始

### 1. 安装准确翻译依赖

```powershell
python -m pip install -r requirements-accurate.txt
```

首次启用“自动深度审校”还需要执行一次：

```powershell
.\prepare_local_reviewer.cmd
```

它会安装 4-bit 依赖并下载 `Qwen/Qwen3-4B-Instruct-2507` 到本机缓存。未准备审校器时仍可正常使用 OPUS-MT，疑难段落会自动回退初译。

项目启动脚本已将 Hugging Face 模型/数据缓存固定到 D 盘纯英文目录 `D:\MyTransformer_HF_Cache`，不会再次占用 C 盘，也可避免 Windows SentencePiece 的中文路径问题；该目录不在 Git 仓库内，不会提交模型权重。

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

随后可用一个前台监视窗口依次训练两个方向各 20 轮：

```powershell
.\start_bidirectional_finetune_20_epochs.cmd
```

也可以在翻译界面选择方向并点击“训练增强模型…”，或分别运行：

```powershell
.\start_accurate_finetune.cmd
.\start_accurate_finetune_en_zh.cmd
```

默认配置针对 RTX 4060 Laptop 8GB：

- 从本地语料选择 12,000 对清洗、去重、黑名单过滤后的训练句对，并限制军事等偏科领域占比；
- 另外保留 256 对独立验证数据；
- FP16、微批量 8、梯度累积 4，每个方向 20 轮（7,500 个优化步骤）；
- 每 375 步（每轮）保存恢复断点，日志实时显示轮次、loss、显存与 ETA；
- 每轮运行独立验证、保留最佳轮并早停；最终还必须通过冻结评测集的综合指标才会启用。

仓库中的当前 `best` 权重仍是早期短程候选；20 轮配置只有完整运行并通过新质量门后才会取代它。冻结测试基线保存在 `eval/baselines/opus_context_v1.json`，不能把训练 loss 当成最终翻译准确率。两个方向的增强权重分别保存在 `output/accurate_finetuned/best` 和 `output/accurate_finetuned_en_zh/best`，不会提交到 Git。

运行冻结评测：

```powershell
python evaluate_translations.py --direction both --split test --reviewer off
python evaluate_translations.py --direction both --split test --reviewer on `
  --baseline eval\baselines\opus_context_v1.json
```

当前 RTX 4060 8GB 冻结测试结果：中译英 chrF++ `68.52 → 76.22`，英译中 `31.16 → 37.19`；两个方向的术语准确率、数字/专名保留率均为 `100%`，严重错译数为 `0`，配对 bootstrap 的 95% 区间下界均大于 0。测试集规模仍较小，这些数字用于项目回归，不等同于大型公开基准。

如需研究对照，可运行 `evaluate_nllb_baseline.py --direction en-zh --allow-download`。NLLB-600M 仅作为离线基线：其模型卡注明 CC BY-NC、研究用途且输入上限为 512 Token，不进入默认文档翻译管线。

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
│   ├── context_rules.en_zh.json      # 英译中语境消歧与危险错译修复
│   ├── translation_quality.py        # 低置信度检测、事实保护、文档上下文
│   ├── local_reviewer.py             # 4-bit Qwen 审校与安全回退
│   ├── translation_eval.py           # chrF++、术语、保留率和晋级门
│   ├── eval/                         # 冻结评测集和当前基线
│   ├── tests/                        # CPU 回归测试
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

- 审校器按需加载时速度会下降；RTX 4060 8GB 默认使用 4-bit 和有限上下文以避免 OOM。
- PDF、扫描件 OCR、复杂文本框和脚注尚未接入当前入口。
- 通用模型不能保证法律、医疗、合同等高风险内容无需人工复核。
- GitHub 仓库不包含训练检查点；首次切换到某个方向时下载对应基础模型，本地微调后生成增强权重。

## 来源与许可

- 上游项目：[K2etn/MyTransformer](https://github.com/K2etn/MyTransformer)
- 预训练模型：[中译英 opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) / [英译中 opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
- 可选审校模型：[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)（Apache 2.0）
- Transformer 论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 训练语料来源包括 [OPUS](https://opus.nlpl.eu/) 与 IWSLT；IWSLT 本地元数据标注为 CC BY-NC-ND，OPUS-100 没有统一覆盖全部子语料的许可证，必须分别核验后使用。

代码按 [MIT License](LICENSE) 发布；该许可证不自动覆盖下载的模型、微调权重或训练语料。上游版权声明与本仓库修改者声明均保留在许可证中。
