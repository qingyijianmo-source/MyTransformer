# MyTransformer

**简体中文 | [English](README_EN.md)**

利用 PyTorch 框架「从 0.5 手写」Transformer(base 模型),尽可能地复现《[Attention Is All You Need](https://arxiv.org/abs/1706.03762)》原论文的结构与细节,并用它实现中→英机器翻译。没有使用 `nn.Transformer` 等现成高层封装,注意力、编码器、解码器、位置编码等模块均基于 PyTorch 基础算子自行搭建。

> 这是一个学习项目:希望通过亲手搭建的过程,逐行理解 Transformer 的工作原理。水平有限,目前仅简单实现了中译英功能,如有错漏,欢迎指正。

## 高精度长文本 / 文档翻译（推荐实际使用）

从零训练的小模型适合学习 Transformer，但不适合正式文档。项目现已加入预训练 OPUS-MT GPU 翻译入口，支持直接粘贴大段文字以及 `.docx`、`.txt`、`.md` 文档，包含自动分句、5-beam search、术语表、格式保留和断点缓存。

```powershell
.\start_accurate_translator.cmd
```

详细使用方法见 [README_ACCURATE.md](README_ACCURATE.md)。原来的手写模型、训练和 SCST 实验均继续保留。

---

## 特性

- **手写核心模块**:多头注意力、编码器/解码器层、位置编码,基于 PyTorch 基础算子搭建,未使用现成封装
- **ByteLevel BPE 分词器**:按语言分别训练(中/英)
- **Noam 学习率调度**:warmup + 反平方根衰减
- **设备自适应精度**:支持的 CUDA 设备使用 bf16,CPU 自动回退到 float32
- **早停机制**:基于阶段性验证 `Checker`
- **健壮的训练/推理入口**:支持断点权重加载、单句命令行推理和端到端冒烟测试

---

## 项目结构

```
MyTransformer/
├── parts.py          # 模型组件:注意力、编码器/解码器层、Embedding 等
├── build.py          # 模型装配 + build_transformer 工厂函数
├── train_test.py     # 数据管线、分词器训练、训练与验证循环
├── run.py            # 交互式中译英推理(贪心解码)
├── accurate_translator.py # 高精度长文本/文档翻译核心
├── translator_gui.py # 大段文字与文档翻译图形界面
├── finetune_accurate.py # 当前 OPUS-MT 的 GPU 微调 + 质量门控
├── config.finetune.json # 快速增强训练配置
├── glossary.json     # 可编辑专业术语表
├── start_accurate_translator.cmd # 推荐启动入口
├── start_accurate_finetune.cmd # 前台显示增强训练进度
├── smoke_test.py     # 小模型端到端快速验证
├── prepare_corpus.py # 下载、清洗并合并通用 OPUS-100 语料
├── prepare_iwslt_fast.py # 准备干净、紧凑的 IWSLT TED 语料
├── fast_rl_pipeline.py # 快速监督训练 + SCST 奖励微调
├── rl_finetune.py    # Self-Critical 策略梯度实现
├── config.json       # 全部超参数,共五块
├── config.general.json # 通用增强语料的独立配置
├── data/             # OPUS OpenOffice en-zh 数据集
├── requirements.txt
└── LICENSE           # MIT
```

`output/`(分词器 + 模型权重)由训练在本地生成,**不**纳入 git。

---

## 环境要求

- Python ≥ 3.10
- 推荐 CUDA GPU(开发机为 RTX 4060 Laptop 8GB;bf16 需 Ampere 及以上架构)

安装依赖:

```bash
pip install -r requirements.txt
```

NVIDIA GPU 环境建议安装 CUDA 13.0 版 PyTorch(无需另装 CUDA Toolkit):

```bash
pip install -r requirements-gpu.txt
```

---

## 快速开始

### 0. 快速验证

建议先运行约十几秒的端到端检查。它会读取真实数据,复用现有分词器(缺失时临时训练),并完成一次小模型的前向、反向和解码,但不会写入模型权重:

```bash
python smoke_test.py
```

看到 `SMOKE TEST PASSED` 即表示数据、分词器、模型和训练链路可以正常工作。

### 1. 训练

```bash
python train_test.py
```

该脚本会:

1. 加载并划分数据集(`data/`)
2. 训练 zh / en BPE 分词器 → `output/tokenizers/`
3. 构建 Transformer,以 Noam 调度和设备自适应精度训练
4. 保存最优权重 → `output/checkpoints/best_state_from_checker.pt`

已有分词器会自动复用。如需重新训练分词器,使用:

```bash
python train_test.py --retrain-tokenizers
```

也可以从自动保存的完整训练状态精确续训(包含优化器、学习率和 batch 位置):

```bash
python train_test.py --resume output/checkpoints/last_training_state.pt
```

脚本每隔 `save_steps` 个优化器 step 自动更新该状态。普通 `epoch_*.pt` 仍可加载模型权重,但会重新初始化优化器。

安全暂停当前训练(当前 step 完成后保存再退出):

```bash
python pause_training.py
```

按当前 8000 词表计算,默认配置约 5640 万参数,CPU 全量训练仍会很慢,建议使用 CUDA GPU。

### 2. 推理

```bash
python run.py
```

也可直接翻译一句并退出:

```bash
python run.py --text "你好吗"
```

仓库不附带训练好的权重;推理前必须先生成 `output/checkpoints/best_state_from_checker.pt`,或通过 `--checkpoint` 指定兼容的权重。

交互式翻译:

```
=====中译英开始!  输入'quit'退出=====
中文>>>你好吗
英文: how are you
```

---

## 配置说明

所有超参数集中在 `config.json`,共五块:

| 块 | 内容 |
|---|---|
| `model` | `d_model`、`d_hidden`、`num_heads`、`drop_prob`、`num_encode_layers`、`num_decode_layers`、`source_seq_len`、`target_seq_len` |
| `tokenizer` | `vocab_size`、`min_frequency`、`tokenizers_dir` |
| `data` | `dataset_path`、`source_language`、`target_language`、`train_size`、`test_size`、`seed` |
| `train` | `batch_size`、`num_epochs`、`warmup`、`warmup_factor`、`label_smoothing`、`dynamic_padding`、`gradient_accumulation_steps`、`save_steps`、`max_grad_norm`、`num_workers`、`check_steps`、`patience_times`、`min_progress_value`、`checkpoints_dir` |
| `run` | 分词器文件名、`selected_checkpoint` |

默认模型配置对齐论文 **base** 模型:`d_model=512`、`d_hidden=2048`、`num_heads=8`、6 层编码器 + 6 层解码器。

### 通用语料增强

生成清洗、去重后的 OpenOffice + OPUS-100 合并语料:

```bash
python prepare_corpus.py
```

增强配置仍使用手写 Transformer 和 12000 词表,但针对低资源语料调整为 `d_model=256`、4 层编码器/解码器并共享目标 Embedding/投影权重,共约 1352 万参数。动态 padding 与梯度累积将有效 batch 提升到 96,训练更稳定也更快。新权重写入独立的 `checkpoints_general_v2`,不会覆盖基础版本或旧增强权重:

```bash
python train_test.py --config config.general.json --retrain-tokenizers
python run.py --config config.general.json
```

暂停后从最近的完整状态继续:

```bash
python train_test.py --config config.general.json --resume output/checkpoints_general_v2/last_training_state.pt
```

推理默认启用重复惩罚、3-gram 去重和特殊 token 屏蔽,避免未充分训练的模型无限重复同一片段。

语料来源和过滤统计记录在 `data/combined_general_en_zh.metadata.json`。OPUS-100 数据集卡没有给出统一许可证，增强语料建议仅用于本地学习和实验。

### 快速训练 + 奖励微调

`config.fast_rl.json` 使用 10 万条清洗后的 IWSLT 2017 中英 TED 句对、约 1146 万参数模型、batch 96 和六轮监督训练。监督阶段完成后,自动在 1024 条样本上执行一轮 SCST(Self-Critical Sequence Training):以贪心译文作为 baseline,用采样译文相对 baseline 的 1/2-gram F1 奖励优势更新策略,同时保留少量交叉熵损失防止策略漂移。SCST 每 16 个批次在固定留出集上检查一次,只保存奖励真正提高的候选;若所有强化学习更新都退化,`best_state_scst.pt` 会自动回退为监督基线。

```bash
python fast_rl_pipeline.py --config config.fast_rl.json
python run.py --config config.fast_rl.json
```

安全暂停与精确续训:

```bash
python pause_training.py
python fast_rl_pipeline.py --config config.fast_rl.json --resume output/checkpoints_iwslt_fast_v2/last_training_state.pt
```

IWSLT 2017 语料许可证为 CC BY-NC-ND 4.0,仅用于非商业实验。序列级奖励训练参考 Minimum Risk Training for NMT 与 SCST。

---

## 致谢

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原论文
- [OPUS OpenOffice](https://opus.nlpl.eu/) — 本项目使用的 en-zh 平行语料
- [PyTorch](https://pytorch.org/) / [Hugging Face tokenizers & datasets](https://huggingface.co/)

---

## License

[MIT](LICENSE) © 2026 K2etn
