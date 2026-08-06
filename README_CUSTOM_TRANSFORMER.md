# 手写 Transformer 实验说明

[返回项目首页](README.md)

这一部分保留并扩展了上游 [K2etn/MyTransformer](https://github.com/K2etn/MyTransformer) 的学习路线：不调用 `torch.nn.Transformer`，使用 PyTorch 基础模块搭建注意力、位置编码、编码器和解码器，并从中英平行语料开始训练。

> 这是教学和实验模型，不是当前文档翻译 GUI 的默认模型。需要准确翻译长文档时，请使用首页推荐的 OPUS-MT 路线。

## 核心实现

- 多头自注意力与交叉注意力。
- 编码器、解码器、前馈网络和残差连接。
- 正弦位置编码。
- 中文和英文独立 ByteLevel BPE 分词器。
- Noam warmup 与反平方根学习率衰减。
- CUDA bf16/FP16 与 CPU float32 回退。
- 动态 padding、梯度累积、裁剪和阶段验证。
- 完整训练状态保存、安全暂停和断点续训。

主要文件：

| 文件 | 作用 |
|---|---|
| `parts.py` | 注意力、编码器/解码器层、Embedding 等组件 |
| `build.py` | 模型装配与 `build_transformer` 工厂函数 |
| `train_test.py` | 数据、分词器、训练和验证循环 |
| `run.py` | 自定义模型中译英推理 |
| `config.json` | 论文 base 风格配置 |
| `config.general.json` | 通用语料小模型配置 |
| `config.fast_rl.json` | 快速监督训练与 SCST 配置 |
| `fast_rl_pipeline.py` | 监督训练 + 奖励微调入口 |
| `rl_finetune.py` | Self-Critical 策略梯度实现 |

## 环境与数据

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

仓库附带紧凑的 OpenOffice 示例语料。生成更丰富的本地训练语料：

```powershell
python prepare_corpus.py
python prepare_iwslt_fast.py
```

大规模 Parquet 语料、分词器和模型权重均属于本地生成文件，不提交到 Git。

## 基础监督训练

先运行约十几秒的端到端检查：

```powershell
python smoke_test.py
```

开始训练：

```powershell
python train_test.py
```

流程包括：

1. 加载并划分中英平行语料；
2. 训练或复用中文、英文 BPE 分词器；
3. 构建自定义 Transformer；
4. 使用 Noam 调度和设备自适应精度训练；
5. 保存验证最优权重与可恢复训练状态。

重新训练分词器：

```powershell
python train_test.py --retrain-tokenizers
```

从完整状态继续：

```powershell
python train_test.py --resume output/checkpoints/last_training_state.pt
```

请求当前训练在本 step 完成后保存并退出：

```powershell
python pause_training.py
```

## 自定义模型推理

交互模式：

```powershell
python run.py
```

翻译一句并退出：

```powershell
python run.py --text "你好吗"
```

推理前必须先生成兼容的训练权重，或通过 `--checkpoint` 显式指定权重。

## 通用语料实验

`config.general.json` 使用较小的 256 维、4 层编码器/解码器配置，适合在 8GB GPU 上快速实验：

```powershell
python prepare_corpus.py
python train_test.py --config config.general.json --retrain-tokenizers
python run.py --config config.general.json
```

继续训练：

```powershell
python train_test.py `
  --config config.general.json `
  --resume output/checkpoints_general_v2/last_training_state.pt
```

## 快速监督训练与 SCST

`fast_rl_pipeline.py` 先执行监督训练，再使用 Self-Critical Sequence Training。贪心译文作为 baseline，采样译文相对 baseline 的 1/2-gram F1 优势作为序列级奖励，并保留交叉熵项限制策略漂移。

```powershell
python prepare_iwslt_fast.py
python fast_rl_pipeline.py --config config.fast_rl.json
python run.py --config config.fast_rl.json
```

暂停后继续：

```powershell
python pause_training.py
python fast_rl_pipeline.py `
  --config config.fast_rl.json `
  --resume output/checkpoints_iwslt_fast_v2/last_training_state.pt
```

SCST 候选只有在固定留出集奖励提高时才保存；若强化学习阶段退化，则回退到监督基线。

## 配置字段

| 配置块 | 内容 |
|---|---|
| `model` | 隐藏维度、前馈维度、注意力头、层数、Dropout 和序列长度 |
| `tokenizer` | 词表大小、最小词频和分词器目录 |
| `data` | 数据路径、语言列、训练/测试规模和随机种子 |
| `train` | 批量、轮数、warmup、精度、梯度累积、保存和早停参数 |
| `run` | 分词器文件和推理检查点 |

## 与当前准确翻译路线的关系

```text
手写实验：data → BPE → build.py → train_test.py/SCST → run.py
实际翻译：OPUS-MT → 可选质量门控微调 → accurate_translator.py → GUI/文档
```

两套权重、配置和入口互不覆盖。训练这一页的模型不会自动提高 `translator_gui.py` 的翻译结果。

## 数据许可提醒

- IWSLT 2017 元数据标注为 CC BY-NC-ND 4.0，仅建议用于相应许可范围内的非商业实验。
- OPUS-100 数据集卡没有提供统一的语料许可，使用前应核对具体来源。
- 代码继续遵循仓库中的 [MIT License](LICENSE)。
