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

“⇄ 交换”会交换左右文本并反转方向。切换方向时，程序先释放旧模型的 GPU 显存，再按需加载另一个模型。“自动深度审校”开启时，只有质量检测命中的疑难段落会进入本地 Qwen，状态栏会显示触发、完成和回退数量。

首次使用审校器先双击 `prepare_local_reviewer.cmd`。若模型未下载、依赖不可用、超时、OOM、输出为空或遗漏数字/缩写，程序会保留 OPUS 初译并继续后续段落。

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

每个输出文档旁会生成 `.translation-cache.jsonl`。缓存签名还包含审校模型配置、相邻段落摘要和文档词汇表指纹，因此更换审校器或文档上下文后不会命中旧管线译文。

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

## 英译中上下文消歧

`context_rules.en_zh.json` 用于处理必须结合语境判断的多义词和文学表达。规则以“词义候选 + 触发语境 + 禁用语境”为主，不再依赖完整测试句匹配。例如建筑语境中的 `mortar` 会改写为建筑灰浆，而带有 `shell`、`rocket`、`attack` 等军事语境时不会触发。

这与全局术语替换不同：上下文规则不会把所有 `mortar` 都固定成“灰浆”。当前规则也覆盖 `shroud`、`flagstones`、`lineage`、`neon-drenched` 和硬汉派隐喻等已发现的高风险表达。

## 分别增强两个方向

在一个前台窗口中依次训练两个方向各 20 轮：

```powershell
.\start_bidirectional_finetune_20_epochs.cmd
```

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

- 12,000 对清洗、去重、错配检测和永久黑名单过滤后的训练句对；
- 256 对训练过程未见过的验证句对；
- FP16、微批量 8、梯度累积 4；
- 20 轮、每轮 375 个优化步骤，每轮保存恢复断点；
- 每轮独立验证并保留当前最佳轮，连续三轮无实质改善会早停；
- 结束后同时检查 loss、chrF++、普通句、术语、数字/专名和严重错误，候选不合格不会覆盖当前 `best`。

下表是旧短程权重的 loss 记录，不是新冻结集上的准确率：

| 方向 | 基础验证损失 | 增强后 | 结果 |
|---|---:|---:|---|
| 中译英 | 1.60919 | 1.53671 | 通过并启用 |
| 英译中 | 2.39090 | 2.28609 | 通过并启用 |

权重分别写入 `output/accurate_finetuned/best` 和 `output/accurate_finetuned_en_zh/best`。新训练先保存在 `candidate`，综合质量门通过后才晋级。训练中断后重新启动会从各自最近的 `resume` 断点继续；增加 `--fresh` 可从官方模型重新开始。

## 文档保留范围

- DOCX：保留段落、标题层级、列表、表格、分页、图片、页眉、页脚和 Word 字段。
- TXT：保留原换行和空行。
- Markdown：保留标题/列表前缀、代码块、行内代码、链接和 URL。
- DOCX 同一段落内的复杂局部字体、粗体或颜色可能统一继承第一个文本运行的样式。
- PDF、扫描 OCR、复杂文本框和脚注尚未接入当前入口。

## 长文档原理与边界

程序先扫描整份文档，提取专名、缩写和术语；再由 OPUS-MT 对完整段落快速初译。质量检测器检查长难句、多义词、语言残留、异常长度、重复、数字和缩写，只把命中的段落连同前后文、初译和文档词汇表交给 Qwen 深度审校。审校结果还要经过事实完整性检查，不合格就回退初译，最后写回原文档结构。

这种方式可以处理很长的文件，但模型不会一次读取整篇文档，因此跨段代词、人名一致性和全文语境仍可能需要人工复核。法律、医疗、合同等高风险文档不应直接把机器译文当成最终版本。

## 验收

运行 CPU 回归测试：

```powershell
python -m pytest
```

运行冻结翻译评测并与基线比较：

```powershell
python evaluate_translations.py --direction both --split test --reviewer on `
  --baseline eval\baselines\opus_context_v1.json
```

本机 RTX 4060 8GB 的当前测试报告：中译英 chrF++ `68.52 → 76.22`，英译中 `31.16 → 37.19`；两方向术语与数字/专名保留率均为 `100%`，严重错误为 `0`，chrF++ 配对 bootstrap 检验通过。报告是小型项目回归集结果，不应外推为通用基准。

同时检查自动识别、中译英增强模型和英译中增强模型：

```powershell
python accurate_smoke_test.py
```

看到 `BIDIRECTIONAL ACCURATE TRANSLATION TEST PASSED` 即表示双向 CUDA 翻译链路正常。
