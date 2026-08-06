# MyTransformer Enhanced

**[简体中文](README.md) | English**

A GPU-accelerated bidirectional Chinese-English translator for long text and documents, with automatic direction detection, terminology control, resumable caching, and quality-gated OPUS-MT fine-tuning. The repository also preserves a hand-built Transformer pipeline for architecture and training experiments.

> This is an independent derivative of [K2etn/MyTransformer](https://github.com/K2etn/MyTransformer), not an official upstream release. The original educational implementation and attribution are preserved; document translation, the GUI, quality-gated fine-tuning, and production-oriented safeguards are additions in this repository.

## Two separate paths

| Path | Intended use | Model | Main entry point |
|---|---|---|---|
| **Accurate document translation** | Chinese/English long text, DOCX, TXT, and Markdown | One 77.9M-parameter OPUS-MT per direction, each independently fine-tunable | `start_accurate_translator.cmd` |
| **Hand-built Transformer lab** | Learning attention, encoder/decoder internals, and training | Custom Transformer trained from scratch | `train_test.py` / `fast_rl_pipeline.py` |

Training the custom model does not modify the accurate document translator. Fine-tuning OPUS-MT does not overwrite custom-model checkpoints.

## Accurate translator features

- Bidirectional large pasted text with line preservation.
- Automatic detection, Chinese→English, English→Chinese, and one-click swapping.
- `.docx`, `.txt`, `.md`, and `.markdown` document input.
- DOCX paragraph, heading, list, table, image, header, and footer preservation.
- CUDA FP16 batches and 5-beam decoding.
- Sentence-aware and token-aware long-text chunking.
- Separate terminology rules in `glossary.json` and `glossary.en_zh.json`.
- JSONL document cache for interruption recovery.
- Foreground fine-tuning monitor with loss, progress, VRAM, and ETA.
- A held-out quality gate that rejects regressing checkpoints.

## Quick start

Install the accurate translation dependencies:

```powershell
python -m pip install -r requirements-accurate.txt
```

Launch the Windows GUI:

```powershell
.\start_accurate_translator.cmd
```

Or use the Python entry point:

```bash
python translator_gui.py
```

Each pinned directional OPUS-MT model is downloaded on first use and then loaded from the local Hugging Face cache. Models are switched on demand instead of remaining on the GPU together.

Translate a document from the command line:

```powershell
python accurate_translator.py --input "D:\docs\report.docx"
```

Direction is detected automatically, or can be selected explicitly:

```powershell
python accurate_translator.py --direction zh-en --text "需要翻译的中文"
python accurate_translator.py --direction en-zh --text "Translate this into Chinese."
```

See [README_ACCURATE.md](README_ACCURATE.md) for the complete Chinese usage guide.

## Fine-tune the current translator

Optionally prepare the larger local corpora:

```powershell
python prepare_iwslt_fast.py
python prepare_corpus.py
```

Then click **训练增强模型…** in the GUI or run:

```powershell
.\start_accurate_finetune.cmd
.\start_accurate_finetune_en_zh.cmd
```

Each direction has an RTX 4060 8GB profile using 12,000 filtered training pairs, 256 held-out validation pairs, FP16, an effective batch size of 32, and 200 optimizer steps. A candidate checkpoint is promoted only when held-out validation loss does not regress against that direction's pinned base model.

Fine-tuned weights are stored separately in `output/accurate_finetuned/best` and `output/accurate_finetuned_en_zh/best`; both are intentionally excluded from Git.

## Upstream versus this repository

| Area | Upstream project | This enhanced repository |
|---|---|---|
| Primary goal | Learn Transformer internals from basic PyTorch operations | Practical document translation plus the retained learning path |
| Default translation core | Locally trained Chinese→English model | Two pinned OPUS-MT models with independent quality-gated checkpoints |
| Input | Primarily interactive sentences | Long text, DOCX, TXT, and Markdown |
| Decoding and stability | Greedy decoding | Beam search, repetition control, glossary normalization, and cache |
| Interface | Command line | Foreground two-pane GUI and document progress |

The original custom-model instructions now live in [README_CUSTOM_TRANSFORMER.md](README_CUSTOM_TRANSFORMER.md), separate from the practical translation workflow.

## Verification

Accurate translator and local checkpoint:

```powershell
python accurate_smoke_test.py
```

Custom-model forward, backward, and decoding pipeline:

```powershell
python smoke_test.py
```

## Known limits

- Very long documents are translated in segments, so cross-paragraph context is limited.
- PDF, scanned OCR input, complex text boxes, and footnotes are not yet supported.
- High-risk legal, medical, or contractual translations still require human review.
- Model checkpoints are not stored in Git; the base model is downloaded on first use.

## Attribution and license

- Upstream: [K2etn/MyTransformer](https://github.com/K2etn/MyTransformer)
- Pretrained models: [Chinese→English](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) / [English→Chinese](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Released under the repository's [MIT License](LICENSE). Both the upstream copyright notice and the modification notice are retained.
