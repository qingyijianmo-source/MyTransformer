# MyTransformer Enhanced

**[简体中文](README.md) | English**

A GPU-accelerated Chinese-to-English translator for long text and documents, with terminology control, resumable caching, and quality-gated OPUS-MT fine-tuning. The repository also preserves a hand-built Transformer pipeline for architecture and training experiments.

> This is an independent derivative of [K2etn/MyTransformer](https://github.com/K2etn/MyTransformer), not an official upstream release. The original educational implementation and attribution are preserved; document translation, the GUI, quality-gated fine-tuning, and production-oriented safeguards are additions in this repository.

## Two separate paths

| Path | Intended use | Model | Main entry point |
|---|---|---|---|
| **Accurate document translation** | Long text, DOCX, TXT, and Markdown | 77.9M-parameter OPUS-MT with optional local fine-tuning | `start_accurate_translator.cmd` |
| **Hand-built Transformer lab** | Learning attention, encoder/decoder internals, and training | Custom Transformer trained from scratch | `train_test.py` / `fast_rl_pipeline.py` |

Training the custom model does not modify the accurate document translator. Fine-tuning OPUS-MT does not overwrite custom-model checkpoints.

## Accurate translator features

- Large pasted text with line preservation.
- `.docx`, `.txt`, `.md`, and `.markdown` document input.
- DOCX paragraph, heading, list, table, image, header, and footer preservation.
- CUDA FP16 batches and 5-beam decoding.
- Sentence-aware and token-aware long-text chunking.
- Editable terminology rules in `glossary.json`.
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

The pinned OPUS-MT base model is downloaded once (about 312 MB) and then loaded from the local Hugging Face cache.

Translate a document from the command line:

```powershell
python accurate_translator.py --input "D:\docs\report.docx"
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
```

The default RTX 4060 8GB profile uses 12,000 filtered training pairs, 256 held-out validation pairs, FP16, an effective batch size of 32, and 200 optimizer steps. A candidate checkpoint is promoted only when held-out validation loss does not regress against the pinned base model.

Fine-tuned weights are stored in `output/accurate_finetuned/best` and are intentionally excluded from Git.

## Upstream versus this repository

| Area | Upstream project | This enhanced repository |
|---|---|---|
| Primary goal | Learn Transformer internals from basic PyTorch operations | Practical document translation plus the retained learning path |
| Default translation core | Locally trained custom model | Pinned OPUS-MT with an optional quality-gated local checkpoint |
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
- Pretrained model: [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Released under the repository's [MIT License](LICENSE). Both the upstream copyright notice and the modification notice are retained.
