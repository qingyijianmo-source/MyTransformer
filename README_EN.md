# MyTransformer

**[简体中文](README.md) | English**

A "from-0.5, hand-written" Transformer (base model) built with the PyTorch framework, reproducing the architecture and details of the original **"[Attention Is All You Need](https://arxiv.org/abs/1706.03762)"** paper as faithfully as possible, and applying it to Chinese → English machine translation. High-level off-the-shelf modules like `nn.Transformer` are not used; attention, encoder, decoder, and positional encoding are all hand-built from PyTorch's basic operators.

> This is a learning project: the goal is to understand how the Transformer works, line by line, by building it by hand. It simply implements CN→EN translation — corrections and suggestions are warmly welcome.

---

## Features

- **Hand-built core modules**: multi-head attention, encoder/decoder layers, positional encoding — built from PyTorch's basic operators, no off-the-shelf high-level modules
- **ByteLevel BPE tokenizer**, trained per language (zh / en)
- **Noam learning-rate schedule**: warmup + inverse-sqrt decay
- **Device-aware precision**: bf16 on supported CUDA devices, automatic float32 fallback on CPU
- **Early stopping** with a staged validation `Checker`
- **Robust entry points** with checkpoint resume, one-shot inference, and an end-to-end smoke test

---

## Project Structure

```
MyTransformer/
├── parts.py          # Core components: attention, encoder/decoder layers, embeddings, etc.
├── build.py          # Transformer assembly + build_transformer factory
├── train_test.py     # Data pipeline, tokenizer training, training & validation loop
├── run.py            # Interactive CN→EN inference (greedy decoding)
├── smoke_test.py     # Fast end-to-end check with a small model
├── prepare_corpus.py # Download, clean, and merge general OPUS-100 data
├── prepare_iwslt_fast.py # Prepare a clean, compact IWSLT TED corpus
├── fast_rl_pipeline.py # Fast supervised training plus SCST reward tuning
├── rl_finetune.py    # Self-critical policy-gradient implementation
├── config.json       # All hyperparameters in five blocks
├── config.general.json # Separate config for the general-domain corpus
├── data/             # OPUS OpenOffice en-zh dataset
├── requirements.txt
└── LICENSE           # MIT
```

`output/` (tokenizers + checkpoints) is generated locally by training and is **not** tracked in git.

---

## Requirements

- Python ≥ 3.10
- CUDA GPU recommended (developed on RTX 4060 Laptop 8GB; bf16 needs Ampere+)

Install dependencies:

```bash
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA 13.0 PyTorch build (a separate CUDA Toolkit is not required):

```bash
pip install -r requirements-gpu.txt
```

---

## Quick Start

### 0. Fast verification

First run the roughly ten-second end-to-end check. It reads the real dataset, reuses existing tokenizers (or trains temporary ones when missing), then performs a small-model forward pass, backward pass, and decode without writing model weights:

```bash
python smoke_test.py
```

`SMOKE TEST PASSED` confirms that the data, tokenizers, model, and training path work together.

### 1. Train

```bash
python train_test.py
```

This will:

1. Load and split the dataset (`data/`)
2. Train zh / en BPE tokenizers → `output/tokenizers/`
3. Build the Transformer and train with the Noam schedule and device-aware precision
4. Save best weights → `output/checkpoints/best_state_from_checker.pt`

Existing tokenizers are reused by default. Use `python train_test.py --retrain-tokenizers` to replace them, or resume the complete training state (optimizer, scheduler, and batch position included) with:

```bash
python train_test.py --resume output/checkpoints/last_training_state.pt
```

This state is refreshed every `save_steps` optimizer steps. Plain `epoch_*.pt` files can still initialize model weights, but start with a fresh optimizer.

Safely pause the active run after its current optimizer step is saved:

```bash
python pause_training.py
```

With the current 8,000-token vocabularies, the default configuration has about 56.4 million parameters. Full CPU training is still slow; a CUDA GPU is recommended.

### 2. Inference

```bash
python run.py
```

Translate one sentence and exit with:

```bash
python run.py --text "你好吗"
```

Pretrained weights are not bundled. Before inference, train `output/checkpoints/best_state_from_checker.pt` or pass a compatible file with `--checkpoint`.

Interactive translation:

```
=====中译英开始!  输入'quit'退出=====
中文>>>你好吗
英文: how are you
```

---

## Configuration

All hyperparameters live in `config.json`, in five blocks:

| Block | Contents |
|---|---|
| `model` | `d_model`, `d_hidden`, `num_heads`, `drop_prob`, `num_encode_layers`, `num_decode_layers`, `source_seq_len`, `target_seq_len` |
| `tokenizer` | `vocab_size`, `min_frequency`, `tokenizers_dir` |
| `data` | `dataset_path`, `source_language`, `target_language`, `train_size`, `test_size`, `seed` |
| `train` | `batch_size`, `num_epochs`, `warmup`, `warmup_factor`, `label_smoothing`, `dynamic_padding`, `gradient_accumulation_steps`, `save_steps`, `max_grad_norm`, `num_workers`, `check_steps`, `patience_times`, `min_progress_value`, `checkpoints_dir` |
| `run` | tokenizer file names, `selected_checkpoint` |

Default model config follows the paper's **base** model: `d_model=512`, `d_hidden=2048`, `num_heads=8`, 6 encoder + 6 decoder layers.

### General-domain corpus

Build a cleaned and deduplicated OpenOffice + OPUS-100 corpus:

```bash
python prepare_corpus.py
```

The enhanced config keeps the hand-written Transformer and 12,000-token vocabulary, but fits the low-resource corpus with `d_model=256`, four encoder/decoder layers, and tied target embedding/projection weights: about 13.52 million parameters. Dynamic padding and gradient accumulation produce an effective batch of 96. New weights use the separate `checkpoints_general_v2` directory, preserving both base and older enhanced weights:

```bash
python train_test.py --config config.general.json --retrain-tokenizers
python run.py --config config.general.json
```

Resume exactly from the latest automatic training state:

```bash
python train_test.py --config config.general.json --resume output/checkpoints_general_v2/last_training_state.pt
```

Inference applies a repetition penalty, 3-gram blocking, and special-token suppression to prevent unbounded repeated fragments from under-trained checkpoints.

Source provenance and filter statistics are stored in `data/combined_general_en_zh.metadata.json`. The OPUS-100 dataset card does not provide a unified license, so this enhanced corpus is recommended for local learning and experimentation only.

### Fast training + reward tuning

`config.fast_rl.json` uses 100,000 cleaned IWSLT 2017 Chinese-English TED pairs, an approximately 11.46-million-parameter model, batch 96, and six supervised epochs. It then runs one SCST (Self-Critical Sequence Training) epoch on 1,024 examples: greedy translations provide the baseline, sampled translations receive a unigram/bigram F1 advantage, and a small cross-entropy term prevents policy drift. SCST evaluates a fixed held-out set every 16 batches and only keeps candidates that improve its reward; if every RL update regresses, `best_state_scst.pt` automatically falls back to the supervised baseline.

```bash
python fast_rl_pipeline.py --config config.fast_rl.json
python run.py --config config.fast_rl.json
```

Safely pause and resume exactly:

```bash
python pause_training.py
python fast_rl_pipeline.py --config config.fast_rl.json --resume output/checkpoints_iwslt_fast_v2/last_training_state.pt
```

IWSLT 2017 is licensed CC BY-NC-ND 4.0 and is used here for non-commercial experimentation. The sequence-level objective follows ideas from Minimum Risk Training for NMT and SCST.

---

## Acknowledgments

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original Transformer paper
- [OPUS OpenOffice](https://opus.nlpl.eu/) — the en-zh parallel corpus used here
- [PyTorch](https://pytorch.org/) / [Hugging Face tokenizers & datasets](https://huggingface.co/)

---

## License

[MIT](LICENSE) © 2026 K2etn
