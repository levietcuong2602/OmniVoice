# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmniVoice is a massively multilingual zero-shot TTS system supporting 600+ languages. It uses a diffusion language model architecture built on a Qwen-0.6B LLM backbone with 8-codebook audio tokenization.

## Setup & Installation

**Requirements**: Python 3.10+, PyTorch 2.4+, torchaudio 2.4+

```bash
# Development install
pip install -e .
# or with uv (recommended)
uv sync
```

## CLI Entry Points

```bash
omnivoice-demo          # Launch Gradio web UI
omnivoice-infer         # Single-item inference
omnivoice-infer-batch   # Batch inference across multiple GPUs
```

## Training & Evaluation

```bash
bash examples/run_emilia.sh    # Train from scratch on Emilia dataset
bash examples/run_finetune.sh  # Fine-tune from pretrained checkpoint
bash examples/run_eval.sh      # Evaluate WER, speaker similarity, UTMOS
```

There is no automated test suite. Validation is done via evaluation scripts in `eval/`.

## Architecture

### Core Pipeline

**Inference**: Text + optional reference audio/instruction → OmniVoice model → 32-step diffusion → 24 kHz WAV output

**Training**: JSONL manifest → WebDataset shards → token extraction (8 codebooks) → token-based batching → HF Accelerate distributed training

### Key Modules

- **`omnivoice/models/omnivoice.py`** — Main `OmniVoice` class: loads from HF Hub (`k2-fsa/OmniVoice`), handles three generation modes (voice cloning, voice design, auto voice), supports long-text chunking with cross-fading
- **`omnivoice/training/`** — `OmniTrainer` wraps HF Accelerate for distributed bf16 training; `TrainingConfig` dataclass defines all hyperparameters; supports checkpoint save/resume
- **`omnivoice/data/`** — `WebDatasetReader` and `JsonlDatasetReader` feed data; token-based batching groups samples by total token count (not example count) for efficiency
- **`omnivoice/utils/lang_map.py`** — Maps 650+ language names/codes to language IDs used for model conditioning
- **`omnivoice/scripts/`** — Data preprocessing: audio denoising, token extraction, JSONL→WebDataset conversion
- **`omnivoice/eval/`** — Evaluation metrics: WER (FunASR/JIWER), speaker similarity (ECAPA-TDNN/WavLM), UTMOS (MOS prediction)

### Model Architecture Notes

- Qwen-0.6B LLM backbone predicts masked audio tokens (diffusion LM, not pure diffusion)
- Audio tokenized into 8 hierarchical codebooks (vocab size 1025 each)
- 600+ languages conditioned via explicit language IDs (not implicit from text)
- Attention: uses Flex attention when available, falls back to SDPA
- Generation config lives in `omnivoice/models/generation_config.py`; key params: `num_step` (diffusion steps, default 32), `speed`, `duration`, `temperature`

## Data Format

Input manifests are JSONL files; each line requires:
```json
{"audio": "path/to/audio.wav", "text": "transcript", "language": "en"}
```

See `docs/data_preparation.md` for full schema and language ID reference.

## Key Documentation

- `docs/training.md` — Training config details, TensorBoard, attention options
- `docs/generation-parameters.md` — All inference parameters with descriptions
- `docs/voice-design.md` — Voice design attribute syntax (gender, age, pitch, style, accents)
- `docs/tips.md` — Best practices for reference audio length and cross-lingual cloning
