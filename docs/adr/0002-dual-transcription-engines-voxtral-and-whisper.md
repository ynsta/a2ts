# ADR 0002: Dual Transcription Engines (Voxtral and Faster-Whisper)

**Status**: Accepted  
**Date**: 2026-09-21  
**Deciders**: Stany Marcel  
**Consulted**: Architecture Spec  

---

## Context & Problem Statement

Transcription of domain-heavy audio (fantasy roleplay, technical architecture discussions) frequently fails on uncommon terminology, proper nouns, and fantasy character names. Standard pre-trained models mishear domain terms as phonetically similar dictionary words.

Additionally, the state of the art in open speech models offers two distinct paradigms:
1. **CTranslate2 Whisper (`faster-whisper`)**: Battle-tested, ultra-fast C++ runtime with word-level timestamps.
2. **Multimodal Audio LLMs (`Voxtral`)**: Mistral AI's audio LLM family (`mistralai/Voxtral-Mini-3B-2507`), which natively integrates language modeling and acoustic encoding.

## Decision Drivers

- Terminology accuracy on Obsidian notes, lore files, and custom glossaries.
- High transcription throughput with word-level timestamp generation.
- Flexibility to choose between lightweight speed (Whisper Turbo) and deep linguistic comprehension (Voxtral).

## Decision

We chose to implement a **pluggable dual-engine architecture** (`BaseTranscriber` interface):
1. **Faster-Whisper**: Primary engine for production speed (`large-v3`, `turbo`) with word-level alignment.
2. **Mistral Voxtral**: Optional engine via Hugging Face `transformers` for multimodal linguistic context.
3. **Obsidian Lore Prompting**: Automatic mining of wikilinks, headings, and frontmatter aliases from Obsidian notes (`contexte/`), tokenized and budgeted via `tiktoken` (OpenAI `cl100k_base`) to inject into the initial ASR prompt.

## Consequences

- **Positive**:
  - Dramatically reduced spelling errors on fantasy names (e.g. "Brakk", "Ilvarris", "Merrow").
  - Seamless fallback between Whisper and Voxtral via `--engine`.
  - Independent caching of raw transcription results per engine checkpoint (`.a2ts/transcripts/<hash>_<engine>.json`).
- **Negative**:
  - Voxtral requires heavier PyTorch/Transformers dependencies and more VRAM than quantized Faster-Whisper.
  - Lore prompts must respect strict token budgets (< 250 tokens) to avoid degrading acoustic attention.
