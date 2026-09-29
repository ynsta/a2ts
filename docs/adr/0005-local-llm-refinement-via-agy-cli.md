# ADR 0005: Local LLM Refinement via agy CLI

**Status**: Accepted  
**Date**: 2026-09-22  
**Deciders**: Stany Marcel  
**Consulted**: Architecture Spec  

---

## Context & Problem Statement

Raw ASR transcripts often require light post-processing:
- Formatting out-of-character (OOC) table banter, rules questions, or side jokes into distinct callouts (`> [!NOTE] Hors-jeu`).
- Standardizing spelling of fantasy terminology against Obsidian notes.
- Punctuation and paragraph structuring.

However, embedding a large language model directly inside the `a2ts` Python process introduces severe drawbacks:
1. High GPU memory contention with ASR/Diarization models.
2. Complex dependency stacks (Ollama, vLLM, llama.cpp, PyTorch transformers).
3. API key management and vulnerability to rate limits if using cloud endpoints.

## Decision Drivers

- Maintain strict GPU memory boundaries.
- Avoid embedding a separate LLM inference engine in the `a2ts` codebase.
- Protect against LLM hallucination and content alteration.

## Decision

We chose to delegate post-processing to an **external subprocess call to the `agy` CLI**:
1. `refiner.py` prepares a structured system prompt containing context lore and format guidelines.
2. Executes `agy` via standard subprocess invocation.
3. **Safety Guard**: Computes normalized Levenshtein distance between raw text and refined text. If text changes drastically (> threshold), the refiner pass is rejected, preserving the exact raw transcript.

## Consequences

- **Positive**:
  - `a2ts` remains lean; zero LLM inference dependencies inside `a2ts`.
  - Full isolation of GPU memory: ASR and diarization complete before `agy` runs.
  - Safe against hallucinations via string distance verification.
- **Negative**:
  - Requires the `agy` CLI binary to be installed on the host system to use `--refine`.
  - Disabled by default (`--no-refine`).
