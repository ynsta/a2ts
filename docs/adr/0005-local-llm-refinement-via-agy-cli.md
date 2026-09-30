# ADR 0005: LLM Refinement via agy CLI and Remote Egress Boundary

**Status**: Accepted  
**Date**: 2026-09-22 (Updated 2026-09-30)  
**Deciders**: Stany Marcel  
**Consulted**: Architecture Spec, Security & Privacy Review  

---

## Context & Problem Statement

Raw ASR transcripts often require light post-processing:
- Formatting out-of-character (OOC) table banter, rules questions, or side jokes into distinct callouts (`> [!NOTE] Hors-jeu / Discussion`).
- Standardizing spelling of fantasy terminology against Obsidian notes.
- Punctuation and paragraph structuring.

However, embedding a large language model directly inside the `a2ts` Python process introduces severe drawbacks:
1. High GPU memory contention with ASR and diarization models.
2. Complex dependency stacks (Ollama, vLLM, llama.cpp, PyTorch transformers).
3. API key management and vulnerability to rate limits if using cloud endpoints directly within the core library.

Furthermore, invoking external LLMs introduces an important **privacy and data sovereignty boundary**: transcript content leaves the local workstation if the LLM backend is cloud-hosted.

## Decision Drivers

- Maintain strict GPU memory boundaries (keep `a2ts` core VRAM <= 6.0 GB).
- Avoid embedding a separate LLM inference engine in the `a2ts` codebase.
- Explicitly document and isolate the network egress boundary.
- Protect against LLM hallucination, dropped speech turns, and transcript tampering.

## Decision

We chose to delegate post-processing to an **external subprocess call to the `agy` CLI**:

1. **Strict Data Boundary**:
   - Core inference (audio extraction, Faster-Whisper / Voxtral transcription, Nemotron-3 / ECAPA diarization, and deterministic RPG normalization) is **100% local, self-hosted, and air-gapped**. Zero audio or text egresses the local machine during standard runs.
   - The `--refine` flag is **disabled by default**.
   - When `--refine` is explicitly enabled, `a2ts` delegates chunked markdown to the `agy` CLI (`agy --model <model> --effort <effort>`), defaulting to `gemini-3.8-flash-low`. When `agy` connects to Google Gemini or another hosted provider, transcript text is transmitted to the provider's API. Users requiring complete air-gapped privacy must keep `--refine` disabled or configure `agy` to use a purely local model backend.

2. **Pre-Refinement Raw Snapshot**:
   - `a2ts` unconditionally writes the raw, consolidated transcript to `<output>.raw.md` before refinement begins. An un-altered, ground-truth local copy is guaranteed to exist on disk regardless of LLM availability or refinement outcome.

3. **Turn Chunking & Safety Verification**:
   - Transcripts are split into small windows of at most 25 turns (`chunk_transcript_markdown`) to prevent context window overflow or model attention decay.
   - Refined chunks are validated via `validate_refiner_chunk`:
     - Every turn header (`### [HH:MM:SS - HH:MM:SS] Speaker`) must match verbatim.
     - Word count delta must remain within 15% of the raw chunk.
   - If validation fails or `agy` raises an error/timeout, `a2ts` logs a warning and falls back to the raw chunk without failing the pipeline.

## Consequences

- **Positive**:
  - `a2ts` remains lean with zero LLM inference dependencies.
  - Complete isolation of GPU memory: ASR and diarization complete before `agy` executes.
  - Raw transcript `<output>.raw.md` is always preserved.
  - Strong safety guards prevent turn dropping, hallucination, or structural mangling.
- **Negative**:
  - Requires the `agy` CLI binary to be installed on the host system to use `--refine`.
  - Network egress occurs when `agy` is backed by hosted models (e.g. Gemini).
  - Disabled by default (`--no-refine`).

