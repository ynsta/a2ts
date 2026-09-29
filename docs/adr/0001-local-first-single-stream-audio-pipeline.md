# ADR 0001: Local-First Single-Stream Audio Pipeline

**Status**: Accepted  
**Date**: 2026-09-21  
**Deciders**: Stany Marcel  
**Consulted**: Architecture Spec  

---

## Context & Problem Statement

Speech transcription of multi-hour tabletop RPG sessions, sensitive personal conversations, and team meetings often faces three obstacles:
1. **Privacy & Data Sovereignty**: Transmitting raw personal audio to third-party cloud APIs (OpenAI, Google Speech, AssemblyAI) creates data privacy risks.
2. **Recurring Token Costs**: Cloud API costs for 3-to-4 hour weekly sessions accumulate rapidly.
3. **Format Heterogeneity**: Unlike Discord bots (`craig2ts`) which produce pre-separated multi-track audio streams (1 file per user), real-world physical recordings produce a single mixed video/audio track (`.mp4`, `.wav`, `.mkv`) requiring acoustic separation.

## Decision Drivers

- Zero data leakage / strict local execution on consumer NVIDIA RTX hardware.
- High computational throughput (substantially faster than real-time).
- Self-contained packaging with deterministic dependency management (`uv`, Python 3.13).

## Considered Options

1. **Cloud Speech API Pipeline** (e.g. OpenAI Whisper API, Deepgram).
2. **Local Multi-track assumption** (requiring multi-mic hardware setups).
3. **Local Single-Stream Pipeline with GPU acceleration** (`a2ts`).

## Decision

We chose **Option 3**: Build `a2ts` as a local-first single-stream pipeline using local CUDA execution. Input media files are probed and resampled via `ffmpeg` into standardized 16kHz mono WAV files, cached content-addressably by file hash in `.a2ts/`.

## Consequences

- **Positive**:
  - Zero ongoing cloud costs.
  - Complete offline sovereignty and privacy.
  - Universal input support (any video/audio container supported by `ffmpeg`).
- **Negative**:
  - Requires local NVIDIA GPU with at least 8-10 GB VRAM.
  - Requires handling speaker attribution acoustically without physical track separation.
