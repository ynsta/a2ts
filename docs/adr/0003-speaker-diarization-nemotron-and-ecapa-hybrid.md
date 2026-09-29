# ADR 0003: Speaker Diarization via Nemotron-3 and SpeechBrain ECAPA Hybrid

**Status**: Accepted  
**Date**: 2026-09-29  
**Deciders**: Stany Marcel  
**Consulted**: Architecture Spec, Benchmark Evaluation  

---

## Context & Problem Statement

Initial implementations of `a2ts` relied exclusively on SpeechBrain's ECAPA-TDNN embedding extractor (`speechbrain/spkrec-ecapa-voxceleb`) coupled with Scikit-Learn's `AgglomerativeClustering`.

While functional, this approach had severe limitations:
1. **ASR Boundary Dependency**: Diarization operated on arbitrary segment cuts emitted by the ASR engine. If two people spoke within a single segment, the acoustic embedding was corrupted.
2. **Threshold Brittleness**: Fixed cosine distance thresholds (`0.60`) frequently caused over-clustering or under-clustering depending on mic quality and room reverberation.
3. **No Overlap Detection**: Cannot detect cross-talk or simultaneous speakers.
4. **Speed**: Segment-by-segment embedding extraction in Python was slow (~30x real-time).

In September 2026, NVIDIA published **Nemotron-3 Diarization** (`nvidia/Nemotron-3-Diarization`), a 100M parameter open-weight Transformer encoder (Sortformer successor) capable of end-to-end speaker activity prediction up to 8 speakers with native overlap detection and > 130x real-time factor on RTX 3080.

However, Nemotron-3 outputs anonymous permutation-ordered labels (`SPEAKER_00`, `SPEAKER_01`) and does not generate acoustic identity embeddings for matching against persistent speaker profile databases (`VoiceProfilesDatabase`).

## Decision

We chose a **Hybrid Architecture**:
1. **Primary Diarization (Nemotron-3)**:
   - Uses `nvidia/Nemotron-3-Diarization` via Hugging Face `transformers` to predict speaker turns at 10ms–80ms resolution directly from raw audio.
   - Detects overlapping speech natively.
   - Runs by default (`--diarizer-engine auto`) on CUDA hardware.
2. **On-Demand Speaker Embedding (SpeechBrain ECAPA-TDNN)**:
   - When `--voice-profiles` is passed and named speakers exist, SpeechBrain ECAPA extracts 192-dimensional embeddings specifically from the clean turns cut by Nemotron-3.
   - Matches centroids against `VoiceProfilesDatabase` via cosine similarity.
   - Completely skipped if voice profiles are not used, preserving pure Nemotron performance.
3. **Graceful Fallback**:
   - Falls back to ECAPA + Agglomerative Clustering on CPU environments or if Nemotron fails to load.

## Consequences

- **Positive**:
  - Benchmark on actual RPG audio demonstrated 4.4x speedup (460 ms vs 2,024 ms on 60s audio, 130.4x real-time).
  - Perfect speaker count resolution (detected exact 4 speakers matching ground truth; ECAPA alone falsely hallucinated 7 clusters).
  - Overlapping speech captured accurately (14 overlap instances detected).
  - Full backward compatibility with existing `voice_profiles.json` databases.
- **Negative**:
  - Requires `transformers >= 5.18.0.dev0` (installed from git HEAD via `pyproject.toml` `[tool.uv.sources]`).
  - Hard limit of 8 concurrent speakers in single audio chunk (sufficient for almost all sessions).
