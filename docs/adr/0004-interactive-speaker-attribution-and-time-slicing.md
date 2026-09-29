# ADR 0004: Interactive Speaker Attribution and Time-Slicing

**Status**: Accepted  
**Date**: 2026-09-22  
**Deciders**: Stany Marcel  
**Consulted**: Architecture Spec  

---

## Context & Problem Statement

In conversational audio lasting several hours, acoustic conditions drift over time:
- Microphones are repositioned or shared between players.
- Voices fatigue or change pitch.
- Acoustic embeddings from early in the session can merge with different speakers later in the session.
- Fully automated diarization algorithms (even modern ones) make occasional cluster assignment mistakes.

Forcing users to accept fully automated attribution results in errors that propagate through the entire transcript. Conversely, forcing manual annotation of thousands of turns is prohibitive.

## Decision Drivers

- Human-in-the-loop verification with minimal cognitive burden.
- Ability to fix cluster assignments without re-running compute-heavy ASR or diarization.
- Support for temporal division of clusters that drift across long recordings.

## Decision

We implemented:
1. **Time-Slice Segmentation (`timeline.py`)**:
   - Audio is divided into 15-minute time windows (`time_slice_id`).
   - Cluster labels can be scoped or reviewed within specific temporal slices.
2. **Interactive Terminal QCM Review (`speaker_review.py`)**:
   - Rich CLI menu displaying speaker turn counts, total speaking duration, and sample quotes.
   - Option to play audio snippets around sample quotes via `ffplay`/`aplay`.
   - Forward and backward label propagation to fill in brief acoustic gaps.
3. **Cluster Splitting (`a2ts split`)**:
   - A dedicated subcommand allowing the user to partition an acoustic cluster at a timestamp `T` (e.g. `a2ts split SPEAKER_00 1800.0 SPEAKER_04`).
4. **Decoupled Review Subcommand (`a2ts review`)**:
   - Reads cached `turns.json` and `speakers_mapping.json`, allowing instant re-attribution without re-transcribing.

## Consequences

- **Positive**:
  - High attribution accuracy with human validation.
  - Interactive review takes less than 2 minutes for a 3-hour session.
  - Attribution mistakes can be corrected in seconds via `a2ts split` or `a2ts review`.
- **Negative**:
  - Requires user interaction in interactive mode (can be disabled with `--no-interactive`).
