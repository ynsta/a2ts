# ADR 0006: Craig Multi-Track Discord Pipeline Architecture

**Status**: Accepted  
**Date**: 2026-09-29  
**Deciders**: Stany Marcel  
**Consulted**: Architecture Spec, Craig Adapter Design  

---

## Context & Problem Statement

Tabletop role-playing sessions and remote team meetings conducted over Discord are frequently recorded using the Craig Discord bot. Craig generates a folder containing discrete, isolated audio tracks for each participant (`1-merrow1.flac`, `2-tessaro.flac`, etc.) alongside recording metadata (`info.txt`).

Processing these multi-track recordings through the standard single-stream pipeline (`a2ts run`) presents significant drawbacks:
1. Merging multiple discrete tracks into a single mixed audio stream artificially creates cross-talk and overlapping speech, throwing away ground-truth microphone isolation.
2. Running acoustic diarization (Nemotron-3 or SpeechBrain ECAPA) on mixed multi-speaker audio is computationally heavy and introduces potential clustering errors (speaker misattribution, under-clustering, or cluster fragmentation).
3. Any small edit or re-run on a single participant's track would require re-transcribing and re-diarizing the entire multi-hour mixed recording.

## Decision Drivers

- Maximize attribution accuracy by leveraging physical microphone isolation as ground truth.
- Eliminate redundant compute: bypass acoustic diarization and clustering models entirely.
- Support discrete per-track caching so editing one track or speaker roster does not invalidate other tracks.
- Automatically resolve Discord usernames to character names, roles, and Game Master status from human-friendly roster notes (`speakers.md`).
- Maintain consistent output format and consolidation pipeline with standard `a2ts run` transcripts.

## Decision

We chose to implement a dedicated multi-track adapter and command, `a2ts craig`:

1. **Physical Track Discovery & Username Extraction**:
   - `a2ts craig <recording_dir>` automatically scans for tracks matching `^(\d+)-(.*)\.flac$` sorted numerically by track index prefix.
   - Extracts Discord usernames directly from filenames (e.g. `1-merrow1.flac` -> `merrow1`).

2. **Roster & Metadata Ingestion**:
   - Parses human-edited `speakers.md` roster files to map usernames to character names, roles, nicknames, and Game Master (`is_dm`) designations.
   - Parses Craig's `info.txt` to capture recording metadata (guild, channel, start time, user IDs).
   - Combines character identities and Obsidian lore into a token-budgeted prompt (180 tokens) via `vocab.py`.

3. **Discrete Track Transcription & Provenance Caching**:
   - Each track is transcribed independently via Faster-Whisper with word-level timestamps, voice activity detection (`vad_filter=True`), and language hints (`language="fr"`).
   - Results are cached per-track in `<recording_dir>/.transcripts/<track_stem>.json` governed by `TrackCacheProvenance`.
   - Modifying a single participant's track or re-running with `--force` on one track leaves other cached tracks untouched.

4. **Bypassing Acoustic Diarization**:
   - Because each track corresponds to exactly one microphone, acoustic diarization and clustering are completely bypassed.
   - Speaker labels are attributed deterministically from `speakers.md` or track usernames.

5. **Chronological Interleaving & Unified Consolidation**:
   - Track segments are merged into a unified chronological turn sequence (`merge_craig_tracks_to_turns`) sorted by `(seg.start, track_id)`.
   - Merged turns pass directly through existing downstream stages:
     - Consecutive utterance debouncing (`consolidator.debounce_consecutive_turns`).
     - Deterministic French RPG normalization (`consolidator.normalize_rpg_terms`).
     - Markdown transcript assembly (`consolidator.render_markdown_transcript`).
     - Optional LLM refinement via `agy` CLI (`refiner.refine_transcript_markdown`).

## Consequences

- **Positive**:
  - 100% speaker attribution accuracy without acoustic clustering ambiguity.
  - Significantly faster execution by omitting heavy diarization models.
  - Per-track caching enables rapid iteration on roster changes, debouncing thresholds, or prompt updates.
  - Complete architectural consistency with single-stream pipeline output and markdown structure.
- **Negative**:
  - Requires input files to follow Craig's multi-track folder structure and naming scheme.
  - Separate CLI subcommand (`a2ts craig`), though options and behaviors align with `a2ts run`.
