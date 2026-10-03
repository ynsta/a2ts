# ADR 0007: Overlap-Aware Word Alignment

**Status**: Accepted
**Date**: 2026-10-03

## Context

Nemotron can emit simultaneous speaker turns. Selecting the first turn containing each word midpoint favors the earlier turn regardless of intersection length. Independently assigning ASR tokens also splits contractions and hyphenated words across speakers. Collecting words by acoustic turn can reorder dialogue when assignments alternate between overlapping turns.

## Decision

Use maximum temporal intersection for lexical token groups, with proximity to the turn center as a deterministic tie-breaker. Keep apostrophe and hyphen continuations together while preserving original word timestamps. Emit chronological contiguous runs and preserve acoustic-turn provenance for consumers of turn embeddings and speaker review mappings.

Expose cross-cluster attribution ambiguity as `AlignedTurn.speaker_uncertain`. Consolidation preserves per-turn flags in session metadata and renders one notice before the first transcript turn. Repeated warnings make heavily overlapping transcripts difficult to read, so individual turns do not render separate warnings.

Hold the application-generated notice outside model editing and restore it once after refinement. Continue protecting legacy per-turn warning callouts. Compare text length symmetrically without counting application metadata and log the actual validation failure category for each rejected chunk.

## Consequences

Attribution no longer depends solely on which overlapping turn starts first. Words stay readable and chronological. Existing cached aligned turns require regeneration to receive the new behavior; ASR and diarization caches can still be reused.

Temporal ranking remains a heuristic. It cannot distinguish actual simultaneous speakers from false acoustic detections or establish which voice produced a word in a mixed recording. Uncertain passages require audio review; no sentence-level or LLM-based relabeling is introduced. The readable notice does not locate every affected passage; session metadata retains that detail. Dense overlaps increase candidate-scanning cost.
