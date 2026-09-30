# Opus Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all verified P1 and key P2 architectural, data-integrity, and safety defects identified in the 2026-09-30 Opus review, while applying reasoned pushback on dice normalizer defaults and model caching claims.

**Architecture:** 
1. Scope session state artifacts (`turns.json`, `speakers_mapping.json`, `session.json`) into `cache_dir / "sessions" / <media_hash> /` to prevent cross-session contamination and profile poisoning.
2. Remove legacy embedding cache alias fallbacks, add typed provenance envelopes for embeddings, and wire `--force` down to extractors.
3. Make diarization provenance transcript-aware for ECAPA and record the resolved engine.
4. Persist manual splits in `SpeakersMapping` and remove dangerous positional index fallbacks.
5. Harden `--refine` with turn chunking, strict validation ratios, and immutable `<output>.raw.md` output.
6. Fix French dice normalizer regex to require roll verb context or explicit dice markers, preventing ordinary phrase corruption.
7. Clean packaging dependencies (drop pandas, add platform markers for CUDA, declare torch), add VAD to Craig tracks, and optimize quadratic loops with bisect.
8. Align canonical documentation with disk reality, update ADR 0005, and enforce `ruff format`.

**Tech Stack:** Python 3.13, Typer, Pydantic v2, faster-whisper, speechbrain ECAPA-TDNN, ruff, pytest.

**Spec:** `docs/spec.md`, `docs/design.md`, `docs/codemap.md`, `/home/stany/Work/a2ts-review-2026-09-30-opus.md`.

## Global Constraints
- Target Python: >= 3.13
- All tests must pass: `uv run pytest -q`
- Ruff lint and formatting must be clean: `uv run ruff check .` and `uv run ruff format --check .`
- Types must pass: `uv run mypy src tests`
- Zero regression on existing 154 tests
- Caveman compression active for conversational responses

---

### Task 1: Scoping Session State to `cache_dir / "sessions" / <media_hash>` (P1-1)

**Files:**
- Modify: `src/a2ts/cli.py`
- Modify: `src/a2ts/speaker_review.py`
- Test: `tests/test_session_scoping.py`

**Interfaces:**
- Consumes: `cache_dir: Path`, `file_hash: str`
- Produces: `session_dir = cache_dir / "sessions" / file_hash` containing `turns.json`, `speakers_mapping.json`, `session.json`

- [x] **Step 1: Write the failing multi-session isolation test**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Update `cli.py` to scope session state and adapt `review`, `split`, `recluster`**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 2: Embedding Cache Provenance, No Legacy Alias, and `--force` Support (P1-2)

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `src/a2ts/diarizer.py`
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_embedding_provenance.py`

**Interfaces:**
- Consumes: `EmbeddingCacheProvenance`
- Produces: `_turn_embeddings_provenance.json`, `_segment_embeddings_provenance.json`, no legacy alias writes/reads

- [x] **Step 1: Write test for embedding provenance and force invalidation**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement `EmbeddingCacheProvenance`, remove legacy alias, add `force`**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 3: Diarization Cache Provenance Transcript Dependency (P1-3)

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `src/a2ts/cli.py`
- Modify: `src/a2ts/diarizer.py`
- Test: `tests/test_diarization_provenance.py`

**Interfaces:**
- Consumes: `transcript_provenance_hash: str | None`, `resolved_engine: str`
- Produces: `DiarizationCacheProvenance` reflecting transcript segmentation dependency

- [x] **Step 1: Write test for transcript dependency in diarization cache**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Update models and diarization cache key**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 4: Persist Splits and Fix Positional Fallbacks (P2-2, P2-3, P2-1)

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `src/a2ts/cli.py`
- Modify: `src/a2ts/speaker_review.py`
- Test: `tests/test_splits_persistence.py`

**Interfaces:**
- Consumes: `ClusterSplit(cluster_id, at, new_cluster_id)`
- Produces: `SpeakersMapping.splits`, no positional fallbacks `turns[tid]`

- [x] **Step 1: Write tests for split persistence and no-fallback behavior**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement split persistence and remove positional fallbacks**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 5: Refiner Hardening & Raw Markdown Output (P1-4, P1-5)

**Files:**
- Modify: `src/a2ts/refiner.py`
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_refiner_safety.py`

**Interfaces:**
- Consumes: `raw_markdown: str`, `agy_model: str`
- Produces: Immutable `<output>.raw.md` on disk, chunk-based refinement, truncation rejection

- [x] **Step 1: Write test for refiner truncation rejection and header validation**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement chunking, strict validation, and raw markdown output**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 6: French RPG Normalizer Hardening (P1-6)

**Files:**
- Modify: `src/a2ts/consolidator.py`
- Modify: `src/a2ts/refiner.py`
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_consolidator_dice.py`

**Interfaces:**
- Consumes: `text: str`, `rpg_normalize: bool = True`
- Produces: Correct dice normalization without corrupting ordinary French phrases

- [x] **Step 1: Write regression tests for false positives**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Update `DICE_PATTERN` and single normalization point**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 7: Craig Quality, Performance & Packaging (P1-7, P2-5, P2-6, P2-8, P2-10)

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/a2ts/craig.py`
- Modify: `src/a2ts/cli.py`
- Modify: `src/a2ts/timeline.py`
- Modify: `src/a2ts/media.py`
- Modify: `src/a2ts/transcriber.py`
- Test: `tests/test_packaging_and_perf.py`

**Interfaces:**
- Consumes: Clean packaging metadata, `bisect` alignment, atomic WAV write
- Produces: Multiplatform wheel metadata, non-hallucinating Craig tracks, fast alignment

- [x] **Step 1: Write test for timeline bisect alignment and packaging import sanity**
- [x] **Step 2: Run test to verify it passes/fails**
- [x] **Step 3: Update `pyproject.toml`, `craig.py`, `timeline.py`, `media.py`**
- [x] **Step 4: Run tests and lock sync**
- [x] **Step 5: Commit**

---

### Task 8: Formatting, Canonical Docs & ADR 0005 Updates (§5, P1-5, §8)

**Files:**
- Modify: `docs/spec.md`
- Modify: `docs/design.md`
- Modify: `docs/runbook.md`
- Modify: `docs/codemap.md`
- Modify: `docs/adr/0005-local-llm-refinement.md`
- Modify: `README.md`
- Move: `docs/plans/2026-09-29-*.md` -> `docs/archive/`
- Touch: `.agents/last-docs-consolidate`

**Interfaces:**
- Consumes: Disk state
- Produces: 100% coherent canonical documentation, archived transient plans

- [x] **Step 1: Run `uv run ruff format .`**
- [x] **Step 2: Update documentation options, defaults, and ADR 0005**
- [x] **Step 3: Verify all gates**
- [x] **Step 4: Commit**
