# [ARCHIVED] Opus Review Fixes Implementation Plan

> [!WARNING]
> **HISTORICAL ARCHIVE**: This implementation plan was fully completed on 2026-09-30 and is archived for historical reference only.
> **DO NOT USE AS LIVING DOCUMENTATION.** Refer to canonical documentation:
> - Specifications (The WHAT): [`../spec.md`](../spec.md)
> - Technical Design (The HOW): [`../design.md`](../design.md)
> - Architectural Decisions (The WHY): [`../adr/`](../adr/)
> - Codebase Map (The WHERE): [`../codemap.md`](../codemap.md)
> - Operational Recipes (The OPS): [`../runbook.md`](../runbook.md)

---

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

**Spec:** `docs/spec.md`, `docs/design.md`, `docs/codemap.md`.

## Global Constraints
- Target Python: >= 3.13
- All tests must pass: `uv run pytest -q`
- Ruff lint and formatting must be clean: `uv run ruff check .` and `uv run ruff format --check .`
- Types must pass: `uv run mypy src tests`
- Zero regression on existing 154 tests
- Caveman compression active for conversational responses

---

### Task 1: Scoping Session State to `cache_dir / "sessions" / <media_hash>` (P1-1)
- [x] **Step 1: Write the failing multi-session isolation test**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Update `cli.py` to scope session state and adapt `review`, `split`, `recluster`**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 2: Embedding Cache Provenance, No Legacy Alias, and `--force` Support (P1-2)
- [x] **Step 1: Write test for embedding provenance and force invalidation**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement `EmbeddingCacheProvenance`, remove legacy alias, add `force`**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 3: Diarization Cache Provenance Transcript Dependency (P1-3)
- [x] **Step 1: Write test for transcript dependency in diarization cache**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Update models and diarization cache key**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 4: Persist Splits and Fix Positional Fallbacks (P2-2, P2-3, P2-1)
- [x] **Step 1: Write tests for split persistence and no-fallback behavior**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement split persistence and remove positional fallbacks**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 5: Refiner Hardening & Raw Markdown Output (P1-4, P1-5)
- [x] **Step 1: Write test for refiner truncation rejection and header validation**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement chunking, strict validation, and raw markdown output**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 6: French RPG Normalizer Hardening (P1-6)
- [x] **Step 1: Write regression tests for false positives**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Update `DICE_PATTERN` and single normalization point**
- [x] **Step 4: Run tests to verify they pass**
- [x] **Step 5: Commit**

---

### Task 7: Craig Quality, Performance & Packaging (P1-7, P2-5, P2-6, P2-8, P2-10)
- [x] **Step 1: Write test for timeline bisect alignment and packaging import sanity**
- [x] **Step 2: Run test to verify it passes/fails**
- [x] **Step 3: Update `pyproject.toml`, `craig.py`, `timeline.py`, `media.py`**
- [x] **Step 4: Run tests and lock sync**
- [x] **Step 5: Commit**

---

### Task 8: Formatting, Canonical Docs & ADR 0005 Updates (§5, P1-5, §8)
- [x] **Step 1: Run `uv run ruff format .`**
- [x] **Step 2: Update documentation options, defaults, and ADR 0005**
- [x] **Step 3: Verify all gates**
- [x] **Step 4: Commit**
