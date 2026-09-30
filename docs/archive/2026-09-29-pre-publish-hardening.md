# Pre-Publication Hardening & Craig Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix P1 correctness defects (embedding indices, cache provenance, voice profile idempotence, Nemotron packaging), restore green mypy/pytest gates, port Craig RPG normalizer, align docs/CLI, and add CI before public release.

**Architecture:** 
- Centralize streaming SHA-256 and structured provenance envelopes in `a2ts.media` and `a2ts.models`.
- Decouple turn embeddings from list positions using stable turn identifiers (`turn.id`).
- Store enrolled sample identifiers (`sample_id`) in `VoiceProfile` to make profile enrollment strictly idempotent.
- Port deterministic French RPG/dice regex normalization from `craig2ts` into `a2ts.consolidator`.
- Reconcile CLI documentation with actual Typer commands and establish GitHub Actions CI.

**Tech Stack:** Python 3.13, faster-whisper, SpeechBrain/ECAPA, Transformers, Typer, Rich, Pydantic v2, Pytest, Ruff, Mypy.

**Spec:** `docs/spec.md`, `docs/design.md`, `../a2ts-review-2026-09-29-codex.md`

## Global Constraints
- Python >=3.13 compatibility.
- Zero mypy type errors (`mypy src tests`).
- 100% pytest test pass rate with strict isolation (no default writes outside tmp_path).
- All code, comments, docstrings, commits, and docs in English.
- No regression in existing CLI interface commands (`run`, `review`, `split`, `recluster`, `extract-audio`, `extract-vocab`, `info`).

---

### Task 1: Streaming SHA-256 Media Hashing

**Files:**
- Modify: `src/a2ts/media.py:10-20`
- Test: `tests/unit/test_media.py`

**Interfaces:**
- Consumes: `Path` to media file
- Produces: `compute_file_hash(path: Path, chunk_size: int = 65536) -> str` (hex digest of full file)

- [ ] **Step 1: Write failing unit test for full file content difference**
Add test in `tests/unit/test_media.py` verifying that two files with identical 64 KiB prefixes but different tails produce different hashes.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run pytest tests/unit/test_media.py -k test_compute_file_hash_diff_tails`

- [ ] **Step 3: Implement streaming SHA-256**
Update `compute_file_hash` in `src/a2ts/media.py` to stream the whole file in 64 KiB chunks through `hashlib.sha256()`.

- [ ] **Step 4: Verify test passes**
Run: `uv run pytest tests/unit/test_media.py`

- [ ] **Step 5: Commit**
`git commit -am "fix(media): use full streaming sha256 for media hash"`

---

### Task 2: Versioned Cache Provenance & Cache Invalidation

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `src/a2ts/cli.py`
- Test: `tests/unit/test_cache.py` or `tests/test_cli.py`

**Interfaces:**
- Consumes: media hash, model name, compute type, prompt hash, diarizer settings, language
- Produces: `TranscriptCacheProvenance`, `DiarizationCacheProvenance` models and envelope checks in `cli.py`, plus `--force` CLI option

- [ ] **Step 1: Define Provenance models in `src/a2ts/models.py`**
Define `TranscriptCacheProvenance` and `DiarizationCacheProvenance` tracking model, compute type, prompt hash, engine, threshold, num_speakers, and media hash.

- [ ] **Step 2: Write tests for cache invalidation on parameter changes and `--force`**
Test that changing `model_name` or `cluster_threshold` or passing `--force` ignores existing cache files and re-runs transcription/diarization.

- [ ] **Step 3: Update `cli.py` caching logic**
Save provenance envelope with cached JSONs; on load, verify parameters match and `force is False`; add `--force` option to `run` command.

- [ ] **Step 4: Verify tests pass**
Run: `uv run pytest tests/test_cli.py -k cache`

- [ ] **Step 5: Commit**
`git commit -am "feat(cache): add versioned provenance envelope and force flag"`

---

### Task 3: Stable Turn IDs for Embeddings & Alignment

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `src/a2ts/diarizer.py`
- Modify: `src/a2ts/timeline.py`
- Modify: `src/a2ts/cli.py`
- Test: `tests/unit/test_diarizer.py`, `tests/unit/test_timeline.py`

**Interfaces:**
- Consumes: `SpeakerTurn` / `AlignedTurn`
- Produces: Turn embeddings dictionary / mapping keyed by stable turn ID (`turn.id`) rather than list index.

- [ ] **Step 1: Write test exposing index shift when empty turns are omitted**
Test `align_words_to_speaker_turns` omitting empty turns and ensure turn embeddings maintain correct association by `turn.id`.

- [ ] **Step 2: Update `extract_embeddings_for_turns` and cache storage**
Store mapping or index list matching `turn.id`s, not implicit array indexes.

- [ ] **Step 3: Update `cli.py` review and recluster stages**
Look up embeddings by turn ID instead of zip/list index.

- [ ] **Step 4: Verify diarizer and timeline tests pass**
Run: `uv run pytest tests/unit/test_diarizer.py tests/unit/test_timeline.py`

- [ ] **Step 5: Commit**
`git commit -am "fix(diarizer): bind embeddings to turn id rather than array position"`

---

### Task 4: Idempotent Voice Profile Enrollment

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `src/a2ts/diarizer.py`
- Test: `tests/unit/test_voice_profiles.py` (or `test_diarizer.py`)

**Interfaces:**
- Consumes: `VoiceProfile`, sample embeddings, session & turn identifiers
- Produces: `enrolled_samples: list[str]` in `VoiceProfile`; repeated enrollment of same sample IDs does not alter centroid or sample count.

- [ ] **Step 1: Write failing test enrolling the same session turns twice**
Verify centroid vector and `sample_count` remain strictly identical after second enrollment.

- [ ] **Step 2: Update `VoiceProfile` and `compute_voice_profiles`**
Add `sample_ids: list[str]` to `VoiceProfile`. In `compute_voice_profiles`, only incorporate embeddings whose sample IDs are not already enrolled.

- [ ] **Step 3: Verify tests pass**
Run: `uv run pytest tests/unit/test_diarizer.py -k voice_profile`

- [ ] **Step 4: Commit**
`git commit -am "fix(profiles): make voice profile enrollment idempotent via sample IDs"`

---

### Task 5: Fix Release Gates (Mypy & Test Isolation)

**Files:**
- Modify: `src/a2ts/diarizer.py`
- Modify: `tests/unit/test_diarizer.py`
- Modify: `tests/integration/test_pipeline_e2e.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Fix mypy errors in `tests/unit/test_diarizer.py`**
Change `compute_voice_profiles` signature parameter `turns: Sequence[SpeakerTurn | AlignedTurn]` so covariant lists pass cleanly.

- [ ] **Step 2: Fix mypy errors in `tests/integration/test_pipeline_e2e.py`**
Add `assert loaded_profiles is not None` before accessing `loaded_profiles.speakers`.

- [ ] **Step 3: Verify mypy is 100% green**
Run: `uv run mypy --cache-dir /tmp/mypy_cache src tests`

- [ ] **Step 4: Fix test isolation in `tests/test_cli.py`**
Ensure tests invoking `run` pass explicit `--output` and `--voice-profiles` inside `tmp_path`, preventing attempts to write to current directory or read-only mounts.

- [ ] **Step 5: Run full pytest suite**
Run: `uv run pytest -q -p no:cacheprovider`
Expected: 100% passing.

- [ ] **Step 6: Commit**
`git commit -am "fix(types,tests): fix mypy errors and test isolation in CLI suite"`

---

### Task 6: Port Craig RPG / Dice Normalization

**Files:**
- Modify: `src/a2ts/consolidator.py`
- Create/Modify: `tests/unit/test_consolidator.py`

**Interfaces:**
- Consumes: transcript text
- Produces: `normalize_rpg_terms(text: str) -> str`

- [ ] **Step 1: Port unit tests for RPG normalization**
Port all test cases from `craig2ts` (dice patterns: 1d20, 2d6, exclusions for nouns like "20 gardes", etc.) into `tests/unit/test_consolidator.py`.

- [ ] **Step 2: Implement `normalize_rpg_terms` in `a2ts.consolidator`**
Add regex pattern and mappings into `a2ts/consolidator.py` and invoke it during transcript consolidation.

- [ ] **Step 3: Verify tests pass**
Run: `uv run pytest tests/unit/test_consolidator.py`

- [ ] **Step 4: Commit**
`git commit -am "feat(consolidator): add deterministic French RPG dice normalization"`

---

### Task 7: Dependencies & Packaging Metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/a2ts/__init__.py`

- [ ] **Step 1: Set `__version__ = "0.2.0.dev1"` in `src/a2ts/__init__.py`**
Remove placeholder "Hello from a2ts!" and expose `__version__`.

- [ ] **Step 2: Align `pyproject.toml` metadata & sdist exclusions**
Add hatchling / build configuration excluding `.agents`, `contexte/speakers.txt`, personal context from sdist. Clearly document Nemotron requirement.

- [ ] **Step 3: Test build**
Run: `uv build`

- [ ] **Step 4: Commit**
`git commit -am "chore(pkg): clean packaging metadata, version, and sdist exclusions"`

---

### Task 8: Canonical Documentation Reconciliation & README Rewrite

**Files:**
- Modify: `README.md`
- Modify: `docs/spec.md`
- Modify: `docs/runbook.md`
- Modify: `docs/design.md`
- Modify: `docs/codemap.md`

- [ ] **Step 1: Rewrite README.md**
Add alpha badge/disclaimer, prerequisites, install profiles, copy-paste quickstart, actual CLI command reference, and privacy/local model boundary.

- [ ] **Step 2: Reconcile spec, runbook, design, and codemap**
Update CLI flags, data structures, and module map to match actual code reality.

- [ ] **Step 3: Commit**
`git commit -am "docs: reconcile spec, runbook, and README with CLI reality"`

---

### Task 9: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create CI workflow**
Add workflow running on push and PR: checkout, install uv, sync dependencies, run `ruff check .`, `mypy src tests`, and `pytest`.

- [ ] **Step 2: Verify lint and tests locally**
Run: `uv run ruff check .`, `uv run mypy --cache-dir /tmp/mypy_cache src tests`, `uv run pytest -q -p no:cacheprovider`

- [ ] **Step 3: Commit**
`git commit -am "ci: add GitHub Actions workflow for lint, types, and tests"`
