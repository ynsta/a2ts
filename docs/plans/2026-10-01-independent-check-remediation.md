# Independent Check Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all residues, incomplete fixes, and edge cases identified in the 2026-10-01 independent check reviews (`a2ts-review-2026-10-01-codex-check.md` and `a2ts-review-2026-10-01-opus-check.md`).

**Architecture:** Strictly close all turn-embedding unvalidated loading fast-paths, enforce entity fingerprint checking on load, strip control-token sequences from biasing prompts, standardize 180-token prompt budgets across all subcommands, harden recluster cache identity and split handling, handle non-UTF-8 mapping files, and fix link and documentation drift.

**Tech Stack:** Python 3.13, Pydantic v2, Pytest, Ruff, Mypy, Faster-Whisper, SpeechBrain, CTranslate2, Typer, Rich.

**Spec:** [docs/spec.md](file:///home/stany/Work/a2ts/docs/spec.md), [docs/design.md](file:///home/stany/Work/a2ts/docs/design.md), [a2ts-review-2026-10-01-codex-check.md](file:///home/stany/Work/a2ts-review-2026-10-01-codex-check.md), [a2ts-review-2026-10-01-opus-check.md](file:///home/stany/Work/a2ts-review-2026-10-01-opus-check.md).

## Global Constraints
- Static checking: `uv run --frozen --no-sync mypy` must pass with 0 issues (`disallow_untyped_defs = true`).
- Lint & format: `uv run --frozen --no-sync ruff check --no-cache` and `uv run --frozen --no-sync ruff format --check --no-cache` must pass with 0 errors.
- Test suite: `uv run --frozen --no-sync pytest -p no:cacheprovider` must remain 100% green without regressions.
- No blind trust: Every step must be independently verified by inspecting disk state and running concrete test probes.

---

### Task 1: Turn & Segment Embedding Validation on All Fast Paths (F-01, F-02, Codex #1)

**Files:**
- Modify: `src/a2ts/diarizer.py:270-320, 390-440, 480-500, 570-600, 800-815`
- Modify: `src/a2ts/cli.py:555-585, 660-705, 900-925`
- Test: `tests/test_embedding_provenance.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- `load_segment_embeddings`:
  - `expected_transcript_provenance_hash` cannot be None (treat None as cache miss).
  - If `segments` is provided: compute fingerprint and verify `prov.entity_fingerprint == expected_fingerprint`.
- `load_turn_embeddings`:
  - `expected_diarization_provenance_hash` cannot be None (treat None as cache miss).
  - If `turns` is provided: compute fingerprint and verify `prov.entity_fingerprint == expected_fingerprint`.
- `extract_embeddings_for_segments`: pass `expected_transcript_provenance_hash=transcript_provenance_hash`, `segments=segments`.
- `extract_embeddings_for_turns`: pass `expected_diarization_provenance_hash=diarization_provenance_hash`, `turns=turns`, `media_hash=media_hash`.
- `diarize_segments`: bounds check `if 0 <= valid_idx < len(full_labels): full_labels[valid_idx] = ...`.
- CLI `run` auto-enrollment:
  - Route through `load_turn_embeddings(..., expected_diarization_provenance_hash=diar_prov_hash, turns=speaker_turns)`.
- CLI `review`:
  - Route through `load_turn_embeddings(..., expected_diarization_provenance_hash=diar_prov_hash, turns=turns)`.

- [ ] **Step 1: Write failing probe test for unvalidated auto-enrollment and wildcard hash**
Add test in `tests/test_cli.py` reproducing probe: stale turn cache with wrong vector, no pre-existing `voice_profiles.json`, fresh diarization hash -> verify enrolled centroid in `voice_profiles.json` matches fresh vector, NOT stale vector. Add test in `tests/test_embedding_provenance.py` that `load_turn_embeddings` with `expected_diarization_provenance_hash=None` returns `None`.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run --frozen --no-sync pytest tests/test_cli.py -k test_run_auto_enroll_stale_turn_cache -v`
Expected: FAIL.

- [ ] **Step 3: Implement strict validation across diarizer and CLI fast paths**
Update `src/a2ts/diarizer.py` and `src/a2ts/cli.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run --frozen --no-sync pytest tests/test_embedding_provenance.py tests/test_cli.py -k "test_run_auto_enroll or test_embedding" -v`
Expected: PASS.

---

### Task 2: Prompt Control Token Sanitization & 180-Token Budget Parity (Codex #8, F-09)

**Files:**
- Modify: `src/a2ts/vocab.py:110-160`
- Modify: `src/a2ts/craig.py:250-265`
- Modify: `src/a2ts/cli.py:240-255`
- Modify: `docs/spec.md:40-60`
- Modify: `docs/design.md:20-60, 90-100, 150-160`
- Modify: `README.md:20-30`
- Test: `tests/unit/test_vocab.py`
- Test: `tests/unit/test_craig.py`

**Interfaces:**
- `clean_entity_name`: strip `<|...|>` pattern so special control tokens cannot enter the prompt string.
- `build_biasing_prompt`: sanitize all candidate strings.
- `get_tokenizer`: check `os.environ.get("HF_HOME")` and `os.environ.get("HF_HUB_CACHE")`.
- `build_craig_prompt`: default `max_tokens=180`.
- `extract-vocab`: default `--max-tokens` to 180.
- Documentation: update 220/224 token references to 180 tokens (buffer 224).

- [ ] **Step 1: Write failing test for control-token stripping and craig 180 token default**
Add test in `tests/unit/test_vocab.py` asserting that entity with `<|endoftext|>` or `<|startoftranscript|>` has tags stripped from final prompt. Add test in `tests/unit/test_craig.py` for 180 token default.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run --frozen --no-sync pytest tests/unit/test_vocab.py -k test_special_tokens -v`
Expected: FAIL.

- [ ] **Step 3: Implement control token sanitization and budget alignment**
Update `src/a2ts/vocab.py`, `src/a2ts/craig.py`, `src/a2ts/cli.py`, and documentation.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run --frozen --no-sync pytest tests/unit/test_vocab.py tests/unit/test_craig.py -v`
Expected: PASS.

---

### Task 3: Recluster Cache Identity & Split Reset (F-10)

**Files:**
- Modify: `src/a2ts/cli.py:1210-1235, 1330-1370`
- Test: `tests/test_cli.py`

**Interfaces:**
- In `recluster`:
  - Calculate `expected_trans_hash` by loading `TranscriptCacheFile` from `transcript_path` (or session metadata), without falling back to `"no_prompt"`.
  - Read diarization cache typed via `DiarizationCacheFile.model_validate(json.loads(...))`.
  - Reset `mapping.splits = []` because cluster IDs have been reassigned.

- [ ] **Step 1: Write failing test for recluster followed by run cache hit and split reset**
Add test in `tests/test_cli.py` verifying that after `a2ts recluster`, running `a2ts run` recognizes the diarization cache and does not re-diarize, and `mapping.splits` is cleared.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run --frozen --no-sync pytest tests/test_cli.py -k test_recluster -v`
Expected: FAIL.

- [ ] **Step 3: Implement recluster fixes**
Update `src/a2ts/cli.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run --frozen --no-sync pytest tests/test_cli.py -k test_recluster -v`
Expected: PASS.

---

### Task 4: Robust Mapping Error Handling, Voxtral Truncation Marker & CLI Polish (F-11, F-18, F-19)

**Files:**
- Modify: `src/a2ts/speaker_review.py:135-165`
- Modify: `src/a2ts/transcriber.py:290-320`
- Modify: `src/a2ts/cli.py:230-245, 430-445, 940-955`
- Test: `tests/unit/test_speaker_review.py`
- Test: `tests/unit/test_transcriber.py`

**Interfaces:**
- In `speaker_review.py`:
  - Catch `UnicodeDecodeError` in `load_speakers_mapping`.
  - Generate timestamped backup: `mapping_path.with_name(f"{mapping_path.stem}.corrupt.{int(time.time())}.json")`.
- In `transcriber.py`:
  - In `VoxtralEngine.transcribe()`: if output was truncated, set `is_truncated = True` on RawSegment or record in metadata so truncated output is not treated as complete.
- In `cli.py`:
  - Catch `RuntimeError` around `extract_audio_to_wav` in `run()` to print friendly error on missing ffmpeg.
  - Wrap profile match print in `escape()`.

- [ ] **Step 1: Write failing test for non-UTF8 mapping file**
Add test in `tests/unit/test_speaker_review.py` with invalid non-UTF8 bytes in `speakers_mapping.json`.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run --frozen --no-sync pytest tests/unit/test_speaker_review.py -k test_corrupt_mapping -v`
Expected: FAIL.

- [ ] **Step 3: Implement error handling, Voxtral truncation guard, and escaping**
Update `speaker_review.py`, `transcriber.py`, `cli.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run --frozen --no-sync pytest tests/unit/test_speaker_review.py tests/unit/test_transcriber.py -v`
Expected: PASS.

---

### Task 5: Link Hygiene & Docs Reconciliation (F-04, F-05, F-14)

**Files:**
- Modify: `docs/archive/2026-10-01-pre-release-reviews-remediation.md`
- Modify: `docs/spec.md`
- Modify: `docs/runbook.md`
- Modify: `.agents/last-docs-consolidate`

**Interfaces:**
- Replace absolute `file:///home/stany/...` links in `docs/archive/2026-10-01-pre-release-reviews-remediation.md` with relative links.
- In `docs/spec.md`: fix `extract-vocab` description (outputs to stdout) and compute-type default.
- In `docs/runbook.md`: clarify Nemotron memory requirements and long-audio limits.
- Update `.agents/last-docs-consolidate`.

- [ ] **Step 1: Update documentation and links**
Update markdown files and verify all links resolve.

- [ ] **Step 2: Run full quality gates**
Run: `uv run --frozen --no-sync ruff check --no-cache && uv run --frozen --no-sync ruff format --check --no-cache && uv run --frozen --no-sync mypy --cache-dir /tmp/a2ts-mypy && uv run --frozen --no-sync pytest -p no:cacheprovider`
Expected: All 100% green.
