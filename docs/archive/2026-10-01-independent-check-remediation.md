# [ARCHIVED] Independent Check Remediation Plan

> **ARCHIVAL NOTICE:** This plan has been fully executed, verified, and reviewed. Canonical specifications are consolidated in `docs/spec.md`, `docs/design.md`, `docs/codemap.md`, `docs/runbook.md`, and `README.md`.

**Goal:** Address all residues, incomplete fixes, and edge cases identified in the 2026-10-01 independent check reviews (`a2ts-review-2026-10-01-codex-check.md` and `a2ts-review-2026-10-01-opus-check.md`).

**Architecture:** Strictly close all turn-embedding unvalidated loading fast-paths, enforce entity fingerprint checking on load, strip control-token sequences from biasing prompts, standardize 180-token prompt budgets across all subcommands, harden recluster cache identity and split handling, handle non-UTF-8 mapping files, and fix link and documentation drift.

**Tech Stack:** Python 3.13, Pydantic v2, Pytest, Ruff, Mypy, Faster-Whisper, SpeechBrain, CTranslate2, Typer, Rich.

**Spec:** [docs/spec.md](../spec.md), [docs/design.md](../design.md), `a2ts-review-2026-10-01-codex-check.md`, `a2ts-review-2026-10-01-opus-check.md`.

## Global Constraints
- Static checking: `uv run --frozen --no-sync mypy` must pass with 0 issues (`disallow_untyped_defs = true`).
- Lint & format: `uv run --frozen --no-sync ruff check --no-cache` and `uv run --frozen --no-sync ruff format --check --no-cache` must pass with 0 errors.
- Test suite: `uv run --frozen --no-sync pytest -p no:cacheprovider` must remain 100% green without regressions.
- No blind trust: Every step must be independently verified by inspecting disk state and running concrete test probes.

---

### Task 1: Turn & Segment Embedding Validation on All Fast Paths (F-01, F-02, Codex #1)

**Files:**
- Modify: `src/a2ts/diarizer.py`
- Modify: `src/a2ts/cli.py`
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

- [x] **Step 1: Write failing probe test for unvalidated auto-enrollment and wildcard hash**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement strict validation across diarizer and CLI fast paths**
- [x] **Step 4: Run test to verify it passes**

---

### Task 2: Prompt Control Token Sanitization & 180-Token Budget Parity (Codex #8, F-09)

**Files:**
- Modify: `src/a2ts/vocab.py`
- Modify: `src/a2ts/craig.py`
- Modify: `src/a2ts/cli.py`
- Modify: `docs/spec.md`
- Modify: `docs/design.md`
- Modify: `README.md`
- Test: `tests/unit/test_vocab.py`
- Test: `tests/unit/test_craig.py`

**Interfaces:**
- `clean_entity_name`: strip `<|...|>` pattern so special control tokens cannot enter the prompt string.
- `build_biasing_prompt`: sanitize all candidate strings.
- `get_tokenizer`: check `os.environ.get("HF_HOME")` and `os.environ.get("HF_HUB_CACHE")`.
- `build_craig_prompt`: default `max_tokens=180`.
- `extract-vocab`: default `--max-tokens` to 180.
- Documentation: update 220/224 token references to 180 tokens (buffer 224).

- [x] **Step 1: Write failing test for control-token stripping and craig 180 token default**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement control token sanitization and budget alignment**
- [x] **Step 4: Run test to verify it passes**

---

### Task 3: Recluster Cache Identity & Split Reset (F-10)

**Files:**
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- In `recluster`:
  - Calculate `expected_trans_hash` by loading `TranscriptCacheFile` from `transcript_path` (or session metadata), without falling back to `"no_prompt"`.
  - Read diarization cache typed via `DiarizationCacheFile.model_validate(json.loads(...))`.
  - Reset `mapping.splits = []` because cluster IDs have been reassigned.

- [x] **Step 1: Write failing test for recluster followed by run cache hit and split reset**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement recluster fixes**
- [x] **Step 4: Run test to verify it passes**

---

### Task 4: Robust Mapping Error Handling, Voxtral Truncation Marker & CLI Polish (F-11, F-18, F-19)

**Files:**
- Modify: `src/a2ts/speaker_review.py`
- Modify: `src/a2ts/transcriber.py`
- Modify: `src/a2ts/cli.py`
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

- [x] **Step 1: Write failing test for non-UTF8 mapping file**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement error handling, Voxtral truncation guard, and escaping**
- [x] **Step 4: Run test to verify it passes**

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

- [x] **Step 1: Update documentation and links**
- [x] **Step 2: Run full quality gates**
