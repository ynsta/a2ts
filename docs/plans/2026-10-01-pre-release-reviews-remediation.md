# Pre-Release Reviews Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all high, blocker, medium, and publication findings identified in the 2026-10-01 Codex and Opus pre-publication reviews for a2ts.

**Architecture:** Strengthen cache provenance bindings (transcript/diarization hashes and entity counts), align speaker resolution between display and enrollment, enforce atomic writes and unique temporary files across all outputs, make device/compute defaults adaptive, add Voxtral and subprocess guardrails, and harmonize documentation, CI, and packaging metadata.

**Tech Stack:** Python 3.13, Pydantic v2, Pytest, Ruff, Mypy, Faster-Whisper, SpeechBrain, CTranslate2, Typer, Rich.

**Spec:** [docs/spec.md](file:///home/stany/Work/a2ts/docs/spec.md), [docs/design.md](file:///home/stany/Work/a2ts/docs/design.md), [a2ts-review-2026-10-01-codex.md](file:///home/stany/Work/a2ts-review-2026-10-01-codex.md), [a2ts-review-2026-10-01-opus.md](file:///home/stany/Work/a2ts-review-2026-10-01-opus.md).

## Global Constraints

- Static checking: `uv run --frozen --no-sync mypy` must pass with 0 issues (`disallow_untyped_defs = true`).
- Lint & format: `uv run --frozen --no-sync ruff check --no-cache` and `uv run --frozen --no-sync ruff format --check --no-cache` must pass with 0 errors.
- Test suite: `uv run --frozen --no-sync pytest -p no:cacheprovider` must remain 100% green without regressions.
- Atomic I/O invariant: All state files, mappings, embeddings, and transcript outputs must use atomic writes via unique sibling temporary files.
- Observable behavior over internal mechanics: test contracts and failure modes rather than private helpers.

---

### Task 1: Embedding Cache Provenance & Foreign Glob Removal (F-01, F-02, Codex #1)

**Files:**
- Modify: `src/a2ts/models.py:190-199`
- Modify: `src/a2ts/diarizer.py:196-234, 320-358, 420-435`
- Modify: `src/a2ts/cli.py:440-470, 785-805, 1095-1115`
- Test: `tests/test_embedding_provenance.py`

**Interfaces:**
- `EmbeddingCacheProvenance`: add `transcript_provenance_hash: str | None = None`, `diarization_provenance_hash: str | None = None`, `entity_fingerprint: str | None = None`.
- `load_segment_embeddings`: validates `media_hash`, `embedding_model`, `count`, `transcript_provenance_hash`.
- `load_turn_embeddings`: validates `media_hash`, `embedding_model`, `count`, `diarization_provenance_hash`.
- CLI `review` and `recluster`: delete glob fallbacks (`*_turn_embeddings.npy`, `*_segment_embeddings.npy`).

- [ ] **Step 1: Write failing tests for segment and turn embedding provenance validation**
Add tests in `tests/test_embedding_provenance.py` verifying:
1. Loading segment embeddings fails / returns None when `transcript_provenance_hash` changes or is missing.
2. Loading turn embeddings fails / returns None when `diarization_provenance_hash` changes or is missing.
3. Review and recluster fail cleanly instead of loading embeddings from other files.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run --frozen --no-sync pytest tests/test_embedding_provenance.py -v`
Expected: FAIL on provenance mismatch checks.

- [ ] **Step 3: Update models and diarizer embedding save/load logic**
Update `EmbeddingCacheProvenance` in `src/a2ts/models.py`. Update `save_segment_embeddings`, `load_segment_embeddings`, `save_turn_embeddings`, and `load_turn_embeddings` in `src/a2ts/diarizer.py`. Remove foreign glob fallbacks in `src/a2ts/cli.py`.

- [ ] **Step 4: Run tests to verify they pass**
Run: `uv run --frozen --no-sync pytest tests/test_embedding_provenance.py -v`
Expected: PASS.

---

### Task 2: Slice-Level Speaker Override Alignment in Voice Enrollment (Codex #2)

**Files:**
- Modify: `src/a2ts/speaker_review.py:38-65`
- Modify: `src/a2ts/diarizer.py:680-730`
- Test: `tests/unit/test_speaker_review.py`
- Test: `tests/unit/test_diarizer.py`

**Interfaces:**
- `resolve_speaker_for_turn(turn_id: int, cluster_id: str, time_slice_id: int, mapping: SpeakersMapping, fallback_speaker: str | None = None) -> str`
- Priority:
  1. `turn_overrides[turn_id]`
  2. `slice_overrides[str(time_slice_id)][cluster_id]`
  3. `cluster_defaults[cluster_id]`
  4. `fallback_speaker`
  5. `cluster_id`

- [ ] **Step 1: Write failing test reproducing slice override enrollment discrepancy**
Add a test in `tests/unit/test_diarizer.py`: Given default `Alice`, slice 1 override `Bob`, and a turn in slice 1 with manual label source, `compute_voice_profiles_from_turns` must enroll voice samples for `Bob`, NOT `Alice`.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run --frozen --no-sync pytest tests/unit/test_diarizer.py -k test_slice_override_enrollment -v`
Expected: FAIL (enrolled for 'Alice').

- [ ] **Step 3: Implement unified speaker resolution helper and use in diarizer & review**
Implement `resolve_speaker_for_turn` in `src/a2ts/speaker_review.py` and call it from `apply_speakers_mapping` and `compute_voice_profiles_from_turns`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run --frozen --no-sync pytest tests/unit/test_diarizer.py -k test_slice_override_enrollment -v`
Expected: PASS.

---

### Task 3: Atomic I/O & Concurrency Hardening (F-06, F-15, Codex #5, #6)

**Files:**
- Modify: `src/a2ts/cache.py:20-27`
- Modify: `src/a2ts/speaker_review.py:80-95, 230-260`
- Modify: `src/a2ts/diarizer.py:285-305, 415-435, 775-800`
- Modify: `src/a2ts/cli.py:640, 935, 974, 999, 1200-1215, 1255`
- Test: `tests/unit/test_cache.py`

**Interfaces:**
- `atomic_write_text(path: Path, content: str) -> None`: uses `tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)` and `os.replace`.
- `atomic_save_numpy(path: Path, arr: np.ndarray) -> None`: saves numpy array via temporary sibling file and `os.replace`.
- Periodic review persistence: save mapping after each cluster reviewed.

- [ ] **Step 1: Write concurrent writer and atomic numpy tests**
Add tests in `tests/unit/test_cache.py` demonstrating concurrent writes to same target path without `FileNotFoundError`, and testing `atomic_save_numpy`.

- [ ] **Step 2: Run tests to verify failure**
Run: `uv run --frozen --no-sync pytest tests/unit/test_cache.py -k test_concurrent -v`
Expected: FAIL.

- [ ] **Step 3: Implement unique temporary file atomic write and numpy save**
Update `atomic_write_text` in `src/a2ts/cache.py`. Add `atomic_save_numpy`. Replace bare `write_text` and direct `np.save` calls across `speaker_review.py`, `diarizer.py`, and `cli.py`.

- [ ] **Step 4: Run tests to verify pass**
Run: `uv run --frozen --no-sync pytest tests/unit/test_cache.py -v`
Expected: PASS.

---

### Task 4: Recluster Fixes & Diarization Fallback Cache (F-10, F-13)

**Files:**
- Modify: `src/a2ts/cli.py:1085-1240`
- Modify: `src/a2ts/cache.py:90-125`
- Test: `tests/test_cli.py`
- Test: `tests/unit/test_cache.py`

**Interfaces:**
- `recluster`:
  - Check if session used Nemotron (no segment embeddings) -> exit with clear explanation.
  - Save diarization cache as `DiarizationCacheFile` with `DiarizationCacheProvenance` using `atomic_write_text`.
  - Prune or reset stale cluster defaults in `SpeakersMapping`.
- Diarizer engine fallback:
  - Cache provenance records resolved engine appropriately.

- [ ] **Step 1: Write failing test for recluster writing valid cache format and handling Nemotron**
Add test in `tests/test_cli.py` checking `a2ts recluster` creates a valid `DiarizationCacheFile` structure (with provenance and turns) rather than a bare list.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run --frozen --no-sync pytest tests/test_cli.py -k test_recluster -v`
Expected: FAIL.

- [ ] **Step 3: Implement recluster fixes**
Update `src/a2ts/cli.py` to write `DiarizationCacheFile` using `save_diarization_cache`, reset obsolete `cluster_defaults`, and guard against Nemotron.

- [ ] **Step 4: Run test to verify pass**
Run: `uv run --frozen --no-sync pytest tests/test_cli.py -k test_recluster -v`
Expected: PASS.

---

### Task 5: Platform & Device Selection Defaults (F-12, Codex #7)

**Files:**
- Modify: `src/a2ts/cli.py:150-170, 350-370, 1300-1320`
- Modify: `src/a2ts/transcriber.py:150-170, 220-240`
- Modify: `README.md:40-60`
- Modify: `docs/spec.md:50-80`
- Test: `tests/test_cli.py`

**Interfaces:**
- CLI arguments `--device` defaults to `"auto"`.
- If device is `"auto"`: resolve to `"cuda"` if `torch.cuda.is_available()` else `"cpu"`.
- CLI argument `--compute-type` defaults to `None`; resolved dynamically based on resolved device: `"float16"` for CUDA, `"int8"` for CPU.
- Pass device and compute_type to Voxtral and Whisper consistently.

- [ ] **Step 1: Write failing test for device=auto and CPU default resolution**
Add test in `tests/test_cli.py` verifying device resolution on non-CUDA / CPU environments.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run --frozen --no-sync pytest tests/test_cli.py -k test_device_auto -v`
Expected: FAIL.

- [ ] **Step 3: Implement device/compute_type auto-detection**
Update `cli.py` and `transcriber.py`. Update README to remove "CPU/Metal" claim.

- [ ] **Step 4: Run test to verify pass**
Run: `uv run --frozen --no-sync pytest tests/test_cli.py -k test_device_auto -v`
Expected: PASS.

---

### Task 6: Voxtral Guardrails & Subprocess Timeouts (F-11, F-18, Codex #3, #9)

**Files:**
- Modify: `src/a2ts/transcriber.py:240-280`
- Modify: `src/a2ts/media.py:30-80`
- Modify: `src/a2ts/cli.py:350-380`
- Test: `tests/unit/test_transcriber.py`
- Test: `tests/unit/test_media.py`

**Interfaces:**
- `VoxtralEngine`: detect generation cap exhaustion; warn/error if `--diarize` is attempted on Voxtral untimed output.
- `media.py`: `ffprobe` (timeout=60s), `ffmpeg` (timeout=300s), catch `FileNotFoundError` and raise actionable `RuntimeError("ffmpeg not found in PATH")`.

- [ ] **Step 1: Write failing tests for media timeouts and missing tool handling**
Add tests in `tests/unit/test_media.py` mocking `subprocess.run` timeout and `FileNotFoundError`.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run --frozen --no-sync pytest tests/unit/test_media.py -k test_timeout -v`
Expected: FAIL.

- [ ] **Step 3: Implement timeouts and exception wrapping in media.py, add Voxtral guardrails**
Update `media.py`, `transcriber.py`, `cli.py`.

- [ ] **Step 4: Run test to verify pass**
Run: `uv run --frozen --no-sync pytest tests/unit/test_media.py -v`
Expected: PASS.

---

### Task 7: Vocab Disallowed Special Tokens & Prompt Budgeting (Codex #8, F-09)

**Files:**
- Modify: `src/a2ts/vocab.py:130-155`
- Test: `tests/unit/test_vocab.py`

**Interfaces:**
- `build_biasing_prompt`:
  - `tokenizer.encode(candidate_str, disallowed_special=())`.
  - Use safety margin (max_tokens=180 or Whisper tokenizer) and ensure key speaker names are preserved.

- [ ] **Step 1: Write failing test with <|endoftext|> in lore entity name**
Add test in `tests/unit/test_vocab.py` with `EntityRecord(name="<|endoftext|>", kind="custom")`.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run --frozen --no-sync pytest tests/unit/test_vocab.py -k test_special_tokens -v`
Expected: FAIL (`ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'`).

- [ ] **Step 3: Implement disallowed_special=() and budget margin**
Update `src/a2ts/vocab.py`.

- [ ] **Step 4: Run test to verify pass**
Run: `uv run --frozen --no-sync pytest tests/unit/test_vocab.py -v`
Expected: PASS.

---

### Task 8: CLI Robustness & Rich Markup Escaping (F-16, F-19, F-20, F-21, F-25)

**Files:**
- Modify: `src/a2ts/speaker_review.py:90-100, 200-300`
- Modify: `src/a2ts/cli.py:120-135, 330-345, 930-945`
- Modify: `src/a2ts/models.py:80-93`
- Test: `tests/unit/test_speaker_review.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Rich markup escaping: `from rich.markup import escape` on user strings and transcripts.
- `load_speakers_mapping`: handles invalid JSON gracefully.
- `SessionMetadata`: add `rpg_normalize: bool = True`.
- `a2ts info`: print detected CUDA devices, VRAM status, available engines.
- `interactive`: default to `sys.stdin.isatty()`.

- [ ] **Step 1: Write failing tests for markup escaping and corrupt mapping handling**
Add tests in `tests/unit/test_speaker_review.py` for transcript text with `[/i]` and malformed JSON in `load_speakers_mapping`.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run --frozen --no-sync pytest tests/unit/test_speaker_review.py -k test_rich_escape -v`
Expected: FAIL.

- [ ] **Step 3: Implement escaping, robust mapping loading, session metadata, info command**
Update `speaker_review.py`, `cli.py`, `models.py`.

- [ ] **Step 4: Run test to verify pass**
Run: `uv run --frozen --no-sync pytest tests/unit/test_speaker_review.py -v`
Expected: PASS.

---

### Task 9: CI Workflow, Lockfile & Dependency Metadata (F-03, F-31, F-29)

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Test: Run `ruff check`, `ruff format --check`, `mypy`, `pytest`

**Interfaces:**
- CI: add `permissions: { contents: read }`, `uv sync --locked --dev`, `uv run ruff format --check .`.
- `pyproject.toml`: align `torch` floors, add `[project.urls]`, classifiers, keywords.

- [ ] **Step 1: Update .github/workflows/ci.yml and pyproject.toml**
Add permissions block, `--locked` flag to sync, format check step, project metadata.

- [ ] **Step 2: Run all static checks locally**
Run: `uv run --frozen --no-sync ruff check --no-cache && uv run --frozen --no-sync ruff format --check --no-cache && uv run --frozen --no-sync mypy --cache-dir /tmp/a2ts-mypy`
Expected: All PASS.

---

### Task 10: Publication Hygiene, Community Files & Disclaimers (F-05, F-07, F-08, User Request)

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `NOTICE`
- Modify: `README.md`
- Modify: `docs/spec.md`
- Modify: `docs/archive/2026-09-29-craig-adapter.md`
- Modify: `docs/README.md`
- Modify: `.agents/last-docs-consolidate`

**Interfaces:**
- Prominent experimental status disclaimer in README and spec ("experimental project, potentially useful for audio/video transcription and tabletop RPG sessions, but provided as-is without strong guarantees of stability or correctness").
- Voice biometrics privacy section in README.
- Apache-2.0 license notice & community files.
- Fix broken link in archive doc.

- [ ] **Step 1: Create community files and update documentation**
Create `CONTRIBUTING.md`, `SECURITY.md`, `NOTICE`. Update `README.md`, `docs/spec.md`, archive link.

- [ ] **Step 2: Run docs verification & full test suite**
Run: `uv run --frozen --no-sync pytest -p no:cacheprovider`
Expected: 100% green.
