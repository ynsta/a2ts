# Craig Multi-Track Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a first-class Craig Discord multi-track recording adapter in `a2ts` via a new `a2ts craig <recording_dir>` command, enabling direct transcription, timeline interleaving, and Markdown rendering without acoustic diarization.

**Architecture:**
- Create `src/a2ts/craig.py` with multi-track discovery (`^\d+-.*\.flac`), Craig `info.txt` parsing, and `speakers.md` role/identity extraction.
- Transcribe individual tracks using Faster-Whisper with in-place `.transcripts/` provenance caching and `--force` support.
- Interleave multi-track segments chronologically into `AlignedTurn` structures with dual speaker labels (`Character (username)` or `MJ (username)`), bypassing acoustic diarization.
- Route turns through existing `debounce_consecutive_turns()` (including deterministic French RPG dice normalization) and render Markdown transcripts.

**Tech Stack:** Python 3.13, Faster-Whisper, Typer, Rich, Pydantic v2, Pytest, Mypy, Ruff.

**Spec:** [docs/plans/2026-09-29-craig-adapter-design.md](file:///home/stany/Work/a2ts/docs/plans/2026-09-29-craig-adapter-design.md)

## Global Constraints
- Python >=3.13 compatibility.
- 0 mypy errors (`mypy src tests`).
- 0 ruff errors (`ruff check .`).
- 100% pytest pass rate with strict isolation.
- English for all code, comments, identifiers, docstrings, commits, and docs.
- No regression in existing CLI commands (`run`, `review`, `recluster`, `split`, `extract-audio`, `extract-vocab`, `info`).

---

### Task 1: Data Models for Craig Multi-Track

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `tests/unit/test_models.py`

**Interfaces:**
- Produces: `SpeakerInfo`, `TrackCacheProvenance`, `TrackCacheFile` in `a2ts.models`

- [ ] **Step 1: Write failing unit test for Craig models**
Add tests in `tests/unit/test_models.py` testing serialization and validation of `SpeakerInfo`, `TrackCacheProvenance`, and `TrackCacheFile`.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run pytest -q -p no:cacheprovider tests/unit/test_models.py -k test_craig_models`

- [ ] **Step 3: Implement models in `src/a2ts/models.py`**
Add `SpeakerInfo`, `TrackCacheProvenance`, `TrackCacheFile`.

- [ ] **Step 4: Run test to verify pass**
Run: `uv run pytest -q -p no:cacheprovider tests/unit/test_models.py`

- [ ] **Step 5: Commit**
`git commit -am "feat(models): add SpeakerInfo, TrackCacheProvenance, and TrackCacheFile"`

---

### Task 2: Craig Track Discovery, Username Parsing & File Readers

**Files:**
- Create: `src/a2ts/craig.py`
- Create: `tests/unit/test_craig.py`

**Interfaces:**
- Produces: `discover_tracks(recording_dir: Path) -> list[Path]`, `parse_track_username(path: Path) -> str`, `parse_speakers_file(path: Path) -> dict[str, SpeakerInfo]`, `parse_info_file(path: Path) -> dict[str, Any]`, `find_speakers_file(recording_dir: Path, custom_path: Path | None = None) -> Path | None`.

- [ ] **Step 1: Write failing unit tests for track discovery and parsing**
Add unit tests in `tests/unit/test_craig.py` covering:
- `test_discover_tracks`: sorting `1-merrow1.flac`, `2-tessaro.flac`, `10-carol.flac`
- `test_parse_track_username`: `1-merrow1.flac` -> `merrow1`
- `test_parse_speakers_file`: parsing real format (`* tessaro: Le MJ ...`, `* merrow1: Merrow Ashdale, barde humain, surnoms: (Mimi)`)
- `test_parse_info_file`: parsing real Craig `info.txt` format (`Guild:`, `Channel:`, `Start time:`, `Tracks:`)
- `test_find_speakers_file`: discovering `speakers.md` in folder or parent

- [ ] **Step 2: Run test to verify failure**
Run: `uv run pytest -q -p no:cacheprovider tests/unit/test_craig.py`

- [ ] **Step 3: Implement discovery and parsing in `src/a2ts/craig.py`**
Implement the 5 functions with robust regex parsing.

- [ ] **Step 4: Run test to verify pass**
Run: `uv run pytest -q -p no:cacheprovider tests/unit/test_craig.py`

- [ ] **Step 5: Commit**
`git commit -am "feat(craig): implement track discovery, username extraction, and file parsers"`

---

### Task 3: Craig Biasing Prompt & Track Transcription with Caching

**Files:**
- Modify: `src/a2ts/craig.py`
- Modify: `tests/unit/test_craig.py`

**Interfaces:**
- Produces: `build_craig_prompt(speakers: dict[str, SpeakerInfo], context_dir: Path | None = None) -> str`, `transcribe_craig_track(...) -> list[RawSegment]`

- [ ] **Step 1: Write failing unit tests for prompt and track caching**
In `tests/unit/test_craig.py`:
- `test_build_craig_prompt`: merges speaker names, character names, and context lore into token-budgeted prompt.
- `test_transcribe_craig_track_caching`: verifies caching in `.transcripts/<track_stem>.json`, provenance validation, and `--force` invalidation.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run pytest -q -p no:cacheprovider tests/unit/test_craig.py -k prompt`

- [ ] **Step 3: Implement prompt builder and cached transcription in `src/a2ts/craig.py`**
Implement `build_craig_prompt` and `transcribe_craig_track` with atomic file I/O and provenance validation.

- [ ] **Step 4: Run test to verify pass**
Run: `uv run pytest -q -p no:cacheprovider tests/unit/test_craig.py`

- [ ] **Step 5: Commit**
`git commit -am "feat(craig): add prompt builder and cached track transcription"`

---

### Task 4: Multi-Track Interleaving & Speaker Turn Formatting

**Files:**
- Modify: `src/a2ts/craig.py`
- Modify: `tests/unit/test_craig.py`

**Interfaces:**
- Produces: `merge_craig_tracks_to_turns(track_segments: list[tuple[str, RawSegment]], speakers: dict[str, SpeakerInfo]) -> list[AlignedTurn]`

- [ ] **Step 1: Write failing unit tests for timeline merging**
In `tests/unit/test_craig.py`:
- `test_merge_craig_tracks_to_turns`: verifies segments from multiple tracks are ordered by `(start, track_id)`, speaker names resolve to `Character (username)` or `MJ (username)`, or fallback `username`.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run pytest -q -p no:cacheprovider tests/unit/test_craig.py -k merge`

- [ ] **Step 3: Implement `merge_craig_tracks_to_turns` in `src/a2ts/craig.py`**
Map segments to `AlignedTurn`, resolving speaker identities and assigning sequential turn IDs.

- [ ] **Step 4: Run test to verify pass**
Run: `uv run pytest -q -p no:cacheprovider tests/unit/test_craig.py`

- [ ] **Step 5: Commit**
`git commit -am "feat(craig): implement multi-track segment interleaving and turn alignment"`

---

### Task 5: Typer CLI Subcommand `a2ts craig` & Pipeline Integration

**Files:**
- Modify: `src/a2ts/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Exposes: `a2ts craig <recording_dir> [OPTIONS]` CLI subcommand

- [ ] **Step 1: Write failing CLI test for `a2ts craig`**
In `tests/test_cli.py`:
- `test_craig_command_e2e_mocked`: mock Faster-Whisper transcribe, verify `a2ts craig <dir>` runs discovery, transcription, debouncing, dice normalization, and outputs `transcript.md`.

- [ ] **Step 2: Run test to verify failure**
Run: `uv run pytest -q -p no:cacheprovider tests/test_cli.py -k test_craig_command`

- [ ] **Step 3: Implement `craig` subcommand in `src/a2ts/cli.py`**
Add `@app.command("craig")` with all options (`--model-name`, `--device`, `--compute-type`, `--context-dir`, `--speakers-file`, `--output`, `--debounce`, `--force`, `--refine`).

- [ ] **Step 4: Run test to verify pass**
Run: `uv run pytest -q -p no:cacheprovider tests/test_cli.py`

- [ ] **Step 5: Commit**
`git commit -am "feat(cli): add craig subcommand for multi-track Discord recordings"`

---

### Task 6: Canonical Documentation Reconciliation & Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/spec.md`
- Modify: `docs/design.md`
- Modify: `docs/codemap.md`
- Modify: `docs/runbook.md`
- Modify: `.agents/last-docs-consolidate`

- [ ] **Step 1: Update documentation**
Document `a2ts craig` subcommand, options, and data flow in `README.md`, `docs/spec.md`, `docs/design.md`, `docs/codemap.md`, and `docs/runbook.md`. Refresh `.agents/last-docs-consolidate`.

- [ ] **Step 2: Run full verification suite**
Run:
- `uv run ruff check --no-cache .`
- `uv run mypy --cache-dir /tmp/mypy_cache src tests`
- `uv run pytest -q -p no:cacheprovider`
- `uv build`

- [ ] **Step 3: Commit**
`git commit -am "docs: document craig multi-track adapter and update canonical docs"`
