# [ARCHIVED] V2 Check Remediation Plan

> [!WARNING]
> **ARCHIVED / HISTORICAL DOCUMENTATION**
> This implementation plan was fully executed and verified on 2026-10-01.
> Do not use this file as a living source of truth. Refer to canonical documentation:
> - Specifications: [`../spec.md`](../spec.md)
> - Technical Design: [`../design.md`](../design.md)
> - Codebase Map: [`../codemap.md`](../codemap.md)
> - Operational Runbook: [`../runbook.md`](../runbook.md)
> - Architecture Decision Records: [`../adr/`](../adr/)

**Goal:** Remediate all v2 review findings from Codex and Opus (`a2ts-review-2026-10-01-codex-check-v2.md` and `a2ts-review-2026-10-01-opus-check-v2.md`): R-1 (review turn-embedding fingerprint mismatch), R-2 (nested control-token bypass), R-3 (Rich markup crash and unescaped prints), F-05 (external broken links), F-19 (backup name collisions), ECAPA fallback hash synchronization, F-29 (transformers version alignment), and F-04/F-06 (privacy and documentation governance).

**Architecture:**
- Route review turn-embedding fingerprint validation through the authoritative `SpeakerTurn` records from `DiarizationCacheFile.turns`.
- Reload actual diarization provenance from `diar_cache` after `diarize_segments` to maintain hash parity on ECAPA fallback.
- Iteratively strip Whisper control tokens until stable and eliminate stray `<|` and `|>`.
- Systematically escape dynamic console prints across `cli.py`.
- Handle backup collision with incrementing suffix.
- Neutralize external review links in documentation archives and document privacy decision.

**Tech Stack:** Python 3.13, Pydantic v2, Pytest, Ruff, Mypy, Faster-Whisper, SpeechBrain, CTranslate2, Typer, Rich.

**Spec:** [docs/spec.md](../spec.md), [docs/design.md](../design.md), [docs/codemap.md](../codemap.md), [docs/runbook.md](../runbook.md).

## Global Constraints
- Static checking: `uv run --frozen --no-sync mypy` must pass with 0 issues (`disallow_untyped_defs = true`).
- Lint & format: `uv run --frozen --no-sync ruff check --no-cache` and `uv run --frozen --no-sync ruff format --check --no-cache` must pass with 0 errors.
- Test suite: `uv run --frozen --no-sync pytest -p no:cacheprovider` must remain 100% green without regressions.
- No blind trust: Every step must be independently verified by inspecting disk state and running concrete test probes.

---

### Task 1: R-1 (Review Turn Embedding Fingerprint Source & Roundtrip Test) and ECAPA Fallback Hash Sync

**Files:**
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- In `review()` (`src/a2ts/cli.py:910-935`):
  - Pass `turns=diar_cache_file.turns` (if `diar_cache_file` is loaded) to `load_turn_embeddings` so the expected fingerprint matches the `SpeakerTurn`s recorded by diarization.
- In `run()` (`src/a2ts/cli.py:530-550`):
  - After `diarize_segments` executes, reload `diar_prov` and `diar_prov_hash` from `diar_cache` if the file exists, ensuring any fallback (e.g. Nemotron to ECAPA) updates the provenance hash before turn embeddings are extracted or loaded.
- In `tests/test_cli.py`:
  - In `test_review_with_voice_profiles`: update fixture so `entity_fingerprint` is computed over `diar_cache.turns` (SpeakerTurns) instead of AlignedTurns.
  - Add `test_run_to_review_voice_profiles_roundtrip`: invoke `run` to generate session and turn embeddings on disk, then invoke `review` to verify turn embeddings are loaded and voice profile is matched and enrolled.
  - Add test verifying ECAPA fallback syncs `diar_prov_hash` between diarization cache and turn embeddings provenance.

- [x] **Step 1: Write failing tests for run-to-review roundtrip and ECAPA fallback hash parity**
- [x] **Step 2: Run tests to verify they fail**
- [x] **Step 3: Implement fingerprint source fix in review() and provenance reload in run()**
- [x] **Step 4: Run tests to verify they pass**

---

### Task 2: R-2 (Codex #8 Nested Control Token Sanitization) & F-29 (Dependency Alignment)

**Files:**
- Modify: `src/a2ts/vocab.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_vocab.py`

**Interfaces:**
- In `src/a2ts/vocab.py`:
  - Implement `strip_control_tokens(text: str) -> str`: loop `CONTROL_TOKEN_PATTERN.sub("", text)` until stable, then strip `<|` and `|>`.
  - Use `strip_control_tokens` in `clean_entity_name()` and `parse_obsidian_note()`.
- In `pyproject.toml`:
  - Update dependency `"transformers>=5.18.0.dev0"` to match `[tool.uv.sources]`, `uv.lock`, and `docs/runbook.md`.
  - Verify `uv lock --check`.
- In `tests/unit/test_vocab.py`:
  - Add tests for nested control tokens (`<<|endoftext|>|endoftext|>`, `<<<|startoftranscript|>|>`, `prefix<|<|foo|>|>suffix`).

- [x] **Step 1: Write failing test for nested control tokens**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement iterative control-token stripping and align pyproject.toml**
- [x] **Step 4: Run test to verify it passes**

---

### Task 3: R-3 (`extract-vocab` Crash & Systematic Rich Markup Escaping in `cli.py`) & F-19 (Corrupt Backup Non-Collision)

**Files:**
- Modify: `src/a2ts/cli.py`
- Modify: `src/a2ts/speaker_review.py`
- Test: `tests/test_cli.py`
- Test: `tests/unit/test_speaker_review.py`

**Interfaces:**
- In `src/a2ts/cli.py`:
  - In `extract_vocab`: wrap prompt in `escape(prompt)` when printing to console.
  - Audit and wrap dynamic interpolations in `console.print` across `cli.py` with `rich.markup.escape()` (media file paths, exceptions, session hashes, candidate strings, engines).
- In `src/a2ts/speaker_review.py`:
  - In `load_speakers_mapping`: if `corrupt_backup.exists()`, increment counter suffix `f"{target_file.stem}.corrupt.{ts}_{counter}.json"` so existing backups are never overwritten.
- In `tests/test_cli.py`:
  - Add test for `extract-vocab` with `[/bold]` and `[Lore]` terms verifying clean execution without `MarkupError`.
- In `tests/unit/test_speaker_review.py`:
  - Add test verifying consecutive corrupt backups within the same second receive distinct filenames.

- [x] **Step 1: Write failing test for extract-vocab with Rich markup and same-second corrupt backup collision**
- [x] **Step 2: Run tests to verify they fail**
- [x] **Step 3: Implement Rich escaping across cli.py and collision-safe backup naming in speaker_review.py**
- [x] **Step 4: Run tests to verify they pass**

---

### Task 4: F-05 (Neutralize External Review Links) & F-04 / F-06 (Privacy & Documentation Governance)

**Files:**
- Modify: `docs/archive/2026-10-01-pre-release-reviews-remediation.md`
- Modify: `docs/archive/2026-10-01-independent-check-remediation.md`
- Modify: `docs/archive/README.md`
- Modify: `SECURITY.md`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/README.md`
- Modify: `.agents/last-docs-consolidate`

**Interfaces:**
- Neutralize external review links in `docs/archive/*.md` to plain text / code references (`a2ts-review-2026-10-01-codex.md`).
- Document explicit F-04 privacy decision in `SECURITY.md`: character names (`Brakk`, `Ilvaris`) and test Discord usernames (`tessaro`, `merrow1`) are synthetic/fictional TTRPG test fixtures; no personal secrets or credentials exist in git history.
- Clean up F-06 internal `isec-iagen-dev` references in user-facing documentation to standard Tier B documentation model.
- Update `.agents/last-docs-consolidate` with current UTC timestamp.

- [x] **Step 1: Update documentation files, neutralize links, and record governance decisions**
- [x] **Step 2: Run full quality gates (lock, ruff, mypy, pytest)**
