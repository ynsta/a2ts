# [ARCHIVED] Pre-Release Reviews Remediation Plan

> **ARCHIVAL NOTICE:** This plan has been fully executed, verified, and reviewed. Canonical specifications are consolidated in `docs/spec.md`, `docs/design.md`, `docs/codemap.md`, and `README.md`.

**Goal:** Remediate all high, blocker, medium, and publication findings identified in the 2026-10-01 Codex and Opus pre-publication reviews for a2ts.

**Architecture:** Strengthen cache provenance bindings (transcript/diarization hashes and entity counts), align speaker resolution between display and enrollment, enforce atomic writes and unique temporary files across all outputs, make device/compute defaults adaptive, add Voxtral and subprocess guardrails, and harmonize documentation, CI, and packaging metadata.

**Tech Stack:** Python 3.13, Pydantic v2, Pytest, Ruff, Mypy, Faster-Whisper, SpeechBrain, CTranslate2, Typer, Rich.

**Spec:** [docs/spec.md](../spec.md), [docs/design.md](../design.md), `a2ts-review-2026-10-01-codex.md`, `a2ts-review-2026-10-01-opus.md`.

## Global Constraints

- Static checking: `uv run --frozen --no-sync mypy` must pass with 0 issues (`disallow_untyped_defs = true`).
- Lint & format: `uv run --frozen --no-sync ruff check --no-cache` and `uv run --frozen --no-sync ruff format --check --no-cache` must pass with 0 errors.
- Test suite: `uv run --frozen --no-sync pytest -p no:cacheprovider` must remain 100% green without regressions.
- Atomic I/O invariant: All state files, mappings, embeddings, and transcript outputs must use atomic writes via unique sibling temporary files.
- Observable behavior over internal mechanics: test contracts and failure modes rather than private helpers.

---

### Task 1: Embedding Cache Provenance & Foreign Glob Removal (F-01, F-02, Codex #1)
- Completed in commit `9f906e1` & `8a0b98d`.

### Task 2: Slice-Level Speaker Override Alignment in Voice Enrollment (Codex #2)
- Completed in commit `45b5275`.

### Task 3: Atomic I/O & Concurrency Hardening (F-06, F-15, Codex #5, #6)
- Completed in commit `ef679d1` & `621306a`.

### Task 4: Recluster Fixes & Diarization Fallback Cache (F-10, F-13)
- Completed in commit `a8034af`.

### Task 5: Platform & Device Selection Defaults (F-12, Codex #7)
- Completed in commit `637eca1`.

### Task 6: Voxtral Guardrails & Subprocess Timeouts (F-11, F-18, Codex #3, #9)
- Completed in commit `f50163b`.

### Task 7: Vocab Disallowed Special Tokens & Prompt Budgeting (Codex #8, F-09)
- Completed in commit `238fd47`.

### Task 8: CLI Robustness & Rich Markup Escaping (F-16, F-19, F-20, F-21, F-25)
- Completed in commit `5afd326`.

### Task 9: CI Workflow, Lockfile & Dependency Metadata (F-03, F-31, F-29)
- Completed in commit `03edf28`.

### Task 10: Publication Hygiene, Community Files & Disclaimers (F-05, F-07, F-08, User Request)
- Completed in commit `29f918a`.
