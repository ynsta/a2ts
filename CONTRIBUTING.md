# Contributing to a2ts

Thank you for your interest in contributing to `a2ts`! We welcome contributions, bug reports, and suggestions.

This document outlines the development workflow, quality gates, and engineering standards for the project.

---

## 1. Development Setup

We use [`uv`](https://docs.astral.sh/uv/) for deterministic Python environment and dependency management.

### Prerequisites

- **Python**: `>= 3.13`
- **System Tools**: `ffmpeg` must be installed and accessible in `$PATH`
- **uv**: Install via Astral's instructions (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Environment Initialization

Clone the repository and install all development dependencies:

```bash
git clone https://github.com/ynsta/a2ts.git
cd a2ts
uv sync --locked --dev
```

If you plan to work on or test the experimental Voxtral engine:

```bash
uv sync --locked --dev --extra voxtral
```

---

## 2. Quality Gates & Pre-Commit Verification

Before submitting any pull request or committing changes, your code **must** pass all four gates with **0 errors**:

### 1. Linting (`ruff check`)
```bash
uv run --frozen --no-sync ruff check --no-cache
```

### 2. Code Formatting (`ruff format`)
```bash
uv run --frozen --no-sync ruff format --check --no-cache
```
To automatically format modified code:
```bash
uv run ruff format
```

### 3. Static Type Checking (`mypy`)
```bash
uv run --frozen --no-sync mypy
```
- All functions (public and private) require explicit type annotations (`disallow_untyped_defs = true`).
- Package typing marker `src/a2ts/py.typed` must remain present.
- Use standard Python 3.13 typing primitives (`list[str]`, `dict[str, Any]`, `tuple[...]`, `typing.Protocol`).

### 4. Automated Test Suite (`pytest`)
```bash
uv run --frozen --no-sync pytest -p no:cacheprovider -q
```
All tests must pass 100% green with zero regressions.

---

## 3. Engineering & Architecture Standards

### Testing Principles: Test the WHAT, Not the HOW

We strictly practice invariant-driven, observable-behavior testing:
- **Observable Behavior Over Mechanics**: Test public contracts, inputs, outputs, generated files, and state transitions. Avoid asserting on private functions, intermediate variables, or call execution order.
- **Refactoring Resilience**: Tests should not break when an internal implementation is refactored, provided external behavior and contracts remain intact. Avoid brittle tests that spy on private methods or mock internal functions.
- **Minimal Mock Boundaries**: Mock **only** external system boundaries (remote APIs, GPU inference models, long-running CLI tools like `ffmpeg`). Never mock internal domain logic or data models.
- **Hermetic & Fast**: Every test must use pytest's `tmp_path` fixture for filesystem operations. Never write to `~/.a2ts`, repository root, or global temporary paths. Tests must not require network access or GPU hardware.
- **No Vanity Coverage**: Do not chase arbitrary coverage percentages (e.g. 100%) at the expense of testing private details. Focus tests on critical invariants, contracts, and error paths.

### Code Quality & Robustness

- **Pydantic Validation**: All external data, cache records, and session payloads must be defined as `pydantic.BaseModel` with strict field types.
- **Atomic I/O**: File writes that mutate persistent caches or output documents (`.json`, `.md`, `.wav`) must use atomic write patterns (`atomic_write_text()` via sibling `.tmp` and `os.replace`).
- **Hardware Defensiveness**: Check `torch.cuda.is_available()` before CUDA operations; always respect user-provided `--device` and provide clean CPU fallback paths.
- **Bounded Subprocesses**: `shell=True` is strictly forbidden. Subprocess calls must use explicit argument lists (`list[str]`), specify timeouts, and handle exceptions cleanly.
- **Path Hardening**: User-provided paths must be resolved and checked against path traversal before file operations.
- **English Default**: All code, docstrings, comments, identifiers, commits, and technical documentation must be written in English.

---

## 4. Documentation Standards

This project adheres to the **Tier B** documentation model.
When modifying existing capabilities or adding new features:
- Update canonical docs in the same change set: `docs/spec.md`, `docs/design.md`, `docs/codemap.md`, and `docs/runbook.md`.
- Architectural choices and trade-offs require an Architectural Decision Record in `docs/adr/`.
- Ensure all internal relative links in documentation remain valid.
