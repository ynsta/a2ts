# Archive Directory

This directory contains historical implementation plans, temporary design sketches, and development artifacts generated during past feature development sessions.

> [!WARNING]
> **Do not use the files in this directory as the source of truth for the project.**
> These documents represent point-in-time development snapshots and may contain outdated assumptions or references that have since been superseded by production code.

For current, authoritative documentation, refer to the canonical suite in [`../`](../README.md):
- **Specification (The What)**: [`../spec.md`](../spec.md)
- **Architecture & Design (The How)**: [`../design.md`](../design.md)
- **Codebase Map (The Map)**: [`../codemap.md`](../codemap.md)
- **Operational Runbook (The Ops)**: [`../runbook.md`](../runbook.md)
- **Architecture Decision Records (The Why)**: [`../adr/`](../adr/)

---

## Historical Inventory

| File | Date | Original Purpose | Current Canonical Replacement |
| :--- | :--- | :--- | :--- |
| [`2026-10-01-independent-check-remediation.md`](./2026-10-01-independent-check-remediation.md) | 2026-10-01 | Implementation plan for independent check reviews remediation | `../spec.md`, `../design.md`, `../codemap.md` |
| [`2026-10-01-pre-release-reviews-remediation.md`](./2026-10-01-pre-release-reviews-remediation.md) | 2026-10-01 | Implementation plan for pre-release reviews remediation | `../spec.md`, `../design.md`, `../codemap.md` |
| [`2026-09-30-opus-review-fixes.md`](./2026-09-30-opus-review-fixes.md) | 2026-09-30 | Implementation plan for Opus review fixes | `../spec.md`, `../design.md`, `../adr/0005-local-llm-refinement-via-agy-cli.md` |
| [`2026-09-29-craig-adapter.md`](./2026-09-29-craig-adapter.md) | 2026-09-29 | Implementation plan for Craig multi-track processing | `../spec.md#38`, `../design.md#28`, `../adr/0006-craig-multi-track-discord-pipeline.md` |
| [`2026-09-29-craig-adapter-design.md`](./2026-09-29-craig-adapter-design.md) | 2026-09-29 | Design scratch for Craig multi-track processing | `../design.md#28`, `../adr/0006-craig-multi-track-discord-pipeline.md` |
| [`2026-09-29-pre-publish-hardening.md`](./2026-09-29-pre-publish-hardening.md) | 2026-09-29 | Pre-release cleanup and packaging fixes | `../runbook.md`, `../pyproject.toml` |
| `superpowers/` | 2026-09-21 | Early design sketches and bootstrap implementation plans | `../spec.md`, `../design.md`, `../adr/` |
