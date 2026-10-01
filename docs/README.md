# a2ts Documentation Suite

**Project**: `a2ts` (Audio to Text / Transcript)  
**Governance Tier**: Tier B  
**Documentation Model**: `isec-iagen-dev` Standard (Spec, Design, ADRs, Runbook, Codemap)

Welcome to the canonical documentation for `a2ts`. This directory serves as the **single source of truth** for all architectural, functional, operational, and design decisions of the project.

---

## 1. Documentation Index

The documentation is organized across five distinct pillars:

```
docs/
├── spec.md                 # THE WHAT: Functional capabilities, user journeys, CLI contract
├── design.md               # THE HOW: System architecture, data flow, models, caching, VRAM
├── codemap.md              # THE MAP: Codebase structure, module responsibilities, call graph
├── runbook.md              # THE OPS: Environment setup, execution recipes, troubleshooting
├── adr/                    # THE WHY: Architecture Decision Records (canonical ADR format)
│   ├── 0001-local-first-single-stream-audio-pipeline.md
│   ├── 0002-dual-transcription-engines-voxtral-and-whisper.md
│   ├── 0003-speaker-diarization-nemotron-and-ecapa-hybrid.md
│   ├── 0004-interactive-speaker-attribution-and-time-slicing.md
│   ├── 0005-local-llm-refinement-via-agy-cli.md
│   └── 0006-craig-multi-track-discord-pipeline.md
└── archive/                # HISTORICAL: Deprecated/archived transient plans and session notes
    └── README.md
```

| Document | Purpose & Audience | Key Contents |
| :--- | :--- | :--- |
| [**`spec.md`**](./spec.md) | **The What** (Functional Spec) | Feature capabilities, Obsidian lore mining, CLI commands (`run`, `review`, `recluster`, `split`), inputs/outputs. |
| [**`design.md`**](./design.md) | **The How** (Architecture & Design) | Pipeline sequence diagram, Pydantic schemas, Nemotron/ECAPA hybrid diarizer, `.a2ts/` caching, VRAM limits. |
| [**`codemap.md`**](./codemap.md) | **The Map** (Codebase Navigation) | Inventory of all `src/a2ts/*.py` modules, responsibilities, inter-module dependencies, test mapping. |
| [**`runbook.md`**](./runbook.md) | **The Ops** (Operations & Recipes) | `uv` installation, NVIDIA CUDA configuration, CLI command recipes, cache cleanup, common error fixes. |
| [**`adr/`**](./adr/) | **The Why** (Architecture Decisions) | Numbered records explaining the rationale, trade-offs, and context behind technical choices. |

---

## 2. Ephemeral vs Canonical Documentation

To prevent drift, confusion, and outdated guidance:
- **Canonical Docs** (`spec.md`, `design.md`, `codemap.md`, `runbook.md`, `adr/`) are living documents maintained alongside the codebase. Any architectural change, feature addition, or behavior update must be reflected here.
- **Session Plans** (`docs/archive/`): Implementation plans generated during agent sessions are transient artifacts. Once implemented, their lasting knowledge is folded into the canonical suite, and the plan is archived.

---

## 3. Maintenance & Review Policy

- **Documentation Consolidation**: Document status is tracked via `.agents/last-docs-consolidate`.
- Whenever non-trivial changes land on the working branch, canonical documentation must be updated and verified before completion.
