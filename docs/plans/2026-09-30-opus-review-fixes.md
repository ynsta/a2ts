# Opus Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all verified P1 and key P2 architectural, data-integrity, and safety defects identified in the 2026-09-30 Opus review, while applying reasoned pushback on dice normalizer defaults and model caching claims.

**Architecture:** 
1. Scope session state artifacts (`turns.json`, `speakers_mapping.json`, `session.json`) into `cache_dir / "sessions" / <media_hash> /` to prevent cross-session contamination and profile poisoning.
2. Remove legacy embedding cache alias fallbacks, add typed provenance envelopes for embeddings, and wire `--force` down to extractors.
3. Make diarization provenance transcript-aware for ECAPA and record the resolved engine.
4. Persist manual splits in `SpeakersMapping` and remove dangerous positional index fallbacks.
5. Harden `--refine` with turn chunking, strict validation ratios, and immutable `<output>.raw.md` output.
6. Fix French dice normalizer regex to require roll verb context or explicit dice markers, preventing ordinary phrase corruption.
7. Clean packaging dependencies (drop pandas, add platform markers for CUDA, declare torch), add VAD to Craig tracks, and optimize quadratic loops with bisect.
8. Align canonical documentation with disk reality, update ADR 0005, and enforce `ruff format`.

**Tech Stack:** Python 3.13, Typer, Pydantic v2, faster-whisper, speechbrain ECAPA-TDNN, ruff, pytest.

**Spec:** `docs/spec.md`, `docs/design.md`, `docs/codemap.md`, `/home/stany/Work/a2ts-review-2026-09-30-opus.md`.

## Global Constraints
- Target Python: >= 3.13
- All tests must pass: `uv run pytest -q`
- Ruff lint and formatting must be clean: `uv run ruff check .` and `uv run ruff format --check .`
- Types must pass: `uv run mypy src tests`
- Zero regression on existing 154 tests
- Caveman compression active for conversational responses

---

### Task 1: Scoping Session State to `cache_dir / "sessions" / <media_hash>` (P1-1)

**Files:**
- Modify: `src/a2ts/cli.py`
- Modify: `src/a2ts/speaker_review.py`
- Test: `tests/test_session_scoping.py`

**Interfaces:**
- Consumes: `cache_dir: Path`, `file_hash: str`
- Produces: `session_dir = cache_dir / "sessions" / file_hash` containing `turns.json`, `speakers_mapping.json`, `session.json`

- [ ] **Step 1: Write the failing multi-session isolation test**

```python
# tests/test_session_scoping.py
from pathlib import Path
import json
from typer.testing import CliRunner
from a2ts.cli import app
from a2ts.models import AlignedTurn, SpeakersMapping

runner = CliRunner()

def test_session_state_scoped_by_hash(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / ".a2ts"
    hash1 = "1111111111111111111111111111111111111111111111111111111111111111"
    hash2 = "2222222222222222222222222222222222222222222222222222222222222222"
    
    sess1_dir = cache_dir / "sessions" / hash1
    sess2_dir = cache_dir / "sessions" / hash2
    sess1_dir.mkdir(parents=True, exist_ok=True)
    sess2_dir.mkdir(parents=True, exist_ok=True)
    
    # Session 1 has Alice
    mapping1 = SpeakersMapping(cluster_defaults={"SPEAKER_00": "Alice"})
    (sess1_dir / "speakers_mapping.json").write_text(json.dumps(mapping1.model_dump()))
    
    # Session 2 should not see Alice
    from a2ts.speaker_review import load_speakers_mapping
    m2 = load_speakers_mapping(sess2_dir / "speakers_mapping.json")
    assert "SPEAKER_00" not in m2.cluster_defaults
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session_scoping.py -v`

- [ ] **Step 3: Update `cli.py` to scope session state and adapt `review`, `split`, `recluster`**

In `src/a2ts/cli.py`:
- In `run`: Define `session_dir = cache_dir / "sessions" / file_hash`, create it, and write `turns.json`, `speakers_mapping.json`, and `session.json` inside `session_dir`.
- In `review`: Support `session_dir` argument. If `(session_dir / "turns.json").exists()`, use `session_dir`. If user passes `cache_dir` that contains `sessions/<hash>`, or a session directory directly, resolve to the specific session directory.
- In `split` and `recluster`: Expect session directory (or resolve `session_dir / "turns.json"`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_session_scoping.py tests/test_cli.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/cli.py src/a2ts/speaker_review.py tests/test_session_scoping.py
git commit -m "fix(cli): scope session state to sessions/<media_hash> to prevent cross-session pollution (P1-1)"
```

---

### Task 2: Embedding Cache Provenance, No Legacy Alias, and `--force` Support (P1-2)

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `src/a2ts/diarizer.py`
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_embedding_provenance.py`

**Interfaces:**
- Consumes: `EmbeddingCacheProvenance`
- Produces: `_turn_embeddings_provenance.json`, `_segment_embeddings_provenance.json`, no legacy alias writes/reads

- [ ] **Step 1: Write test for embedding provenance and force invalidation**

```python
# tests/test_embedding_provenance.py
from pathlib import Path
import numpy as np
from a2ts.models import EmbeddingCacheProvenance, SpeakerTurn
from a2ts.diarizer import extract_embeddings_for_turns

def test_embedding_provenance_and_force(tmp_path: Path, monkeypatch):
    prefix = tmp_path / "test"
    turns = [SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00", cluster_id="SPEAKER_00", turn_id=0)]
    # Verify legacy alias <prefix>_embeddings.npy is NOT written
    # Verify force=True causes re-computation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedding_provenance.py -v`

- [ ] **Step 3: Implement `EmbeddingCacheProvenance`, remove legacy alias, add `force`**

- In `src/a2ts/models.py`: Add `EmbeddingCacheProvenance` model.
- In `src/a2ts/diarizer.py`:
  - Delete `np.save(Path(f"{cache_prefix}_embeddings.npy"), emb_matrix)` and `atomic_write_text(Path(f"{cache_prefix}_indices.json"), ...)`.
  - Delete alias fallback reads.
  - Write `_turn_embeddings_provenance.json` and `_segment_embeddings_provenance.json`.
  - Add `force: bool = False` to `extract_embeddings_for_segments` and `extract_embeddings_for_turns`.
- In `src/a2ts/cli.py`: Pass `force=force` from CLI to embedding extractors.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_provenance.py tests/test_diarizer.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/models.py src/a2ts/diarizer.py src/a2ts/cli.py tests/test_embedding_provenance.py
git commit -m "fix(diarizer): add embedding provenance, delete legacy alias fallback, honor force (P1-2)"
```

---

### Task 3: Diarization Cache Provenance Transcript Dependency (P1-3)

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `src/a2ts/cli.py`
- Modify: `src/a2ts/diarizer.py`
- Test: `tests/test_diarization_provenance.py`

**Interfaces:**
- Consumes: `transcript_provenance_hash: str | None`, `resolved_engine: str`
- Produces: `DiarizationCacheProvenance` reflecting transcript segmentation dependency

- [ ] **Step 1: Write test for transcript dependency in diarization cache**

```python
# tests/test_diarization_provenance.py
from a2ts.models import DiarizationCacheProvenance

def test_diarization_provenance_includes_transcript_hash():
    prov = DiarizationCacheProvenance(
        media_hash="abc",
        engine="ecapa",
        resolved_engine="ecapa",
        cluster_threshold=0.6,
        device="cpu",
        transcript_provenance_hash="tx_123",
    )
    assert prov.transcript_provenance_hash == "tx_123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diarization_provenance.py -v`

- [ ] **Step 3: Update models and diarization cache key**

- In `src/a2ts/models.py`: Add `transcript_provenance_hash: str | None = None` and `resolved_engine: str | None = None` to `DiarizationCacheProvenance`.
- In `src/a2ts/cli.py`: For ECAPA, compute transcript provenance hash and include it when checking/saving diarization cache.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diarization_provenance.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/models.py src/a2ts/cli.py src/a2ts/diarizer.py tests/test_diarization_provenance.py
git commit -m "fix(diarizer): bind diarization cache provenance to transcript identity (P1-3)"
```

---

### Task 4: Persist Splits and Fix Positional Fallbacks (P2-2, P2-3, P2-1)

**Files:**
- Modify: `src/a2ts/models.py`
- Modify: `src/a2ts/cli.py`
- Modify: `src/a2ts/speaker_review.py`
- Test: `tests/test_splits_persistence.py`

**Interfaces:**
- Consumes: `ClusterSplit(cluster_id, at, new_cluster_id)`
- Produces: `SpeakersMapping.splits`, no positional fallbacks `turns[tid]`

- [ ] **Step 1: Write tests for split persistence and no-fallback behavior**

```python
# tests/test_splits_persistence.py
from a2ts.models import SpeakersMapping, ClusterSplit

def test_speakers_mapping_persists_splits():
    mapping = SpeakersMapping(splits=[ClusterSplit(cluster_id="SPEAKER_00", at=120.0, new_cluster_id="SPEAKER_99")])
    dump = mapping.model_dump()
    loaded = SpeakersMapping.model_validate(dump)
    assert len(loaded.splits) == 1
    assert loaded.splits[0].new_cluster_id == "SPEAKER_99"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_splits_persistence.py -v`

- [ ] **Step 3: Implement split persistence and remove positional fallbacks**

- In `src/a2ts/models.py`: Add `ClusterSplit` and `splits: list[ClusterSplit] = Field(default_factory=list)` to `SpeakersMapping`.
- In `src/a2ts/cli.py`:
  - `split`: Save split rule into `mapping.splits` and apply it to turns.
  - Remove positional fallback `elif tid < len(turns): cid = turns[tid].cluster_id` in `review` and `recluster`. Skip unknown turn IDs.
  - Auto-enrollment in `cli.py`: Enroll only `manual` labels; do not enroll closed-set guesses.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_splits_persistence.py tests/test_speaker_review.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/models.py src/a2ts/cli.py src/a2ts/speaker_review.py tests/test_splits_persistence.py
git commit -m "fix(cli): persist split rules in mapping and remove positional fallbacks (P2-2, P2-3, P2-1)"
```

---

### Task 5: Refiner Hardening & Raw Markdown Output (P1-4, P1-5)

**Files:**
- Modify: `src/a2ts/refiner.py`
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_refiner_safety.py`

**Interfaces:**
- Consumes: `raw_markdown: str`, `agy_model: str`
- Produces: Immutable `<output>.raw.md` on disk, chunk-based refinement, truncation rejection

- [ ] **Step 1: Write test for refiner truncation rejection and header validation**

```python
# tests/test_refiner_safety.py
from a2ts.refiner import validate_refiner_chunk

def test_truncation_is_rejected():
    original = "### [00:00:00 - 00:00:10] Alice\nBonjour tout le monde. Ceci est un long test avec beaucoup de mots."
    truncated = "### [00:00:00 - 00:00:10] Alice\nBonjour"
    assert not validate_refiner_chunk(original, truncated)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_refiner_safety.py -v`

- [ ] **Step 3: Implement chunking, strict validation, and raw markdown output**

- In `src/a2ts/refiner.py`:
  - Implement `validate_refiner_chunk(original: str, refined: str) -> bool`:
    - Checks exact match of `### [HH:MM:SS - HH:MM:SS] Name` headers.
    - Checks word count ratio: `abs(len(refined.split()) - len(original.split())) / len(original.split()) <= 0.15`.
  - Implement `chunk_transcript_markdown(markdown: str, max_turns: int = 25) -> list[str]`.
  - Refine chunk-by-chunk with timeout proportional to chunk size. Fall back to normalized raw chunk if validation fails.
- In `src/a2ts/cli.py`:
  - When `--refine` is enabled, always write `<output_stem>.raw.md` before refinement.
  - Print notice about `--refine` using external `agy` model.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_refiner_safety.py tests/test_refiner.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/refiner.py src/a2ts/cli.py tests/test_refiner_safety.py
git commit -m "fix(refiner): chunk refinement, reject truncation, and save immutable raw markdown (P1-4, P1-5)"
```

---

### Task 6: French RPG Normalizer Hardening (P1-6)

**Files:**
- Modify: `src/a2ts/consolidator.py`
- Modify: `src/a2ts/refiner.py`
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_consolidator_dice.py`

**Interfaces:**
- Consumes: `text: str`, `rpg_normalize: bool = True`
- Produces: Correct dice normalization without corrupting ordinary French phrases

- [ ] **Step 1: Write regression tests for false positives**

```python
# tests/test_consolidator_dice.py
from a2ts.consolidator import normalize_rpg_terms

def test_french_speech_not_mangled():
    assert normalize_rpg_terms("c est une des dix merveilles") == "c est une des dix merveilles"
    assert normalize_rpg_terms("il a eu deux des six votes") == "il a eu deux des six votes"
    assert normalize_rpg_terms("trois des huit portes sont ouvertes") == "trois des huit portes sont ouvertes"
    assert normalize_rpg_terms("un des vingt chevaliers est mort") == "un des vingt chevaliers est mort"

def test_actual_dice_rolls_normalized():
    assert normalize_rpg_terms("il lance deux dés six") == "il lance 2d6"
    assert normalize_rpg_terms("fais un jet de dés 20") == "fais un jet de 1d20"
    assert normalize_rpg_terms("un jet de délai") == "un jet de dés"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_consolidator_dice.py -v`

- [ ] **Step 3: Update `DICE_PATTERN` and single normalization point**

- In `src/a2ts/consolidator.py`:
  - Require roll verb context for bare "des N" (`lance`, `roule`, `jet`, `tire`, `fait`, `fais`, `dégâts`), or explicit accented `dé/dés`.
  - Merge phonetic rules from `refiner.py`.
  - Remove duplicate call in `debounce_consecutive_turns` (normalize only once during final render).
- In `src/a2ts/cli.py`:
  - Add `--rpg-normalize / --no-rpg-normalize` flag (default True).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_consolidator_dice.py tests/test_consolidator.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/consolidator.py src/a2ts/refiner.py src/a2ts/cli.py tests/test_consolidator_dice.py
git commit -m "fix(consolidator): require roll context in dice normalizer and prevent speech corruption (P1-6)"
```

---

### Task 7: Craig Quality, Performance & Packaging (P1-7, P2-5, P2-6, P2-8, P2-10)

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/a2ts/craig.py`
- Modify: `src/a2ts/cli.py`
- Modify: `src/a2ts/timeline.py`
- Modify: `src/a2ts/media.py`
- Modify: `src/a2ts/transcriber.py`
- Test: `tests/test_packaging_and_perf.py`

**Interfaces:**
- Consumes: Clean packaging metadata, `bisect` alignment, atomic WAV write
- Produces: Multiplatform wheel metadata, non-hallucinating Craig tracks, fast alignment

- [ ] **Step 1: Write test for timeline bisect alignment and packaging import sanity**

```python
# tests/test_packaging_and_perf.py
from a2ts.timeline import align_words_to_speaker_turns
from a2ts.models import RawSegment, WordTimestamp, SpeakerTurn

def test_alignment_with_many_turns():
    # Verify alignment works accurately with sorted turn bisect
    words = [WordTimestamp(word="hello", start=1.0, end=1.5, probability=0.9)]
    segments = [RawSegment(start=1.0, end=1.5, text="hello", words=words)]
    turns = [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00", cluster_id="SPEAKER_00", turn_id=0)]
    aligned = align_words_to_speaker_turns(segments, turns)
    assert len(aligned) == 1
    assert aligned[0].text == "hello"
```

- [ ] **Step 2: Run test to verify it passes/fails**

Run: `uv run pytest tests/test_packaging_and_perf.py -v`

- [ ] **Step 3: Update `pyproject.toml`, `craig.py`, `timeline.py`, `media.py`**

- In `pyproject.toml`:
  - Remove `pandas>=2.2.0`.
  - Add platform markers to `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`: `; sys_platform == 'linux' or sys_platform == 'win32'`.
  - Declare `torch>=2.2.0` in dependencies.
- In `src/a2ts/craig.py` & `cli.py`:
  - Pass `vad_filter=True` and `language` to track transcribe.
  - Wire `--refine-effort` through.
  - Drop Discord usernames from prompt.
- In `src/a2ts/timeline.py`:
  - Optimize word alignment loop using `bisect_right` on turn start times.
- In `src/a2ts/media.py`:
  - Write WAV to `.tmp.wav` and `os.replace` to destination.
- In `src/a2ts/transcriber.py`:
  - Restrict `ensure_cuda_libs` to cublas and cudnn only.

- [ ] **Step 4: Run tests and lock sync**

Run: `uv run pytest -q && uv run ruff check .`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/a2ts/craig.py src/a2ts/cli.py src/a2ts/timeline.py src/a2ts/media.py src/a2ts/transcriber.py tests/test_packaging_and_perf.py
git commit -m "fix(packaging,craig,perf): clean deps, add Craig VAD, and optimize word alignment (P1-7, P2-5, P2-6)"
```

---

### Task 8: Formatting, Canonical Docs & ADR 0005 Updates (§5, P1-5, §8)

**Files:**
- Modify: `docs/spec.md`
- Modify: `docs/design.md`
- Modify: `docs/runbook.md`
- Modify: `docs/codemap.md`
- Modify: `docs/adr/0005-local-llm-refinement.md`
- Modify: `README.md`
- Move: `docs/plans/2026-09-29-*.md` -> `docs/archive/`
- Touch: `.agents/last-docs-consolidate`

**Interfaces:**
- Consumes: Disk state
- Produces: 100% coherent canonical documentation, archived transient plans

- [ ] **Step 1: Run `uv run ruff format .`**
- [ ] **Step 2: Update documentation options, defaults, and ADR 0005**
  - Fix CLI option defaults in `spec.md`, `README.md`, `runbook.md`.
  - Update ADR 0005 to document `agy` delegation and remote model boundary.
  - Update `codemap.md` with new symbols.
  - Move stale plans to `docs/archive/`.
  - Update `.agents/last-docs-consolidate`.
- [ ] **Step 3: Verify all gates**
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run mypy src tests`
  - `uv run pytest -q`
- [ ] **Step 4: Commit**

```bash
git add docs/ README.md .agents/last-docs-consolidate
git commit -m "docs: consolidate canonical docs, update ADR 0005, and enforce ruff format"
```
