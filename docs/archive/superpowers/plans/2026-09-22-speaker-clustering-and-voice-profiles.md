# Speaker Clustering and Voice Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve speaker clustering quality by exposing tunable distance thresholds, caching raw embeddings for instant re-clustering, and providing persistent voice profile enrollment and signature matching across sessions.

**Architecture:** 
1. Decouple ECAPA-TDNN embedding extraction and caching from Agglomerative Clustering in `src/a2ts/diarizer.py`.
2. Persist 192-dim unit-normalized embeddings to `.a2ts/diarization/<hash>_embeddings.npy` along with segment indices.
3. Allow user to tune clustering threshold (`--cluster-threshold`) and force speaker count (`--num-speakers`), plus add `a2ts recluster` command to re-cluster cached sessions in seconds.
4. Add voice profile models and centroid calculations (`VoiceProfile`, `VoiceProfilesDatabase`): compute speaker voice centroids from user-assigned turns after interactive review, save to `contexte/voice_profiles.json`, and pre-match segment embeddings against known voice profiles on subsequent runs using cosine similarity.

**Tech Stack:** Python 3.11+, Pydantic v2, NumPy, scikit-learn, SpeechBrain ECAPA-TDNN, Typer, Rich.

**Spec:** Brainstorming Option C validated with user in session on 2026-09-22.

## Global Constraints

- All code, comments, identifiers, docstrings, and commit messages in English.
- Use `uv run` for all commands; never use raw `pip install`.
- Every task must strictly follow TDD: write failing test, verify FAIL, implement code, verify PASS.
- Code must pass `uv run pytest -v`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy src/`.
- Maintain full backward compatibility with existing CLI arguments and cache formats.

---

### Task 1: Voice Profile Domain Models

**Files:**
- Modify: `src/a2ts/models.py`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Produces: `VoiceProfile`, `VoiceProfilesDatabase`

- [ ] **Step 1: Write failing unit test for VoiceProfile and VoiceProfilesDatabase**

In `tests/unit/test_models.py`:
```python
def test_voice_profile_and_database() -> None:
    from a2ts.models import VoiceProfile, VoiceProfilesDatabase

    vp = VoiceProfile(
        speaker_name="Brakk",
        centroid=[0.1, 0.2, -0.3],
        sample_count=5,
    )
    assert vp.speaker_name == "Brakk"
    assert len(vp.centroid) == 3
    assert vp.sample_count == 5

    db = VoiceProfilesDatabase(
        speakers={"Brakk": vp},
        version=1,
    )
    assert "Brakk" in db.speakers
    assert db.speakers["Brakk"].sample_count == 5
    dumped = db.model_dump()
    assert dumped["speakers"]["Brakk"]["speaker_name"] == "Brakk"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_models.py::test_voice_profile_and_database -v`
Expected: FAIL with `ImportError: cannot import name 'VoiceProfile' from 'a2ts.models'`

- [ ] **Step 3: Implement VoiceProfile and VoiceProfilesDatabase in src/a2ts/models.py**

In `src/a2ts/models.py`:
```python
from datetime import UTC, datetime


class VoiceProfile(BaseModel):
    """Voice signature for an enrolled speaker."""

    speaker_name: str
    centroid: list[float]
    sample_count: int = 1


class VoiceProfilesDatabase(BaseModel):
    """Database of enrolled speaker voice profiles."""

    version: int = 1
    updated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    speakers: dict[str, VoiceProfile] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Run linters and commit**

```bash
uv run ruff check src/a2ts/models.py tests/unit/test_models.py
uv run ruff format --check src/a2ts/models.py tests/unit/test_models.py
uv run mypy src/a2ts/models.py
git add src/a2ts/models.py tests/unit/test_models.py
git commit -m "feat: add VoiceProfile and VoiceProfilesDatabase models"
```

---

### Task 2: Embedding Caching & Tunable Diarization

**Files:**
- Modify: `src/a2ts/diarizer.py`
- Test: `tests/unit/test_diarizer.py`

**Interfaces:**
- Produces:
  - `extract_embeddings(audio_path: Path, segments: list[RawSegment], device: str = "cuda", cache_prefix: Path | None = None) -> tuple[np.ndarray, list[int]]`
  - `diarize_segments(audio_path: Path, segments: list[RawSegment], num_speakers: int | None = None, distance_threshold: float = 0.60, device: str = "cuda", cache_path: Path | None = None, embeddings_cache_prefix: Path | None = None) -> list[SpeakerTurn]`

- [ ] **Step 1: Write failing unit tests for extract_embeddings and tunable clustering**

In `tests/unit/test_diarizer.py`:
```python
def test_extract_embeddings_caches_to_npy(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch
    from a2ts.diarizer import extract_embeddings
    from a2ts.models import RawSegment

    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="Hello"),
        RawSegment(id=1, start=1.5, end=2.5, text="World"),
    ]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    cache_prefix = tmp_path / "test_emb"

    fake_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with (
        patch("soundfile.read", return_value=(np.zeros(32000, dtype=np.float32), 16000)),
        patch("a2ts.diarizer.get_embedding_model") as mock_model,
    ):
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.side_effect = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        ]
        mock_model.return_value = mock_classifier

        emb_matrix, valid_indices = extract_embeddings(
            audio_path, segments, device="cpu", cache_prefix=cache_prefix
        )

        assert emb_matrix.shape == (2, 2)
        assert valid_indices == [0, 1]
        assert (tmp_path / "test_emb_embeddings.npy").is_file()
        assert (tmp_path / "test_emb_indices.json").is_file()

        # Second call should load from cache without calling get_embedding_model
        mock_model.reset_mock()
        emb2, indices2 = extract_embeddings(
            audio_path, segments, device="cpu", cache_prefix=cache_prefix
        )
        assert np.allclose(emb_matrix, emb2)
        assert indices2 == valid_indices
        mock_model.assert_not_called()


def test_diarize_segments_with_num_speakers(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch
    from a2ts.diarizer import diarize_segments
    from a2ts.models import RawSegment

    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="A"),
        RawSegment(id=1, start=1.0, end=2.0, text="B"),
        RawSegment(id=2, start=2.0, end=3.0, text="C"),
    ]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()

    # 3 distinct vectors forced into 2 clusters
    cached_emb = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    with patch("a2ts.diarizer.extract_embeddings", return_value=(cached_emb, [0, 1, 2])):
        turns = diarize_segments(
            audio_path,
            segments,
            num_speakers=2,
            device="cpu",
        )
        assert len(turns) == 3
        clusters = {t.cluster_id for t in turns}
        assert len(clusters) == 2
        # segments 0 and 1 should cluster together
        assert turns[0].cluster_id == turns[1].cluster_id
        assert turns[2].cluster_id != turns[0].cluster_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_diarizer.py::test_extract_embeddings_caches_to_npy -v`
Expected: FAIL with `ImportError: cannot import name 'extract_embeddings' from 'a2ts.diarizer'`

- [ ] **Step 3: Implement extract_embeddings and update diarize_segments in src/a2ts/diarizer.py**

In `src/a2ts/diarizer.py`:
- Create `extract_embeddings(audio_path, segments, device="cuda", cache_prefix=None) -> tuple[np.ndarray, list[int]]`:
  - Checks if `f"{cache_prefix}_embeddings.npy"` and `f"{cache_prefix}_indices.json"` exist.
  - If exist, loads using `np.load` and `json.loads` and returns `(emb_matrix, valid_indices)`.
  - Otherwise, reads audio, extracts embeddings with ECAPA-TDNN, normalizes them, and if `cache_prefix` is set, saves to `.npy` and `.json`.
- Refactor `diarize_segments(...)`:
  - Takes `num_speakers: int | None = None, distance_threshold: float = 0.60, device: str = "cuda", cache_path: Path | None = None, embeddings_cache_prefix: Path | None = None`.
  - Calls `extract_embeddings(audio_path, segments, device=device, cache_prefix=embeddings_cache_prefix)`.
  - Performs Agglomerative Clustering using either `n_clusters=num_speakers` or `distance_threshold=distance_threshold`.
  - Reconstructs turns and writes `cache_path` if given.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_diarizer.py -v`
Expected: PASS (all tests in `test_diarizer.py` pass)

- [ ] **Step 5: Run linters and commit**

```bash
uv run ruff check src/a2ts/diarizer.py tests/unit/test_diarizer.py
uv run ruff format --check src/a2ts/diarizer.py tests/unit/test_diarizer.py
uv run mypy src/a2ts/diarizer.py
git add src/a2ts/diarizer.py tests/unit/test_diarizer.py
git commit -m "feat: add extract_embeddings with npy caching and tunable diarize_segments"
```

---

### Task 3: Voice Profile Enrollment & Signature Matching

**Files:**
- Modify: `src/a2ts/diarizer.py`
- Test: `tests/unit/test_diarizer.py`

**Interfaces:**
- Produces:
  - `compute_voice_profiles(embeddings: np.ndarray, valid_indices: list[int], turns: list[SpeakerTurn | AlignedTurn], speakers_mapping: SpeakersMapping | dict[str, str], existing_db: VoiceProfilesDatabase | None = None) -> VoiceProfilesDatabase`
  - `save_voice_profiles(db: VoiceProfilesDatabase, path: Path) -> None`
  - `load_voice_profiles(path: Path) -> VoiceProfilesDatabase | None`
  - `match_embeddings_to_profiles(embeddings: np.ndarray, db: VoiceProfilesDatabase, similarity_threshold: float = 0.60) -> dict[int, tuple[str, float]]`

- [ ] **Step 1: Write failing unit tests for compute_voice_profiles, save/load, and match**

In `tests/unit/test_diarizer.py`:
```python
def test_voice_profiles_computation_and_matching(tmp_path: Path) -> None:
    from a2ts.diarizer import (
        compute_voice_profiles,
        save_voice_profiles,
        load_voice_profiles,
        match_embeddings_to_profiles,
    )
    from a2ts.models import SpeakerTurn, SpeakersMapping

    # Two 2D unit vectors
    embeddings = np.array([
        [1.0, 0.0],
        [0.8, 0.6],
        [0.0, 1.0],
    ], dtype=np.float32)
    valid_indices = [0, 1, 2]

    turns = [
        SpeakerTurn(id=0, start=0.0, end=1.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=1, start=1.0, end=2.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=2, start=2.0, end=3.0, cluster_id="SPEAKER_01"),
    ]
    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Brakk", "SPEAKER_01": "Merrow"}
    )

    db = compute_voice_profiles(embeddings, valid_indices, turns, mapping)
    assert "Brakk" in db.speakers
    assert "Merrow" in db.speakers
    assert db.speakers["Brakk"].sample_count == 2
    assert db.speakers["Merrow"].sample_count == 1

    # Check centroid normalization
    brakk_centroid = np.array(db.speakers["Brakk"].centroid)
    assert np.isclose(np.linalg.norm(brakk_centroid), 1.0)

    # Save and reload
    db_path = tmp_path / "voice_profiles.json"
    save_voice_profiles(db, db_path)
    loaded = load_voice_profiles(db_path)
    assert loaded is not None
    assert "Brakk" in loaded.speakers

    # Test matching
    test_embs = np.array([
        [0.99, 0.05],  # Very close to Brakk
        [0.02, 0.99],  # Very close to Merrow
        [-1.0, 0.0],   # Opposite / unknown
    ], dtype=np.float32)
    matches = match_embeddings_to_profiles(test_embs, loaded, similarity_threshold=0.60)
    assert matches[0][0] == "Brakk"
    assert matches[1][0] == "Merrow"
    assert 2 not in matches  # Did not match
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_diarizer.py::test_voice_profiles_computation_and_matching -v`
Expected: FAIL with `ImportError: cannot import name 'compute_voice_profiles' from 'a2ts.diarizer'`

- [ ] **Step 3: Implement compute_voice_profiles, save/load, and match in src/a2ts/diarizer.py**

In `src/a2ts/diarizer.py`:
- Implement `compute_voice_profiles`:
  - Resolve speaker name for each turn from `speakers_mapping` (dict or `SpeakersMapping.cluster_defaults` / `turn_overrides`).
  - Ignore any speaker that starts with `"SPEAKER_"` (unassigned generic clusters).
  - For each speaker, collect their segment embeddings.
  - Calculate centroid $\mu = \frac{1}{N}\sum e_i$.
  - If `existing_db` has this speaker, combine via weighted average:
    $\mu_{\text{comb}} = \frac{N_{\text{old}} \mu_{\text{old}} + N_{\text{new}} \mu_{\text{new}}}{N_{\text{old}} + N_{\text{new}}}$.
  - Normalize to unit vector: $\hat{\mu} = \frac{\mu}{\|\mu\|_2}$.
  - Return `VoiceProfilesDatabase`.
- Implement `save_voice_profiles` and `load_voice_profiles`:
  - Atomic JSON read/write using Pydantic `model_dump_json(indent=2)`.
- Implement `match_embeddings_to_profiles`:
  - Build centroid matrix $C$ ($K \times D$) and list of speaker names.
  - Compute cosine similarity matrix: $S = E \cdot C^T$ ($N \times K$).
  - For each embedding $i$, find $k^* = \arg\max_k S_{i,k}$.
  - If $S_{i, k^*} \ge \text{similarity\_threshold}$, record `matches[i] = (speaker_names[k^*], float(S_{i, k^*}))`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_diarizer.py -v`
Expected: PASS

- [ ] **Step 5: Run linters and commit**

```bash
uv run ruff check src/a2ts/diarizer.py tests/unit/test_diarizer.py
uv run ruff format --check src/a2ts/diarizer.py tests/unit/test_diarizer.py
uv run mypy src/a2ts/diarizer.py
git add src/a2ts/diarizer.py tests/unit/test_diarizer.py
git commit -m "feat: add voice profile enrollment, persistence, and similarity matching"
```

---

### Task 4: CLI Integration: Tunable Flags & Voice Profile Auto-Enrollment

**Files:**
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Updates `a2ts run`:
  - `--cluster-threshold`: float = 0.60
  - `--num-speakers`: int | None = None
  - `--voice-profiles`: Path = Path("contexte/voice_profiles.json")
  - `--profile-threshold`: float = 0.60
- Updates `a2ts review`:
  - `--voice-profiles`: Path = Path("contexte/voice_profiles.json")

- [ ] **Step 1: Write failing tests for CLI run flags and voice profile enrollment**

In `tests/test_cli.py`:
```python
def test_run_with_cluster_threshold_and_voice_profiles(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch
    from typer.testing import CliRunner
    from a2ts.cli import app
    from a2ts.models import RawSegment, SpeakerTurn, SpeakersMapping

    runner = CliRunner()
    media_file = tmp_path / "session.mp3"
    media_file.touch()
    profiles_path = tmp_path / "voice_profiles.json"

    with (
        patch("a2ts.cli.compute_file_hash", return_value="abc1234"),
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine") as mock_engine,
        patch("a2ts.cli.diarize_segments") as mock_diarize,
        patch("a2ts.cli.load_voice_profiles", return_value=None),
        patch("a2ts.cli.run_interactive_review", return_value=SpeakersMapping(cluster_defaults={"SPEAKER_00": "Brakk"})),
        patch("a2ts.cli.compute_voice_profiles") as mock_compute_vp,
        patch("a2ts.cli.save_voice_profiles") as mock_save_vp,
    ):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            RawSegment(id=0, start=0.0, end=2.0, text="Hello Brakk", words=[])
        ]
        mock_engine.return_value = mock_transcriber
        mock_diarize.return_value = [
            SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")
        ]

        result = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--cache-dir",
                str(tmp_path / ".a2ts"),
                "--cluster-threshold",
                "0.65",
                "--num-speakers",
                "4",
                "--voice-profiles",
                str(profiles_path),
                "--no-refine",
            ],
        )

        assert result.exit_code == 0
        # Verify diarize_segments called with threshold and num_speakers
        mock_diarize.assert_called_once()
        kwargs = mock_diarize.call_args.kwargs
        assert kwargs["distance_threshold"] == 0.65
        assert kwargs["num_speakers"] == 4

        # Verify voice profile computation was invoked and saved
        mock_compute_vp.assert_called_once()
        mock_save_vp.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_run_with_cluster_threshold_and_voice_profiles -v`
Expected: FAIL with unrecognized options `--cluster-threshold` or assertions.

- [ ] **Step 3: Implement CLI updates in src/a2ts/cli.py**

In `src/a2ts/cli.py`:
- Add options to `run`:
  - `cluster_threshold: Annotated[float, typer.Option(help="Diarization cosine distance threshold (higher = merges more)")] = 0.60`
  - `num_speakers: Annotated[int | None, typer.Option(help="Target speaker count for clustering")] = None`
  - `voice_profiles: Annotated[Path, typer.Option(help="Path to voice profiles database JSON")] = Path("contexte/voice_profiles.json")`
  - `profile_threshold: Annotated[float, typer.Option(help="Cosine similarity threshold for matching voice profiles")] = 0.60`
- Pass `distance_threshold=cluster_threshold`, `num_speakers=num_speakers`, and `embeddings_cache_prefix=diarization_dir / file_hash` to `diarize_segments`.
- Before interactive review:
  - If `voice_profiles.is_file()`:
    - Load `db = load_voice_profiles(voice_profiles)`.
    - Check if cached embeddings exist (`embeddings_npy = diarization_dir / f"{file_hash}_embeddings.npy"`).
    - If both exist, match embeddings: for clusters where the majority of valid embeddings match a known voice profile with score $\ge \text{profile\_threshold}$, pre-seed `mapping.cluster_defaults[cluster_id] = matched_speaker`.
- After interactive review:
  - Check if cached embeddings and valid indices exist.
  - If so, call `db = compute_voice_profiles(embeddings, valid_indices, aligned_turns, mapping, existing_db=loaded_db)`.
  - Call `save_voice_profiles(db, voice_profiles)`.
  - Console prints: `✓ Voice profiles updated for X enrolled speakers.`
- Also update `review` command with `--voice-profiles` option to update profiles after manual review.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run linters and commit**

```bash
uv run ruff check src/a2ts/cli.py tests/test_cli.py
uv run ruff format --check src/a2ts/cli.py tests/test_cli.py
uv run mypy src/a2ts/cli.py
git add src/a2ts/cli.py tests/test_cli.py
git commit -m "feat: expose cluster threshold and auto-enroll voice profiles in CLI"
```

---

### Task 5: Fast Re-cluster Command: `a2ts recluster`

**Files:**
- Modify: `src/a2ts/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `a2ts recluster` command

- [ ] **Step 1: Write failing unit test for `a2ts recluster` command**

In `tests/test_cli.py`:
```python
def test_recluster_command(tmp_path: Path) -> None:
    from typer.testing import CliRunner
    from a2ts.cli import app
    from a2ts.models import RawSegment, SessionMetadata, SpeakersMapping

    runner = CliRunner()
    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir()
    diar_dir = session_dir / "diarization"
    diar_dir.mkdir()

    # Create dummy cached session metadata
    meta = SessionMetadata(
        media_path=str(tmp_path / "media.mp3"),
        media_hash="hash123",
        duration_seconds=10.0,
        engine="whisper",
        model_name="large-v3",
        output_path=str(tmp_path / "recluster_out.md"),
    )
    (session_dir / "session.json").write_text(meta.model_dump_json(), encoding="utf-8")

    # Create dummy raw segments cache
    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir()
    raw_segs = [
        RawSegment(id=0, start=0.0, end=1.0, text="Turn 1", words=[]),
        RawSegment(id=1, start=1.5, end=2.5, text="Turn 2", words=[]),
    ]
    (transcripts_dir / "hash123_whisper.json").write_text(
        json.dumps([s.model_dump() for s in raw_segs]), encoding="utf-8"
    )

    # Create dummy embeddings cache
    np.save(diar_dir / "hash123_embeddings.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    (diar_dir / "hash123_indices.json").write_text("[0, 1]", encoding="utf-8")

    # Run recluster command
    result = runner.invoke(
        app,
        [
            "recluster",
            str(session_dir),
            "--cluster-threshold",
            "0.70",
            "--output",
            str(tmp_path / "recluster_out.md"),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "recluster_out.md").is_file()
    assert (session_dir / "turns.json").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_recluster_command -v`
Expected: FAIL with `No such command 'recluster'`

- [ ] **Step 3: Implement `recluster` command in src/a2ts/cli.py**

In `src/a2ts/cli.py`:
- Add `@app.command() def recluster(...)`:
  - `session_dir: Annotated[Path, typer.Argument(help="Path to session cache directory (.a2ts)")] = Path(".a2ts")`
  - `cluster_threshold: Annotated[float, typer.Option(help="Diarization cosine distance threshold")] = 0.60`
  - `num_speakers: Annotated[int | None, typer.Option(help="Target speaker count")] = None`
  - `slice_minutes: Annotated[float | None, typer.Option(help="Time slice window")] = None`
  - `output: Annotated[Path | None, typer.Option(help="Output markdown path")] = None`
  - Loads `session.json` to get `media_hash` and `engine`.
  - Loads raw segments from `transcripts/{media_hash}_{engine}.json`.
  - Loads `embeddings.npy` and `indices.json` from `diarization/{media_hash}_*`.
  - Performs fast `AgglomerativeClustering` directly on embeddings with the new threshold / `num_speakers`.
  - Runs `assign_time_slices` and `align_words_to_speaker_turns`.
  - Saves updated `turns.json`.
  - Loads existing `speakers_mapping.json` (if present) and applies mapping.
  - Consolidates, debounces turns, and renders markdown transcript to `output`!
  - Console prints: `✓ Re-clustered into X speakers in < 0.1s. Saved to <output>.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py::test_recluster_command -v`
Expected: PASS

- [ ] **Step 5: Run linters and commit**

```bash
uv run ruff check src/a2ts/cli.py tests/test_cli.py
uv run ruff format --check src/a2ts/cli.py tests/test_cli.py
uv run mypy src/a2ts/cli.py
git add src/a2ts/cli.py tests/test_cli.py
git commit -m "feat: add a2ts recluster command for fast session re-clustering"
```

---

### Task 6: Full Integration Test & Verification

**Files:**
- Modify: `tests/integration/test_pipeline_e2e.py`
- All code files.

- [ ] **Step 1: Write E2E integration test verifying voice profile cross-run auto-assignment**

In `tests/integration/test_pipeline_e2e.py`:
- Test pipeline run 1:
  - Audio transcribed, embeddings extracted and cached.
  - Interactive review assigns `SPEAKER_00` -> `Brakk`.
  - Voice profile saved to `voice_profiles.json`.
- Test pipeline run 2 (different file or re-run):
  - Same embedding matches `Brakk` voice profile.
  - Speaker mapping is pre-seeded with `Brakk` automatically.
- Test `recluster` command:
  - Invokes `recluster` on session dir and verifies turns and output markdown updated.

- [ ] **Step 2: Run full test suite and linters**

```bash
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
```
Expected: 100% PASS, 0 lint errors, 0 type errors.

- [ ] **Step 3: Commit and self-review git diff**

```bash
git add tests/integration/test_pipeline_e2e.py
git commit -m "test: add E2E test for voice profile enrollment and fast reclustering"
```
