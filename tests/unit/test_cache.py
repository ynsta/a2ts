import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pytest

from a2ts.cache import (
    atomic_save_numpy,
    atomic_write_text,
    compute_prompt_hash,
    load_diarization_cache,
    load_transcript_cache,
    save_diarization_cache,
    save_transcript_cache,
)
from a2ts.models import (
    DiarizationCacheProvenance,
    RawSegment,
    SpeakerTurn,
    TranscriptCacheProvenance,
)


def test_compute_prompt_hash() -> None:
    assert compute_prompt_hash(None) == "no_prompt"
    h1 = compute_prompt_hash("Brakk MJ Oskel")
    assert len(h1) == 16
    h2 = compute_prompt_hash("Brakk MJ Oskel")
    assert h1 == h2
    assert h1 != compute_prompt_hash("Different prompt")


def test_transcript_cache_save_and_load(tmp_path: Path) -> None:
    cache_path = tmp_path / "transcripts" / "test.json"
    prov = TranscriptCacheProvenance(
        media_hash="hash123",
        engine="whisper",
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="prompt123",
    )
    segments = [
        RawSegment(id=0, start=0.0, end=2.0, text="Hello"),
        RawSegment(id=1, start=2.5, end=4.0, text="World"),
    ]

    save_transcript_cache(cache_path, prov, segments)
    assert cache_path.is_file()

    loaded = load_transcript_cache(cache_path, prov, force=False)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].text == "Hello"
    assert loaded[1].text == "World"


def test_transcript_cache_invalidation_on_model_or_prompt(tmp_path: Path) -> None:
    cache_path = tmp_path / "transcripts" / "test.json"
    prov1 = TranscriptCacheProvenance(
        media_hash="hash123",
        engine="whisper",
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="prompt123",
    )
    segments = [RawSegment(id=0, start=0.0, end=2.0, text="Hello")]
    save_transcript_cache(cache_path, prov1, segments)

    # Different model
    prov_diff_model = TranscriptCacheProvenance(
        media_hash="hash123",
        engine="whisper",
        model_name="medium",
        compute_type="float16",
        prompt_hash="prompt123",
    )
    assert load_transcript_cache(cache_path, prov_diff_model) is None

    # Different prompt
    prov_diff_prompt = TranscriptCacheProvenance(
        media_hash="hash123",
        engine="whisper",
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="diff_prompt",
    )
    assert load_transcript_cache(cache_path, prov_diff_prompt) is None


def test_transcript_cache_force_and_corruption(tmp_path: Path) -> None:
    cache_path = tmp_path / "transcripts" / "test.json"
    prov = TranscriptCacheProvenance(
        media_hash="hash123",
        engine="whisper",
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="prompt123",
    )
    segments = [RawSegment(id=0, start=0.0, end=2.0, text="Hello")]
    save_transcript_cache(cache_path, prov, segments)

    # Force bypass
    assert load_transcript_cache(cache_path, prov, force=True) is None

    # Corrupted JSON
    cache_path.write_text("{invalid json", encoding="utf-8")
    assert load_transcript_cache(cache_path, prov, force=False) is None


def test_diarization_cache_save_and_load(tmp_path: Path) -> None:
    cache_path = tmp_path / "diarization" / "test.json"
    prov = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="auto",
        cluster_threshold=0.60,
        num_speakers=2,
        device="cuda",
    )
    turns = [
        SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=1, start=2.5, end=4.0, cluster_id="SPEAKER_01"),
    ]

    save_diarization_cache(cache_path, prov, turns)
    assert cache_path.is_file()

    loaded = load_diarization_cache(cache_path, prov, force=False)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].cluster_id == "SPEAKER_00"


def test_diarization_cache_invalidation_and_force(tmp_path: Path) -> None:
    cache_path = tmp_path / "diarization" / "test.json"
    prov = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="auto",
        cluster_threshold=0.60,
        num_speakers=2,
        device="cuda",
    )
    turns = [SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")]
    save_diarization_cache(cache_path, prov, turns)

    # Different threshold
    prov_diff_thresh = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="auto",
        cluster_threshold=0.40,
        num_speakers=2,
        device="cuda",
    )
    assert load_diarization_cache(cache_path, prov_diff_thresh) is None

    # Different speaker count
    prov_diff_speakers = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="auto",
        cluster_threshold=0.60,
        num_speakers=4,
        device="cuda",
    )
    assert load_diarization_cache(cache_path, prov_diff_speakers) is None

    # Force bypass
    assert load_diarization_cache(cache_path, prov, force=True) is None


def test_atomic_write_text_concurrent(tmp_path: Path) -> None:
    target_file = tmp_path / "concurrent.txt"
    num_threads = 10
    num_writes = 30
    payloads = [f"thread-payload-{i}\n" for i in range(num_writes)]

    def write_payload(payload: str) -> None:
        atomic_write_text(target_file, payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(write_payload, p) for p in payloads]
        for f in concurrent.futures.as_completed(futures):
            # Verify no exceptions (e.g. FileNotFoundError on rename) were raised
            f.result()

    assert target_file.is_file()
    final_content = target_file.read_text(encoding="utf-8")
    assert final_content in payloads
    # Ensure no leftover temporary files in directory
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_atomic_write_text_cleanup_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_file = tmp_path / "error_test.txt"

    def fail_replace(
        _src: os.PathLike[str] | str, _dst: os.PathLike[str] | str
    ) -> None:
        raise OSError("Simulated disk error during replace")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="Simulated disk error"):
        atomic_write_text(target_file, "should not persist")

    assert not target_file.exists()
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_atomic_save_numpy(tmp_path: Path) -> None:
    target_file = tmp_path / "embeddings.npy"
    data = np.array([[1.0, 2.5, -3.2], [0.5, 4.0, 1.2]], dtype=np.float32)

    atomic_save_numpy(target_file, data)

    assert target_file.is_file()
    loaded = np.load(target_file)
    np.testing.assert_array_equal(loaded, data)
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_atomic_save_numpy_cleanup_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_file = tmp_path / "embeddings_error.npy"
    data = np.ones((3, 3), dtype=np.float32)

    def fail_replace(
        _src: os.PathLike[str] | str, _dst: os.PathLike[str] | str
    ) -> None:
        raise OSError("Simulated disk error during replace")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="Simulated disk error"):
        atomic_save_numpy(target_file, data)

    assert not target_file.exists()
    assert list(tmp_path.glob(".*.tmp*")) == []
