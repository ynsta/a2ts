"""Integration tests for a2ts end-to-end pipeline and speaker clustering lifecycle."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from typer.testing import CliRunner

from a2ts.cli import app
from a2ts.diarizer import load_voice_profiles
from a2ts.models import AlignedTurn, RawSegment, WordTimestamp
from a2ts.speaker_review import apply_speakers_mapping, load_speakers_mapping

runner = CliRunner()


@patch("a2ts.cli.extract_audio_to_wav")
@patch("a2ts.cli.get_engine")
def test_cli_run_pipeline_end_to_end(
    mock_get_engine: MagicMock, mock_extract: MagicMock, tmp_path: Path
) -> None:
    # 1. Setup dummy video and context dir
    video_file = tmp_path / "test_session.mkv"
    video_file.write_bytes(b"dummy mkv video content")
    context_dir = tmp_path / "contexte"
    context_dir.mkdir()
    (context_dir / "lore.md").write_text(
        "Rencontre avec [[Kaelen]] et [[Garrick]].", encoding="utf-8"
    )

    out_file = tmp_path / "output_transcript.md"

    # 2. Mock audio extraction and engine
    mock_wav = tmp_path / "sample.wav"
    mock_wav.touch()
    mock_extract.return_value = mock_wav

    dummy_engine = MagicMock()
    dummy_engine.transcribe.return_value = [
        RawSegment(
            id=0,
            start=0.0,
            end=5.0,
            text="Bienvenue à tous pour la session.",
            words=[
                WordTimestamp(word="Bienvenue", start=0.0, end=1.0),
                WordTimestamp(word="à", start=1.1, end=1.5),
                WordTimestamp(word="tous", start=1.6, end=2.5),
                WordTimestamp(word="pour", start=2.6, end=3.0),
                WordTimestamp(word="la", start=3.1, end=3.5),
                WordTimestamp(word="session.", start=3.6, end=5.0),
            ],
        )
    ]
    mock_get_engine.return_value = dummy_engine

    # 3. Run CLI command
    result = runner.invoke(
        app,
        [
            "run",
            str(video_file),
            "--context-dir",
            str(context_dir),
            "--output",
            str(out_file),
            "--cache-dir",
            str(tmp_path / ".a2ts"),
            "--no-interactive",
            "--no-refine",
            "--no-diarize",
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "Bienvenue à tous pour la session." in content


@patch("a2ts.cli.probe_media", return_value={"format": {"duration": "10.0"}})
@patch("a2ts.speaker_review.play_audio_clip_async")
@patch("a2ts.speaker_review.Prompt.ask")
@patch("a2ts.diarizer.get_embedding_model")
@patch("a2ts.diarizer.sf.read")
@patch("a2ts.cli.get_engine")
@patch("a2ts.cli.extract_audio_to_wav")
def test_voice_profiles_enrollment_and_recluster_e2e(
    mock_extract: MagicMock,
    mock_get_engine: MagicMock,
    mock_sf_read: MagicMock,
    mock_get_model: MagicMock,
    mock_ask: MagicMock,
    mock_play: MagicMock,
    mock_probe: MagicMock,
    tmp_path: Path,
) -> None:
    # 1. Setup paths
    context_dir = tmp_path / "contexte"
    context_dir.mkdir()
    (context_dir / "lore.md").write_text(
        "Rencontre avec [[Brakk]] et [[Oskel]].", encoding="utf-8"
    )
    (context_dir / "speakers.txt").write_text("Brakk\nOskel\n", encoding="utf-8")

    vp_file = tmp_path / "voice_profiles.json"

    video1 = tmp_path / "session1.mkv"
    video1.write_bytes(b"dummy video 1 content")
    session1_cache = tmp_path / "cache1"
    out1 = tmp_path / "out1.md"

    video2 = tmp_path / "session2.mkv"
    video2.write_bytes(b"dummy video 2 content - distinct audio")
    session2_cache = tmp_path / "cache2"
    out2 = tmp_path / "out2.md"

    mock_wav = tmp_path / "sample.wav"
    mock_wav.touch()
    mock_extract.return_value = mock_wav

    # 15s of 16kHz audio
    mock_sf_read.return_value = (np.zeros(16000 * 15, dtype=np.float32), 16000)

    # Fixed 192-dim speaker embeddings
    emb_brakk = np.zeros(192, dtype=np.float32)
    emb_brakk[0] = 1.0
    emb_than = np.zeros(192, dtype=np.float32)
    emb_than[1] = 1.0
    emb_other = np.zeros(192, dtype=np.float32)
    emb_other[2] = 1.0

    mock_classifier = MagicMock()
    mock_get_model.return_value = mock_classifier

    # --- RUN 1: Transcription, extraction, interactive enrollment ---
    session1_segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=3.0,
            text="Je suis Brakk le guerrier.",
            words=[
                WordTimestamp(word="Je", start=0.0, end=0.5),
                WordTimestamp(word="suis", start=0.6, end=1.0),
                WordTimestamp(word="Brakk", start=1.1, end=1.8),
                WordTimestamp(word="le", start=1.9, end=2.2),
                WordTimestamp(word="guerrier.", start=2.3, end=3.0),
            ],
        ),
        RawSegment(
            id=1,
            start=3.5,
            end=6.0,
            text="Et moi je suis Oskel le magicien.",
            words=[
                WordTimestamp(word="Et", start=3.5, end=3.8),
                WordTimestamp(word="moi", start=3.9, end=4.2),
                WordTimestamp(word="je", start=4.3, end=4.6),
                WordTimestamp(word="suis", start=4.7, end=5.0),
                WordTimestamp(word="Oskel", start=5.1, end=5.5),
                WordTimestamp(word="le", start=5.6, end=5.7),
                WordTimestamp(word="magicien.", start=5.8, end=6.0),
            ],
        ),
        RawSegment(
            id=2,
            start=6.5,
            end=9.0,
            text="Brakk reprend la parole.",
            words=[
                WordTimestamp(word="Brakk", start=6.5, end=7.2),
                WordTimestamp(word="reprend", start=7.3, end=8.0),
                WordTimestamp(word="la", start=8.1, end=8.4),
                WordTimestamp(word="parole.", start=8.5, end=9.0),
            ],
        ),
    ]

    dummy_engine1 = MagicMock()
    dummy_engine1.transcribe.return_value = session1_segments
    mock_get_engine.return_value = dummy_engine1

    # In Run 1, seg 0 -> brakk, seg 1 -> oskel, seg 2 -> brakk (segments, then turns)
    mock_classifier.encode_batch.side_effect = [
        torch.tensor(emb_brakk).unsqueeze(0),
        torch.tensor(emb_than).unsqueeze(0),
        torch.tensor(emb_brakk).unsqueeze(0),
        torch.tensor(emb_brakk).unsqueeze(0),
        torch.tensor(emb_than).unsqueeze(0),
        torch.tensor(emb_brakk).unsqueeze(0),
    ]

    # Interactive review: assign SPEAKER_00 -> Brakk, skip SPEAKER_01
    mock_ask.side_effect = ["Brakk", "s"]

    result1 = runner.invoke(
        app,
        [
            "run",
            str(video1),
            "--device",
            "cpu",
            "--context-dir",
            str(context_dir),
            "--cache-dir",
            str(session1_cache),
            "--voice-profiles",
            str(vp_file),
            "--output",
            str(out1),
            "--no-refine",
            "--no-auto-play",
        ],
    )
    assert result1.exit_code == 0
    assert out1.is_file()
    content1 = out1.read_text(encoding="utf-8")
    assert "] Brakk" in content1
    assert "] SPEAKER_01" in content1

    # Verify voice_profiles.json created and enrolled Brakk
    assert vp_file.is_file()
    db1 = load_voice_profiles(vp_file)
    assert db1 is not None
    assert "Brakk" in db1.speakers
    assert db1.speakers["Brakk"].sample_count == 2
    assert "SPEAKER_01" not in db1.speakers

    # Verify session 1 cache artifacts
    sess1_dir = next((session1_cache / "sessions").iterdir())
    assert (sess1_dir / "turns.json").is_file()
    assert (sess1_dir / "speakers_mapping.json").is_file()
    mapping1 = load_speakers_mapping(sess1_dir / "speakers_mapping.json")
    assert mapping1.cluster_defaults.get("SPEAKER_00") == "Brakk"

    # Verify embeddings cache files exist
    turn_files = list((session1_cache / "diarization").glob("*_turn_embeddings.npy"))
    assert len(turn_files) >= 1
    seg_files = list((session1_cache / "diarization").glob("*_segment_embeddings.npy"))
    assert len(seg_files) >= 1

    # --- RUN 2: Voice profile pre-matches without manual prompt ---
    session2_segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=3.0,
            text="Brakk revient pour une nouvelle quête.",
            words=[
                WordTimestamp(word="Brakk", start=0.0, end=0.8),
                WordTimestamp(word="revient", start=0.9, end=1.5),
                WordTimestamp(word="pour", start=1.6, end=1.9),
                WordTimestamp(word="une", start=2.0, end=2.2),
                WordTimestamp(word="nouvelle", start=2.3, end=2.6),
                WordTimestamp(word="quête.", start=2.7, end=3.0),
            ],
        ),
        RawSegment(
            id=1,
            start=3.5,
            end=6.0,
            text="Un voyageur mystérieux répond.",
            words=[
                WordTimestamp(word="Un", start=3.5, end=3.8),
                WordTimestamp(word="voyageur", start=3.9, end=4.5),
                WordTimestamp(word="mystérieux", start=4.6, end=5.3),
                WordTimestamp(word="répond.", start=5.4, end=6.0),
            ],
        ),
    ]

    dummy_engine2 = MagicMock()
    dummy_engine2.transcribe.return_value = session2_segments
    mock_get_engine.return_value = dummy_engine2

    # In Run 2, seg 0 has Brakk's embedding, seg 1 has other embedding (segments, then turns)
    mock_classifier.encode_batch.side_effect = [
        torch.tensor(emb_brakk).unsqueeze(0),
        torch.tensor(emb_other).unsqueeze(0),
        torch.tensor(emb_brakk).unsqueeze(0),
        torch.tensor(emb_other).unsqueeze(0),
    ]

    result2 = runner.invoke(
        app,
        [
            "run",
            str(video2),
            "--device",
            "cpu",
            "--context-dir",
            str(context_dir),
            "--cache-dir",
            str(session2_cache),
            "--voice-profiles",
            str(vp_file),
            "--output",
            str(out2),
            "--no-interactive",
            "--no-refine",
            "--no-auto-play",
        ],
    )
    assert result2.exit_code == 0
    assert "Matched voice profile for SPEAKER_00: Brakk" in result2.output
    assert out2.is_file()
    content2 = out2.read_text(encoding="utf-8")
    assert "] Brakk" in content2
    assert "] SPEAKER_01" in content2

    sess2_dir = next((session2_cache / "sessions").iterdir())
    mapping2 = load_speakers_mapping(sess2_dir / "speakers_mapping.json")
    assert mapping2.cluster_defaults.get("SPEAKER_00") == "Brakk"

    # --- FAST RECLUSTER: Recluster session 1 using cached embeddings ---
    recluster_out = tmp_path / "recluster_out.md"
    extract_calls_before = mock_extract.call_count
    transcribe_calls_before = dummy_engine1.transcribe.call_count
    model_calls_before = mock_get_model.call_count

    result_recluster = runner.invoke(
        app,
        [
            "recluster",
            str(session1_cache),
            "--num-speakers",
            "1",
            "--voice-profiles",
            str(vp_file),
            "--output",
            str(recluster_out),
        ],
    )
    assert result_recluster.exit_code == 0
    assert "✓ Re-clustered into 1 speakers" in result_recluster.output
    assert recluster_out.is_file()

    # Verify no audio re-extraction or transcription engine execution during recluster
    assert mock_extract.call_count == extract_calls_before
    assert dummy_engine1.transcribe.call_count == transcribe_calls_before
    assert mock_get_model.call_count == model_calls_before

    # Verify all turns merged into SPEAKER_00 / Brakk
    recluster_content = recluster_out.read_text(encoding="utf-8")
    assert "] Brakk" in recluster_content
    assert "SPEAKER_01" not in recluster_content

    re_turns = [
        AlignedTurn.model_validate(t)
        for t in json.loads((sess1_dir / "turns.json").read_text(encoding="utf-8"))
    ]
    assert len(re_turns) == 3
    assert all(t.cluster_id == "SPEAKER_00" for t in re_turns)
    mapped_re_turns = apply_speakers_mapping(re_turns, mapping1)
    assert all(t.speaker == "Brakk" for t in mapped_re_turns)

    # Verify reclustering back to 2 speakers restores both clusters
    recluster_2spk_out = tmp_path / "recluster_2spk.md"
    result_recluster_2spk = runner.invoke(
        app,
        [
            "recluster",
            str(session1_cache),
            "--num-speakers",
            "2",
            "--voice-profiles",
            str(vp_file),
            "--output",
            str(recluster_2spk_out),
        ],
    )
    assert result_recluster_2spk.exit_code == 0
    assert "✓ Re-clustered into 2 speakers" in result_recluster_2spk.output
    content_2spk = recluster_2spk_out.read_text(encoding="utf-8")
    assert "] Brakk" in content_2spk
    assert "] SPEAKER_01" in content_2spk
