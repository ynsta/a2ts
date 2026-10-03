from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from a2ts.models import RawSegment, WordTimestamp
from a2ts.transcriber import (
    ParakeetEngine,
    TranscriberEngine,
    VoxtralEngine,
    WhisperEngine,
    create_transcriber,
    ensure_cuda_libs,
    ensure_pyav_compat,
    get_engine,
)


def test_transcriber_engine_protocol_conformance() -> None:
    class DummyEngine:
        def transcribe(
            self,
            audio_path: Path,
            prompt: str | None = None,
            language: str = "fr",
            duration: float | None = None,
        ) -> list[RawSegment]:
            return [RawSegment(id=0, start=0.0, end=1.0, text="Test")]

    engine: TranscriberEngine = DummyEngine()
    result = engine.transcribe(Path("test.wav"))
    assert len(result) == 1
    assert result[0].text == "Test"


def test_get_engine_whisper() -> None:
    engine = get_engine("whisper", model_name="tiny", device="cpu")
    assert isinstance(engine, WhisperEngine)
    assert engine.model_name == "tiny"
    assert engine.device == "cpu"


def test_get_engine_voxtral() -> None:
    engine = get_engine("voxtral", model_name="dummy/voxtral", load_4bit=False)
    assert isinstance(engine, VoxtralEngine)
    assert engine.model_name == "dummy/voxtral"
    assert engine.load_4bit is False


def test_get_engine_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine("unknown_engine")


def test_whisper_engine_transcribe() -> None:
    engine = WhisperEngine(model_name="tiny", device="cpu")

    mock_word1 = MagicMock()
    mock_word1.word = " Bonjour "
    mock_word1.start = 0.5
    mock_word1.end = 1.0
    mock_word1.probability = 0.95

    mock_word2 = MagicMock()
    mock_word2.word = " monde "
    mock_word2.start = 1.1
    mock_word2.end = 1.8
    mock_word2.probability = 0.98

    mock_segment = MagicMock()
    mock_segment.start = 0.5
    mock_segment.end = 1.8
    mock_segment.text = " Bonjour monde "
    mock_segment.words = [mock_word1, mock_word2]

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())

    with patch.object(engine, "_get_model", return_value=mock_model):
        results = engine.transcribe(
            Path("audio.wav"), prompt="Salutations", language="fr"
        )

    mock_model.transcribe.assert_called_once_with(
        "audio.wav",
        language="fr",
        initial_prompt="Salutations",
        vad_filter=True,
        word_timestamps=True,
    )
    assert len(results) == 1
    seg = results[0]
    assert seg.id == 0
    assert seg.start == 0.5
    assert seg.end == 1.8
    assert seg.text == "Bonjour monde"
    assert len(seg.words) == 2
    assert seg.words[0] == WordTimestamp(
        word="Bonjour", start=0.5, end=1.0, probability=0.95
    )
    assert seg.words[1] == WordTimestamp(
        word="monde", start=1.1, end=1.8, probability=0.98
    )


def test_whisper_engine_transcribe_no_words() -> None:
    engine = WhisperEngine(model_name="tiny", device="cpu")

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = "Sans mots"
    mock_segment.words = None

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())

    with patch.object(engine, "_get_model", return_value=mock_model):
        results = engine.transcribe(Path("audio.wav"))

    assert len(results) == 1
    assert results[0].words == []
    assert results[0].text == "Sans mots"


def test_voxtral_engine_transcribe_with_prompt() -> None:
    engine = VoxtralEngine(model_name="dummy/voxtral")

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_inputs = MagicMock()
    mock_inputs.input_ids.shape = [1, 10]
    mock_processor.apply_chat_template.return_value.to.return_value = mock_inputs

    mock_outputs = MagicMock()
    mock_model.generate.return_value = mock_outputs
    mock_processor.batch_decode.return_value = [" Transcription Voxtral réussie "]

    with patch.object(engine, "_load_model", return_value=(mock_processor, mock_model)):
        results = engine.transcribe(
            Path("/tmp/audio.wav"), prompt="Contexte JdR", language="fr"
        )

    assert len(results) == 1
    assert results[0].id == 0
    assert results[0].text == "Transcription Voxtral réussie"
    assert results[0].start == 0.0
    assert results[0].end == 0.0

    call_args = mock_processor.apply_chat_template.call_args[0][0]
    user_message = call_args[0]["content"]
    assert user_message[0]["type"] == "audio"
    assert user_message[1]["type"] == "text"
    assert "Contexte et vocabulaire: Contexte JdR" in user_message[1]["text"]


def test_voxtral_engine_transcribe_no_prompt() -> None:
    engine = VoxtralEngine(model_name="dummy/voxtral")

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_inputs = MagicMock()
    mock_inputs.input_ids.shape = [1, 5]
    mock_processor.apply_chat_template.return_value.to.return_value = mock_inputs
    mock_processor.batch_decode.return_value = []

    with patch.object(engine, "_load_model", return_value=(mock_processor, mock_model)):
        results = engine.transcribe(Path("/tmp/audio.wav"), prompt=None)

    assert len(results) == 1
    assert results[0].text == ""

    call_args = mock_processor.apply_chat_template.call_args[0][0]
    user_message = call_args[0]["content"]
    assert (
        user_message[1]["text"]
        == "Transcris fidèlement cet enregistrement audio en français."
    )


def test_ensure_cuda_libs() -> None:
    # ensure_cuda_libs should execute safely without exceptions
    ensure_cuda_libs()


def test_create_transcriber_voxtral_forwards_device_and_compute_type() -> None:
    engine = create_transcriber(
        engine="voxtral",
        model_name="dummy/voxtral",
        device="cpu",
        compute_type="int8",
        load_4bit=False,
    )
    assert isinstance(engine, VoxtralEngine)
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"
    assert engine.model_name == "dummy/voxtral"


def test_create_transcriber_whisper_forwards_device_and_compute_type() -> None:
    engine = create_transcriber(
        engine="whisper",
        model_name="tiny",
        device="cpu",
        compute_type="int8",
    )
    assert isinstance(engine, WhisperEngine)
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"
    assert engine.model_name == "tiny"


def test_voxtral_engine_defaults() -> None:
    engine = VoxtralEngine()
    assert engine.device == "auto"
    assert engine.compute_type == "float16"


def test_voxtral_engine_truncation_warning(caplog: pytest.LogCaptureFixture) -> None:
    engine = VoxtralEngine(model_name="dummy/voxtral", max_new_tokens=10)

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_inputs = MagicMock()
    mock_inputs.input_ids.shape = [1, 5]
    mock_processor.apply_chat_template.return_value.to.return_value = mock_inputs

    # Simulate generating 10 tokens (reaching max_new_tokens cap)
    mock_generated = MagicMock()
    mock_generated.__len__.return_value = 10
    mock_generated.shape = [1, 10]
    mock_generated.__getitem__.return_value = [1] * 10

    mock_outputs = MagicMock()
    # outputs[:, 5:] returns mock_generated
    mock_outputs.__getitem__.return_value = mock_generated
    mock_model.generate.return_value = mock_outputs
    mock_processor.batch_decode.return_value = ["Truncated text..."]

    with (
        patch.object(engine, "_load_model", return_value=(mock_processor, mock_model)),
        caplog.at_level("WARNING"),
    ):
        results = engine.transcribe(Path("/tmp/audio.wav"), prompt=None)

    assert len(results) == 1
    assert "truncated" in caplog.text.lower() or "max_new_tokens" in caplog.text.lower()
    assert results[0].is_truncated is True


def test_voxtral_engine_long_audio_warning(caplog: pytest.LogCaptureFixture) -> None:
    engine = VoxtralEngine(model_name="dummy/voxtral")

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_inputs = MagicMock()
    mock_inputs.input_ids.shape = [1, 5]
    mock_processor.apply_chat_template.return_value.to.return_value = mock_inputs

    mock_outputs = MagicMock()
    mock_outputs.__getitem__.return_value = MagicMock(shape=[1, 2])
    mock_model.generate.return_value = mock_outputs
    mock_processor.batch_decode.return_value = ["Short text"]

    with (
        patch.object(engine, "_load_model", return_value=(mock_processor, mock_model)),
        caplog.at_level("WARNING"),
    ):
        results = engine.transcribe(Path("/tmp/audio.wav"), prompt=None, duration=240.0)

    assert len(results) == 1
    assert (
        "Voxtral is an experimental engine designed for short clips; long recordings risk truncation at 1024 tokens"
        in caplog.text
    )


def test_ensure_pyav_compat_handles_metadata_errors() -> None:
    import av

    calls: list[dict[str, Any]] = []

    def mock_orig_open(*args: Any, **kwargs: Any) -> str:
        calls.append(kwargs.copy())
        if "metadata_errors" in kwargs:
            raise TypeError(
                "open() got an unexpected keyword argument 'metadata_errors'"
            )
        return "success_container"

    # Patch with our mock
    with patch.object(av, "open", mock_orig_open):
        ensure_pyav_compat()
        # Calling av.open with metadata_errors should succeed via the compat wrapper
        res = av.open("fake_path.wav", mode="r", metadata_errors="ignore")
        assert res == "success_container"
        assert len(calls) == 2
        # First call contained metadata_errors
        assert "metadata_errors" in calls[0]
        # Second call had metadata_errors removed
        assert "metadata_errors" not in calls[1]


def test_ensure_pyav_compat_preserves_other_typeerrors() -> None:
    import av

    def mock_broken_open(*args: Any, **kwargs: Any) -> None:
        raise TypeError("unrelated type error")

    with patch.object(av, "open", mock_broken_open):
        ensure_pyav_compat()
        with pytest.raises(TypeError, match="unrelated type error"):
            av.open("fake_path.wav")


def test_get_engine_parakeet() -> None:
    # Default model
    engine = get_engine("parakeet", device="cpu")
    assert isinstance(engine, ParakeetEngine)
    assert engine.model_name == "nvidia/parakeet-tdt-0.6b-v3"
    assert engine.device == "cpu"

    # Alias mapping for legacy/cloud NIM model name
    engine_alias = get_engine(
        "parakeet",
        model_name="nvidia/parakeet-1.1b-rnnt-multilingual-asr",
        device="cpu",
    )
    assert isinstance(engine_alias, ParakeetEngine)
    assert engine_alias.model_name == "nvidia/parakeet-tdt-0.6b-v3"

    # Custom model preserved
    engine_custom = get_engine(
        "parakeet", model_name="custom/parakeet-model", device="cpu"
    )
    assert isinstance(engine_custom, ParakeetEngine)
    assert engine_custom.model_name == "custom/parakeet-model"


def test_parakeet_engine_missing_nemo_raises_importerror() -> None:
    engine = ParakeetEngine()
    with (
        patch.dict(
            "sys.modules",
            {"nemo": None, "nemo.collections": None, "nemo.collections.asr": None},
        ),
        pytest.raises(ImportError, match="NVIDIA NeMo is required"),
    ):
        engine._load_model()


def test_parakeet_engine_transcribe_with_timestamps() -> None:
    from a2ts.models import SpeakerTurn
    from a2ts.timeline import align_words_to_speaker_turns

    engine = ParakeetEngine(
        model_name="nvidia/parakeet-1.1b-rnnt-multilingual-asr", device="cpu"
    )

    mock_hyp = MagicMock()
    mock_hyp.text = "Bonjour tout le monde"
    mock_hyp.timestamp = {
        "word": [
            {"word": "Bonjour", "start": 0.5, "end": 1.0},
            {"word": "tout", "start": 1.1, "end": 1.4},
            {"word": "le", "start": 1.4, "end": 1.6},
            {"word": "monde", "start": 1.6, "end": 2.2},
        ]
    }

    mock_model = MagicMock()
    mock_model.transcribe.return_value = [mock_hyp]

    with patch.object(engine, "_load_model", return_value=mock_model):
        results = engine.transcribe(Path("test.wav"), prompt=None, language="fr")

    assert len(results) == 1
    seg = results[0]
    assert seg.text == "Bonjour tout le monde"
    assert seg.start == 0.5
    assert seg.end == 2.2
    assert len(seg.words) == 4
    assert seg.words[0] == WordTimestamp(
        word="Bonjour", start=0.5, end=1.0, probability=1.0
    )
    assert seg.words[3] == WordTimestamp(
        word="monde", start=1.6, end=2.2, probability=1.0
    )

    # Test alignment with speaker turns (diarization verification)
    turns = [
        SpeakerTurn(
            id=0, start=0.4, end=1.05, cluster_id="SPEAKER_00", resolved_speaker="Alice"
        ),
        SpeakerTurn(
            id=1, start=1.08, end=2.3, cluster_id="SPEAKER_01", resolved_speaker="Bob"
        ),
    ]
    aligned = align_words_to_speaker_turns(results, turns)
    assert len(aligned) == 2
    assert aligned[0].speaker == "Alice"
    assert aligned[0].text == "Bonjour"
    assert aligned[1].speaker == "Bob"
    assert aligned[1].text == "tout le monde"


def test_parakeet_engine_chunked_streaming(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf  # type: ignore[import-untyped]

    wav_file = tmp_path / "long.wav"
    # Create 400 seconds of audio at 16kHz
    sr = 16000
    sf.write(wav_file, np.zeros(400 * sr, dtype=np.float32), sr)

    engine = ParakeetEngine(device="cpu")

    mock_hyp_chunk1 = MagicMock()
    mock_hyp_chunk1.text = "Première partie"
    mock_hyp_chunk1.timestamp = {
        "word": [
            {"word": "Première", "start": 1.0, "end": 2.0},
            {"word": "partie", "start": 2.5, "end": 3.0},
        ]
    }

    mock_hyp_chunk2 = MagicMock()
    mock_hyp_chunk2.text = "Deuxième partie"
    mock_hyp_chunk2.timestamp = {
        "word": [
            {"word": "Deuxième", "start_offset": 5.0, "end_offset": 6.0},
            {"word": "partie", "start_offset": 6.5, "end_offset": 7.0},
        ]
    }

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = [
        [mock_hyp_chunk1],
        [mock_hyp_chunk2],
        [mock_hyp_chunk2],
    ]

    with patch.object(engine, "_load_model", return_value=mock_model):
        results = engine.transcribe(wav_file, prompt=None, language="fr")

    assert len(results) == 1
    seg = results[0]
    assert "Première partie" in seg.text
    assert "Deuxième partie" in seg.text
    # Words in chunk 1 (offset 0.0s)
    assert seg.words[0].word == "Première"
    assert seg.words[0].start == 1.0
    # Words in chunk 2 (offset 180.0s)
    assert seg.words[2].word == "Deuxième"
    assert seg.words[2].start == 180.0 + 5.0
    assert seg.words[2].end == 180.0 + 6.0
