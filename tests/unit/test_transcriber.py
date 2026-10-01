from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from a2ts.models import RawSegment, WordTimestamp
from a2ts.transcriber import (
    TranscriberEngine,
    VoxtralEngine,
    WhisperEngine,
    create_transcriber,
    ensure_cuda_libs,
    get_engine,
)


def test_transcriber_engine_protocol_conformance() -> None:
    class DummyEngine:
        def transcribe(
            self,
            audio_path: Path,
            prompt: str | None = None,
            language: str = "fr",
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
