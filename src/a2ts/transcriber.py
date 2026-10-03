"""Dual transcription engine architecture for a2ts (Whisper & Voxtral)."""

import ctypes
import logging
import sys
from pathlib import Path
from typing import Any, Protocol, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from a2ts.models import RawSegment, WordTimestamp

logger = logging.getLogger(__name__)
console = Console()


def ensure_cuda_libs() -> None:
    """Preload bundled NVIDIA CUDA libraries (cublas and cudnn) if available."""
    for path_entry in sys.path:
        nvidia_dir = Path(path_entry) / "nvidia"
        if nvidia_dir.is_dir():
            for lib_dir in nvidia_dir.glob("*/lib"):
                if lib_dir.is_dir():
                    candidates: list[Path] = []
                    candidates.extend(lib_dir.glob("libcublas.so*"))
                    candidates.extend(lib_dir.glob("libcudnn*.so*"))
                    for so_file in sorted(candidates):
                        try:
                            ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                        except (OSError, RuntimeError):
                            continue


def ensure_pyav_compat() -> None:
    """Ensure PyAV compatibility with faster-whisper.

    PyAV >= 19.0.0 removed the 'metadata_errors' and 'metadata_encoding' parameters
    from av.open(). faster-whisper passes metadata_errors='ignore', which causes a
    TypeError on PyAV >= 19.0.0.
    We wrap av.open to strip metadata_errors and metadata_encoding if rejected with TypeError.
    """
    try:
        import av
    except ImportError:
        return

    orig_open = av.open
    if getattr(orig_open, "_a2ts_compat", False):
        return

    def _safe_av_open(*args: Any, **kwargs: Any) -> Any:
        try:
            return orig_open(*args, **kwargs)
        except TypeError as err:
            err_msg = str(err)
            if "metadata_errors" in err_msg or "metadata_encoding" in err_msg:
                kwargs.pop("metadata_errors", None)
                kwargs.pop("metadata_encoding", None)
                return orig_open(*args, **kwargs)
            raise

    _safe_av_open._a2ts_compat = True  # type: ignore[attr-defined]
    _safe_av_open._orig_open = orig_open  # type: ignore[attr-defined]
    av.open = _safe_av_open


# Apply PyAV compatibility eagerly on module import if PyAV is installed
ensure_pyav_compat()


class TranscriberEngine(Protocol):
    """Protocol defining the transcription engine interface."""

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
        duration: float | None = None,
    ) -> list[RawSegment]:
        """Transcribe an audio file into timestamped RawSegment objects."""
        ...


class WhisperEngine:
    """Faster-Whisper engine with CTranslate2 and CUDA 12 support."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            ensure_cuda_libs()
            ensure_pyav_compat()
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            with console.status(
                f"[bold cyan]Loading Whisper model '{self.model_name}' on {self.device} ({self.compute_type})...[/bold cyan]",
                spinner="dots",
            ):
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            console.print(f"[green]✓ Whisper '{self.model_name}' loaded.[/green]")
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
        duration: float | None = None,
    ) -> list[RawSegment]:
        """Transcribe audio using faster-whisper."""
        model = self._get_model()
        transcribe_out = model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt,
            vad_filter=True,
            word_timestamps=True,
        )
        if isinstance(transcribe_out, tuple):
            segments_gen, info = transcribe_out
        else:
            segments_gen, info = transcribe_out, None

        results: list[RawSegment] = []
        raw_duration = getattr(info, "duration", None)
        computed_duration = (
            float(raw_duration) if isinstance(raw_duration, (int, float)) else None
        )
        total_duration = duration if duration is not None else computed_duration

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Transcribing (Whisper)[/bold blue]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("transcribe", total=total_duration)
            for i, s in enumerate(segments_gen):
                if (
                    total_duration
                    and hasattr(s, "end")
                    and isinstance(s.end, (int, float))
                ):
                    progress.update(task, completed=min(s.end, total_duration))
                words = [
                    WordTimestamp(
                        word=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in (s.words or [])
                ]
                results.append(
                    RawSegment(
                        id=i,
                        start=s.start,
                        end=s.end,
                        text=s.text.strip(),
                        words=words,
                    )
                )
            if duration:
                progress.update(task, completed=duration)

        console.print(
            f"[green]✓ Whisper transcription complete ({len(results)} segments).[/green]\n"
        )
        return results


class VoxtralEngine:
    """Mistral Voxtral multimodal audio engine with 4-bit quantization support."""

    def __init__(
        self,
        model_name: str = "mistralai/Voxtral-Mini-3B-2507",
        load_4bit: bool = True,
        device: str = "auto",
        compute_type: str = "float16",
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.load_4bit = load_4bit
        self.device = device
        self.compute_type = compute_type
        self.max_new_tokens = max_new_tokens
        self._processor: Any = None
        self._model: Any = None

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                VoxtralForConditionalGeneration,
            )

            is_cuda = (self.device in ("cuda", "auto")) and torch.cuda.is_available()
            with console.status(
                f"[bold cyan]Loading Voxtral model '{self.model_name}' on {self.device}...[/bold cyan]",
                spinner="dots",
            ):
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                quant_config = None
                if self.load_4bit and is_cuda:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                device_map = "auto" if is_cuda else "cpu"
                self._model = VoxtralForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    device_map=device_map,
                )
            console.print("[green]✓ Voxtral model loaded.[/green]")
        return self._processor, self._model

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
        duration: float | None = None,
    ) -> list[RawSegment]:
        """Transcribe audio using Voxtral model with real-time feedback."""
        import torch

        effective_duration = duration
        if effective_duration is None:
            try:
                from a2ts.media import get_media_duration

                effective_duration = get_media_duration(audio_path)
            except (RuntimeError, FileNotFoundError, OSError):
                effective_duration = None

        if effective_duration is not None and effective_duration > 180.0:
            warning_msg = (
                "Voxtral is an experimental engine designed for short clips; "
                "long recordings risk truncation at 1024 tokens"
            )
            logger.warning(warning_msg)
            console.print(f"[bold yellow]Warning: {warning_msg}[/bold yellow]")

        processor, model = self._load_model()

        instruction = "Transcris fidèlement cet enregistrement audio en français."
        if prompt:
            instruction += f" Contexte et vocabulaire: {prompt}"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(audio_path.resolve())},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        target_device = (
            "cuda"
            if (self.device in ("cuda", "auto")) and torch.cuda.is_available()
            else "cpu"
        )
        with console.status(
            "[bold cyan]Encoding audio waveform and prompt...[/bold cyan]",
            spinner="dots",
        ):
            inputs = processor.apply_chat_template(conversation).to(target_device)

        streamer = None
        try:
            from transformers import TextStreamer

            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                streamer = TextStreamer(tok, skip_prompt=True)
        except (ImportError, AttributeError, TypeError, ValueError):
            streamer = None

        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if streamer is not None:
            gen_kwargs["streamer"] = streamer
            console.print(
                "\n[bold green]=== Voxtral Live Transcription Stream ===[/bold green]"
            )
            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)
            console.print("\n[green]✓ Voxtral generation complete.[/green]\n")
        else:
            with (
                console.status(
                    "[bold green]Voxtral generating transcription on GPU...[/bold green]",
                    spinner="dots",
                ),
                torch.inference_mode(),
            ):
                outputs = model.generate(**inputs, **gen_kwargs)
            console.print("[green]✓ Voxtral generation complete.[/green]\n")

        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        is_truncated = False
        try:
            if hasattr(generated_ids, "shape") and len(generated_ids.shape) >= 2:
                is_truncated = int(generated_ids.shape[1]) >= self.max_new_tokens
            elif hasattr(generated_ids, "__getitem__"):
                sub = generated_ids[0]
                if hasattr(sub, "__len__"):
                    is_truncated = len(sub) >= self.max_new_tokens
                elif hasattr(sub, "shape") and len(sub.shape) >= 1:
                    is_truncated = int(sub.shape[0]) >= self.max_new_tokens
            elif hasattr(generated_ids, "__len__"):
                is_truncated = len(generated_ids) >= self.max_new_tokens
        except (TypeError, IndexError, ValueError):
            is_truncated = False

        if is_truncated:
            trunc_warning = (
                f"Voxtral generation reached max_new_tokens cap ({self.max_new_tokens}); "
                "transcription output may be truncated."
            )
            logger.warning(trunc_warning)
            console.print(f"[bold red]Warning: {trunc_warning}[/bold red]")

        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        text = decoded[0].strip() if decoded else ""

        return [
            RawSegment(id=0, start=0.0, end=0.0, text=text, is_truncated=is_truncated)
        ]


class ParakeetEngine:
    """NVIDIA NeMo Parakeet ASR engine with RNN-T / TDT multilingual support."""

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "auto",
        compute_type: str = "float16",
    ) -> None:
        if model_name in (
            "nvidia/parakeet-1.1b-rnnt-multilingual-asr",
            "parakeet-1.1b-rnnt-multilingual-asr",
        ):
            model_name = "nvidia/parakeet-tdt-0.6b-v3"
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            ensure_cuda_libs()
            try:
                import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found,import-untyped]
                import torch
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    "NVIDIA NeMo is required for the Parakeet engine. "
                    "Install it with: uv sync --extra parakeet (or: uv tool install --editable '.[parakeet]' --reinstall)"
                ) from None

            with console.status(
                f"[bold cyan]Loading NeMo Parakeet model '{self.model_name}'...[/bold cyan]",
                spinner="dots",
            ):
                try:
                    model_cls = getattr(
                        nemo_asr.models, "EncDecRNNTBPEModel", nemo_asr.models.ASRModel
                    )
                    self._model = model_cls.from_pretrained(model_name=self.model_name)
                except (AttributeError, RuntimeError, ValueError):
                    self._model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=self.model_name
                    )

                if self.device in ("cuda", "auto") and torch.cuda.is_available():
                    self._model = self._model.cuda()
                    if self.compute_type in ("float16", "fp16") and hasattr(
                        self._model, "half"
                    ):
                        self._model = self._model.half()
                else:
                    self._model = self._model.cpu()

                self._model.eval()

            console.print(f"[green]✓ Parakeet '{self.model_name}' loaded.[/green]")
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
        duration: float | None = None,
    ) -> list[RawSegment]:
        """Transcribe audio using NeMo Parakeet model with word timestamps and safe audio chunking."""
        import numpy as np
        import soundfile as sf

        model = self._load_model()

        resolved_audio = audio_path.resolve()
        try:
            info = sf.info(str(resolved_audio))
            total_duration = float(info.duration)
            sr = info.samplerate
            total_samples = info.frames
        except (RuntimeError, OSError, ValueError):
            total_duration = duration or 0.0
            sr = 16000
            total_samples = 0

        # Chunk audio into 180s (3 min) slices to prevent CUDA STFT allocation overflow on long files
        chunk_seconds = 180.0
        all_words: list[WordTimestamp] = []
        text_parts: list[str] = []

        with console.status(
            "[bold blue]Transcribing (Parakeet TDT)...[/bold blue]",
            spinner="dots",
        ):
            chunks_data: list[tuple[Any, float]] = []
            if total_samples == 0:
                try:
                    outputs = model.transcribe(
                        [str(resolved_audio)],
                        timestamps=True,
                        return_hypotheses=True,
                    )
                except TypeError:
                    outputs = model.transcribe([str(resolved_audio)])
                chunks_data.append((outputs, 0.0))
            else:
                chunk_samples = int(chunk_seconds * sr)
                with sf.SoundFile(str(resolved_audio)) as f:
                    while True:
                        offset_samples = f.tell()
                        if offset_samples >= total_samples:
                            break
                        offset_sec = float(offset_samples) / sr
                        data = f.read(frames=chunk_samples, dtype="float32")
                        if len(data) == 0:
                            break
                        if data.ndim > 1:
                            data = np.mean(data, axis=1)

                        try:
                            outputs = model.transcribe(
                                [data],
                                timestamps=True,
                                return_hypotheses=True,
                            )
                        except TypeError:
                            outputs = model.transcribe([data])

                        chunks_data.append((outputs, offset_sec))

            for outputs, offset_sec in chunks_data:
                if not outputs:
                    continue
                hyp = outputs[0]
                if hasattr(hyp, "text"):
                    chunk_text = str(hyp.text).strip()
                elif isinstance(hyp, dict):
                    chunk_text = str(hyp.get("text", "")).strip()
                else:
                    chunk_text = str(hyp).strip()

                if chunk_text:
                    text_parts.append(chunk_text)

                timestamps_dict = getattr(hyp, "timestamp", None)
                if isinstance(hyp, dict) and timestamps_dict is None:
                    timestamps_dict = hyp.get("timestamp")

                if timestamps_dict and isinstance(timestamps_dict, dict):
                    word_stamps = timestamps_dict.get("word") or []
                    for w in word_stamps:
                        if isinstance(w, dict):
                            w_text = w.get("word", "")
                            w_start = offset_sec + float(w.get("start", w.get("start_offset", 0.0)))
                            w_end = offset_sec + float(w.get("end", w.get("end_offset", 0.0)))
                        else:
                            w_text = getattr(w, "word", "")
                            w_start = offset_sec + float(getattr(w, "start", getattr(w, "start_offset", 0.0)))
                            w_end = offset_sec + float(getattr(w, "end", getattr(w, "end_offset", 0.0)))
                        all_words.append(
                            WordTimestamp(
                                word=str(w_text).strip(),
                                start=w_start,
                                end=w_end,
                                probability=1.0,
                            )
                        )

        full_text = " ".join(text_parts)
        start_time = all_words[0].start if all_words else 0.0
        end_time = all_words[-1].end if all_words else (total_duration or duration or 0.0)

        return [
            RawSegment(
                id=0,
                start=start_time,
                end=end_time,
                text=full_text,
                words=all_words,
            )
        ]


def get_engine(engine_type: str, **kwargs: Any) -> TranscriberEngine:
    """Factory creating configured TranscriberEngine instance."""
    normalized = engine_type.lower().strip()
    if normalized == "whisper":
        return WhisperEngine(**kwargs)
    elif normalized == "voxtral":
        return VoxtralEngine(**kwargs)
    elif normalized == "parakeet":
        return ParakeetEngine(**kwargs)
    raise ValueError(
        f"Unknown engine: {engine_type}. Expected 'parakeet', 'voxtral' or 'whisper'."
    )


def create_transcriber(
    engine: str = "whisper",
    model_name: str | None = None,
    device: str = "auto",
    compute_type: str = "float16",
    **kwargs: Any,
) -> TranscriberEngine:
    """Factory creating configured TranscriberEngine instance with device and compute type forwarding."""
    engine_kwargs: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        **kwargs,
    }
    if model_name is not None:
        engine_kwargs["model_name"] = model_name

    cli_mod = sys.modules.get("a2ts.cli")
    if cli_mod is not None and getattr(cli_mod, "get_engine", None) is not get_engine:
        fn: Any = cli_mod.get_engine
        return cast(TranscriberEngine, fn(engine, **engine_kwargs))

    return get_engine(engine, **engine_kwargs)
