"""Dual transcription engine architecture for a2ts (Whisper & Voxtral)."""

import ctypes
import logging
import sys
from pathlib import Path
from typing import Any, Protocol

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
    """Preload bundled NVIDIA CUDA libraries if available."""
    for path_entry in sys.path:
        nvidia_dir = Path(path_entry) / "nvidia"
        if nvidia_dir.is_dir():
            for lib_dir in nvidia_dir.glob("*/lib"):
                if lib_dir.is_dir():
                    for so_file in sorted(lib_dir.glob("*.so*")):
                        try:
                            ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                        except (OSError, RuntimeError):
                            continue


class TranscriberEngine(Protocol):
    """Protocol defining the transcription engine interface."""

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
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
        duration = (
            float(raw_duration) if isinstance(raw_duration, (int, float)) else None
        )

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
            task = progress.add_task("transcribe", total=duration)
            for i, s in enumerate(segments_gen):
                if duration and hasattr(s, "end") and isinstance(s.end, (int, float)):
                    progress.update(task, completed=min(s.end, duration))
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
    ) -> None:
        self.model_name = model_name
        self.load_4bit = load_4bit
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

            with console.status(
                f"[bold cyan]Loading Voxtral model '{self.model_name}' into GPU (4-bit NF4)...[/bold cyan]",
                spinner="dots",
            ):
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                quant_config = None
                if self.load_4bit:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                self._model = VoxtralForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    device_map="auto",
                )
            console.print("[green]✓ Voxtral model loaded.[/green]")
        return self._processor, self._model

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
    ) -> list[RawSegment]:
        """Transcribe audio using Voxtral model with real-time feedback."""
        import torch

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

        with console.status(
            "[bold cyan]Encoding audio waveform and prompt...[/bold cyan]",
            spinner="dots",
        ):
            inputs = processor.apply_chat_template(conversation).to("cuda")

        streamer = None
        try:
            from transformers import TextStreamer

            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                streamer = TextStreamer(tok, skip_prompt=True)
        except (ImportError, AttributeError, TypeError, ValueError):
            streamer = None

        gen_kwargs: dict[str, Any] = {"max_new_tokens": 1024}
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

        decoded = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        text = decoded[0].strip() if decoded else ""

        return [RawSegment(id=0, start=0.0, end=0.0, text=text)]


def get_engine(engine_type: str, **kwargs: Any) -> TranscriberEngine:
    """Factory creating configured TranscriberEngine instance."""
    normalized = engine_type.lower().strip()
    if normalized == "whisper":
        return WhisperEngine(**kwargs)
    elif normalized == "voxtral":
        return VoxtralEngine(**kwargs)
    raise ValueError(f"Unknown engine: {engine_type}. Expected 'voxtral' or 'whisper'.")
