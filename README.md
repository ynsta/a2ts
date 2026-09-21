# a2ts (Audio to Text / Transcript)

Contextualized Single-Stream Audio & Video Transcriber with **Voxtral** (Mistral AI) and **Whisper**.

`a2ts` processes single audio or video streams (local recordings, session captures, video clips), leverages Obsidian / Markdown lore for contextual vocabulary biasing, and transcribes speech locally using either Mistral's **Voxtral** family (e.g. `Voxtral-Mini-3B`) or **Faster-Whisper** (`large-v3` / `turbo`) on local GPU hardware (NVIDIA RTX CUDA).

---

## Key Differences from `craig2ts`

| Feature | `craig2ts` | `a2ts` |
| :--- | :--- | :--- |
| **Input Source** | Multi-track `.flac` from Craig Discord bot (1 track per speaker) | Single audio or video stream (`.mp4`, `.mkv`, `.flac`, `.wav`, `.mp3`) |
| **Speaker Attribution** | Trivial (1 track = 1 Discord speaker) | Single mixed audio track (requires Diarization or LLM attribution pass) |
| **Engines** | `faster-whisper` | **Voxtral** (Mistral multimodal audio LLM) & `faster-whisper` |
| **Future Vision** | Separate Discord session pipeline | Potential fusion into a unified transcription suite |

---

## Features

- **Audio Extraction**: Automatic audio extraction from video container formats (`ffmpeg`).
- **Context Lore Extraction**: Reuses/shares vocabulary extraction patterns from Obsidian notes (`[[wikilinks]]`, aliases, entities) to bias transcription towards domain terms.
- **Voxtral & Whisper Inference**: Run local GPU transcription without cloud API costs or privacy leaks.
- **Local LLM Refiner**: Optional refinement pass using local `agy` CLI for cleanup, formatting, and speaker annotation.

---

## Quick Start

### Installation

Ensure [uv](https://docs.astral.sh/uv/) is installed:

```bash
uv sync
```

To include Voxtral dependencies (PyTorch + Hugging Face Transformers):

```bash
uv sync --extra voxtral
```

### CLI Usage

```bash
uv run a2ts info
```
