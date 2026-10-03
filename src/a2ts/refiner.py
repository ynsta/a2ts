"""Local LLM transcript refiner and phonetic normalizer."""

import difflib
import logging
import re
import shutil
import subprocess
import tempfile

from a2ts.consolidator import normalize_rpg_terms

logger = logging.getLogger(__name__)

TURN_HEADER_PATTERN = re.compile(r"^###\s+\[[\d:.]+\s*-\s*[\d:.]+\].*$", re.MULTILINE)


def normalize_rpg_phonetics(text: str) -> str:
    """Deterministically correct speech-to-text artifacts for French RPG terms."""
    return normalize_rpg_terms(text)


def compute_edit_distance_ratio(original: str, refined: str) -> float:
    """Calculate relative character-level difference ratio (0.0 = identical)."""
    matcher = difflib.SequenceMatcher(None, original, refined)
    return 1.0 - matcher.ratio()


def extract_turn_headers(markdown: str) -> list[str]:
    """Find all turn header lines matching timestamp format."""
    return [m.strip() for m in TURN_HEADER_PATTERN.findall(markdown)]


def strip_markdown_code_fences(text: str) -> str:
    """Strip enclosing markdown code fences (e.g. ```markdown ... ```) if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def strip_callout_syntax(text: str) -> str:
    """Strip markdown callout header lines and quote prefixes for dialogue comparison."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("> [!NOTE]", ">[!NOTE]")):
            continue
        if stripped.startswith(">"):
            stripped = stripped.lstrip("> ").strip()
        lines.append(stripped)
    return "\n".join(lines)


def chunk_transcript_markdown(markdown: str, max_turns: int = 25) -> list[str]:
    """Split markdown into chunks containing at most max_turns turns each, keeping frontmatter intact."""
    if not markdown:
        return []
    max_turns = max(1, max_turns)
    matches = list(TURN_HEADER_PATTERN.finditer(markdown))
    if not matches:
        return [markdown]
    if len(matches) <= max_turns:
        return [markdown]

    chunks: list[str] = []
    for i in range(0, len(matches), max_turns):
        start_pos = 0 if i == 0 else matches[i].start()
        next_i = i + max_turns
        end_pos = matches[next_i].start() if next_i < len(matches) else len(markdown)
        chunks.append(markdown[start_pos:end_pos])
    return chunks


def validate_refiner_chunk(original: str, refined: str) -> bool:
    """Validate refined chunk retains all turn headers and stays within 15% word count delta."""
    clean_refined = strip_markdown_code_fences(refined)
    if extract_turn_headers(original) != extract_turn_headers(clean_refined):
        return False
    orig_words = len(original.split())
    # Strip callout syntax so added callout headers don't falsely exceed the 15% tolerance
    content_refined = strip_callout_syntax(clean_refined)
    ref_words = len(content_refined.split())
    if orig_words == 0:
        return ref_words == 0
    allowed_delta = max(5, int(orig_words * 0.15))
    return abs(ref_words - orig_words) <= allowed_delta


def refine_transcript_markdown(
    raw_markdown: str,
    agy_model: str = "gemini-3.8-flash-low",
    chunk_turns: int = 25,
    effort: str = "low",
) -> str:
    """Refine transcript via local agy CLI with deterministic normalization fallback."""
    normalized = normalize_rpg_phonetics(raw_markdown)

    if not shutil.which("agy"):
        return normalized

    chunks = chunk_transcript_markdown(normalized, max_turns=chunk_turns)
    if not chunks:
        return normalized

    processed_chunks: list[str] = []
    for chunk in chunks:
        clean_chunk = chunk.strip()
        if not clean_chunk:
            continue

        prompt = (
            "Tu es un assistant d'édition pour des retranscriptions de sessions de jeu de rôle.\n"
            "Nettoie et formate ce texte en respectant scrupuleusement les règles suivantes:\n"
            "1. Ne modifie pas les timestamps ### [HH:MM:SS - HH:MM:SS] Nom.\n"
            "2. Encadre les discussions hors-jeu / techniques (jets de dés, règles, apartés) dans des blocs callout:\n"
            "> [!NOTE] Hors-jeu / Discussion\n"
            "3. Ne retire aucune information ni dialogue.\n"
            "4. Réponds UNIQUEMENT avec la transcription formatée, sans bloc de code markdown englobant (pas de ```markdown), sans salutation ni texte d'introduction ou de conclusion.\n\n"
            f"Voici la transcription brute:\n\n{clean_chunk}"
        )

        timeout = max(60, min(180, 45 + len(clean_chunk.split()) // 10))
        cmd = [
            "agy",
            "--model",
            agy_model,
            "--effort",
            effort,
            "--disable-slash-commands",
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
                cwd=tempfile.gettempdir(),
                timeout=timeout,
            )
            refined_chunk = strip_markdown_code_fences(proc.stdout.strip())
            if validate_refiner_chunk(clean_chunk, refined_chunk):
                processed_chunks.append(refined_chunk)
            else:
                logger.warning(
                    "Refined chunk validation failed (headers or length mismatch), falling back to raw chunk."
                )
                processed_chunks.append(clean_chunk)
        except (subprocess.SubprocessError, OSError) as err:
            logger.warning(
                "agy refinement failed for chunk, falling back to raw chunk: %s",
                err,
            )
            processed_chunks.append(clean_chunk)

    if not processed_chunks:
        return normalized
    return "\n\n".join(processed_chunks)
