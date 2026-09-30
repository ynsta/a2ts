"""Local LLM transcript refiner and phonetic normalizer."""

import difflib
import logging
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)

PHONETIC_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bun des vingt\b", re.IGNORECASE), "1d20"),
    (re.compile(r"\bun dé vingt\b", re.IGNORECASE), "1d20"),
    (re.compile(r"\bdeux des six\b", re.IGNORECASE), "2d6"),
    (re.compile(r"\bjets? de délai\b", re.IGNORECASE), "jet de dés"),
    (re.compile(r"\bjet d'initiative\b", re.IGNORECASE), "jet d'initiative"),
]

TURN_HEADER_PATTERN = re.compile(
    r"^###\s+\[[\d:.]+\s*-\s*[\d:.]+\].*$", re.MULTILINE
)


def normalize_rpg_phonetics(text: str) -> str:
    """Deterministically correct speech-to-text artifacts for French RPG terms."""
    result = text
    for pattern, replacement in PHONETIC_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


def compute_edit_distance_ratio(original: str, refined: str) -> float:
    """Calculate relative character-level difference ratio (0.0 = identical)."""
    matcher = difflib.SequenceMatcher(None, original, refined)
    return 1.0 - matcher.ratio()


def extract_turn_headers(markdown: str) -> list[str]:
    """Find all turn header lines matching timestamp format."""
    return [m.strip() for m in TURN_HEADER_PATTERN.findall(markdown)]


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
    if extract_turn_headers(original) != extract_turn_headers(refined):
        return False
    orig_words = len(original.split())
    ref_words = len(refined.split())
    if orig_words == 0:
        return ref_words == 0
    return abs(ref_words - orig_words) / orig_words <= 0.15


def refine_transcript_markdown(
    raw_markdown: str,
    agy_model: str = "gemini-3.8-flash-low",
    chunk_turns: int = 25,
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
            "3. Ne retire aucune information ni dialogue.\n\n"
            f"Voici la transcription brute:\n\n{clean_chunk}"
        )

        timeout = max(30, min(120, len(clean_chunk.split()) // 20))
        cmd = ["agy", "--model", agy_model, "--effort", "low"]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
            )
            refined_chunk = proc.stdout.strip()
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
