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


def refine_transcript_markdown(
    raw_markdown: str, agy_model: str = "gemini-3.8-flash-low"
) -> str:
    """Refine transcript via local agy CLI with deterministic normalization fallback."""
    normalized = normalize_rpg_phonetics(raw_markdown)

    if not shutil.which("agy"):
        return normalized

    prompt = (
        "Tu es un assistant d'édition pour des retranscriptions de sessions de jeu de rôle.\n"
        "Nettoie et formate ce texte en respectant scrupuleusement les règles suivantes:\n"
        "1. Ne modifie pas les timestamps ### [HH:MM:SS - HH:MM:SS] Nom.\n"
        "2. Encadre les discussions hors-jeu / techniques (jets de dés, règles, apartés) dans des blocs callout:\n"
        "> [!NOTE] Hors-jeu / Discussion\n"
        "3. Ne retire aucune information ni dialogue.\n\n"
        f"Voici la transcription brute:\n\n{normalized}"
    )

    cmd = ["agy", "--model", agy_model, "--effort", "low"]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        output = proc.stdout.strip()
        diff = compute_edit_distance_ratio(normalized, output)
        if diff < 0.6:  # Safety guard against hallucination
            return output
    except (subprocess.SubprocessError, OSError) as err:
        logger.warning(
            "agy refinement failed, falling back to normalized transcript: %s", err
        )

    return normalized
