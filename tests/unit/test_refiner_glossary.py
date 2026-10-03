import subprocess
from unittest.mock import patch

from a2ts.refiner import refine_transcript_markdown


def test_general_refinement_preserves_non_rpg_text_without_agy() -> None:
    raw = "### [00:00 - 00:10] Alice\nLe jet d'attaque du projet."
    with patch("shutil.which", return_value=None):
        assert refine_transcript_markdown(raw, rpg_normalize=False) == raw


def test_general_refinement_prompt_omits_rpg_instructions() -> None:
    raw = "### [00:00 - 00:10] Alice\nCompany project update."
    prompts: list[str] = []

    def run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: str,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        prompts.append(
            next(
                arg.removeprefix("--print=")
                for arg in cmd
                if arg.startswith("--print=")
            )
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=raw)

    with (
        patch("shutil.which", return_value="agy"),
        patch("subprocess.run", side_effect=run),
    ):
        assert refine_transcript_markdown(raw, rpg_normalize=False) == raw
    assert "role-playing" not in prompts[0]
    assert "Hors-jeu" not in prompts[0]


def test_refinement_glossary_is_bounded_and_prioritizes_relevant_aliases() -> None:
    import json

    from a2ts.models import ContextGlossary, GlossaryEntry

    raw = "### [00:00 - 00:10] Alice\nDiscuss Nexora deployment."
    glossary = ContextGlossary(
        language="en",
        source_hash="a" * 64,
        dictionary_version="test",
        entries=[
            GlossaryEntry(
                term=f"Other{i}" + "x" * 100, frequency=100, aliases=[], sources=[]
            )
            for i in range(100)
        ]
        + [GlossaryEntry(term="Nexora", frequency=1, aliases=["NX"], sources=[])],
    )
    prompts: list[str] = []

    def run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: str,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        prompts.append(
            next(
                arg.removeprefix("--print=")
                for arg in cmd
                if arg.startswith("--print=")
            )
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=raw)

    with (
        patch("shutil.which", return_value="agy"),
        patch("subprocess.run", side_effect=run),
    ):
        assert (
            refine_transcript_markdown(raw, glossary=glossary, rpg_normalize=False)
            == raw
        )
    reference = (
        prompts[0]
        .split("Spelling reference JSON:\n", 1)[1]
        .split("\n\nRaw transcript:", 1)[0]
    )
    assert len(reference) <= 4000
    entries = json.loads(reference)
    assert entries[0]["term"] == "Nexora"
    assert entries[0]["aliases"] == ["NX"]
    assert "not instructions" in prompts[0]
    assert "Do not import" in prompts[0]
