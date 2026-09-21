"""Basic tests for a2ts CLI."""

from typer.testing import CliRunner

from a2ts.cli import app

runner = CliRunner()


def test_info_command() -> None:
    """Test that info command executes cleanly."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "ready" in result.output.lower()
