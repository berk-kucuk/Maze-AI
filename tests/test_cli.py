"""Tests for the terminal entry points: fix, ask, shell integration."""

from __future__ import annotations

import pytest

from maze_ai import cli
from maze_ai.app import _flag_value, _request_from


# ── argument shapes the shell integration produces ─────────────────────────
def test_flag_value_reads_the_text_after_a_flag():
    assert _flag_value(["--ask", "how", "are", "you"], "--ask") == "how are you"


def test_flag_value_skips_standalone_options():
    # `maze-ai --fix --quiet "npm instal"` is exactly what mzfix runs.
    assert _flag_value(["--fix", "--quiet", "npm instal"], "--fix") == "npm instal"


def test_flag_value_stops_at_the_next_real_option():
    assert _flag_value(["--ask", "hello", "--shell-init", "zsh"], "--ask") == "hello"


def test_flag_value_missing_flag():
    assert _flag_value(["--debug"], "--ask") == ""


@pytest.mark.parametrize("argv,expected", [
    ([], "show"),
    (["--ask"], "ask"),
    (["--ask", "what is a symlink"], "ask:what is a symlink"),
    (["--clipboard"], "clipboard"),
    (["--screenshot"], "screenshot"),
])
def test_request_from_argv(argv, expected):
    assert _request_from(argv) == expected


# ── pulling a runnable command out of a model reply ────────────────────────
@pytest.mark.parametrize("lines,expected", [
    (["sudo pacman -Syu --noconfirm", "typo in the flag."], "sudo pacman -Syu --noconfirm"),
    (["```bash", "npm install", "```"], "npm install"),
    (["$ ls -la ~", "listing your home."], "ls -la ~"),
    (["grep -r \"TODO\" --include=\"*.py\""], "grep -r \"TODO\" --include=\"*.py\""),
    # The explanation came first; the command is still found.
    (["The flag was wrong.", "docker compose up -d"], "docker compose up -d"),
])
def test_command_line_extraction(lines, expected):
    assert cli._command_line(lines) == expected


@pytest.mark.parametrize("lines", [
    ["The pattern is interpreted by the shell before grep sees it."],
    ["Bu komut zaten doğru görünüyor."],
    [""],
    [],
])
def test_prose_is_not_mistaken_for_a_command(lines):
    # Quiet mode puts this straight onto the user's prompt — a sentence there
    # would be worse than printing nothing.
    assert cli._command_line(lines) == ""


# ── shell integration ──────────────────────────────────────────────────────
@pytest.mark.parametrize("shell", ["zsh", "bash", "fish"])
def test_shell_init_prints_the_helpers(shell, capsys):
    assert cli.shell_init(shell) == 0
    out = capsys.readouterr().out
    assert "mzfix" in out and "maze-ai --fix" in out


def test_shell_init_rejects_an_unknown_shell(capsys):
    assert cli.shell_init("csh") == 2
    assert "Unknown shell" in capsys.readouterr().err


# ── fix ────────────────────────────────────────────────────────────────────
def test_fix_needs_a_command(capsys):
    assert cli.fix_command("") == 2


def test_fix_prints_command_then_explanation(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_complete",
                        lambda *a, **k: "sudo pacman -S firefox\nThe package name was wrong.")
    monkeypatch.setattr(cli, "_copy_to_clipboard", lambda text: False)
    monkeypatch.setattr(cli, "_stdin_text", lambda **k: "")
    assert cli.fix_command("sudo pacman -S firefx") == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "sudo pacman -S firefox"
    assert "package name" in " ".join(out[1:])


def test_quiet_fix_prints_only_the_command(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_complete",
                        lambda *a, **k: "ls -la ~\nYou wanted the long listing.")
    monkeypatch.setattr(cli, "_stdin_text", lambda **k: "")
    assert cli.fix_command("ls -la ~~", quiet=True) == 0
    assert capsys.readouterr().out.strip() == "ls -la ~"


def test_quiet_fix_stays_silent_on_prose(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_complete",
                        lambda *a, **k: "I could not work out what you meant.")
    monkeypatch.setattr(cli, "_stdin_text", lambda **k: "")
    assert cli.fix_command("???", quiet=True) == 1
    assert capsys.readouterr().out.strip() == ""


def test_fix_passes_piped_output_to_the_model(monkeypatch, capsys):
    seen = {}

    def fake_complete(system, user, config=None):
        seen["user"] = user
        return "tar -xzf archive.tar.gz"

    monkeypatch.setattr(cli, "_complete", fake_complete)
    monkeypatch.setattr(cli, "_copy_to_clipboard", lambda text: False)
    monkeypatch.setattr(cli, "_stdin_text", lambda **k: "tar: Error opening archive")
    cli.fix_command("tar -xf archive.tar.gz")
    assert "Error opening archive" in seen["user"]


def test_fix_reports_a_backend_error(monkeypatch, capsys):
    from maze_ai.llm.base import LLMError

    def boom(*a, **k):
        raise LLMError("Ollama is not running")

    monkeypatch.setattr(cli, "_complete", boom)
    monkeypatch.setattr(cli, "_stdin_text", lambda **k: "")
    assert cli.fix_command("ls") == 1
    assert "Ollama is not running" in capsys.readouterr().err


# ── file-manager menu installation from the command line ───────────────────
def test_install_menus_flag(monkeypatch, capsys):
    import sys

    from maze_ai import app as app_module
    from maze_ai import desktop_integration

    monkeypatch.setattr(desktop_integration, "install", lambda: {"dolphin": ""})
    monkeypatch.setattr(sys, "argv", ["maze-ai", "--install-menus"])
    assert app_module.main() == 0
    assert "dolphin: installed" in capsys.readouterr().out


def test_install_menus_reports_failure(monkeypatch, capsys):
    import sys

    from maze_ai import app as app_module
    from maze_ai import desktop_integration

    monkeypatch.setattr(desktop_integration, "install",
                        lambda: {"dolphin": "read-only file system"})
    monkeypatch.setattr(sys, "argv", ["maze-ai", "--install-menus"])
    assert app_module.main() == 1


def test_remove_menus_flag(monkeypatch, capsys):
    import sys

    from maze_ai import app as app_module
    from maze_ai import desktop_integration

    monkeypatch.setattr(desktop_integration, "uninstall", lambda: ["dolphin"])
    monkeypatch.setattr(sys, "argv", ["maze-ai", "--remove-menus"])
    assert app_module.main() == 0
    assert "Removed: dolphin" in capsys.readouterr().out
