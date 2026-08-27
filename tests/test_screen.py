"""Tests for reading the screen — text first, so every model can do it."""

from __future__ import annotations

import pytest

from maze_ai.agent import tools
from maze_ai.agent.agent import Agent
from maze_ai.agent.tools import ToolResult
from maze_ai.config import MODE_AUTO
from maze_ai.llm.base import LLMBackend, LLMReply, ToolCall


@pytest.fixture
def fake_capture(monkeypatch, tmp_path):
    """A capture that returns a fixed file instead of grabbing the real screen."""
    shot = tmp_path / "screen.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(tools, "capture_region",
                        lambda save_path="": ToolResult(True, str(shot)))
    monkeypatch.setattr(
        tools, "screenshot",
        lambda save_path="", region=False, **kw: ToolResult(
            True, f"Saved a screenshot to {shot}."),
    )
    monkeypatch.setattr(tools, "_active_window_title", lambda: "Konsole — bash")
    return shot


def test_screen_text_is_returned_for_any_model(fake_capture, monkeypatch):
    monkeypatch.setattr(tools, "ocr_image",
                        lambda **kw: ToolResult(True, "error: unit not found"))
    result = tools.read_screen()
    assert result.ok
    assert "error: unit not found" in result.output
    assert "Konsole" in result.output          # the window title is context too
    assert result.images == [str(fake_capture)]


def test_region_capture_is_used_when_asked(fake_capture, monkeypatch):
    monkeypatch.setattr(tools, "ocr_image", lambda **kw: ToolResult(True, "hello"))
    assert str(fake_capture) in tools.read_screen(region=True).output


def test_a_failed_capture_is_reported(monkeypatch):
    monkeypatch.setattr(tools, "screenshot",
                        lambda save_path="", **kw: ToolResult(
                            False, "No screenshot tool found."))
    result = tools.read_screen()
    assert not result.ok and "No screenshot tool" in result.output


def test_missing_ocr_still_offers_the_picture(fake_capture, monkeypatch):
    monkeypatch.setattr(
        tools, "ocr_image",
        lambda **kw: ToolResult(False, "Tesseract has no language data installed."),
    )
    result = tools.read_screen()
    assert result.ok                       # a vision model can still work
    assert "language data" in result.output
    assert "describe the attached capture" in result.output
    assert result.images


def test_blank_screen_says_so(fake_capture, monkeypatch):
    monkeypatch.setattr(tools, "ocr_image",
                        lambda **kw: ToolResult(True, "(no text detected in the image)"))
    assert "No text could be read" in tools.read_screen().output


# ── OCR failure messages point at the fix ──────────────────────────────────
def test_missing_language_data_is_explained(monkeypatch, tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG")
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(tools, "_tesseract_langs", lambda: {"osd"})
    result = tools.ocr_image(path=str(image))
    assert not result.ok
    assert "tesseract-data-eng" in result.output


def test_any_installed_language_is_better_than_none(monkeypatch, tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG")
    seen = {}

    class Proc:
        returncode = 0
        stdout = "some text"
        stderr = ""

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return Proc()

    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(tools, "_tesseract_langs", lambda: {"osd", "deu"})
    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    assert tools.ocr_image(path=str(image)).ok
    assert "deu" in seen["argv"]


# ── the capture reaches a model that can see ───────────────────────────────
class VisionBackend(LLMBackend):
    name = "vision-fake"
    supports_native_tools = True
    supports_vision = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages):
        return ""

    def chat_ex(self, messages, *, tools=None, schema=None, stream=False,
                on_text=None, on_thinking=None):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else LLMReply(text="done")

    def available_models(self):
        return ["vision-fake"]


def test_the_capture_is_handed_to_a_vision_model(fake_capture, monkeypatch):
    monkeypatch.setattr(tools, "ocr_image", lambda **kw: ToolResult(True, "hello"))
    backend = VisionBackend([
        LLMReply(tool_calls=[ToolCall("read_screen", {})]),
        LLMReply(text="I can see it."),
    ])
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    agent.run("what's on my screen", lambda ev: None, lambda req: True)

    images = [m for m in backend.calls[-1] if m.get("images")]
    assert images, "the screenshot should be attached for a model with eyes"
    assert images[0]["images"] == [str(fake_capture)]


def test_no_image_is_sent_to_a_text_only_model(fake_capture, monkeypatch):
    monkeypatch.setattr(tools, "ocr_image", lambda **kw: ToolResult(True, "hello"))
    backend = VisionBackend([
        LLMReply(tool_calls=[ToolCall("read_screen", {})]),
        LLMReply(text="I read it."),
    ])
    backend.supports_vision = False
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    agent.run("what's on my screen", lambda ev: None, lambda req: True)

    assert not any(m.get("images") for m in backend.calls[-1])
    # …but the OCR text still made it, so the answer is possible anyway.
    assert any("hello" in (m.get("content") or "") for m in backend.calls[-1])


# ── one selector, and only when asked ──────────────────────────────────────
def test_a_cancelled_selection_does_not_open_another_tool(monkeypatch, tmp_path):
    started: list[str] = []

    def fake_run(argv, out, timeout=120):
        started.append(argv[0])
        return False          # user pressed Escape

    monkeypatch.setattr(tools.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(tools, "_run_capture", fake_run)
    result = tools.capture_region(str(tmp_path / "x.png"))

    assert started == ["spectacle"], "a second crosshair must never appear"
    assert not result.ok
    assert "cancelled" in result.output.lower()
    assert "Do not try again" in result.output


def test_the_available_selector_is_used(monkeypatch, tmp_path):
    started: list[str] = []

    def fake_run(argv, out, timeout=120):
        started.append(argv[0])
        return True

    # No KDE here: maim is the one installed.
    monkeypatch.setattr(tools.shutil, "which",
                        lambda name: "/usr/bin/maim" if name == "maim" else None)
    monkeypatch.setattr(tools, "_run_capture", fake_run)
    assert tools.capture_region(str(tmp_path / "x.png")).ok
    assert started == ["maim"]


def test_no_selector_at_all_says_what_to_install(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    result = tools.capture_region(str(tmp_path / "x.png"))
    assert not result.ok and "spectacle" in result.output


def test_read_screen_defaults_to_the_whole_screen(fake_capture, monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "ocr_image", lambda **kw: ToolResult(True, "text"))
    monkeypatch.setattr(tools, "capture_region",
                        lambda save_path="": calls.append("region") or ToolResult(True, "x"))
    tools.read_screen()
    assert calls == [], "asking what is on screen must not interrupt the user"


def test_the_tool_description_discourages_region(monkeypatch):
    spec = tools.TOOLS["read_screen"]
    assert "whole screen" in spec.description
    assert "explicitly asked" in spec.description
    assert "ONLY" in spec.args["region"]


# ── reading one application's window ───────────────────────────────────────
def test_a_window_is_focused_then_captured(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(tools.shutil, "which",
                        lambda name: "/usr/bin/kdotool" if name == "kdotool" else None)

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(tools.subprocess, "run",
                        lambda argv, **kw: calls.append(argv[0]) or Proc())
    monkeypatch.setattr(tools.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(tools, "_capture_active_window", lambda out: True)
    monkeypatch.setattr(tools, "ocr_image", lambda **kw: ToolResult(True, "$ git status"))

    result = tools.read_window(name="konsole", save_path=str(tmp_path / "w.png"))
    assert "kdotool" in calls
    assert "git status" in result.output
    assert "Requested window: konsole" in result.output


def test_without_a_window_tool_it_falls_back_and_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    monkeypatch.setattr(tools, "screenshot",
                        lambda save_path="", **kw: ToolResult(True, f"Saved to {save_path}."))
    monkeypatch.setattr(tools, "ocr_image", lambda **kw: ToolResult(True, "some text"))

    result = tools.read_window(name="firefox", save_path=str(tmp_path / "w.png"))
    assert result.ok
    assert "whole screen instead" in result.output
    assert "kdotool" in result.output      # tells the user how to fix it


def test_a_missing_window_is_reported(monkeypatch, tmp_path):
    class Proc:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(tools.shutil, "which",
                        lambda name: "/usr/bin/kdotool" if name == "kdotool" else None)
    monkeypatch.setattr(tools.subprocess, "run", lambda argv, **kw: Proc())
    monkeypatch.setattr(tools, "screenshot",
                        lambda save_path="", **kw: ToolResult(True, "Saved."))
    monkeypatch.setattr(tools, "ocr_image", lambda **kw: ToolResult(True, "text"))
    result = tools.read_window(name="ghost", save_path=str(tmp_path / "w.png"))
    assert "No window matching 'ghost'" in result.output


# ── shell history ──────────────────────────────────────────────────────────
def test_zsh_extended_history_is_parsed():
    raw = ": 1712345678:0;git status\n: 1712345679:0;ls -la ~\n"
    assert tools._parse_history(raw, "zsh") == ["git status", "ls -la ~"]


def test_fish_history_is_parsed():
    raw = "- cmd: git push\n  when: 1712345678\n- cmd: make\n  when: 1712345679\n"
    assert tools._parse_history(raw, "fish") == ["git push", "make"]


def test_plain_bash_history_is_parsed():
    assert tools._parse_history("cd /tmp\nls\n", "bash") == ["cd /tmp", "ls"]


def test_recent_commands_returns_the_tail(monkeypatch, tmp_path):
    history = tmp_path / ".zsh_history"
    history.write_text("".join(f": 17123456{i:02d}:0;command {i}\n" for i in range(20)))
    monkeypatch.setattr(tools, "_HISTORY_FILES", ((str(history), "zsh"),))
    result = tools.recent_commands(count=3)
    assert result.ok
    assert "command 19" in result.output
    assert "command 5" not in result.output


def test_recent_commands_without_a_history_file(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_HISTORY_FILES", ((str(tmp_path / "nope"), "zsh"),))
    assert not tools.recent_commands().ok
