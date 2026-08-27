"""Tests for chat rendering: code-block splitting and streaming behaviour.

These need Qt, which is present wherever the app itself runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maze_ai.ui.chat_view import (  # noqa: E402
    ChatView,
    CodeBlock,
    _Bubble,
    split_code_blocks,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# ── splitting ──────────────────────────────────────────────────────────────
def test_split_separates_prose_and_code():
    segments = split_code_blocks("Run this:\n\n```bash\nls -la\n```\n\nThen look.")
    assert segments == [
        ("text", "Run this:", ""),
        ("code", "ls -la", "bash"),
        ("text", "Then look.", ""),
    ]


def test_split_plain_text_has_no_code():
    assert split_code_blocks("just prose") == [("text", "just prose", "")]


def test_unterminated_fence_still_yields_code():
    # Happens whenever a fence is the last thing in an answer.
    segments = split_code_blocks("here:\n```py\nprint(1)")
    assert segments[-1] == ("code", "print(1)", "py")


def test_multiple_blocks():
    segments = split_code_blocks("```sh\na\n```\ntext\n```sh\nb\n```")
    assert [kind for kind, _, _ in segments] == ["code", "text", "code"]


# ── bubbles ────────────────────────────────────────────────────────────────
def test_bubble_builds_a_code_widget(app):
    bubble = _Bubble("intro\n```sh\necho hi\n```\nouttro", user=False, actions=True)
    kinds = [type(w).__name__ for w in bubble._extra]
    assert "CodeBlock" in kinds
    block = next(w for w in bubble._extra if isinstance(w, CodeBlock))
    assert block.code == "echo hi"


def test_bubble_without_code_uses_one_label(app):
    bubble = _Bubble("**bold** text", user=False)
    assert bubble._extra == []
    assert bubble.label.textFormat() == Qt.TextFormat.MarkdownText


def test_streaming_uses_plain_text_then_renders_markdown(app):
    bubble = _Bubble("", user=False)
    bubble.set_streaming_text("partial ```sh")
    assert bubble.label.textFormat() == Qt.TextFormat.PlainText
    assert bubble._extra == []          # no half-parsed code widget mid-stream
    bubble.set_text("done\n```sh\nls\n```")
    assert any(isinstance(w, CodeBlock) for w in bubble._extra)


def test_switching_back_to_plain_text_clears_code_widgets(app):
    bubble = _Bubble("```sh\nls\n```", user=False)
    assert bubble._extra
    bubble.set_text("no code any more")
    assert bubble._extra == []


# ── the view ───────────────────────────────────────────────────────────────
def test_stream_then_finalize(app):
    view = ChatView()
    view.append_stream("hel")
    view.append_stream("lo")
    # Text is revealed over the next few frames, so the queue holds the rest.
    assert view._stream_text + view._stream_pending == "hello"
    assert view.finalize_stream("hello **there**") is True
    assert view._stream_bubble is None
    assert not view._stream_timer.isActive()


def test_finalize_without_a_stream_is_a_no_op(app):
    view = ChatView()
    assert view.finalize_stream("x") is False


def test_clear_stops_streaming(app):
    view = ChatView()
    view.append_stream("chunk")
    view.clear()
    assert view._stream_bubble is None
    assert not view._stream_timer.isActive()


# ── streaming feel ─────────────────────────────────────────────────────────
def pump(ms: int):
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_a_burst_is_revealed_gradually(app):
    view = ChatView()
    view.append_stream("x" * 400)
    # The first frame shows something, but nowhere near everything.
    assert 0 < len(view._stream_text) < 400
    assert view._stream_pending


def test_the_queue_drains_over_time(app):
    view = ChatView()
    view.append_stream("y" * 200)
    first = len(view._stream_text)
    pump(120)
    assert len(view._stream_text) > first
    pump(400)
    assert view._stream_pending == "", "the backlog should catch up quickly"


def test_a_slow_trickle_still_appears(app):
    view = ChatView()
    for chunk in ("Mer", "haba", " dünya"):
        view.append_stream(chunk)
        pump(40)
    assert view._stream_text + view._stream_pending == "Merhaba dünya"


def test_finalize_shows_everything_immediately(app):
    view = ChatView()
    view.append_stream("z" * 500)
    assert view.finalize_stream("") is True
    assert view._stream_pending == ""
    assert not view._stream_timer.isActive()


def test_the_timer_parks_itself_when_nothing_arrives(app):
    view = ChatView()
    view.append_stream("hi")
    pump(60)
    assert view._stream_timer.isActive()
    pump(900)          # nothing new for a while
    assert not view._stream_timer.isActive()


def test_streaming_resumes_after_a_pause(app):
    view = ChatView()
    view.append_stream("first")
    pump(900)
    view.append_stream(" second")
    pump(200)
    assert "second" in view._stream_text


def test_the_view_follows_the_text_but_not_the_reader(app):
    view = ChatView()
    view.resize(400, 300)
    for _ in range(40):
        view.add_ai("a paragraph of text " * 10)
    bar = view.verticalScrollBar()
    bar.setValue(0)                      # the user scrolled up to read
    view.append_stream("new answer")
    view._glide_to_bottom()
    assert bar.value() == 0, "streaming must not yank the view away from the reader"
