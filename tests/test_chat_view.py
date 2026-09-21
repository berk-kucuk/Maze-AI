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


# ── progressive code blocks while still streaming ───────────────────────────
# Previously every code answer rendered as plain text with literal backticks
# for the whole reply, then snapped into a monospaced CodeBlock the instant it
# finished — a visible "the message just changed font" glitch right on landing.
def test_a_closed_fence_becomes_a_real_code_block_mid_stream(app):
    bubble = _Bubble("", user=False)
    bubble.set_streaming_text("Here:\n```sh\nls -la\n```\n")
    assert any(isinstance(w, CodeBlock) for w in bubble._extra)
    block = next(w for w in bubble._extra if isinstance(w, CodeBlock))
    assert block.code == "ls -la"


def test_an_open_fence_stays_plain_until_it_closes(app):
    bubble = _Bubble("", user=False)
    bubble.set_streaming_text("Here:\n```sh\nls -l")   # no closing fence yet
    assert bubble._extra == []
    assert bubble.label.textFormat() == Qt.TextFormat.PlainText


def test_prose_after_a_closed_block_keeps_streaming_as_plain_text(app):
    bubble = _Bubble("", user=False)
    bubble.set_streaming_text("```sh\nls\n```\nAnd then some ")
    tail = bubble._tail_label
    assert tail is not None
    assert tail.textFormat() == Qt.TextFormat.PlainText
    # The trailing space is real, live content — the model is still typing,
    # trimming it would flicker the text as the next word lands.
    assert tail.text() == "And then some "

    bubble.set_streaming_text("```sh\nls\n```\nAnd then some more text")
    assert bubble._tail_label is tail, "the tail label should be updated, not rebuilt"
    assert "more text" in tail.text()


def test_the_code_block_is_not_rebuilt_once_it_has_closed(app):
    # Rebuilding on every frame for the rest of the answer would recreate the
    # same CodeBlock dozens of times as trailing prose streams in.
    bubble = _Bubble("", user=False)
    bubble.set_streaming_text("```sh\nls\n```\nmore")
    block = next(w for w in bubble._extra if isinstance(w, CodeBlock))
    bubble.set_streaming_text("```sh\nls\n```\nmore text still arriving")
    assert next(w for w in bubble._extra if isinstance(w, CodeBlock)) is block


def test_a_second_fence_closing_adds_its_own_block(app):
    bubble = _Bubble("", user=False)
    bubble.set_streaming_text("```sh\na\n```\ntext\n```py\nb")
    assert [w.code for w in bubble._extra if isinstance(w, CodeBlock)] == ["a"]
    bubble.set_streaming_text("```sh\na\n```\ntext\n```py\nb\n```")
    assert [w.code for w in bubble._extra if isinstance(w, CodeBlock)] == ["a", "b"]


def test_finalizing_after_progressive_code_blocks_still_matches_set_text(app):
    # The streaming path and the final render must agree, or the message
    # visibly changes shape one more time right as it lands.
    text = "```sh\nls\n```\nAnd a closing line."
    bubble = _Bubble("", user=False)
    bubble.set_streaming_text(text)
    streamed_blocks = [w.code for w in bubble._extra if isinstance(w, CodeBlock)]
    bubble.set_text(text)
    final_blocks = [w.code for w in bubble._extra if isinstance(w, CodeBlock)]
    assert streamed_blocks == final_blocks


def test_many_rapid_rebuilds_do_not_crash(app):
    # Regression guard: _clear_extra() uses deleteLater(), and rebuilding
    # widgets many times in a tight loop is exactly the pattern that has
    # crashed elsewhere in this codebase (Quick Ask) when a widget's Python
    # wrapper was collected before its deferred delete ran. This drives the
    # same rebuild path dozens of times with the event loop pumped between
    # each, the way real streaming frames actually arrive.
    from PySide6.QtCore import QEventLoop, QTimer

    bubble = _Bubble("", user=False)
    text = ""
    for i in range(40):
        text += f"```sh\ncmd{i}\n```\nsome trailing prose after block {i}\n"
        bubble.set_streaming_text(text)
        loop = QEventLoop()
        QTimer.singleShot(0, loop.quit)
        loop.exec()
    blocks = [w.code for w in bubble._extra if isinstance(w, CodeBlock)]
    assert blocks == [f"cmd{i}" for i in range(40)]


# ── segment ordering ────────────────────────────────────────────────────────
def body_order(bubble):
    """The visible widgets top to bottom, as the user actually sees them."""
    out = []
    for i in range(bubble._body.count()):
        widget = bubble._body.itemAt(i).widget()
        if widget is None or not widget.isVisibleTo(bubble):
            continue
        if isinstance(widget, CodeBlock):
            out.append(("code", widget.code))
        elif widget.text():
            out.append(("text", widget.text()))
    return out


def test_prose_after_a_code_block_renders_below_it(app):
    # The permanent label sits at the top of the layout, so reusing it for a
    # prose chunk that came *after* the code hoisted the explanation above the
    # command it was describing — and answers that open with a code block are
    # the common shape.
    bubble = _Bubble("```sh\nls\n```\nThis lists files.", user=False)
    assert body_order(bubble) == [("code", "ls"), ("text", "This lists files.")]


def test_prose_before_a_code_block_still_renders_above_it(app):
    bubble = _Bubble("Here it is:\n```sh\nls\n```", user=False)
    assert body_order(bubble) == [("text", "Here it is:"), ("code", "ls")]


def test_prose_around_a_code_block_keeps_its_order(app):
    bubble = _Bubble("First:\n```sh\nls\n```\nThen this.", user=False)
    assert body_order(bubble) == [
        ("text", "First:"), ("code", "ls"), ("text", "Then this."),
    ]


@pytest.mark.parametrize("text", [
    "```sh\nls\n```\nThis lists files.",
    "Here it is:\n```sh\nls\n```",
    "First:\n```sh\nls\n```\nThen this.",
    "```sh\na\n```\nbetween\n```py\nb\n```",
    "```sh\nls\n```",
    "no code at all",
])
def test_streaming_and_final_agree_on_layout(app, text):
    # If these disagree the message visibly rearranges itself the instant it
    # lands, which is the whole class of glitch this file guards against.
    streaming = _Bubble("", user=False)
    streaming.set_streaming_text(text)
    final = _Bubble("", user=False)
    final.set_text(text)

    kinds_streaming = [kind for kind, _ in body_order(streaming)]
    kinds_final = [kind for kind, _ in body_order(final)]
    assert kinds_streaming == kinds_final
