"""Tests for the approval dialog — what the user is actually shown."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QPushButton

from maze_ai.agent import ApprovalRequest
from maze_ai.agent.agent import REASON_COMMAND, REASON_DESTRUCTIVE, REASON_SENSITIVE
from maze_ai.ui.approval import ApprovalDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def buttons(dialog) -> list[str]:
    return [b.text() for b in dialog.findChildren(QPushButton)]


def test_file_write_shows_the_content_not_just_the_path(app, tmp_path):
    target = tmp_path / "conf.txt"
    target.write_text("debug = true\n")
    request = ApprovalRequest("write_file", {"path": str(target), "content": "debug = false\n"})
    ApprovalDialog(request)  # builds without error
    detail = request.detail()
    assert "-debug = true" in detail and "+debug = false" in detail


def test_new_file_is_shown_as_an_addition(tmp_path):
    request = ApprovalRequest("write_file", {"path": str(tmp_path / "new.txt"), "content": "hi"})
    assert "new file" in request.detail()
    assert "+ hi" in request.detail()


def test_delete_explains_what_is_destroyed(tmp_path):
    directory = tmp_path / "stuff"
    directory.mkdir()
    (directory / "a.txt").write_text("x" * 100)
    detail = ApprovalRequest("delete_path", {"path": str(directory)}).detail()
    assert "directory" in detail and "1 file" in detail


def test_reason_is_displayed(app):
    request = ApprovalRequest(
        "run_command", {"command": "cat ~/.ssh/id_rsa"}, REASON_SENSITIVE, "/.ssh",
    )
    dialog = ApprovalDialog(request)
    # A secret-touching command must not offer a permanent bypass — the prompt
    # existing every time is the entire protection.
    assert "Always allow" not in buttons(dialog)


def test_dangerous_command_cannot_be_always_allowed(app):
    request = ApprovalRequest(
        "run_command", {"command": "rm -rf ~/data"}, REASON_DESTRUCTIVE,
    )
    assert "Always allow" not in buttons(ApprovalDialog(request))


def test_ordinary_command_can_be_always_allowed(app):
    request = ApprovalRequest("run_command", {"command": "npm test"}, REASON_COMMAND)
    assert "Always allow" in buttons(ApprovalDialog(request))


def test_describe_is_a_one_liner(app):
    request = ApprovalRequest("edit_file", {"path": "~/a.py", "old": "x", "new": "y"})
    assert request.describe() == "edit → ~/a.py"


# ── editing a command before it runs ───────────────────────────────────────
def test_command_can_be_edited_before_approval(app):
    request = ApprovalRequest("run_command", {"command": "rm -r ~/data"}, REASON_COMMAND)
    dialog = ApprovalDialog(request)
    assert dialog.editable
    dialog.detail.setPlainText("rm -r ~/data/tmp")
    dialog.accept()
    # The agent executes this very dict, so the edit is what runs.
    assert request.args["command"] == "rm -r ~/data/tmp"


def test_unedited_command_is_untouched(app):
    request = ApprovalRequest("run_command", {"command": "ls -la"}, REASON_COMMAND)
    dialog = ApprovalDialog(request)
    dialog.accept()
    assert request.args["command"] == "ls -la"


def test_file_writes_are_not_editable(app, tmp_path):
    request = ApprovalRequest("write_file", {"path": str(tmp_path / "a"), "content": "x"})
    dialog = ApprovalDialog(request)
    assert not dialog.editable
    assert dialog.detail.isReadOnly()
