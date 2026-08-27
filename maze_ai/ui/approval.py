"""Approval dialog shown before the agent runs a side-effecting tool.

The dialog's job is informed consent: it must show what will actually happen,
not just which tool is about to run. A file write is presented as a diff
against what is on disk, a delete says how much it would destroy, and anything
that needs approval for a *safety* reason (a secret, data leaving the machine)
says so in plain words.
"""

from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..agent import UNSKIPPABLE_REASONS, ApprovalRequest
from ..i18n import tr
from .effects import AuroraCard
from .theme import DANGER, LINE, OK, STYLESHEET, TEXT, TEXT_DIM

_TITLES = {
    "run_command": "Run this command?",
    "launch_app": "Launch this application?",
    "write_file": "Write this file?",
    "edit_file": "Apply this edit?",
    "append_file": "Append to this file?",
    "delete_path": "Delete this?",
    "move_path": "Move this?",
    "copy_path": "Copy this?",
    "create_dir": "Create this directory?",
    "clipboard_copy": "Copy this to the clipboard?",
    "fetch_url": "Allow this network request?",
    "undo_file_change": "Restore this file?",
}

_MONO = "'JetBrains Mono','DejaVu Sans Mono',monospace"


def _as_html(detail: str) -> str:
    """Render the action's detail, colouring diff lines added/removed."""
    lines = []
    for raw in (detail or "").splitlines() or [""]:
        text = html.escape(raw) or "&nbsp;"
        if raw.startswith(("+++", "---")):
            color = TEXT_DIM
        elif raw.startswith("+"):
            color = OK
        elif raw.startswith("-"):
            color = DANGER
        elif raw.startswith("@@"):
            color = TEXT_DIM
        else:
            color = TEXT
        lines.append(f'<div style="color:{color}; white-space:pre-wrap;">{text}</div>')
    return (
        f'<div style="font-family:{_MONO}; font-size:10pt;">' + "".join(lines) + "</div>"
    )


class ApprovalDialog(QDialog):
    def __init__(self, request: ApprovalRequest, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.resize(620, 420)
        self.setMinimumSize(460, 300)
        self.setStyleSheet(STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = AuroraCard(self)
        root.addWidget(card)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(34, 28, 34, 28)
        lay.setSpacing(12)

        title = QLabel(tr(_TITLES.get(request.tool, "Approve this action?")))
        title.setObjectName("h1")
        lay.addWidget(title)

        # One line naming the target, so the header alone identifies the action.
        summary = QLabel(request.describe() or request.tool)
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color: {TEXT_DIM}; font-family: {_MONO}; font-size: 9.5pt;")
        lay.addWidget(summary)

        # Why we're asking. For safety-driven prompts this is the whole point:
        # an unexpected dialog should explain itself.
        if request.reason:
            reason = tr(request.reason).format(detail=request.reason_detail)
            why = QLabel(tr("⚠  Asking because {reason}.").format(reason=reason))
            why.setWordWrap(True)
            why.setStyleSheet(f"color: {DANGER}; font-size: 9.5pt;")
            lay.addWidget(why)

        self.request = request
        # A command is editable right here: spotting a wrong flag and fixing it
        # beats denying, re-asking and hoping the model gets it right.
        self.editable = request.tool == "run_command"
        detail = QTextEdit()
        detail.setReadOnly(not self.editable)
        if self.editable:
            detail.setPlainText(request.detail())
            detail.setStyleSheet(
                f"QTextEdit {{ background: #000; border: 1px solid {LINE};"
                f"border-radius: 10px; color: {TEXT}; padding: 10px;"
                f"font-family: {_MONO}; font-size: 10pt; }}"
            )
        else:
            detail.setHtml(_as_html(request.detail()))
            detail.setStyleSheet(
                f"QTextEdit {{ background: #000; border: 1px solid {LINE};"
                f"border-radius: 10px; color: {TEXT}; padding: 10px; }}"
            )
        self.detail = detail
        lay.addWidget(detail, 1)

        if self.editable:
            hint = QLabel(tr("You can edit the command before running it."))
            hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8.5pt;")
            lay.addWidget(hint)

        # Set when the user picks "Always allow" so the caller can remember this
        # exact command and skip the prompt next time.
        self.always_allow = False

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        deny = QPushButton(tr("Deny"))
        deny.setObjectName("danger")
        deny.clicked.connect(self.reject)
        buttons.addWidget(deny)

        # "Always allow" only makes sense for repeatable shell commands — and
        # never for one flagged as dangerous or secret-touching, where the whole
        # point of the prompt is that it happens every time.
        if (
            request.tool == "run_command"
            and request.args.get("command", "").strip()
            and request.reason not in UNSKIPPABLE_REASONS
        ):
            always = QPushButton(tr("Always allow"))
            always.setToolTip(tr("Approve and never ask for this exact command again"))
            always.clicked.connect(self._always)
            buttons.addWidget(always)

        approve = QPushButton(tr("Approve & Run"))
        approve.setObjectName("primary")
        approve.clicked.connect(self.accept)
        approve.setFocus()
        buttons.addWidget(approve)
        lay.addLayout(buttons)

    def accept(self) -> None:
        """Apply any edit the user made before letting the command run."""
        if getattr(self, "editable", False):
            edited = self.detail.toPlainText().strip()
            if edited and edited != self.request.describe():
                # The args dict is the one the agent is about to execute.
                self.request.args["command"] = edited
        super().accept()

    def _always(self) -> None:
        self.always_allow = True
        self.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # Esc denies (Qt's default reject), Enter approves — but only from the
        # buttons, never while the user is scrolling the detail pane.
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)
