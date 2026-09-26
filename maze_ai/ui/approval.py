"""Approval dialog shown before the agent runs a side-effecting tool.

The dialog's job is informed consent: it must show what will actually happen,
not just which tool is about to run. A file write is presented as a diff
against what is on disk, a delete says how much it would destroy, and anything
that needs approval for a *safety* reason (a secret, data leaving the machine)
says so in plain words.

Two things protect the consent itself:

* Every piece of text here that came from the model is shown as plain text.
  A ``QLabel`` left on auto-detect would render a command containing markup
  (``<span style="display:none">``, ``<br>``, …) as rich text, letting a
  prompt-injected model make the summary say one thing while another runs.
* The dialog pops up by itself, in the middle of whatever the user was doing.
  Approve is therefore not armed for the first moments: an Enter meant for
  the message box must not approve a command nobody has read. Enter on its
  own never approves at all — that takes Ctrl+Enter or a click.
"""

from __future__ import annotations

import html

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
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
from . import icons
from .dialogs import shortcut_text
from .effects import AuroraCard
from .richtext import harden_labels, plain_label
from .theme import (
    DANGER,
    DANGER_BG,
    FONT_MONO,
    LINE,
    OK,
    STYLESHEET,
    TEXT,
    TEXT_DIM,
    TEXT_FAINT,
    WARN,
)

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

_ICONS = {
    "run_command": "terminal",
    "launch_app": "sparkle",
    "delete_path": "trash",
    "fetch_url": "globe",
    "clipboard_copy": "copy",
}

#: How long Approve stays disarmed after the dialog appears.
ARM_DELAY_MS = 700

_MONO = FONT_MONO


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
        self.resize(660, 460)
        self.setMinimumSize(480, 320)
        self.setStyleSheet(STYLESHEET)
        self.setWindowTitle(tr("Approval required"))
        self._armed = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = AuroraCard(self, animated=False)
        root.addWidget(card)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(card.margin + 24, card.margin + 20,
                               card.margin + 24, card.margin + 20)
        lay.setSpacing(12)

        danger = request.reason in UNSKIPPABLE_REASONS
        head = QHBoxLayout()
        head.setSpacing(12)
        badge = QLabel()
        badge.setFixedSize(38, 38)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setPixmap(icons.pixmap(
            "warning" if danger else _ICONS.get(request.tool, "shield"),
            DANGER if danger else TEXT, 19,
        ))
        badge.setStyleSheet(
            f"background: {DANGER_BG if danger else 'rgba(255,255,255,0.05)'};"
            f"border: 1px solid {'#5a2229' if danger else LINE}; border-radius: 11px;"
        )
        head.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)

        titles = QVBoxLayout()
        titles.setSpacing(3)
        title = plain_label(tr(_TITLES.get(request.tool, "Approve this action?")))
        title.setObjectName("h2")
        titles.addWidget(title)
        # One line naming the target, so the header alone identifies the action.
        # Plain text: this string is written by the model.
        summary = plain_label(request.describe() or request.tool)
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        summary.setStyleSheet(f"color: {TEXT_DIM}; font-family: {_MONO}; font-size: 9.5pt;")
        self.summary = summary
        titles.addWidget(summary)
        head.addLayout(titles, 1)
        lay.addLayout(head)

        # Why we're asking. For safety-driven prompts this is the whole point:
        # an unexpected dialog should explain itself.
        if request.reason:
            reason = tr(request.reason).format(detail=request.reason_detail)
            why = plain_label(tr("⚠  Asking because {reason}.").format(reason=reason))
            why.setWordWrap(True)
            color = DANGER if danger else WARN
            why.setStyleSheet(
                f"color: {color}; font-size: 9.5pt; background: rgba(255,255,255,0.03);"
                f"border: 1px solid {LINE}; border-radius: 9px; padding: 7px 10px;"
            )
            lay.addWidget(why)

        self.request = request
        # A command is editable right here: spotting a wrong flag and fixing it
        # beats denying, re-asking and hoping the model gets it right.
        self.editable = request.tool == "run_command"
        detail = QTextEdit()
        detail.setReadOnly(not self.editable)
        detail.setAcceptRichText(False)
        detail.setTabChangesFocus(True)
        if self.editable:
            detail.setPlainText(request.detail())
            detail.setStyleSheet(
                f"QTextEdit {{ background: #000; border: 1px solid {LINE};"
                f"border-radius: 10px; color: {TEXT}; padding: 10px;"
                f"font-family: {_MONO}; font-size: 10pt; }}"
                "QTextEdit:focus { border-color: #5a5a64; }"
            )
        else:
            detail.setHtml(_as_html(request.detail()))
            detail.setStyleSheet(
                f"QTextEdit {{ background: #000; border: 1px solid {LINE};"
                f"border-radius: 10px; color: {TEXT}; padding: 10px; }}"
            )
        self.detail = detail
        lay.addWidget(detail, 1)

        hints = QHBoxLayout()
        hints.setSpacing(12)
        if self.editable:
            hint = plain_label(tr("You can edit the command before running it."))
            hint.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 8.5pt;")
            hints.addWidget(hint)
        hints.addStretch(1)
        keys = plain_label(
            tr("{approve} approve · {deny} deny").format(
                approve=shortcut_text("Ctrl+Return"), deny=shortcut_text("Esc")
            )
        )
        keys.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 8.5pt;")
        hints.addWidget(keys)
        lay.addLayout(hints)

        # Set when the user picks "Always allow" so the caller can remember this
        # exact command and skip the prompt next time.
        self.always_allow = False

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        deny = QPushButton(tr("Deny"))
        deny.setObjectName("danger")
        deny.setCursor(Qt.CursorShape.PointingHandCursor)
        deny.clicked.connect(self.reject)
        buttons.addWidget(deny)
        self.deny_btn = deny

        # "Always allow" only makes sense for repeatable shell commands — and
        # never for one flagged as dangerous or secret-touching, where the whole
        # point of the prompt is that it happens every time.
        self.always_btn: QPushButton | None = None
        if (
            request.tool == "run_command"
            and request.args.get("command", "").strip()
            and request.reason not in UNSKIPPABLE_REASONS
        ):
            always = QPushButton(tr("Always allow"))
            always.setToolTip(
                tr("Approve and never ask for this exact command again")
                + f"  ({shortcut_text('Alt+A')})"
            )
            always.setCursor(Qt.CursorShape.PointingHandCursor)
            always.clicked.connect(self._always)
            always.setEnabled(False)
            buttons.addWidget(always)
            self.always_btn = always
            QShortcut(QKeySequence("Alt+A"), self, activated=self._always)

        approve = QPushButton(tr("Approve & Run"))
        approve.setObjectName("primary")
        approve.setCursor(Qt.CursorShape.PointingHandCursor)
        approve.setIcon(icons.icon("check", "#050506", 16, disabled=TEXT_FAINT, stroke=2.2))
        approve.setIconSize(QSize(16, 16))
        approve.clicked.connect(self.accept)
        approve.setEnabled(False)
        buttons.addWidget(approve)
        self.approve_btn = approve
        lay.addLayout(buttons)

        # No button is a default: a bare Enter must never approve.
        for button in (deny, approve, *( [self.always_btn] if self.always_btn else [])):
            button.setAutoDefault(False)
            button.setDefault(False)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._approve_key)
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self._approve_key)
        deny.setFocus()
        harden_labels(self)

        QTimer.singleShot(ARM_DELAY_MS, self._arm)

    def _arm(self) -> None:
        self._armed = True
        self.approve_btn.setEnabled(True)
        if self.always_btn is not None:
            self.always_btn.setEnabled(True)

    def _approve_key(self) -> None:
        if self._armed:
            self.accept()

    def accept(self) -> None:
        """Apply any edit the user made before letting the command run."""
        if getattr(self, "editable", False):
            edited = self.detail.toPlainText().strip()
            if edited and edited != self.request.describe():
                # The args dict is the one the agent is about to execute.
                self.request.args["command"] = edited
        super().accept()

    def _always(self) -> None:
        if not self._armed or self.always_btn is None:
            return
        self.always_allow = True
        self.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # Esc denies (Qt's default reject). A bare Enter does nothing here: it
        # is exactly the key a user is most likely to be pressing elsewhere at
        # the moment this dialog steals focus.
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            event.accept()
            return
        super().keyPressEvent(event)
