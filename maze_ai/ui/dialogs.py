"""Shared dialog chrome: one frameless card look for every popup.

Stock ``QMessageBox`` / ``QInputDialog`` windows ignore the app's frameless
glass style and come out as grey system boxes in the middle of a black app.
Everything here shares the same card, header, keyboard behaviour (Esc
cancels, Enter confirms) and plain-text-only labels.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
)
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from . import icons
from .effects import AuroraCard
from .richtext import harden_labels, plain_label
from .theme import (
    DANGER,
    FONT_MONO,
    LINE,
    OK,
    PANEL,
    STYLESHEET,
    TEXT,
    TEXT_DIM,
    TEXT_FAINT,
)


def shortcut_text(sequence: str) -> str:
    """A key sequence the way this desktop spells it (Ctrl+Shift+R, …)."""
    return QKeySequence(sequence).toString(QKeySequence.SequenceFormat.NativeText)


def with_shortcut(tip: str, sequence: str) -> str:
    """Tool-tip text with the shortcut appended: "New chat  (Ctrl+N)"."""
    return f"{tip}  ({shortcut_text(sequence)})" if sequence else tip


#: How keys are spelled on the caps — the names people actually see on keyboards.
_KEY_NAMES = {"Return": "Enter", "Up": "↑", "Down": "↓", "Left": "←", "Right": "→",
              "Backspace": "⌫", "PgUp": "PgUp", "PgDown": "PgDn"}


def keycap(text: str) -> QLabel:
    parts = [_KEY_NAMES.get(part, part) for part in text.split("+")] if text != "+" else [text]
    label = plain_label("+".join(parts))
    label.setObjectName("kbd")
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return label


class FramelessDialog(QDialog):
    """A modal glass card with a title row, a close button and a body layout."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        subtitle: str = "",
        width: int = 460,
        icon_name: str = "",
        icon_color: str = TEXT,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.setStyleSheet(STYLESHEET)
        self.setWindowTitle(title)
        self.resize(width, 10)
        self.setMinimumWidth(min(width, 420) if width > 420 else width)
        self._preferred_width = width
        self._drag: QPoint | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.card = AuroraCard(self, animated=False)
        root.addWidget(self.card)

        outer = QVBoxLayout(self.card)
        outer.setContentsMargins(self.card.margin + 22, self.card.margin + 18,
                                 self.card.margin + 22, self.card.margin + 20)
        outer.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(12)
        if icon_name:
            badge = QLabel()
            badge.setFixedSize(34, 34)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setPixmap(icons.pixmap(icon_name, icon_color, 18))
            badge.setStyleSheet(
                f"background: rgba(255,255,255,0.05); border: 1px solid {LINE};"
                "border-radius: 10px;"
            )
            head.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(3)
        self.title_label = plain_label(title)
        self.title_label.setObjectName("h2")
        self.title_label.setWordWrap(True)
        titles.addWidget(self.title_label)
        if subtitle:
            sub = plain_label(subtitle)
            sub.setWordWrap(True)
            sub.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9.5pt;")
            titles.addWidget(sub)
        head.addLayout(titles, 1)
        close = QToolButton()
        close.setObjectName("icon")
        close.setIcon(icons.icon("close", TEXT_DIM, 16, hover=TEXT))
        close.setIconSize(QSize(16, 16))
        close.setFixedSize(30, 30)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setToolTip(with_shortcut(tr("Close"), "Esc"))
        close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close.clicked.connect(self.reject)
        head.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(head)

        self.body = QVBoxLayout()
        self.body.setSpacing(12)
        outer.addLayout(self.body, 1)

        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(8)
        outer.addLayout(self.buttons)

    def showEvent(self, event) -> None:  # noqa: N802
        harden_labels(self)
        super().showEvent(event)
        # Size to content (never narrower than asked), centred over the parent.
        self.adjustSize()
        if self.width() < self._preferred_width:
            self.resize(self._preferred_width, self.height())
        parent = self.parentWidget()
        if parent is not None and parent.isVisible():
            host = parent.window().frameGeometry()
            self.move(host.center() - self.rect().center())

    # Drag anywhere on the card background.
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.windowHandle()
            if handle is not None and handle.startSystemMove():
                return
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = None


class ConfirmDialog(FramelessDialog):
    """Yes/no with an optional monospaced detail (a path, a URL, a command)."""

    def __init__(
        self,
        title: str,
        message: str = "",
        *,
        detail: str = "",
        confirm: str = "",
        cancel: str = "",
        danger: bool = False,
        icon_name: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            title, parent, icon_name=icon_name or ("warning" if danger else ""),
            icon_color=DANGER if danger else TEXT,
        )
        if message:
            text = plain_label(message)
            text.setWordWrap(True)
            text.setStyleSheet(f"color: {TEXT_DIM};")
            self.body.addWidget(text)
        if detail:
            box = plain_label(detail)
            box.setWordWrap(True)
            box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box.setStyleSheet(
                f"background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px;"
                f"padding: 10px 12px; font-family: {FONT_MONO}; font-size: 9.5pt;"
                f"color: {TEXT};"
            )
            self.body.addWidget(box)

        self.buttons.addStretch(1)
        self.cancel_btn = QPushButton(cancel or tr("Cancel"))
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self.reject)
        self.buttons.addWidget(self.cancel_btn)
        self.confirm_btn = QPushButton(confirm or tr("OK"))
        self.confirm_btn.setObjectName("danger" if danger else "primary")
        self.confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.confirm_btn.setDefault(True)
        self.confirm_btn.clicked.connect(self.accept)
        self.buttons.addWidget(self.confirm_btn)
        # A destructive confirm starts on Cancel, so a stray Enter is harmless.
        (self.cancel_btn if danger else self.confirm_btn).setFocus()
        if danger:
            self.confirm_btn.setDefault(False)
            self.cancel_btn.setDefault(True)


class PromptDialog(FramelessDialog):
    """A single line of text input (rename a chat, …)."""

    def __init__(self, title: str, label: str = "", text: str = "", *,
                 confirm: str = "", max_length: int = 80,
                 parent: QWidget | None = None) -> None:
        super().__init__(title, parent, icon_name="edit")
        if label:
            hint = plain_label(label)
            hint.setStyleSheet(f"color: {TEXT_DIM};")
            self.body.addWidget(hint)
        self.field = QLineEdit(text)
        self.field.setMaxLength(max_length)
        self.field.selectAll()
        self.field.returnPressed.connect(self.accept)
        self.body.addWidget(self.field)
        self.buttons.addStretch(1)
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.reject)
        self.buttons.addWidget(cancel)
        ok = QPushButton(confirm or tr("Save"))
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        self.buttons.addWidget(ok)
        self.field.setFocus()

    def value(self) -> str:
        return self.field.text().strip()


class ShortcutsDialog(FramelessDialog):
    """Every keyboard shortcut, grouped, as key caps."""

    def __init__(self, groups: list[tuple[str, list[tuple[str, str]]]],
                 parent: QWidget | None = None) -> None:
        super().__init__(
            tr("Keyboard shortcuts"), parent, width=820, icon_name="keyboard",
            subtitle=tr("Everything in Maze AI can be done without the mouse."),
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 8, 0)
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)

        # Two columns of groups, split where the rows (not the groups) balance.
        sizes = [len(items) + 1 for _, items in groups]
        half = min(range(1, len(groups) + 1),
                   key=lambda i: max(sum(sizes[:i]), sum(sizes[i:])))
        for column, chunk in enumerate((groups[:half], groups[half:])):
            row = 0
            for title, items in chunk:
                header = plain_label(title.upper())
                header.setObjectName("section")
                if row:
                    header.setContentsMargins(0, 12, 0, 0)
                grid.addWidget(header, row, column * 2, 1, 2)
                row += 1
                for label, sequence in items:
                    name = plain_label(label)
                    name.setStyleSheet(f"color: {TEXT}; font-size: 9.5pt;")
                    grid.addWidget(name, row, column * 2)
                    caps = QHBoxLayout()
                    caps.setSpacing(4)
                    caps.addStretch(1)
                    for i, alt in enumerate(sequence.split(" / ")):
                        if i:
                            sep = plain_label(tr("or"))
                            sep.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 8.5pt;")
                            caps.addWidget(sep)
                        caps.addWidget(keycap(shortcut_text(alt) or alt))
                    grid.addLayout(caps, row, column * 2 + 1)
                    row += 1
        grid.setRowStretch(grid.rowCount(), 1)
        scroll.setWidget(host)
        scroll.setMinimumHeight(min(600, host.sizeHint().height() + 8))
        scroll.setMinimumWidth(host.sizeHint().width() + 16)
        self.body.addWidget(scroll)

        self.buttons.addStretch(1)
        done = QPushButton(tr("Done"))
        done.setObjectName("primary")
        done.clicked.connect(self.accept)
        self.buttons.addWidget(done)
        done.setFocus()


class Toast(QLabel):
    """A short-lived message floating at the bottom of a window."""

    def __init__(self, host: QWidget, text: str, kind: str = "info") -> None:
        super().__init__(host)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setText(text)
        color = {"danger": DANGER, "ok": OK}.get(kind, TEXT)
        self.setStyleSheet(
            f"background-color: #17171c; color: {color}; border: 1px solid #34343c;"
            "border-radius: 10px; padding: 9px 16px; font-size: 9.5pt;"
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.adjustSize()
        bottom = host.height() - self.height() - 96
        self.move((host.width() - self.width()) // 2, max(12, bottom))
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        self._fade = QPropertyAnimation(effect, b"opacity", self)
        self._fade.setDuration(180)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.show()
        self.raise_()
        self._fade.start()
        QTimer.singleShot(2200, self._out)

    def _out(self) -> None:
        self._fade.stop()
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.deleteLater)
        self._fade.start()


def notify_toast(widget: QWidget | None, text: str, kind: str = "info") -> None:
    """Show a toast on the window that contains ``widget`` (no-op without one)."""
    if widget is None:
        return
    host = widget.window()
    if host is None or not host.isVisible():
        return
    for old in host.findChildren(Toast):
        old.hide()
        old.deleteLater()
    Toast(host, text, kind)
