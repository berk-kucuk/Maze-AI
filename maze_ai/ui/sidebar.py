"""History sidebar: New-chat button + list of past conversations."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
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

from ..history import Conversation
from ..i18n import tr
from .theme import LINE, TEXT, TEXT_DIM, TEXT_FAINT, WHITE


def _relative_time(ts: float) -> str:
    delta = max(0, int(time.time() - ts))
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    if delta < 7 * 86400:
        return f"{delta // 86400}d ago"
    return time.strftime("%d %b", time.localtime(ts))


class ElidedLabel(QLabel):
    """A single-line label that elides overflowing text with an ellipsis.

    Uses an ``Ignored`` horizontal size policy so the label never demands more
    width than the row can give — this is what keeps the ✕ delete button from
    being pushed off-screen by long conversation titles.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._relayout()

    def setText(self, text: str) -> None:  # noqa: N802
        self._full_text = text
        self.setToolTip(text)
        self._relayout()

    def _relayout(self) -> None:
        fm = QFontMetrics(self.font())
        elided = fm.elidedText(self._full_text, Qt.TextElideMode.ElideRight, max(0, self.width()))
        super().setText(elided)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()


class ChatListItem(QFrame):
    selected = Signal(str)
    deleted = Signal(str)
    renamed = Signal(str)
    exported = Signal(str)

    def __init__(self, conv: Conversation, active: bool) -> None:
        super().__init__()
        self.conv_id = conv.id
        self.setObjectName("chatitem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_active(active)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 9, 8, 9)
        lay.setSpacing(6)

        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = ElidedLabel(conv.title)
        self.title.setStyleSheet(
            f"color: {TEXT}; background: transparent; font-size: 10pt; font-weight: 600;"
        )
        self.title.setWordWrap(False)
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setToolTip(conv.title + "\n(double-click to rename)")
        col.addWidget(self.title)
        meta = QLabel(_relative_time(conv.updated))
        meta.setStyleSheet(f"color: {TEXT_FAINT}; background: transparent; font-size: 8.5pt;")
        col.addWidget(meta)
        lay.addLayout(col, 1)

        self.export_btn = QToolButton()
        self.export_btn.setText("⭳")
        self.export_btn.setObjectName("chatdel")
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.setFixedSize(22, 22)
        self.export_btn.setToolTip(tr("Export chat to Markdown"))
        self.export_btn.clicked.connect(lambda: self.exported.emit(self.conv_id))
        self.export_btn.hide()
        lay.addWidget(self.export_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        self.del_btn = QToolButton()
        self.del_btn.setText("✕")
        self.del_btn.setObjectName("chatdel")
        self.del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.del_btn.setFixedSize(22, 22)
        self.del_btn.setToolTip(tr("Delete chat"))
        self.del_btn.clicked.connect(lambda: self.deleted.emit(self.conv_id))
        self.del_btn.hide()
        lay.addWidget(self.del_btn, 0, Qt.AlignmentFlag.AlignVCenter)

    def _set_active(self, active: bool) -> None:
        bg = "rgba(255,255,255,0.09)" if active else "transparent"
        border = WHITE if active else "transparent"
        self.setStyleSheet(
            f"QFrame#chatitem {{ background: {bg}; border-radius: 10px; "
            f"border-left: 2px solid {border}; }}"
            f"QFrame#chatitem:hover {{ background: rgba(255,255,255,0.06); }}"
            f"QToolButton#chatdel {{ background: transparent; border: none; "
            f"color: {TEXT_FAINT}; border-radius: 6px; }}"
            f"QToolButton#chatdel:hover {{ background: rgba(255,92,92,0.18); color: #ff8080; }}"
        )

    def enterEvent(self, event) -> None:  # noqa: N802
        self.del_btn.show()
        self.export_btn.show()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.del_btn.hide()
        self.export_btn.hide()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.conv_id)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.renamed.emit(self.conv_id)


class HistorySidebar(QWidget):
    new_chat_requested = Signal()
    chat_selected = Signal(str)
    chat_deleted = Signal(str)
    chat_renamed = Signal(str)
    chat_exported = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(238)
        self._all: list[Conversation] = []
        self._active_id = ""
        self._filter = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 10, 6)
        root.setSpacing(10)

        self.new_btn = QPushButton("  ＋   " + tr("New chat"))
        self.new_btn.setObjectName("primary")
        self.new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_btn.setFixedHeight(40)
        self.new_btn.clicked.connect(self.new_chat_requested.emit)
        root.addWidget(self.new_btn)

        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("Search chats…"))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search)
        root.addWidget(self.search)

        header = QLabel(tr("HISTORY"))
        header.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 8.5pt; font-weight: 700; "
            "letter-spacing: 0.6px; background: transparent;"
        )
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")
        self._list_host = QWidget()
        self._list_host.setStyleSheet("background: transparent;")
        self._list = QVBoxLayout(self._list_host)
        self._list.setContentsMargins(0, 0, 4, 0)
        self._list.setSpacing(3)
        self._list.addStretch(1)
        scroll.setWidget(self._list_host)
        root.addWidget(scroll, 1)

        self._empty = QLabel(tr("No saved chats yet."))
        self._empty.setStyleSheet(f"color: {TEXT_FAINT}; background: transparent; font-size: 9pt;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # A hairline on the right edge separating the sidebar from the chat.
        self.setStyleSheet(
            f"HistorySidebar {{ border-right: 1px solid {LINE}; }}"
        )

    def set_conversations(self, conversations: list[Conversation], active_id: str) -> None:
        self._all = conversations
        self._active_id = active_id
        self._rebuild()

    def _on_search(self, text: str) -> None:
        self._filter = (text or "").strip().lower()
        self._rebuild()

    def _matches(self, conv: Conversation) -> bool:
        if not self._filter:
            return True
        if self._filter in conv.title.lower():
            return True
        # Also search message bodies so users can find a chat by its content.
        return any(self._filter in (m.get("content") or "").lower() for m in conv.messages)

    def _rebuild(self) -> None:
        # Detach the reusable empty-state label first (no-op if not currently in
        # the layout). Otherwise the sweep below would deleteLater() it, and the
        # NEXT refresh — new chat, deleted chat, finished turn — would touch a
        # destroyed C++ object and silently abort, freezing the history list until
        # the app is restarted.
        self._list.removeWidget(self._empty)
        self._empty.hide()

        # Clear existing item widgets (keep the trailing stretch).
        while self._list.count() > 1:
            item = self._list.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        shown = [c for c in self._all if self._matches(c)]
        if not shown:
            self._empty.setText(
                tr("No matching chats.") if self._filter else tr("No saved chats yet.")
            )
            self._list.insertWidget(0, self._empty)
            self._empty.show()
            return
        self._empty.hide()
        for conv in shown:
            item = ChatListItem(conv, active=conv.id == self._active_id)
            item.selected.connect(self.chat_selected.emit)
            item.deleted.connect(self.chat_deleted.emit)
            item.renamed.connect(self.chat_renamed.emit)
            item.exported.connect(self.chat_exported.emit)
            self._list.insertWidget(self._list.count() - 1, item)
