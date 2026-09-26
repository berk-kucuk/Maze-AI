"""History sidebar: brand, New chat, search, and past conversations by date."""

from __future__ import annotations

import time
from datetime import date, datetime

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QAction, QFontMetrics, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..history import Conversation
from ..i18n import tr
from . import icons
from .dialogs import shortcut_text, with_shortcut
from .richtext import plain_label, plain_tooltip
from .theme import BG, DANGER, LINE, LOGO_PATH, SIDEBAR, TEXT, TEXT_DIM, TEXT_FAINT

SIDEBAR_WIDTH = 264


def _relative_time(ts: float) -> str:
    delta = max(0, int(time.time() - ts))
    if delta < 60:
        return tr("just now")
    if delta < 3600:
        return tr("{n} min ago").format(n=delta // 60)
    if delta < 86400:
        return tr("{n} h ago").format(n=delta // 3600)
    if delta < 7 * 86400:
        return tr("{n} d ago").format(n=delta // 86400)
    return time.strftime("%d.%m.%Y", time.localtime(ts))


def date_group(ts: float, today: date | None = None) -> str:
    """Which bucket a conversation falls in: Today, Yesterday, …"""
    today = today or date.today()
    day = datetime.fromtimestamp(ts).date()
    age = (today - day).days
    if age <= 0:
        return "Today"
    if age == 1:
        return "Yesterday"
    if age < 7:
        return "Previous 7 days"
    if age < 30:
        return "Previous 30 days"
    return "Older"


class ElidedLabel(QLabel):
    """A single-line label that elides overflowing text with an ellipsis.

    Uses an ``Ignored`` horizontal size policy so the label never demands more
    width than the row can give — this is what keeps the row's action button
    from being pushed off-screen by long conversation titles.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._relayout()

    def setText(self, text: str) -> None:  # noqa: N802
        self._full_text = text
        self.setToolTip(plain_tooltip(text))
        self._relayout()

    def full_text(self) -> str:
        return self._full_text

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
        self.active = active
        self.setObjectName("chatitem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(38)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: self._menu(self.mapToGlobal(pos)))
        self._set_active(active)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 4, 0)
        lay.setSpacing(4)

        self.title = ElidedLabel(conv.title)
        weight = 600 if active else 400
        color = TEXT if active else "#c9c9d0"
        self.title.setStyleSheet(
            f"color: {color}; background: transparent; font-size: 10pt; font-weight: {weight};"
        )
        self.title.setToolTip(plain_tooltip(
            f"{conv.title}\n{_relative_time(conv.updated)} · {tr('double-click to rename')}"
        ))
        lay.addWidget(self.title, 1)

        self.more_btn = QToolButton()
        self.more_btn.setObjectName("chatmore")
        self.more_btn.setIcon(icons.icon("more", TEXT_DIM, 16, hover=TEXT))
        self.more_btn.setIconSize(QSize(16, 16))
        self.more_btn.setFixedSize(28, 28)
        self.more_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.more_btn.setToolTip(tr("More"))
        self.more_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.more_btn.clicked.connect(
            lambda: self._menu(self.more_btn.mapToGlobal(QPoint(0, self.more_btn.height())))
        )
        # Parent it (via the layout) before touching visibility: showing a
        # parentless widget opens it as its own top-level window.
        lay.addWidget(self.more_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self.more_btn.setVisible(active)

        # Kept for callers and tests that reach for the individual actions.
        self.export_btn = self.more_btn
        self.del_btn = self.more_btn

    def _set_active(self, active: bool) -> None:
        bg = "rgba(255,255,255,0.075)" if active else "transparent"
        self.setStyleSheet(
            f"QFrame#chatitem {{ background: {bg}; border-radius: 9px; border: none; }}"
            "QFrame#chatitem:hover { background: rgba(255,255,255,0.05); }"
            "QToolButton#chatmore { background: transparent; border: none; border-radius: 7px; }"
            "QToolButton#chatmore:hover { background: rgba(255,255,255,0.09); }"
        )

    def _menu(self, where: QPoint) -> None:
        menu = QMenu(self)
        rename = QAction(icons.icon("edit", TEXT_DIM, 16), tr("Rename"), menu)
        rename.setShortcut("F2")
        rename.triggered.connect(lambda: self.renamed.emit(self.conv_id))
        menu.addAction(rename)
        export = QAction(icons.icon("download", TEXT_DIM, 16), tr("Export to Markdown"), menu)
        export.setShortcut("Ctrl+E")
        export.triggered.connect(lambda: self.exported.emit(self.conv_id))
        menu.addAction(export)
        menu.addSeparator()
        delete = QAction(icons.icon("trash", DANGER, 16), tr("Delete chat"), menu)
        delete.setShortcut("Ctrl+Shift+Backspace")
        delete.triggered.connect(lambda: self.deleted.emit(self.conv_id))
        menu.addAction(delete)
        menu.exec(where)

    def enterEvent(self, event) -> None:  # noqa: N802
        self.more_btn.show()

    def leaveEvent(self, event) -> None:  # noqa: N802
        if not self.active:
            self.more_btn.hide()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.conv_id)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.renamed.emit(self.conv_id)


class _SearchField(QLineEdit):
    """Esc clears the search first, then hands focus back to the composer."""

    escaped = Signal()

    def event(self, event) -> bool:
        # Claim Esc before the window-level Esc shortcut (stop / hide) sees it.
        if event.type() == event.Type.ShortcutOverride and event.key() == Qt.Key.Key_Escape:
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            if self.text():
                self.clear()
            else:
                self.escaped.emit()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Down):
            # Enter / ↓ in the search box opens the first match.
            parent = self.parent()
            while parent is not None and not isinstance(parent, HistorySidebar):
                parent = parent.parent()
            if isinstance(parent, HistorySidebar):
                parent.open_first_match()
            event.accept()
            return
        super().keyPressEvent(event)


class HistorySidebar(QWidget):
    new_chat_requested = Signal()
    chat_selected = Signal(str)
    chat_deleted = Signal(str)
    chat_renamed = Signal(str)
    chat_exported = Signal(str)
    #: The user is done searching (Esc on an empty search box).
    search_left = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setObjectName("sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#sidebar {{ background: {SIDEBAR}; border-right: 1px solid {LINE};"
            "border-top-left-radius: 14px; border-bottom-left-radius: 14px; }"
        )
        self._all: list[Conversation] = []
        self._shown: list[Conversation] = []
        self._active_id = ""
        self._filter = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 14, 10, 12)
        root.setSpacing(10)

        # ── brand ──
        brand = QHBoxLayout()
        brand.setContentsMargins(6, 0, 0, 4)
        brand.setSpacing(10)
        logo = QLabel()
        pix = QPixmap(LOGO_PATH)
        if not pix.isNull():
            scaled = pix.scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(2.0)
            logo.setPixmap(scaled)
        logo.setFixedSize(28, 28)
        brand.addWidget(logo)
        name = plain_label("Maze AI")
        name.setStyleSheet("font-size: 12pt; font-weight: 700; letter-spacing: 0.2px;")
        brand.addWidget(name)
        brand.addStretch(1)
        root.addLayout(brand)

        self.new_btn = QPushButton(tr("New chat"))
        self.new_btn.setObjectName("primary")
        self.new_btn.setIcon(icons.icon("plus", BG, 16, stroke=2.2))
        self.new_btn.setIconSize(QSize(16, 16))
        self.new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_btn.setFixedHeight(38)
        self.new_btn.setToolTip(with_shortcut(tr("New chat"), "Ctrl+N"))
        self.new_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.new_btn.clicked.connect(self.new_chat_requested.emit)
        root.addWidget(self.new_btn)

        self.search = _SearchField()
        self.search.setPlaceholderText(
            tr("Search chats…") + f"   {shortcut_text('Ctrl+F')}"
        )
        self.search.setClearButtonEnabled(True)
        self.search.addAction(icons.icon("search", TEXT_FAINT, 16),
                              QLineEdit.ActionPosition.LeadingPosition)
        self.search.setStyleSheet(
            "QLineEdit { padding: 7px 8px; border-radius: 9px; font-size: 9.5pt; }"
        )
        self.search.textChanged.connect(self._on_search)
        self.search.escaped.connect(self.search_left.emit)
        root.addWidget(self.search)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")
        self._list_host = QWidget()
        self._list_host.setStyleSheet("background: transparent;")
        self._list = QVBoxLayout(self._list_host)
        self._list.setContentsMargins(0, 0, 2, 0)
        self._list.setSpacing(1)
        self._list.addStretch(1)
        scroll.setWidget(self._list_host)
        root.addWidget(scroll, 1)

        self._empty = plain_label(tr("No saved chats yet."))
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet(f"color: {TEXT_FAINT}; background: transparent; font-size: 9pt;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._empty.setContentsMargins(0, 18, 0, 0)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # The empty parts of the rail (the brand row) move the window, like the
        # title bar next to it.
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        toggle = getattr(self.window(), "toggle_maximized", None)
        if event.button() == Qt.MouseButton.LeftButton and toggle is not None:
            toggle()
        event.accept()

    def set_conversations(self, conversations: list[Conversation], active_id: str) -> None:
        self._all = conversations
        self._active_id = active_id
        self._rebuild()

    def _on_search(self, text: str) -> None:
        self._filter = (text or "").strip().lower()
        self._rebuild()

    def focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def open_first_match(self) -> None:
        if self._shown:
            self.chat_selected.emit(self._shown[0].id)

    def neighbour(self, current_id: str, step: int) -> str:
        """The id ``step`` places away from ``current_id`` in the visible list."""
        ids = [c.id for c in self._shown]
        if not ids:
            return ""
        if current_id not in ids:
            return ids[0] if step > 0 else ids[-1]
        index = ids.index(current_id) + step
        if 0 <= index < len(ids):
            return ids[index]
        return ""

    def _matches(self, conv: Conversation) -> bool:
        if not self._filter:
            return True
        if self._filter in conv.title.lower():
            return True
        # Also search message bodies so users can find a chat by its content.
        return any(self._filter in (m.get("content") or "").lower() for m in conv.messages)

    def _group_header(self, text: str) -> QLabel:
        header = plain_label(tr(text).upper())
        header.setObjectName("section")
        header.setContentsMargins(12, 12, 0, 4)
        header.setStyleSheet(
            f"color: {TEXT_FAINT}; font-size: 7.5pt; font-weight: 700; letter-spacing: 0.9px;"
        )
        return header

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
        self._shown = shown
        if not shown:
            self._empty.setText(
                tr("No matching chats.") if self._filter else tr("No saved chats yet.")
            )
            self._list.insertWidget(0, self._empty)
            self._empty.show()
            return
        self._empty.hide()
        today = date.today()
        group = None
        for conv in shown:
            bucket = date_group(conv.updated, today)
            if bucket != group:
                group = bucket
                self._list.insertWidget(self._list.count() - 1, self._group_header(bucket))
            item = ChatListItem(conv, active=conv.id == self._active_id)
            item.selected.connect(self.chat_selected.emit)
            item.deleted.connect(self.chat_deleted.emit)
            item.renamed.connect(self.chat_renamed.emit)
            item.exported.connect(self.chat_exported.emit)
            self._list.insertWidget(self._list.count() - 1, item)
