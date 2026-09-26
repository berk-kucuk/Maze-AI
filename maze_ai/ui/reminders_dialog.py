"""Reminders panel: view, add and remove pending reminders from the UI."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..reminders import ReminderStore, parse_when
from . import icons
from .effects import AuroraCard
from .richtext import harden_labels, plain_label
from .theme import DANGER, LINE, STYLESHEET, TEXT, TEXT_DIM, TEXT_FAINT


class RemindersDialog(QDialog):
    def __init__(self, store: ReminderStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.resize(520, 560)
        self.setStyleSheet(STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = AuroraCard(self)
        root.addWidget(card)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(card.margin + 22, card.margin + 18,
                               card.margin + 22, card.margin + 20)
        lay.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel(tr("Reminders"))
        title.setObjectName("h1")
        header.addWidget(title)
        header.addStretch(1)
        close = QToolButton()
        close.setObjectName("icon")
        close.setIcon(icons.icon("close", TEXT_DIM, 16, hover=TEXT))
        close.setIconSize(QSize(16, 16))
        close.setAutoRaise(True)
        close.setToolTip(tr("Close") + "  (Esc)")
        close.setFixedSize(32, 32)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(self.accept)
        header.addWidget(close)
        lay.addLayout(header)

        # ── add row ──
        add_row = QHBoxLayout()
        self.text_in = QLineEdit()
        self.text_in.setPlaceholderText(tr("Remind me to…"))
        self.text_in.returnPressed.connect(self._add)
        add_row.addWidget(self.text_in, 1)
        self.when_in = QLineEdit()
        self.when_in.setPlaceholderText(tr("in 30 minutes"))
        self.when_in.setFixedWidth(130)
        self.when_in.returnPressed.connect(self._add)
        add_row.addWidget(self.when_in)
        add_btn = QPushButton(tr("Add"))
        add_btn.setObjectName("primary")
        add_btn.clicked.connect(self._add)
        add_row.addWidget(add_btn)
        lay.addLayout(add_row)

        self.hint = QLabel("")
        self.hint.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 9pt;")
        lay.addWidget(self.hint)

        # ── list ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        self._host = QWidget()
        self._host.setStyleSheet("background: transparent;")
        self._list = QVBoxLayout(self._host)
        self._list.setContentsMargins(0, 0, 4, 0)
        self._list.setSpacing(6)
        self._list.addStretch(1)
        scroll.setWidget(self._host)
        lay.addWidget(scroll, 1)

        self._reload()
        harden_labels(self)
        self.text_in.setFocus()

    def _add(self) -> None:
        text = self.text_in.text().strip()
        when = self.when_in.text().strip()
        if not text:
            self.hint.setText(tr("Enter what to be reminded about."))
            return
        due = parse_when(when)
        if due is None:
            self.hint.setText(
                tr("Couldn't read the time. Try 'in 10 minutes', '18:30', "
                   "'tomorrow 09:00'.")
            )
            return
        self.store.add(text, due)
        self.text_in.clear()
        self.when_in.clear()
        self.hint.setText("")
        self._reload()

    def _reload(self) -> None:
        while self._list.count() > 1:
            item = self._list.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.store.load()
        pending = self.store.pending()
        if not pending:
            empty = QLabel(tr("No pending reminders."))
            empty.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 9pt;")
            empty.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            self._list.insertWidget(0, empty)
            return
        for r in pending:
            self._list.insertWidget(self._list.count() - 1, self._row(r.id, r.text, r.when_str()))

    def _row(self, rid: str, text: str, when: str) -> QFrame:
        row = QFrame()
        row.setObjectName("card")
        row.setStyleSheet(
            f"QFrame#card {{ background: rgba(255,255,255,0.03); "
            f"border: 1px solid {LINE}; border-radius: 10px; }}"
        )
        h = QHBoxLayout(row)
        h.setContentsMargins(12, 8, 8, 8)
        col = QVBoxLayout()
        col.setSpacing(2)
        # Reminder text is often written by the agent: plain text only.
        t = plain_label(text)
        t.setWordWrap(True)
        t.setStyleSheet(f"color: {TEXT}; background: transparent; font-size: 10pt;")
        col.addWidget(t)
        w = plain_label(f"⏰ {when}")
        w.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; font-size: 8.5pt;")
        col.addWidget(w)
        h.addLayout(col, 1)
        rm = QToolButton()
        rm.setIcon(icons.icon("trash", TEXT_FAINT, 15, hover=DANGER))
        rm.setIconSize(QSize(15, 15))
        rm.setAutoRaise(True)
        rm.setToolTip(tr("Delete"))
        rm.setObjectName("chatdel")
        rm.setFixedSize(24, 24)
        rm.setCursor(Qt.CursorShape.PointingHandCursor)
        rm.setStyleSheet(
            f"QToolButton {{ background: transparent; border: none; color: {TEXT_FAINT}; border-radius: 6px; }}"
            f"QToolButton:hover {{ background: rgba(255,92,92,0.18); color: {DANGER}; }}"
        )
        rm.clicked.connect(lambda: self._remove(rid))
        h.addWidget(rm, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    def _remove(self, rid: str) -> None:
        self.store.remove(rid)
        self._reload()
