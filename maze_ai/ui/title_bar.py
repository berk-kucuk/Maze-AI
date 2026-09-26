"""Custom frameless title bar: chat title, model badge, tools, window controls.

It sits over the chat column (the sidebar carries the brand), doubles as the
window's drag handle, and a double-click maximizes like any native title bar.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QWidget,
)

from ..i18n import tr
from . import icons
from .dialogs import with_shortcut
from .sidebar import ElidedLabel
from .theme import TEXT, TEXT_DIM


class TitleBar(QWidget):
    minimize_clicked = Signal()
    maximize_clicked = Signal()
    close_clicked = Signal()
    settings_clicked = Signal()
    sidebar_clicked = Signal()
    reminders_clicked = Signal()
    shortcuts_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(52)
        self._drag_offset = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 8, 0)
        layout.setSpacing(4)

        self.sidebar_btn = self._icon_button(
            "sidebar", with_shortcut(tr("Toggle chat history"), "Ctrl+B")
        )
        self.sidebar_btn.clicked.connect(self.sidebar_clicked.emit)
        layout.addWidget(self.sidebar_btn)
        layout.addSpacing(6)

        self.title = ElidedLabel("")
        self.title.setStyleSheet("font-size: 11pt; font-weight: 600; background: transparent;")
        self.title.setMinimumWidth(80)
        layout.addWidget(self.title, 1)
        layout.addSpacing(8)

        # The model in use; clicking it opens Settings, where it is changed.
        self.badge = QPushButton("")
        self.badge.setObjectName("chip")
        self.badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self.badge.setIcon(icons.icon("sparkle", TEXT_DIM, 14))
        self.badge.setIconSize(QSize(14, 14))
        self.badge.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.badge.setToolTip(with_shortcut(tr("Change the model in Settings"), "Ctrl+,"))
        self.badge.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addSpacing(6)

        self.reminders_btn = self._icon_button(
            "alarm", with_shortcut(tr("Reminders"), "Ctrl+Shift+R")
        )
        self.reminders_btn.clicked.connect(self.reminders_clicked.emit)
        layout.addWidget(self.reminders_btn)

        self.shortcuts_btn = self._icon_button(
            "keyboard", with_shortcut(tr("Keyboard shortcuts"), "Ctrl+/")
        )
        self.shortcuts_btn.clicked.connect(self.shortcuts_clicked.emit)
        layout.addWidget(self.shortcuts_btn)

        self.settings_btn = self._icon_button("settings", with_shortcut(tr("Settings"), "Ctrl+,"))
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.settings_btn)

        sep = QFrame()
        sep.setObjectName("vsep")
        sep.setFixedHeight(18)
        layout.addSpacing(6)
        layout.addWidget(sep, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addSpacing(6)

        self.min_btn = self._icon_button("minimize", tr("Minimize"), size=16)
        self.min_btn.clicked.connect(self.minimize_clicked.emit)
        layout.addWidget(self.min_btn)

        self.max_btn = self._icon_button("maximize", with_shortcut(tr("Maximize"), "F11"), size=14)
        self.max_btn.clicked.connect(self.maximize_clicked.emit)
        layout.addWidget(self.max_btn)

        self.close_btn = self._icon_button(
            "close", with_shortcut(tr("Hide to tray"), "Ctrl+W"), obj="winclose", size=16
        )
        self.close_btn.clicked.connect(self.close_clicked.emit)
        layout.addWidget(self.close_btn)

    def _icon_button(self, name: str, tip: str, obj: str = "winctl", size: int = 18) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName(obj)
        hover = "#ffffff" if obj == "winclose" else TEXT
        btn.setIcon(icons.icon(name, TEXT_DIM, size, hover=hover))
        btn.setIconSize(QSize(size, size))
        btn.setToolTip(tip)
        btn.setFixedSize(QSize(34, 34))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Hover tint needs the Active icon mode, which QToolButton only uses
        # when it is auto-raised.
        btn.setAutoRaise(True)
        return btn

    def set_backend_badge(self, text: str) -> None:
        self.badge.setText(f" {text}" if text else "")
        self.badge.setVisible(bool(text))

    def set_title(self, text: str) -> None:
        self.title.setText(text)

    def set_maximized(self, maximized: bool) -> None:
        self.max_btn.setIcon(
            icons.icon("restore" if maximized else "maximize", TEXT_DIM, 14, hover=TEXT)
        )
        self.max_btn.setToolTip(
            with_shortcut(tr("Restore") if maximized else tr("Maximize"), "F11")
        )

    # ── window dragging ──────────────────────────────────────────────────
    # Uses the compositor's native move (works on both Wayland and X11);
    # falls back to manual repositioning if no window handle is available.
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        handle = self.window().windowHandle()
        if handle is not None and handle.startSystemMove():
            event.accept()
            return
        self._drag_offset = (
            event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
        )
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_offset = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.maximize_clicked.emit()
        event.accept()
