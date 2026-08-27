"""Custom frameless title bar: logo, title, backend badge, window controls."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

from ..i18n import tr
from .theme import LOGO_PATH, TEXT_DIM


def _alarm_icon(color: str = TEXT_DIM, px: int = 20) -> QIcon:
    """A monochrome alarm-clock icon drawn at runtime (no coloured emoji).

    Face + two hands, two bells on top and two little feet — rendered in the
    given colour so it matches the other monochrome title-bar controls.
    """
    ratio = 4  # supersample for crisp edges on hi-dpi
    size = px * ratio
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    col = QColor(color)
    pen = QPen(col)
    pen.setWidthF(size * 0.065)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

    cx, cy = size * 0.5, size * 0.58
    r = size * 0.29
    off = r * 0.78

    # bells (filled circles) at the top-left / top-right
    br = size * 0.115
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(col))
    for bx in (cx - off, cx + off):
        p.drawEllipse(QPointF(bx, cy - off), br, br)

    # clock face
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(cx, cy), r, r)

    # feet
    fl = size * 0.11
    p.drawLine(QPointF(cx - off, cy + r * 0.75), QPointF(cx - off - fl, cy + r * 0.75 + fl))
    p.drawLine(QPointF(cx + off, cy + r * 0.75), QPointF(cx + off + fl, cy + r * 0.75 + fl))

    # hands
    hp = QPen(col)
    hp.setWidthF(size * 0.055)
    hp.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(hp)
    p.drawLine(QPointF(cx, cy), QPointF(cx, cy - r * 0.52))
    p.drawLine(QPointF(cx, cy), QPointF(cx + r * 0.40, cy + r * 0.06))
    p.end()
    return QIcon(pm)


class TitleBar(QWidget):
    minimize_clicked = Signal()
    close_clicked = Signal()
    settings_clicked = Signal()
    sidebar_clicked = Signal()
    reminders_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(52)
        self._drag_offset = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 12, 0)
        layout.setSpacing(10)

        self.sidebar_btn = self._ctl_button("☰", tr("Toggle chat history"), "winctl")
        self.sidebar_btn.clicked.connect(self.sidebar_clicked.emit)
        layout.addWidget(self.sidebar_btn)

        logo = QLabel()
        pix = QPixmap(LOGO_PATH)
        if not pix.isNull():
            logo.setPixmap(pix.scaled(26, 26, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        logo.setFixedSize(30, 30)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)

        title = QLabel("Maze AI")
        title.setStyleSheet("font-size: 12pt; font-weight: 700; letter-spacing: 0.5px;")
        layout.addWidget(title)

        self.badge = QLabel("")
        self.badge.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 9pt; "
            "background: rgba(255,255,255,0.05); border-radius: 8px; padding: 3px 10px;"
        )
        layout.addSpacing(4)
        layout.addWidget(self.badge)

        layout.addStretch(1)

        self.reminders_btn = self._ctl_button("", tr("Reminders"), "winctl")
        self.reminders_btn.setIcon(_alarm_icon())
        self.reminders_btn.setIconSize(QSize(20, 20))
        self.reminders_btn.clicked.connect(self.reminders_clicked.emit)
        layout.addWidget(self.reminders_btn)

        self.settings_btn = self._ctl_button("⚙", tr("Settings"), "winctl")
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.settings_btn)

        self.min_btn = self._ctl_button("﹣", tr("Minimize to tray"), "winctl")
        self.min_btn.clicked.connect(self.minimize_clicked.emit)
        layout.addWidget(self.min_btn)

        self.close_btn = self._ctl_button("✕", tr("Hide to tray"), "winclose")
        self.close_btn.clicked.connect(self.close_clicked.emit)
        layout.addWidget(self.close_btn)

    def _ctl_button(self, glyph: str, tip: str, obj: str) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName(obj if obj == "winctl" else "winctl")
        if obj == "winclose":
            btn.setObjectName("winclose")
        btn.setText(glyph)
        btn.setToolTip(tip)
        btn.setFixedSize(QSize(34, 34))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def set_backend_badge(self, text: str) -> None:
        self.badge.setText(text)

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
        event.accept()
