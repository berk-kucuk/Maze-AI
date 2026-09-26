"""Visual effects: the card that hosts every window, and small indicators.

The card is near-black glass with a faint light drifting behind it — the
Maze look — painted with QPainter so it needs no compositor support. The
drift only runs while its window is visible *and* focused: a background
window repainting itself twenty times a second is a battery drain, not a
feature.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QWidget

RADIUS = 14
MARGIN = 22  # room around the card for the drop shadow


class AuroraCard(QFrame):
    """Rounded translucent card with a slow, subtle monochrome light field."""

    def __init__(self, parent: QWidget | None = None, *, animated: bool = True) -> None:
        super().__init__(parent)
        self._margin = MARGIN
        self._radius = RADIUS
        self.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        self._phase = 0.0
        self._animated = animated

        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(56)
        self._shadow.setColor(QColor(0, 0, 0, 200))
        self._shadow.setOffset(0, 12)
        self.setGraphicsEffect(self._shadow)

        self._timer = QTimer(self)
        self._timer.setInterval(66)  # ~15 fps is plenty for a slow drift
        self._timer.timeout.connect(self._tick)

    # ── lifecycle: only animate what the user is looking at ─────────────
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        top = self.window()
        if top is not None and top is not self:
            top.installEventFilter(self)
        self._sync_timer()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._timer.stop()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.ActivationChange, QEvent.Type.WindowStateChange):
            self._sync_timer()
        return False

    def _sync_timer(self) -> None:
        top = self.window()
        wanted = (
            self._animated
            and self.isVisible()
            and top is not None
            and top.isActiveWindow()
            and not top.isMinimized()
        )
        if wanted and not self._timer.isActive():
            self._timer.start()
        elif not wanted and self._timer.isActive():
            self._timer.stop()

    def _tick(self) -> None:
        self._phase += 0.006
        self.update()

    def set_flush(self, flush: bool) -> None:
        """Maximized windows fill the screen edge to edge: no margin, no corners."""
        self._margin = 0 if flush else MARGIN
        self._radius = 0 if flush else RADIUS
        self.setContentsMargins(self._margin, self._margin, self._margin, self._margin)
        self._shadow.setEnabled(not flush)
        self.update()

    @property
    def margin(self) -> int:
        return self._margin

    # ── painting ─────────────────────────────────────────────────────────
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        m = self._margin
        rect = QRectF(self.rect().adjusted(m, m, -m, -m))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)
        painter.setClipPath(path)

        base = QLinearGradient(rect.topLeft(), rect.bottomRight())
        base.setColorAt(0.0, QColor("#0a0a0c"))
        base.setColorAt(1.0, QColor("#040405"))
        painter.fillRect(rect, base)

        # Two faint lights: one above the top-left, one low on the right.
        w, h = rect.width(), rect.height()
        blobs = [
            (0.18 + 0.06 * math.sin(self._phase),
             -0.05 + 0.04 * math.cos(self._phase * 0.8),
             0.55, 17),
            (0.88 + 0.05 * math.cos(self._phase * 0.7),
             0.85 + 0.06 * math.sin(self._phase * 1.1),
             0.50, 10),
        ]
        painter.setPen(Qt.PenStyle.NoPen)
        for fx, fy, fr, alpha in blobs:
            cx = rect.x() + fx * w
            cy = rect.y() + fy * h
            radius = fr * max(w, h)
            grad = QRadialGradient(QPointF(cx, cy), radius)
            grad.setColorAt(0.0, QColor(255, 255, 255, alpha))
            grad.setColorAt(0.5, QColor(255, 255, 255, alpha // 3))
            grad.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

        if self._radius:
            # Hairline rim, brighter along the top edge like lit glass.
            rim = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            rim.setColorAt(0.0, QColor(255, 255, 255, 46))
            rim.setColorAt(0.12, QColor(255, 255, 255, 16))
            rim.setColorAt(1.0, QColor(255, 255, 255, 9))
            painter.setClipping(False)
            painter.setPen(QPen(QBrush(rim), 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5),
                                    self._radius, self._radius)
        painter.end()


class GlowDot(QWidget):
    """A small pulsing dot used as a 'thinking' / status indicator."""

    def __init__(self, color: str = "#ffffff", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._color = QColor(color)
        self._phase = 0.0
        self._active = False
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def _tick(self) -> None:
        self._phase += 0.14
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        c = QPointF(self.rect().center())
        if self._active:
            pulse = 0.5 + 0.5 * math.sin(self._phase)
            glow = QRadialGradient(c, 7)
            col = QColor(self._color)
            col.setAlpha(int(120 * pulse))
            glow.setColorAt(0.0, col)
            glow.setColorAt(1.0, QColor(self._color.red(), self._color.green(),
                                        self._color.blue(), 0))
            painter.setBrush(glow)
            painter.drawEllipse(c, 7, 7)
            r = 3 + pulse
        else:
            r = 3.0
        painter.setBrush(self._color)
        painter.drawEllipse(c, r, r)
        painter.end()


class TypingDots(QWidget):
    """Three dots breathing in sequence — the 'assistant is working' cue."""

    def __init__(self, color: str = "#ededf0", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(30, 14)
        self._color = QColor(color)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        self._phase += 0.16
        self.update()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._timer.stop()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._timer.start()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        cy = self.height() / 2
        for i in range(3):
            level = 0.5 + 0.5 * math.sin(self._phase - i * 0.9)
            col = QColor(self._color)
            col.setAlphaF(0.25 + 0.75 * level)
            painter.setBrush(col)
            painter.drawEllipse(QPointF(5 + i * 10, cy - level * 1.5), 2.6, 2.6)
        painter.end()
