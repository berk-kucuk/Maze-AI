"""Visual effects: the animated aurora / frosted-glass card that hosts the UI.

The look is pure black with slow-drifting soft white light blobs behind a
translucent panel — a monochrome take on the "blurred glass + moving light"
aesthetic, painted with QPainter so it needs no compositor support.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, QTimer
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

RADIUS = 18
MARGIN = 22  # room around the card for the drop shadow


class AuroraCard(QFrame):
    """Rounded translucent card with an animated monochrome light field."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        self._phase = 0.0

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(60)
        shadow.setColor(QColor(0, 0, 0, 210))
        shadow.setOffset(0, 10)
        self.setGraphicsEffect(shadow)

        self._timer = QTimer(self)
        self._timer.setInterval(50)  # ~20 fps — smooth but light on CPU
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        self._phase += 0.008
        self.update()

    # ── painting ─────────────────────────────────────────────────────────
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        rect = self.rect().adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(float(rect.x()), float(rect.y()),
                            float(rect.width()), float(rect.height()), RADIUS, RADIUS)
        painter.setClipPath(path)

        # Base fill — a subtle vertical gradient of near-blacks.
        base = QLinearGradient(rect.topLeft(), rect.bottomRight())
        base.setColorAt(0.0, QColor("#0a0a0c"))
        base.setColorAt(1.0, QColor("#040405"))
        painter.fillRect(rect, base)

        # Drifting light blobs.
        w, h = rect.width(), rect.height()
        blobs = [
            (0.22 + 0.10 * math.sin(self._phase),
             0.18 + 0.06 * math.cos(self._phase * 0.8),
             0.55, 26),
            (0.82 + 0.08 * math.cos(self._phase * 0.7),
             0.30 + 0.10 * math.sin(self._phase * 1.1),
             0.48, 18),
            (0.55 + 0.14 * math.sin(self._phase * 0.5 + 1.5),
             0.92 + 0.05 * math.cos(self._phase * 0.9),
             0.62, 14),
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

        # Hairline border with a bright top edge (glass rim).
        rim = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        rim.setColorAt(0.0, QColor(255, 255, 255, 55))
        rim.setColorAt(0.15, QColor(255, 255, 255, 18))
        rim.setColorAt(1.0, QColor(255, 255, 255, 10))
        pen = QPen(QBrush(rim), 1.2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
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
        c = self.rect().center()
        if self._active:
            pulse = 0.5 + 0.5 * math.sin(self._phase)
            glow = QRadialGradient(QPointF(c), 7)
            col = QColor(self._color)
            col.setAlpha(int(120 * pulse))
            glow.setColorAt(0.0, col)
            glow.setColorAt(1.0, QColor(self._color.red(), self._color.green(),
                                        self._color.blue(), 0))
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(c), 7, 7)
            r = 3 + pulse
        else:
            r = 3.0
        painter.setBrush(self._color)
        painter.drawEllipse(QPointF(c), r, r)
        painter.end()
