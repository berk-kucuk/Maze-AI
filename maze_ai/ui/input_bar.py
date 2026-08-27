"""Composer: multi-line input with Enter-to-send and a send button."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from .theme import LINE, PANEL, TEXT_DIM, TEXT_FAINT

# Files that make sense as a vision/OCR attachment rather than a path.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


class _Composer(QPlainTextEdit):
    submit = Signal()
    image_pasted = Signal(str)      # a screenshot pasted straight from the clipboard

    def __init__(self) -> None:
        super().__init__()
        self.setPlaceholderText(
            tr("Ask Maze AI to do something…  (Enter to send, Shift+Enter for newline)")
        )
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(48)
        self.textChanged.connect(self._autosize)

    def _autosize(self) -> None:
        # QPlainTextEdit's document().size().height() is a LINE count (visual
        # lines, wrapping included) — NOT pixels — so turn it into a real pixel
        # height via the font's line spacing. Without this the box stayed frozen
        # at one line and typed text scrolled out of view, looking like it was
        # lost. Grows from ~1 line up to ~6 lines, then scrolls.
        lines = max(1.0, self.document().size().height())
        line_px = self.fontMetrics().lineSpacing()
        chrome = 2 * self.document().documentMargin() + 2 * self.frameWidth() + 16
        height = int(lines * line_px + chrome)
        self.setFixedHeight(max(48, min(160, height)))

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            event.accept()
            self.submit.emit()
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            # Take a screenshot with the system shortcut, hit Ctrl+V here, ask
            # about it. Without this, pasting an image would do nothing at all.
            saved = self._save_pasted_image()
            if saved:
                event.accept()
                self.image_pasted.emit(saved)
                return
        super().keyPressEvent(event)

    @staticmethod
    def _save_pasted_image() -> str:
        """Write an image sitting on the clipboard to a file; "" if there is none."""
        mime = QGuiApplication.clipboard().mimeData()
        if mime is None or not mime.hasImage():
            return ""
        image = QGuiApplication.clipboard().image()
        if image.isNull():
            return ""
        target = (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "maze-ai" / "pasted"
        )
        try:
            target.mkdir(parents=True, exist_ok=True)
            path = target / f"paste-{datetime.now():%Y%m%d-%H%M%S-%f}.png"
            if image.save(str(path), "PNG"):
                return str(path)
        except OSError:
            pass
        return ""


class InputBar(QFrame):
    send = Signal(str)
    stop = Signal()

    #: Emitted when files are dropped that aren't images — the composer gets
    #: their paths so the agent can read them.
    files_dropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._attachments: list[str] = []
        self._vision = False  # whether the active backend accepts images
        # Drag a file onto the composer to work with it. Expected of any modern
        # chat window, and the shortest path from "this file" to "do something".
        self.setAcceptDrops(True)
        self.setStyleSheet(
            f"InputBar {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 16px; }}"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(4)

        # A small strip showing attached images (hidden when there are none).
        self.attach_label = QPushButton("")
        self.attach_label.setFlat(True)
        self.attach_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.attach_label.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {TEXT_FAINT};"
            "font-size: 8.5pt; text-align: left; padding: 0 4px; }"
        )
        self.attach_label.setToolTip(tr("Click to clear attachments"))
        self.attach_label.clicked.connect(self.clear_attachments)
        self.attach_label.hide()
        outer.addWidget(self.attach_label)

        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.attach_btn = QToolButton()
        self.attach_btn.setText("＋")
        self.attach_btn.setToolTip(tr("Attach an image (vision models)"))
        self.attach_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.attach_btn.setFixedSize(40, 40)
        self.attach_btn.setStyleSheet(
            f"QToolButton {{ background: transparent; border: 1px solid {LINE};"
            f"border-radius: 10px; color: {TEXT_DIM}; font-size: 15pt; }}"
            "QToolButton:hover { background: rgba(255,255,255,0.06); }"
            "QToolButton:disabled { color: rgba(255,255,255,0.15); }"
        )
        self.attach_btn.clicked.connect(self._attach)
        lay.addWidget(self.attach_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        self.composer = _Composer()
        self.composer.setStyleSheet(
            "QPlainTextEdit { background: transparent; border: none; padding: 4px 6px; }"
        )
        self.composer.submit.connect(self._emit)
        self.composer.image_pasted.connect(self.add_files_or_images)
        lay.addWidget(self.composer, 1)

        # While the agent is working, the Send button becomes a Stop button.
        self.stop_btn = QPushButton(tr("Stop"))
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.clicked.connect(self.stop.emit)
        self.stop_btn.hide()
        lay.addWidget(self.stop_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        self.send_btn = QPushButton(tr("Send"))
        self.send_btn.setObjectName("primary")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setFixedHeight(40)
        self.send_btn.clicked.connect(self._emit)
        lay.addWidget(self.send_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        outer.addLayout(lay)

    # ── attachments ──────────────────────────────────────────────────────
    def set_vision(self, enabled: bool) -> None:
        """Enable/disable the attach button based on the backend's capability."""
        self._vision = enabled
        self.attach_btn.setEnabled(enabled)
        self.attach_btn.setToolTip(
            tr("Attach an image") if enabled
            else tr("The current model doesn't support images")
        )
        if not enabled:
            self.clear_attachments()

    # ── drag & drop ──────────────────────────────────────────────────────
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        mime = event.mimeData()
        paths = [
            url.toLocalFile() for url in mime.urls() if url.isLocalFile()
        ] if mime.hasUrls() else []
        if not paths and mime.hasImage():
            saved = self.composer._save_pasted_image()
            if saved:
                paths = [saved]
        if paths:
            self.add_files_or_images(paths)
            event.acceptProposedAction()

    def add_files_or_images(self, paths: list[str] | str) -> None:
        """Route dropped/pasted files: images become attachments, the rest paths.

        A non-image file is mentioned by path in the message instead — the agent
        can read, edit or run it with the tools it already has, which works for
        any file type and any model.
        """
        if isinstance(paths, str):
            paths = [paths]
        images: list[str] = []
        others: list[str] = []
        for raw in paths:
            path = Path(raw)
            if not raw:
                continue
            if path.suffix.lower() in IMAGE_SUFFIXES and self._vision:
                images.append(str(path))
            else:
                others.append(str(path))
        if images:
            self._attachments.extend(images)
            self._refresh_attach_label()
        if others:
            existing = self.composer.toPlainText()
            listed = " ".join(f'"{p}"' if " " in p else p for p in others)
            self.composer.setPlainText(
                (existing + ("\n" if existing and not existing.endswith("\n") else "")
                 + listed + " ").lstrip()
            )
            self.files_dropped.emit(others)
        self.composer.setFocus()

    def _attach(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("Attach image(s)"), "",
            tr("Images (*.png *.jpg *.jpeg *.gif *.webp *.bmp)"),
        )
        if paths:
            self._attachments.extend(paths)
            self._refresh_attach_label()

    def clear_attachments(self) -> None:
        self._attachments = []
        self._refresh_attach_label()

    def take_attachments(self) -> list[str]:
        """Return the current attachments and clear them."""
        items = list(self._attachments)
        self.clear_attachments()
        return items

    def _refresh_attach_label(self) -> None:
        n = len(self._attachments)
        if n:
            names = ", ".join(p.rsplit("/", 1)[-1] for p in self._attachments[:3])
            more = f" +{n - 3}" if n > 3 else ""
            self.attach_label.setText(f"📎 {names}{more}   ✕")
            self.attach_label.show()
        else:
            self.attach_label.hide()

    def _emit(self) -> None:
        text = self.composer.toPlainText().strip()
        if text:
            self.composer.clear()
            self.send.emit(text)

    def set_busy(self, busy: bool) -> None:
        self.composer.setReadOnly(busy)
        self.send_btn.setEnabled(not busy)
        self.send_btn.setVisible(not busy)
        self.stop_btn.setVisible(busy)
        self.attach_btn.setEnabled(self._vision and not busy)

    def focus_input(self) -> None:
        self.composer.setFocus()
