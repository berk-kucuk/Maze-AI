"""Small widget subclasses that fix annoying default behaviours."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QSpinBox


class NoWheelComboBox(QComboBox):
    """A combo box that ignores mouse-wheel scrolling and opens on any click.

    Two fixes over the stock ``QComboBox``:

    * Mouse-wheel scrolling is ignored so the value can't silently change while
      the user scrolls the page it lives on (the event bubbles up to the
      scroll area instead).
    * A left-click anywhere opens the list, on RELEASE rather than press. Inside
      this frameless, translucent dialog on Wayland the stock behaviour popped
      the list open on the press, then the matching release was delivered to the
      just-opened popup and dismissed it instantly — so items could never be
      clicked (you had to press-and-hold to select). Opening on release, when
      the button is already up, leaves no stray event to close the popup, so
      selection works. This applies to editable combos too (Qt otherwise only
      opens them via the tiny drop arrow).
    """

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        event.ignore()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # Swallow the press when the list is closed; we open it on release.
        if event.button() == Qt.MouseButton.LeftButton and not self.view().isVisible():
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and not self.view().isVisible():
            self.showPopup()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class NoWheelSpinBox(QSpinBox):
    """A spin box that ignores mouse-wheel scrolling (see NoWheelComboBox)."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()
