"""A small set of line icons, drawn from inline SVG at runtime.

Unicode glyphs (☰ ⚙ ✕ ＋) render in whatever font happens to cover them, at
different weights and baselines on every machine. These are one family: a
24-unit grid, 1.8 stroke, round caps — tinted per use, crisp at any DPI.
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .theme import TEXT_DIM

# Path data on a 24×24 grid. Each value is the inner SVG markup.
_PATHS: dict[str, str] = {
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
    "sidebar": '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M9.5 4.5v15"/>',
    "settings": (
        '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/>'
        '<circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>'
    ),
    "alarm": (
        '<circle cx="12" cy="13" r="7"/><path d="M12 9.5V13l2.2 1.6"/>'
        '<path d="M4.5 5.5l2.5-2M19.5 5.5l-2.5-2"/>'
    ),
    "keyboard": (
        '<rect x="2.5" y="6" width="19" height="12" rx="2.5"/>'
        '<path d="M6.5 10h.01M10 10h.01M13.5 10h.01M17 10h.01M8 14.2h8"/>'
    ),
    "close": '<path d="M6 6l12 12M18 6L6 18"/>',
    "minimize": '<path d="M6 12.5h12"/>',
    "maximize": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    "restore": (
        '<rect x="5.5" y="8.5" width="10" height="10" rx="1.5"/>'
        '<path d="M9 5.5h8a1.5 1.5 0 0 1 1.5 1.5v8"/>'
    ),
    "copy": (
        '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/>'
        '<path d="M15.5 8.5V6a1.5 1.5 0 0 0-1.5-1.5H6A1.5 1.5 0 0 0 4.5 6v8'
        'A1.5 1.5 0 0 0 6 15.5h2.5"/>'
    ),
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "refresh": (
        '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.5 4.5v4h-4"/>'
    ),
    "arrow-up": '<path d="M12 19V5.5M6 11.5l6-6 6 6"/>',
    "stop": '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/>',
    "image": (
        '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/>'
        '<circle cx="9" cy="10" r="1.6"/><path d="M20.5 16l-5-5-9 8.5"/>'
    ),
    "trash": (
        '<path d="M4.5 7h15M10 7V5h4v2M6.5 7l1 12.5h9l1-12.5"/>'
        '<path d="M10.2 11v5M13.8 11v5"/>'
    ),
    "edit": '<path d="M14.5 5.5l4 4L9 19H5v-4z"/><path d="M12.5 7.5l4 4"/>',
    "download": '<path d="M12 4.5v11M7 10.5l5 5 5-5M5 19.5h14"/>',
    "more": (
        '<circle cx="6" cy="12" r="1.2" fill="currentColor"/>'
        '<circle cx="12" cy="12" r="1.2" fill="currentColor"/>'
        '<circle cx="18" cy="12" r="1.2" fill="currentColor"/>'
    ),
    "terminal": (
        '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/>'
        '<path d="M7 9.5l3 2.5-3 2.5M12.5 15h4.5"/>'
    ),
    "file": (
        '<path d="M13.5 3.5H7A1.5 1.5 0 0 0 5.5 5v14A1.5 1.5 0 0 0 7 20.5h10'
        'a1.5 1.5 0 0 0 1.5-1.5V8.5z"/><path d="M13.5 3.5v5h5"/>'
    ),
    "globe": (
        '<circle cx="12" cy="12" r="8.5"/>'
        '<path d="M3.5 12h17M12 3.5c2.5 2.6 3.6 5.4 3.6 8.5s-1.1 5.9-3.6 8.5'
        'c-2.5-2.6-3.6-5.4-3.6-8.5s1.1-5.9 3.6-8.5z"/>'
    ),
    "sparkle": (
        '<path d="M12 3.5l1.9 5.3 5.3 1.9-5.3 1.9L12 17.9l-1.9-5.3-5.3-1.9 '
        '5.3-1.9z"/><path d="M18.5 16v4M16.5 18h4"/>'
    ),
    "shield": (
        '<path d="M12 3.5l7 2.8v5.2c0 4.4-3 7.8-7 9-4-1.2-7-4.6-7-9V6.3z"/>'
        '<path d="M9 12l2.2 2.2L15.5 10"/>'
    ),
    "chat": (
        '<path d="M5 5.5h14A1.5 1.5 0 0 1 20.5 7v8.5A1.5 1.5 0 0 1 19 17h-8.5'
        'L6 20.5V17H5a1.5 1.5 0 0 1-1.5-1.5V7A1.5 1.5 0 0 1 5 5.5z"/>'
    ),
    "chevron-down": '<path d="M6.5 9.5l5.5 5.5 5.5-5.5"/>',
    "chevron-right": '<path d="M9.5 6.5l5.5 5.5-5.5 5.5"/>',
    "warning": (
        '<path d="M12 4l9 15.5H3z"/><path d="M12 10v4.5M12 17.2h.01"/>'
    ),
    "link": (
        '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/>'
        '<path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>'
    ),
}


def names() -> list[str]:
    return sorted(_PATHS)


def _svg(name: str, color: str, stroke: float) -> bytes:
    body = _PATHS[name].replace("currentColor", color)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    ).encode()


@lru_cache(maxsize=256)
def pixmap(name: str, color: str = TEXT_DIM, size: int = 18, stroke: float = 1.8) -> QPixmap:
    """Render one icon at ``size`` logical pixels, sharp on hi-dpi screens."""
    app = QGuiApplication.instance()
    ratio = 2.0
    if app is not None:
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            ratio = max(2.0, screen.devicePixelRatio())
    px = int(size * ratio)
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(_svg(name, color, stroke)))
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, px, px))
    painter.end()
    pm.setDevicePixelRatio(ratio)
    return pm


def icon(name: str, color: str = TEXT_DIM, size: int = 18, *, hover: str = "",
         disabled: str = "", stroke: float = 1.8) -> QIcon:
    """A QIcon with optional hover (Active) and Disabled tints."""
    result = QIcon()
    result.addPixmap(pixmap(name, color, size, stroke), QIcon.Mode.Normal)
    if hover:
        result.addPixmap(pixmap(name, hover, size, stroke), QIcon.Mode.Active)
        result.addPixmap(pixmap(name, hover, size, stroke), QIcon.Mode.Selected)
    if disabled:
        result.addPixmap(pixmap(name, disabled, size, stroke), QIcon.Mode.Disabled)
    return result
