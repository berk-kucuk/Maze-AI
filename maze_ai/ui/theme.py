"""Maze monochrome design system: tokens + the global stylesheet.

Everything visual is derived from the tokens below, so the whole app shares
one spacing scale, one set of radii and one type ramp. The look is OLED
black with a restrained grey ladder; white is reserved for the one primary
action on screen, and colour appears only where it carries meaning
(success, danger, a link).
"""

from __future__ import annotations

from pathlib import Path

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
LOGO_PATH = str(RESOURCES / "logo.png")
# QSS needs files for its images; forward slashes work on every platform.
_CHEVRON = (RESOURCES / "chevron-down.svg").as_posix()
_CHECK = (RESOURCES / "check.svg").as_posix()

# ── surfaces ─────────────────────────────────────────────────────────────
BG = "#050506"            # window base
SIDEBAR = "#09090b"       # the history rail, one step above the base
PANEL = "#0e0e11"         # inputs, cards
PANEL_HI = "#16161a"      # hover / raised
PANEL_TOP = "#1d1d22"     # pressed / selected / user bubble
LINE = "#232329"          # hairline borders
LINE_HI = "#34343c"       # focused / hovered borders

# ── text ─────────────────────────────────────────────────────────────────
TEXT = "#ededf0"          # primary
TEXT_DIM = "#9a9aa4"      # secondary
TEXT_FAINT = "#5f5f69"    # tertiary, hints
WHITE = "#ffffff"

# ── semantic ─────────────────────────────────────────────────────────────
ACCENT = "#ffffff"        # the primary action
DANGER = "#ff6b6b"
DANGER_BG = "#2a1215"
OK = "#6ee7a0"
WARN = "#f5c451"
LINK = "#9ec5ff"          # links must read as links, even in monochrome

# ── messages ─────────────────────────────────────────────────────────────
USER_BUBBLE = PANEL_TOP
USER_TEXT = TEXT
AI_BUBBLE = "transparent"

# ── type ─────────────────────────────────────────────────────────────────
FONT_UI = "'Inter', 'Inter Variable', 'Segoe UI', 'Noto Sans', sans-serif"
FONT_MONO = "'JetBrains Mono', 'Fira Code', 'DejaVu Sans Mono', monospace"

# ── radii ────────────────────────────────────────────────────────────────
R_SM = 6
R_MD = 10
R_LG = 14
R_XL = 20

# QSS applied to the whole app (the QApplication and every top-level window).
# Frameless card corners are painted by AuroraCard, so containers here are
# transparent by design and only controls get a surface.
STYLESHEET = f"""
* {{
    font-family: {FONT_UI};
    outline: none;
    color: {TEXT};
}}

QWidget {{
    background: transparent;
    font-size: 10.5pt;
}}

QToolTip {{
    background-color: {PANEL_HI};
    color: {TEXT};
    border: 1px solid {LINE_HI};
    padding: 6px 9px;
    border-radius: {R_SM}px;
    font-size: 9pt;
}}

/* ── menus (context menus, tray) — opaque, or they are unreadable ── */
QMenu {{
    background-color: #121216;
    border: 1px solid {LINE_HI};
    border-radius: {R_MD}px;
    padding: 6px;
}}
QMenu::item {{
    background-color: transparent;
    padding: 7px 28px 7px 12px;
    border-radius: {R_SM}px;
    color: {TEXT};
}}
QMenu::item:selected {{ background-color: {PANEL_TOP}; }}
QMenu::item:disabled {{ color: {TEXT_FAINT}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 5px 8px; }}

/* ── scrollbars: thin, quiet, grow on hover ── */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px 1px 2px 0;
}}
QScrollBar::handle:vertical {{
    background: {LINE}; border-radius: 4px; min-height: 36px; margin: 0 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {LINE_HI}; margin: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 0 2px 1px 2px;
}}
QScrollBar::handle:horizontal {{
    background: {LINE}; border-radius: 4px; min-width: 36px; margin: 2px 0;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ── inputs ── */
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: {R_MD}px;
    padding: 8px 12px;
    selection-background-color: #3b3b44;
    selection-color: {WHITE};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover {{ border-color: {LINE_HI}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid #5a5a64;
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {TEXT_FAINT}; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    border: none;
    width: 28px;
}}
QComboBox::down-arrow {{
    image: url("{_CHEVRON}");
    width: 12px;
    height: 12px;
    margin-right: 10px;
}}
/* Dropdown popup — must be fully opaque or the list is unreadable over the
   translucent window. Use solid background-color (not the shorthand). */
QComboBox QAbstractItemView {{
    background-color: #121216;
    border: 1px solid {LINE_HI};
    border-radius: {R_MD}px;
    selection-background-color: {PANEL_TOP};
    selection-color: {WHITE};
    outline: none;
    padding: 4px;
}}
QComboBox QAbstractItemView::item {{
    background-color: transparent;
    border: none;
    border-radius: {R_SM}px;
    min-height: 28px;
    padding: 3px 8px;
    color: {TEXT};
}}
QComboBox QAbstractItemView::item:selected,
QComboBox QAbstractItemView::item:hover {{
    background-color: {PANEL_TOP};
    color: {WHITE};
}}

/* ── buttons ── */
QPushButton {{
    background: {PANEL_HI};
    border: 1px solid {LINE};
    border-radius: {R_MD}px;
    padding: 8px 16px;
    color: {TEXT};
    font-weight: 500;
}}
QPushButton:hover {{ background: #1f1f25; border-color: {LINE_HI}; }}
QPushButton:pressed {{ background: #101014; }}
QPushButton:focus {{ border-color: #6a6a74; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: {PANEL}; border-color: {LINE}; }}

QPushButton#primary {{
    background: {ACCENT};
    color: {BG};
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton#primary:hover {{ background: #e6e6ea; border-color: #e6e6ea; }}
QPushButton#primary:pressed {{ background: #cfcfd4; }}
QPushButton#primary:focus {{ border: 2px solid #8a8a94; }}
QPushButton#primary:disabled {{ background: #2c2c33; border-color: #2c2c33; color: {TEXT_FAINT}; }}

QPushButton#danger {{ color: {DANGER}; border-color: #4a2328; background: transparent; }}
QPushButton#danger:hover {{ background: {DANGER_BG}; border-color: #6b2c33; }}

QPushButton#ghost {{ background: transparent; border: 1px solid transparent; color: {TEXT_DIM}; }}
QPushButton#ghost:hover {{ background: {PANEL_HI}; color: {TEXT}; }}

QPushButton#chip {{
    background: rgba(255,255,255,0.04);
    border: 1px solid {LINE};
    border-radius: 13px;
    padding: 4px 12px;
    color: {TEXT_DIM};
    font-size: 9pt;
    font-weight: 500;
}}
QPushButton#chip:hover {{ background: rgba(255,255,255,0.08); color: {TEXT}; border-color: {LINE_HI}; }}

/* ── icon buttons (title bar, toolbars) ── */
QToolButton#winctl, QToolButton#winclose, QToolButton#icon {{
    background: transparent; border: none; border-radius: {R_SM + 2}px;
    color: {TEXT_DIM}; font-size: 13pt; padding: 0;
}}
QToolButton#winctl:hover, QToolButton#icon:hover {{ background: {PANEL_HI}; color: {TEXT}; }}
QToolButton#winctl:pressed, QToolButton#icon:pressed {{ background: {PANEL_TOP}; }}
QToolButton#winctl:checked, QToolButton#icon:checked {{ background: {PANEL_HI}; color: {TEXT}; }}
QToolButton#winclose:hover {{ background: #c42b1c; color: {WHITE}; }}
QToolButton#winctl:disabled, QToolButton#icon:disabled {{ color: #34343a; }}

/* Small text actions under messages and code blocks. */
QToolButton#action {{
    background: transparent; border: none; border-radius: {R_SM}px;
    color: {TEXT_FAINT}; font-size: 9pt; padding: 3px 7px;
}}
QToolButton#action:hover {{ color: {TEXT}; background: rgba(255,255,255,0.07); }}

QCheckBox {{ spacing: 9px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 5px;
    border: 1px solid {LINE_HI}; background: {PANEL};
}}
QCheckBox::indicator:hover {{ border-color: #5a5a64; }}
QCheckBox::indicator:checked {{
    background: {WHITE}; border-color: {WHITE}; image: url("{_CHECK}");
}}

QProgressBar {{
    background: {PANEL}; border: 1px solid {LINE}; border-radius: 5px;
    height: 8px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {WHITE}; border-radius: 4px; }}

QLabel#h1 {{ font-size: 16pt; font-weight: 700; letter-spacing: -0.2px; }}
QLabel#h2 {{ font-size: 12pt; font-weight: 600; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#faint {{ color: {TEXT_FAINT}; font-size: 9pt; }}
QLabel#section {{
    color: {TEXT_FAINT}; font-size: 8pt; font-weight: 700; letter-spacing: 0.8px;
}}
QLabel#kbd {{
    background: {PANEL_HI}; border: 1px solid {LINE_HI}; border-bottom-width: 2px;
    border-radius: 5px; padding: 1px 6px; color: {TEXT_DIM};
    font-family: {FONT_MONO}; font-size: 8.5pt;
}}

QFrame#hsep {{ background: {LINE}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#vsep {{ background: {LINE}; max-width: 1px; min-width: 1px; border: none; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_DIM};
    padding: 8px 16px; border: none; margin-right: 4px;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {WHITE}; }}
"""
