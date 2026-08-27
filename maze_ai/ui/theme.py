"""Monochrome (black & white) theme: palette constants + global stylesheet."""

from __future__ import annotations

from pathlib import Path

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
LOGO_PATH = str(RESOURCES / "logo.png")

# ── palette ──────────────────────────────────────────────────────────────
BG = "#050505"          # near-black window base
PANEL = "#0d0d0f"       # slightly lifted panels
PANEL_HI = "#16161a"    # hover / raised
LINE = "#26262b"        # hairline borders
TEXT = "#f2f2f4"        # primary text (near-white)
TEXT_DIM = "#8a8a92"    # secondary text
TEXT_FAINT = "#5a5a62"  # tertiary
WHITE = "#ffffff"
USER_BUBBLE = "#f4f4f6"     # user message: white bubble, dark text
USER_TEXT = "#0a0a0a"
AI_BUBBLE = "#131317"       # AI message: dark bubble, light text
DANGER = "#ff5c5c"
OK = "#7CFC9A"

# QSS applied to the whole app. Frameless card corners are painted by the
# AuroraCard widget, so most widgets here are transparent by design.
STYLESHEET = f"""
* {{
    font-family: 'Inter', 'Segoe UI', 'Noto Sans', sans-serif;
    outline: none;
    color: {TEXT};
}}

QWidget {{
    background: transparent;
    font-size: 10.5pt;
}}

QToolTip {{
    background: {PANEL_HI};
    color: {TEXT};
    border: 1px solid {LINE};
    padding: 5px 8px;
    border-radius: 6px;
}}

/* ── scrollbars ── */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 4px 2px 4px 0;
}}
QScrollBar::handle:vertical {{
    background: {LINE}; border-radius: 5px; min-height: 32px;
}}
QScrollBar::handle:vertical:hover {{ background: #3a3a42; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ── inputs ── */
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 8px 12px;
    selection-background-color: {WHITE};
    selection-color: {BG};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid #4a4a52;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    border: none;
    width: 28px;
}}
QComboBox::down-arrow {{
    image: none;
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid {TEXT};
    margin-right: 10px;
}}
QComboBox::down-arrow:hover {{ border-top-color: {WHITE}; }}
/* Dropdown popup — must be fully opaque or the list is unreadable over the
   translucent window. Use solid background-color (not the shorthand). */
QComboBox QAbstractItemView {{
    background-color: #14141a;
    border: 1px solid {LINE};
    border-radius: 8px;
    selection-background-color: {WHITE};
    selection-color: {BG};
    outline: none;
    padding: 4px;
}}
QComboBox QAbstractItemView::item {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    min-height: 26px;
    padding: 3px 8px;
    color: {TEXT};
}}
QComboBox QAbstractItemView::item:selected,
QComboBox QAbstractItemView::item:hover {{
    background-color: {WHITE};
    color: {BG};
}}

/* ── buttons ── */
QPushButton {{
    background: {PANEL_HI};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 8px 16px;
    color: {TEXT};
}}
QPushButton:hover {{ background: #202027; border-color: #3a3a42; }}
QPushButton:pressed {{ background: #0f0f13; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: {PANEL}; }}

QPushButton#primary {{
    background: {WHITE};
    color: {BG};
    border: none;
    font-weight: 700;
}}
QPushButton#primary:hover {{ background: #e2e2e6; }}
QPushButton#primary:pressed {{ background: #cfcfd4; }}
QPushButton#primary:disabled {{ background: #3a3a42; color: {TEXT_FAINT}; }}

QPushButton#danger {{ color: {DANGER}; border-color: #4a2a2a; }}
QPushButton#danger:hover {{ background: #241414; }}

/* ── titlebar window controls ── */
QToolButton#winctl {{
    background: transparent; border: none; border-radius: 8px;
    color: {TEXT_DIM}; font-size: 14pt; padding: 0;
}}
QToolButton#winctl:hover {{ background: {PANEL_HI}; color: {TEXT}; }}
QToolButton#winclose:hover {{ background: #3a1414; color: {DANGER}; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 5px;
    border: 1px solid {LINE}; background: {PANEL};
}}
QCheckBox::indicator:checked {{ background: {WHITE}; border-color: {WHITE}; }}

QLabel#h1 {{ font-size: 15pt; font-weight: 700; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#faint {{ color: {TEXT_FAINT}; font-size: 9pt; }}

QFrame#hsep {{ background: {LINE}; max-height: 1px; border: none; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_DIM};
    padding: 8px 16px; border: none; margin-right: 4px;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {WHITE}; }}
"""
