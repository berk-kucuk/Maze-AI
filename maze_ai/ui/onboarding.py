"""First-run welcome: a short intro and a nudge to pick an AI backend."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from .effects import AuroraCard
from .richtext import harden_labels
from .theme import LOGO_PATH, STYLESHEET, TEXT_DIM


class OnboardingDialog(QDialog):
    """Returns Accepted if the user wants to open Settings to configure a backend."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.resize(580, 520)
        self.setStyleSheet(STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = AuroraCard(self)
        root.addWidget(card)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(card.margin + 30, card.margin + 28,
                               card.margin + 30, card.margin + 24)
        lay.setSpacing(14)

        logo = QLabel()
        pix = QPixmap(LOGO_PATH)
        if not pix.isNull():
            logo.setPixmap(pix.scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        logo.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(logo)

        title = QLabel(tr("Welcome to Maze AI"))
        title.setObjectName("h1")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(title)

        body = QLabel(tr(
            "Your private, agentic assistant for Maze Linux. It can run commands, "
            "manage files, launch apps, search the web, take screenshots and set "
            "reminders — right from this window.\n\n"
            "First, choose where the model runs:\n"
            "• **Ollama** — fully local & private (download a model)\n"
            "• **Gemini** or an **OpenAI-compatible** API — hosted, just add a key\n\n"
            "You stay in control: in **Ask** mode every action is confirmed, and "
            "destructive commands always require approval."
        ))
        body.setTextFormat(Qt.TextFormat.MarkdownText)
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10pt;")
        lay.addWidget(body)

        lay.addStretch(1)

        buttons = QHBoxLayout()
        skip = QPushButton(tr("Skip for now"))
        skip.clicked.connect(self.reject)
        buttons.addWidget(skip)
        buttons.addStretch(1)
        go = QPushButton(tr("Choose a backend"))
        go.setObjectName("primary")
        go.setMinimumWidth(160)
        go.clicked.connect(self.accept)
        buttons.addWidget(go)
        lay.addLayout(buttons)
        go.setDefault(True)
        go.setFocus()
        harden_labels(self)
