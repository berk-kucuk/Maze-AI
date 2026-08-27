"""System tray icon with show / settings / quit actions."""

from __future__ import annotations

from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from ..i18n import tr
from .theme import LOGO_PATH


class Tray(QSystemTrayIcon):
    def __init__(self, window, app: QApplication) -> None:
        super().__init__(QIcon(LOGO_PATH), app)
        self.window = window
        self.setToolTip("Maze AI")

        menu = QMenu()
        show = QAction(tr("Open Maze AI"), menu)
        show.triggered.connect(self.show_window)
        menu.addAction(show)

        quick = QAction(tr("Quick Ask"), menu)
        quick.triggered.connect(lambda: self.window.quick_ask())
        menu.addAction(quick)

        clip = QAction(tr("Ask about the clipboard"), menu)
        clip.triggered.connect(lambda: self.window.quick_ask(mode="clipboard"))
        menu.addAction(clip)

        shot = QAction(tr("Ask about a screen area"), menu)
        shot.triggered.connect(lambda: self.window.quick_ask(mode="screenshot"))
        menu.addAction(shot)

        menu.addSeparator()
        settings = QAction(tr("Settings"), menu)
        settings.triggered.connect(self._open_settings)
        menu.addAction(settings)

        menu.addSeparator()
        quit_action = QAction(tr("Quit"), menu)
        quit_action.triggered.connect(window.quit_app)
        menu.addAction(quit_action)

        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_window()

    def show_window(self) -> None:
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def _open_settings(self) -> None:
        self.show_window()
        self.window.open_settings()

    def notify(self, title: str, message: str) -> None:
        self.showMessage(title, message, QIcon(LOGO_PATH), 4000)
