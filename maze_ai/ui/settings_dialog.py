"""Settings dialog — backend selection, API keys, models, agent behaviour."""

from __future__ import annotations

import json
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import autostart, desktop_integration
from ..agent.tools import TOOL_GROUPS, tool_schemas, tools_for_groups
from ..config import LANGUAGES, MODE_ASK, MODE_AUTO, MODE_CHAT, Config
from ..i18n import UI_LANGUAGES, tr
from ..llm import GeminiBackend, OllamaBackend, OpenAIBackend
from ..llm.gemini_backend import KNOWN_MODELS as GEMINI_MODELS
from ..llm.hardware import GB, describe_hardware
from ..llm.ollama_backend import POPULAR_MODELS as OLLAMA_POPULAR
from ..llm.ollama_backend import short_size
from ..llm.openai_backend import KNOWN_MODELS as OPENAI_MODELS
from ..llm.openai_backend import PRESETS as OPENAI_PRESETS
from .effects import AuroraCard
from .theme import LINE, STYLESHEET, TEXT, TEXT_DIM, TEXT_FAINT
from .widgets import NoWheelComboBox, NoWheelSpinBox
from .worker import BenchmarkWorker, OllamaPullWorker


def _label(text: str) -> QLabel:
    lbl = QLabel(tr(text))
    lbl.setStyleSheet(
        f"color: {TEXT_DIM}; font-size: 9pt; font-weight: 700; "
        "letter-spacing: 0.4px; background: transparent;"
    )
    return lbl


def _section(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    """A titled settings card; returns the frame and its content layout."""
    frame = QFrame()
    frame.setObjectName("card")
    frame.setStyleSheet(
        f"QFrame#card {{ background: rgba(255,255,255,0.022); "
        f"border: 1px solid {LINE}; border-radius: 14px; }}"
    )
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(18, 15, 18, 17)
    lay.setSpacing(9)
    head = QLabel(tr(title))
    head.setStyleSheet(
        f"font-size: 11.5pt; font-weight: 700; color: {TEXT}; background: transparent;"
    )
    lay.addWidget(head)
    if subtitle:
        sub = QLabel(tr(subtitle))
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 9pt; background: transparent;")
        lay.addWidget(sub)
    return frame, lay


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.resize(600, 680)
        self.setMinimumSize(520, 520)
        self.setStyleSheet(STYLESHEET)
        self._drag = None
        #: name -> capability/size details, filled by the model refresh.
        self._model_details: dict[str, dict] = {}
        self._confirm_delete = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = AuroraCard(self)
        root.addWidget(card)

        outer = QVBoxLayout(card)
        outer.setContentsMargins(34, 30, 34, 30)
        outer.setSpacing(14)

        # ── header ───────────────────────────────────────────────────────
        header = QHBoxLayout()
        htext = QVBoxLayout()
        htext.setSpacing(2)
        title = QLabel(tr("Settings"))
        title.setObjectName("h1")
        htext.addWidget(title)
        subtitle = QLabel(tr("Backends, models and agent behaviour"))
        subtitle.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 9.5pt;")
        htext.addWidget(subtitle)
        header.addLayout(htext)
        header.addStretch(1)
        close = QToolButton()
        close.setObjectName("winclose")
        close.setText("✕")
        close.setFixedSize(34, 34)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(self.reject)
        header.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(header)

        # ── scrollable body ──────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        form = QVBoxLayout(body)
        form.setContentsMargins(0, 0, 6, 0)
        form.setSpacing(14)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # ── backend section ──────────────────────────────────────────────
        sec_backend, lay = _section(
            "AI Backend", "Choose where the model runs. Switch any time."
        )
        self.backend_box = NoWheelComboBox()
        self.backend_box.addItem(tr("Ollama  ·  local, private models"), "ollama")
        self.backend_box.addItem(tr("Gemini  ·  Google hosted API"), "gemini")
        self.backend_box.addItem(
            tr("OpenAI-compatible  ·  OpenAI, OpenRouter, Groq, LM Studio…"), "openai"
        )
        self.backend_box.currentIndexChanged.connect(self._on_backend_changed)
        lay.addWidget(self.backend_box)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._ollama_pane())
        self.stack.addWidget(self._gemini_pane())
        self.stack.addWidget(self._openai_pane())
        lay.addWidget(self.stack)
        form.addWidget(sec_backend)

        # ── agent section ────────────────────────────────────────────────
        sec_agent, lay = _section(
            "Agent", "How much freedom the assistant has to act on your machine."
        )
        self.mode_box = NoWheelComboBox()
        self.mode_box.addItem(tr("Ask before acting  ·  recommended"), MODE_ASK)
        self.mode_box.addItem(tr("Autonomous  ·  run tools without asking"), MODE_AUTO)
        self.mode_box.addItem(tr("Chat only  ·  no tools"), MODE_CHAT)
        self.mode_box.currentIndexChanged.connect(self._update_mode_desc)
        lay.addWidget(self.mode_box)
        self.mode_desc = QLabel("")
        self.mode_desc.setWordWrap(True)
        self.mode_desc.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        lay.addWidget(self.mode_desc)

        lay.addSpacing(4)
        lay.addWidget(_label("RESPONSE LANGUAGE"))
        self.lang_box = NoWheelComboBox()
        for code, label in LANGUAGES:
            self.lang_box.addItem(label, code)
        lay.addWidget(self.lang_box)

        lay.addSpacing(4)
        lay.addWidget(_label("INTERFACE LANGUAGE"))
        self.ui_lang_box = NoWheelComboBox()
        for code, label in UI_LANGUAGES:
            self.ui_lang_box.addItem(tr(label), code)
        self.ui_lang_box.setToolTip(tr("Takes effect the next time Maze AI starts."))
        lay.addWidget(self.ui_lang_box)
        ui_lang_hint = QLabel(tr("Takes effect the next time Maze AI starts."))
        ui_lang_hint.setObjectName("faint")
        lay.addWidget(ui_lang_hint)

        row = QHBoxLayout()
        row.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(5)
        left.addWidget(_label("MAX STEPS"))
        self.steps_spin = NoWheelSpinBox()
        self.steps_spin.setRange(1, 50)
        left.addWidget(self.steps_spin)
        right = QVBoxLayout()
        right.setSpacing(5)
        right.addWidget(_label("COMMAND TIMEOUT (S)"))
        self.timeout_spin = NoWheelSpinBox()
        self.timeout_spin.setRange(5, 3600)
        right.addWidget(self.timeout_spin)
        row.addLayout(left)
        row.addLayout(right)
        lay.addSpacing(2)
        lay.addLayout(row)
        form.addWidget(sec_agent)

        # ── advanced / behaviour section ─────────────────────────────────
        sec_adv, lay = _section(
            "Behaviour & safety", "Streaming, custom instructions and command guards."
        )
        self.stream_check = QCheckBox(tr("Stream responses as they're generated"))
        lay.addWidget(self.stream_check)
        self.block_danger_check = QCheckBox(
            tr("Always confirm destructive commands (even in autonomous mode)")
        )
        lay.addWidget(self.block_danger_check)
        self.auto_readonly_check = QCheckBox(
            tr("Auto-approve safe read-only commands (skip the prompt)")
        )
        lay.addWidget(self.auto_readonly_check)
        self.native_tools_check = QCheckBox(
            tr("Use the model's own tool calling when it supports it")
        )
        self.native_tools_check.setToolTip(
            tr("Native function calling is far more reliable than asking a model to "
               "hand-write JSON, and it keeps the tool catalogue out of the context "
               "window. Falls back to the JSON protocol automatically.")
        )
        lay.addWidget(self.native_tools_check)
        self.constrain_json_check = QCheckBox(
            tr("Force valid JSON on models without tool calling")
        )
        self.constrain_json_check.setToolTip(
            tr("Constrains generation to the protocol schema, so a small local model "
               "cannot produce truncated or fenced JSON.")
        )
        lay.addWidget(self.constrain_json_check)
        self.guard_secrets_check = QCheckBox(
            tr("Always confirm access to keys, tokens and private history")
        )
        self.guard_secrets_check.setToolTip(
            tr("Reading ~/.ssh, ~/.gnupg, .env files, browser profiles or shell "
            "history always asks first — in every mode. This is the step a "
            "malicious web page would need to steal your credentials.")
        )
        lay.addWidget(self.guard_secrets_check)
        self.egress_check = QCheckBox(tr("Confirm network requests that carry data out"))
        self.egress_check.setToolTip(
            tr("A fetch whose URL carries a payload (or that saves the download to "
            "a file) needs approval, so content the assistant read cannot talk "
            "it into sending your data somewhere.")
        )
        lay.addWidget(self.egress_check)

        lay.addSpacing(6)
        lay.addWidget(_label("TOOLS THE AGENT MAY USE"))
        self.group_boxes: dict[str, QCheckBox] = {}
        group_labels = {
            "shell": "Shell commands and launching apps",
            "files": "Reading and changing files",
            "web": "Web search and fetching pages",
            "desktop": "Screenshots, OCR, clipboard, notifications",
            "reminders": "Reminders and to-dos",
        }
        for group in TOOL_GROUPS:
            box = QCheckBox(tr(group_labels.get(group, group)))
            box.toggled.connect(self._update_tool_cost)
            self.group_boxes[group] = box
            lay.addWidget(box)
        self.tool_cost = QLabel("")
        self.tool_cost.setObjectName("faint")
        self.tool_cost.setWordWrap(True)
        lay.addWidget(self.tool_cost)

        lay.addSpacing(4)
        lay.addWidget(_label("CUSTOM INSTRUCTIONS"))
        self.custom_instructions = QPlainTextEdit()
        self.custom_instructions.setPlaceholderText(
            "Standing instructions for every chat — e.g. “Always use the fish shell”, "
            "“Prefer concise answers”, project context…"
        )
        self.custom_instructions.setFixedHeight(90)
        lay.addWidget(self.custom_instructions)
        form.addWidget(sec_adv)

        # ── shortcuts & terminal ─────────────────────────────────────────
        sec_short, lay = _section(
            "Shortcuts & terminal",
            "Bind these to a key in your desktop's shortcut settings, and the "
            "assistant is one keystroke away from anywhere.",
        )
        for command, description in (
            ("maze-ai --ask", "Quick Ask — a floating question bar"),
            ("maze-ai --clipboard", "Ask about whatever you just copied"),
            ("maze-ai --screenshot", "Drag a box on screen and ask about it"),
        ):
            lay.addLayout(self._command_row(command, description))

        lay.addSpacing(6)
        lay.addWidget(_label("SHELL INTEGRATION"))
        shell_hint = QLabel(tr(
            "Adds `mz` (ask), `mzask` (answer in the terminal) and `mzfix` — "
            "which puts the corrected version of your last failed command "
            "straight onto your prompt."
        ))
        shell_hint.setWordWrap(True)
        shell_hint.setObjectName("faint")
        lay.addWidget(shell_hint)
        for command, description in (
            ('eval "$(maze-ai --shell-init zsh)"', "~/.zshrc"),
            ('eval "$(maze-ai --shell-init bash)"', "~/.bashrc"),
            ("maze-ai --shell-init fish | source", "~/.config/fish/config.fish"),
        ):
            lay.addLayout(self._command_row(command, description))
        lay.addSpacing(6)
        lay.addWidget(_label("FILE MANAGER"))
        fm_hint = QLabel(tr(
            "Add “Ask Maze AI” to the right-click menu of your file manager."
        ))
        fm_hint.setWordWrap(True)
        fm_hint.setObjectName("faint")
        lay.addWidget(fm_hint)
        self.fm_status = QLabel(desktop_integration.describe())
        self.fm_status.setWordWrap(True)
        self.fm_status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        lay.addWidget(self.fm_status)
        fm_row = QHBoxLayout()
        fm_row.setSpacing(8)
        self.fm_install = QPushButton(tr("Add to file manager"))
        self.fm_install.clicked.connect(self._install_menus)
        fm_row.addWidget(self.fm_install)
        self.fm_remove = QPushButton(tr("Remove from file manager"))
        self.fm_remove.clicked.connect(self._remove_menus)
        fm_row.addWidget(self.fm_remove)
        fm_row.addStretch(1)
        lay.addLayout(fm_row)
        form.addWidget(sec_short)

        # ── window section ───────────────────────────────────────────────
        sec_win, lay = _section("Window & notifications")
        self.autostart_check = QCheckBox(tr("Start Maze AI on login (in the tray)"))
        lay.addWidget(self.autostart_check)
        self.tray_check = QCheckBox(tr("Closing the window hides it to the system tray"))
        lay.addWidget(self.tray_check)
        self.greet_check = QCheckBox(tr("Greet me with a notification on startup"))
        lay.addWidget(self.greet_check)
        form.addWidget(sec_win)

        form.addStretch(1)

        # ── footer ───────────────────────────────────────────────────────
        sep = QFrame()
        sep.setObjectName("hsep")
        outer.addWidget(sep)
        footer = QHBoxLayout()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        footer.addWidget(self.status, 1)
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.reject)
        footer.addWidget(cancel)
        save = QPushButton(tr("Save"))
        save.setObjectName("primary")
        save.setMinimumWidth(96)
        save.clicked.connect(self._save)
        footer.addWidget(save)
        outer.addLayout(footer)

        # Keep long combo items (esp. the model picker) from forcing the whole
        # dialog wider than its viewport — size to a small minimum and let the
        # layout stretch them to the available width instead.
        for cb in self.findChildren(QComboBox):
            cb.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            cb.setMinimumContentsLength(8)
            cb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            # Force the dropdown popup opaque (it renders over a translucent
            # window, so a transparent view would be unreadable).
            view = cb.view()
            view.setStyleSheet(
                f"background-color: #14141a; color: {TEXT}; "
                f"border: 1px solid {LINE};"
            )
            popup = view.window()
            if popup is not None:
                popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
                popup.setAutoFillBackground(True)

        self._load()

    # ── file manager menus ───────────────────────────────────────────────
    def _install_menus(self) -> None:
        results = desktop_integration.install()
        if not results:
            self.status.setText(tr("No supported file manager found."))
            return
        failures = {k: v for k, v in results.items() if v}
        added = [k for k, v in results.items() if not v]
        if added:
            self.status.setText(
                tr("Added to: {managers}").format(managers=", ".join(added))
                + "  " + tr("Restart your file manager to see the new menu.")
            )
        for error in failures.values():
            self.status.setText(tr("Could not add the menu: {error}").format(error=error))
        self.fm_status.setText(desktop_integration.describe())

    def _remove_menus(self) -> None:
        removed = desktop_integration.uninstall()
        self.status.setText(
            tr("Removed from: {managers}").format(managers=", ".join(removed))
            if removed else tr("Nothing to remove.")
        )
        self.fm_status.setText(desktop_integration.describe())

    def _command_row(self, command: str, description: str) -> QHBoxLayout:
        """A copyable command line with a plain-language label."""
        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(command)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setStyleSheet(
            f"color: {TEXT}; font-family: 'JetBrains Mono','DejaVu Sans Mono',monospace;"
            f"font-size: 9pt; background: rgba(255,255,255,0.04);"
            f"border: 1px solid {LINE}; border-radius: 8px; padding: 5px 9px;"
        )
        row.addWidget(label)
        note = QLabel(tr(description))
        note.setWordWrap(True)
        note.setObjectName("faint")
        row.addWidget(note, 1)
        copy = QToolButton()
        copy.setText(tr("⧉ Copy"))
        copy.setCursor(Qt.CursorShape.PointingHandCursor)
        copy.setStyleSheet(
            f"QToolButton {{ background: transparent; border: none; color: {TEXT_FAINT};"
            "font-size: 8.5pt; padding: 2px 6px; border-radius: 6px; }"
            f"QToolButton:hover {{ color: {TEXT}; background: rgba(255,255,255,0.06); }}"
        )

        def _copy() -> None:
            QGuiApplication.clipboard().setText(command)
            copy.setText(tr("✓ Copied"))
            QTimer.singleShot(1200, lambda: copy.setText(tr("⧉ Copy")))

        copy.clicked.connect(_copy)
        row.addWidget(copy)
        return row

    # ── panes ────────────────────────────────────────────────────────────
    def _ollama_pane(self) -> QWidget:
        pane = QWidget()
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(8)
        lay.addWidget(_label("OLLAMA HOST"))
        self.ollama_host = QLineEdit()
        self.ollama_host.setPlaceholderText("http://localhost:11434")
        lay.addWidget(self.ollama_host)

        # ── installed model selection ──
        lay.addWidget(_label("ACTIVE MODEL  (installed)"))
        mrow = QHBoxLayout()
        self.ollama_model = NoWheelComboBox()
        self.ollama_model.setEditable(True)
        self.ollama_model.currentTextChanged.connect(self._describe_ollama_model)
        mrow.addWidget(self.ollama_model, 1)
        refresh = QPushButton("↻")
        refresh.setFixedWidth(42)
        refresh.setToolTip(tr("List installed models"))
        refresh.clicked.connect(self._refresh_ollama_models)
        mrow.addWidget(refresh)
        self.ollama_delete = QPushButton("🗑")
        self.ollama_delete.setFixedWidth(42)
        self.ollama_delete.setToolTip(tr("Delete this model from disk"))
        self.ollama_delete.clicked.connect(self._delete_ollama_model)
        mrow.addWidget(self.ollama_delete)
        lay.addLayout(mrow)

        # What this model can actually do — capabilities decide whether the
        # agent can use native tool calling or has to fall back to JSON.
        self.model_caps = QLabel("")
        self.model_caps.setWordWrap(True)
        self.model_caps.setObjectName("faint")
        lay.addWidget(self.model_caps)

        # Where it will run. On a local box this is the single most useful
        # fact in the dialog: VRAM decides whether answers take 3s or 30s.
        self.hardware_line = QLabel(describe_hardware())
        self.hardware_line.setWordWrap(True)
        self.hardware_line.setObjectName("faint")
        lay.addWidget(self.hardware_line)

        self.fit_line = QLabel("")
        self.fit_line.setWordWrap(True)
        self.fit_line.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        lay.addWidget(self.fit_line)

        runtime_row = QHBoxLayout()
        runtime_row.setSpacing(8)
        self.runtime_line = QLabel("")
        self.runtime_line.setWordWrap(True)
        self.runtime_line.setObjectName("faint")
        runtime_row.addWidget(self.runtime_line, 1)
        self.bench_btn = QPushButton(tr("Measure"))
        self.bench_btn.setToolTip(
            tr("Run a short generation and report load time, prefill and "
               "generation speed, and whether the model stayed on the GPU.")
        )
        self.bench_btn.clicked.connect(self._benchmark_model)
        runtime_row.addWidget(self.bench_btn)
        self.unload_btn = QPushButton(tr("Unload"))
        self.unload_btn.setToolTip(tr("Drop the model from memory and free the VRAM"))
        self.unload_btn.clicked.connect(self._unload_model)
        runtime_row.addWidget(self.unload_btn)
        lay.addLayout(runtime_row)

        # ── context window ──
        lay.addWidget(_label("CONTEXT WINDOW  (tokens)"))
        self.ctx_auto = QCheckBox(tr("Auto — from the model's limit and this machine's RAM"))
        self.ctx_auto.toggled.connect(self._on_ctx_auto)
        lay.addWidget(self.ctx_auto)
        self.ollama_ctx = NoWheelSpinBox()
        self.ollama_ctx.setRange(2048, 131072)
        self.ollama_ctx.setSingleStep(1024)
        self.ollama_ctx.setToolTip(
            tr("How many tokens the model can hold at once. Raise this if you hit "
            "'exceeds the available context size' errors (needs more RAM/VRAM). "
            "8192 is a safe default; images and long chats need more.")
        )
        lay.addWidget(self.ollama_ctx)

        # ── keeping the model warm ──
        lay.addWidget(_label("KEEP THE MODEL IN MEMORY"))
        self.keep_alive_box = NoWheelComboBox()
        for label, value in (
            ("5 minutes", "5m"), ("30 minutes", "30m"), ("2 hours", "2h"),
            ("Forever (until you quit Ollama)", "-1"), ("Never — unload after each reply", "0"),
        ):
            self.keep_alive_box.addItem(tr(label), value)
        self.keep_alive_box.setToolTip(
            tr("A cold model spends 5–15 seconds loading before its first token. "
               "Keeping it resident trades RAM for an instant reply.")
        )
        lay.addWidget(self.keep_alive_box)
        self.preload_check = QCheckBox(tr("Load the model at startup, before the first question"))
        lay.addWidget(self.preload_check)
        self.think_check = QCheckBox(tr("Let thinking models reason first (slower, often better)"))
        lay.addWidget(self.think_check)

        lay.addWidget(_label("GPU LAYERS  (0 = let Ollama decide)"))
        self.num_gpu = NoWheelSpinBox()
        self.num_gpu.setRange(0, 999)
        self.num_gpu.setToolTip(
            tr("Forces how many layers are offloaded to the GPU. Leave at 0 unless "
               "Ollama's own estimate is wrong: raise it to use VRAM it left idle, "
               "lower it if loading fails with an out-of-memory error.")
        )
        lay.addWidget(self.num_gpu)

        # ── download a new model ──
        sep = QFrame()
        sep.setObjectName("hsep")
        lay.addSpacing(4)
        lay.addWidget(sep)
        lay.addWidget(_label("DOWNLOAD A MODEL"))
        drow = QHBoxLayout()
        self.pull_box = NoWheelComboBox()
        self.pull_box.setEditable(True)
        for tag, desc in OLLAMA_POPULAR:
            self.pull_box.addItem(f"{tag}   —   {desc}", tag)
        self.pull_box.setCurrentIndex(0)
        drow.addWidget(self.pull_box, 1)
        self.pull_btn = QPushButton(tr("Download"))
        self.pull_btn.setFixedWidth(110)
        self.pull_btn.clicked.connect(self._start_pull)
        drow.addWidget(self.pull_btn)
        lay.addLayout(drow)

        self.pull_bar = QProgressBar()
        self.pull_bar.setTextVisible(True)
        self.pull_bar.setFixedHeight(16)
        self.pull_bar.hide()
        lay.addWidget(self.pull_bar)
        self.pull_status = QLabel("")
        self.pull_status.setObjectName("faint")
        lay.addWidget(self.pull_status)

        browse = QLabel(tr("Browse more at ollama.com/library"))
        browse.setObjectName("faint")
        lay.addWidget(browse)
        return pane

    def _gemini_pane(self) -> QWidget:
        pane = QWidget()
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(8)
        lay.addWidget(_label("GEMINI API KEY"))
        krow = QHBoxLayout()
        self.gemini_key = QLineEdit()
        self.gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key.setPlaceholderText("AIza…")
        krow.addWidget(self.gemini_key, 1)
        self.reveal = QToolButton()
        self.reveal.setText("👁")
        self.reveal.setCheckable(True)
        self.reveal.setFixedWidth(42)
        self.reveal.toggled.connect(
            lambda on: self.gemini_key.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
            )
        )
        krow.addWidget(self.reveal)
        lay.addLayout(krow)

        hint = QLabel(tr("Get a key at aistudio.google.com/apikey"))
        hint.setObjectName("faint")
        lay.addWidget(hint)

        lay.addWidget(_label("MODEL"))
        mrow = QHBoxLayout()
        self.gemini_model = NoWheelComboBox()
        self.gemini_model.setEditable(True)
        mrow.addWidget(self.gemini_model, 1)
        refresh = QPushButton("↻")
        refresh.setFixedWidth(42)
        refresh.setToolTip(tr("Fetch available models"))
        refresh.clicked.connect(self._refresh_gemini_models)
        mrow.addWidget(refresh)
        lay.addLayout(mrow)
        return pane

    def _openai_pane(self) -> QWidget:
        pane = QWidget()
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(8)

        lay.addWidget(_label("API BASE URL"))
        self.openai_base = NoWheelComboBox()
        self.openai_base.setEditable(True)
        for name, url in OPENAI_PRESETS:
            self.openai_base.addItem(f"{url}   —   {name}", url)
        lay.addWidget(self.openai_base)

        lay.addWidget(_label("API KEY  (blank for most local servers)"))
        krow = QHBoxLayout()
        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key.setPlaceholderText("sk-…")
        krow.addWidget(self.openai_key, 1)
        reveal = QToolButton()
        reveal.setText("👁")
        reveal.setCheckable(True)
        reveal.setFixedWidth(42)
        reveal.toggled.connect(
            lambda on: self.openai_key.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
            )
        )
        krow.addWidget(reveal)
        lay.addLayout(krow)

        lay.addWidget(_label("MODEL"))
        mrow = QHBoxLayout()
        self.openai_model = NoWheelComboBox()
        self.openai_model.setEditable(True)
        mrow.addWidget(self.openai_model, 1)
        refresh = QPushButton("↻")
        refresh.setFixedWidth(42)
        refresh.setToolTip(tr("Fetch available models"))
        refresh.clicked.connect(self._refresh_openai_models)
        mrow.addWidget(refresh)
        lay.addLayout(mrow)

        hint = QLabel(tr("Works with any OpenAI-compatible /v1 endpoint."))
        hint.setObjectName("faint")
        lay.addWidget(hint)
        return pane

    def _current_openai_base(self) -> str:
        data = self.openai_base.currentData()
        if data:
            return str(data)
        return self.openai_base.currentText().split("—")[0].strip()

    def _refresh_openai_models(self) -> None:
        self.status.setText(tr("Querying the API…"))
        backend = OpenAIBackend(
            api_key=self.openai_key.text().strip(),
            base_url=self._current_openai_base(),
        )
        models = backend.available_models()
        current = self.openai_model.currentText()
        self.openai_model.clear()
        self.openai_model.addItems(models)
        if current:
            self.openai_model.setCurrentText(current)
        self.status.setText(f"Found {len(models)} model(s).")

    # ── logic ────────────────────────────────────────────────────────────
    def _on_backend_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    # ── model details ────────────────────────────────────────────────────
    def _ollama_backend(self, model: str = "") -> OllamaBackend:
        """A reused probe backend, so its capability cache survives clicks."""
        host = self.ollama_host.text().strip() or None
        probe = getattr(self, "_probe", None)
        if probe is None or probe.host != (host or probe.host):
            probe = OllamaBackend(host)
            self._probe = probe
        probe.model = model or self.ollama_model.currentText().strip()
        return probe

    def _describe_ollama_model(self) -> None:
        """Show what the selected model supports, and how it will be driven."""
        name = self.ollama_model.currentText().strip()
        if not name:
            self.model_caps.setText("")
            return
        info = dict(self._model_details.get(name) or {})
        # /api/tags under-reports capabilities on some servers, so ask
        # /api/show about the model the user actually picked (cached).
        detail = self._ollama_backend(name).model_info(name)
        if detail:
            info.update({k: v for k, v in detail.items() if v})
        if not info:
            self.model_caps.setText("")
            return
        caps = info.get("capabilities") or []
        bits = []
        if info.get("parameter_size"):
            bits.append(str(info["parameter_size"]))
        if info.get("size"):
            bits.append(short_size(int(info["size"])))
        limit = int(info.get("context_length") or 0)
        if limit:
            bits.append(tr("context {tokens}").format(tokens=f"{limit:,}".replace(",", " ")))
            bits.append(tr("auto → {tokens}").format(
                tokens=self._ollama_backend(name).best_context(name)))
        if "tools" in caps:
            bits.append(tr("native tool calling"))
        else:
            bits.append(tr("JSON protocol (no tool calling)"))
        if "vision" in caps:
            bits.append(tr("vision"))
        if "thinking" in caps:
            bits.append(tr("thinking"))
        self.model_caps.setText("  ·  ".join(bits))
        self._refresh_placement()

    def _refresh_placement(self) -> None:
        """Fill in the fit prediction and the live placement for this model."""
        name = self.ollama_model.currentText().strip()
        if not name:
            self.fit_line.setText("")
            self.runtime_line.setText("")
            return
        backend = self._ollama_backend(name)
        usable, other = backend.usable_vram()
        weights = backend.weights_bytes(name)
        context = (
            backend.best_context(name) if self.ctx_auto.isChecked()
            else int(self.ollama_ctx.value())
        )
        report = backend.fit(name, context)
        verdicts = {
            "gpu": tr("Fits in VRAM — will run fully on the GPU."),
            "tight": tr("Only just fits in VRAM; other apps may push it out."),
            "spill": tr("Too big at this context — part of it will run on the CPU."),
            "cpu": tr("Will not fit in VRAM — this model runs on the CPU."),
            "unknown": "",
        }
        line = verdicts.get(report.verdict, "")
        if weights and usable:
            line += "  " + tr("Needs ~{need} GB, {usable} GB of VRAM usable").format(
                need=f"{report.need_bytes / GB:.1f}", usable=f"{usable / GB:.1f}"
            )
        if other:
            line += "  " + tr("(other apps hold {other} GB)").format(
                other=f"{other / GB:.1f}"
            )
        if report.verdict in ("spill", "cpu") and self.ctx_auto.isChecked():
            best = backend.best_context(name)
            if best and best != context:
                line += "  " + tr("Suggested context: {tokens}").format(tokens=best)
        self.fit_line.setText(line.strip())

        runtime = backend.runtime(name)
        if runtime.loaded:
            minutes = int(runtime.expires_in() // 60)
            self.runtime_line.setText(
                tr("In memory now · {processor} · unloads in {minutes} min").format(
                    processor=runtime.processor(), minutes=minutes
                )
            )
        else:
            self.runtime_line.setText(tr("Not loaded"))
        self.unload_btn.setEnabled(runtime.loaded)

    def _unload_model(self) -> None:
        name = self.ollama_model.currentText().strip()
        if not name:
            return
        backend = self._ollama_backend(name)
        if backend.unload():
            self.status.setText(tr("Unloaded '{model}' — VRAM freed.").format(model=name))
        else:
            self.status.setText(tr("Could not unload '{model}'.").format(model=name))
        self._refresh_placement()

    def _benchmark_model(self) -> None:
        """Time a short generation on the selected model."""
        name = self.ollama_model.currentText().strip()
        if not name or getattr(self, "_bench_worker", None) and self._bench_worker.isRunning():
            return
        self.bench_btn.setEnabled(False)
        self.status.setText(tr("Measuring '{model}'…").format(model=name))
        self._bench_worker = BenchmarkWorker(self._ollama_backend(name))
        self._bench_worker.finished_ok.connect(self._on_benchmark)
        self._bench_worker.failed.connect(self._on_benchmark_failed)
        self._bench_worker.start()

    def _on_benchmark(self, result: dict) -> None:
        self.bench_btn.setEnabled(True)
        self.status.setText(
            tr("{model}: {load}s to load · {prefill} tok/s reading · "
               "{generate} tok/s writing · {processor}").format(
                model=result["model"],
                load=f"{result['load_seconds']:.1f}",
                prefill=f"{result['prefill_tps']:.0f}",
                generate=f"{result['generate_tps']:.1f}",
                processor=result["processor"] or tr("unknown"),
            )
        )
        self._refresh_placement()

    def _on_benchmark_failed(self, error: str) -> None:
        self.bench_btn.setEnabled(True)
        self.status.setText(f"✕ {error}")

    def _delete_ollama_model(self) -> None:
        name = self.ollama_model.currentText().strip()
        if not name:
            return
        if self._confirm_delete != name:
            # Two clicks: the first arms it, the second does it. A model is a
            # multi-gigabyte download — worth a deliberate second press.
            self._confirm_delete = name
            self.status.setText(
                tr("Click 🗑 again to delete '{model}' from disk.").format(model=name)
            )
            return
        self._confirm_delete = ""
        if self._ollama_backend(name).delete_model(name):
            self.status.setText(tr("Deleted '{model}'.").format(model=name))
            self._refresh_ollama_models()
        else:
            self.status.setText(tr("Could not delete '{model}'.").format(model=name))

    def _update_tool_cost(self) -> None:
        """Show what the enabled tools cost in context, in tokens.

        On a local model this is the most useful number in the dialog: every
        tool definition is carried in the window on every single turn.
        """
        groups = [g for g, box in self.group_boxes.items() if box.isChecked()]
        names = tools_for_groups(groups)
        if not names:
            self.tool_cost.setText(tr("No tools — the assistant can only talk."))
            return
        size = len(json.dumps(tool_schemas(names))) // 4
        self.tool_cost.setText(
            tr("{count} tools · about {tokens} tokens of context per message")
            .format(count=len(names), tokens=size)
        )

    def _on_ctx_auto(self, checked: bool) -> None:
        self.ollama_ctx.setEnabled(not checked)
        self._refresh_placement()

    def _refresh_ollama_models(self) -> None:
        self.status.setText(tr("Querying Ollama…"))
        backend = OllamaBackend(self.ollama_host.text().strip() or None)
        details = backend.installed_models()
        self._model_details = {d["name"]: d for d in details}
        loaded = set(backend.loaded_models())
        models = [d["name"] for d in details]
        current = self.ollama_model.currentText()
        self.ollama_model.clear()
        self.ollama_model.addItems(models)
        if current:
            self.ollama_model.setCurrentText(current)
        self._describe_ollama_model()
        self._refresh_placement()
        if loaded:
            self.status.setText(
                tr("In memory now: {models}").format(models=", ".join(sorted(loaded)))
            )
            return
        self.status.setText(
            tr("Found {count} model(s).").format(count=len(models)) if models
            else tr("No models found — is Ollama running?")
        )

    # ── model download ───────────────────────────────────────────────────
    def _selected_pull_tag(self) -> str:
        # Prefer the item's stored tag; fall back to typed text (before the em-dash).
        data = self.pull_box.currentData()
        if data:
            return str(data)
        return self.pull_box.currentText().split("—")[0].strip()

    def _start_pull(self) -> None:
        if getattr(self, "_pull_worker", None) and self._pull_worker.isRunning():
            return
        tag = self._selected_pull_tag()
        if not tag:
            self.pull_status.setText(tr("Enter a model name to download."))
            return
        host = self.ollama_host.text().strip() or "http://localhost:11434"
        self.pull_btn.setEnabled(False)
        self.pull_btn.setText("…")
        self.pull_bar.show()
        self.pull_bar.setRange(0, 0)  # indeterminate until we get byte totals
        self.pull_bar.setFormat(tr("Connecting…"))
        self.pull_status.setText(
            tr("Contacting Ollama to download '{model}'…").format(model=tag)
        )
        # Speed/heartbeat tracking so the user can see the download is live
        # even while a single layer is streaming.
        self._pull_last_bytes = 0
        self._pull_last_time = time.monotonic()
        self._pull_speed = 0.0

        self._pull_worker = OllamaPullWorker(host, tag)
        self._pull_worker.progress.connect(self._on_pull_progress)
        self._pull_worker.finished_ok.connect(self._on_pull_ok)
        self._pull_worker.failed.connect(self._on_pull_failed)
        self._pull_worker.start()

    # Turn Ollama's raw status codes into readable phase labels.
    _PULL_PHASE = {
        "pulling manifest": "Fetching manifest…",
        "verifying sha256 digest": "Verifying…",
        "writing manifest": "Finishing up…",
        "removing any unused layers": "Cleaning up…",
        "success": "Done",
    }

    def _on_pull_progress(self, status: str, completed: int, total: int) -> None:
        phase = tr(self._PULL_PHASE.get(status, "")) or status or tr("Downloading…")

        if total > 0:
            completed = min(completed, total)
            pct = min(100, completed * 100 // total)
            self.pull_bar.setRange(0, 100)
            self.pull_bar.setValue(pct)

            # Update the rolling download speed from bytes since the last tick.
            now = time.monotonic()
            dt = now - self._pull_last_time
            if dt >= 0.4:
                delta = completed - self._pull_last_bytes
                inst = delta / dt if dt > 0 else 0.0
                # Smooth it so the readout doesn't jump around every update.
                self._pull_speed = inst if self._pull_speed == 0 else (
                    0.6 * self._pull_speed + 0.4 * inst
                )
                self._pull_last_bytes = completed
                self._pull_last_time = now

            mb = 1024 * 1024
            speed = f" · {self._pull_speed / mb:.1f} MB/s" if self._pull_speed > 0 else ""
            self.pull_bar.setFormat(f"{pct}%")
            self.pull_status.setText(
                tr("Downloading · {done}/{total} MB ({pct}%){speed}").format(
                    done=f"{completed / mb:.0f}", total=f"{total / mb:.0f}",
                    pct=pct, speed=speed,
                )
            )
        else:
            # No byte totals yet (manifest / verify / write phases). Keep the
            # bar animated so it's obvious work is ongoing, and name the phase.
            self.pull_bar.setRange(0, 0)
            self.pull_bar.setFormat(phase)
            self.pull_status.setText(phase)

    def _on_pull_ok(self, model: str) -> None:
        self.pull_btn.setEnabled(True)
        self.pull_btn.setText("Download")
        self.pull_bar.setRange(0, 100)
        self.pull_bar.setValue(100)
        self.pull_bar.setFormat(tr("Done ✓"))
        self.pull_status.setText(
            tr("✓ '{model}' downloaded and ready.").format(model=model)
        )
        # Refresh the installed list and select the freshly pulled model.
        self._refresh_ollama_models()
        self.ollama_model.setCurrentText(model)

    def _on_pull_failed(self, error: str) -> None:
        self.pull_btn.setEnabled(True)
        self.pull_btn.setText("Download")
        self.pull_bar.hide()
        self.pull_status.setText(f"✕ {error}")

    # ── agent mode description ───────────────────────────────────────────
    _MODE_DESC = {
        MODE_ASK: "You approve every command, app launch and file write before it runs.",
        MODE_AUTO: "The agent runs tools on its own. Fast, but review what it does.",
        MODE_CHAT: "Pure conversation — the assistant cannot touch your system.",
    }

    def _update_mode_desc(self) -> None:
        self.mode_desc.setText(tr(self._MODE_DESC.get(self.mode_box.currentData(), "")))

    def _refresh_gemini_models(self) -> None:
        self.status.setText(tr("Querying Gemini…"))
        backend = GeminiBackend(self.gemini_key.text().strip())
        models = backend.available_models()
        current = self.gemini_model.currentText()
        self.gemini_model.clear()
        self.gemini_model.addItems(models)
        if current:
            self.gemini_model.setCurrentText(current)
        self.status.setText(f"Found {len(models)} model(s).")

    def _load(self) -> None:
        c = self.config
        backend = c.get("backend")
        bidx = {"ollama": 0, "gemini": 1, "openai": 2}.get(backend, 0)
        self.backend_box.setCurrentIndex(bidx)
        self.stack.setCurrentIndex(bidx)

        self.ollama_host.setText(c.get("ollama_host"))
        self.ollama_model.addItem(c.get("ollama_model"))
        self.ollama_model.setCurrentText(c.get("ollama_model"))
        ctx = c.get("ollama_num_ctx")
        auto = not isinstance(ctx, int)
        self.ctx_auto.setChecked(auto)
        self.ollama_ctx.setEnabled(not auto)
        self.ollama_ctx.setValue(int(ctx) if isinstance(ctx, int) else 8192)
        keep_index = self.keep_alive_box.findData(str(c.get("ollama_keep_alive")))
        self.keep_alive_box.setCurrentIndex(keep_index if keep_index >= 0 else 1)
        self.preload_check.setChecked(bool(c.get("ollama_preload")))
        self.think_check.setChecked(bool(c.get("ollama_think")))
        self.num_gpu.setValue(int(c.get("ollama_num_gpu") or 0))
        # Populate the capability line for the model already selected.
        self._model_details = {}

        self.gemini_key.setText(c.get("gemini_api_key"))
        self.gemini_model.addItems(GEMINI_MODELS)
        self.gemini_model.setCurrentText(c.get("gemini_model"))

        self.openai_key.setText(c.get("openai_api_key"))
        self.openai_base.setCurrentText(c.get("openai_base_url"))
        self.openai_model.addItems(OPENAI_MODELS)
        self.openai_model.setCurrentText(c.get("openai_model"))

        self.stream_check.setChecked(bool(c.get("stream_responses")))
        self.block_danger_check.setChecked(bool(c.get("block_dangerous_commands")))
        self.auto_readonly_check.setChecked(bool(c.get("auto_approve_readonly")))
        enabled = set(c.get("tool_groups") or TOOL_GROUPS)
        for group, box in self.group_boxes.items():
            box.setChecked(group in enabled)
        self._update_tool_cost()
        self.native_tools_check.setChecked(bool(c.get("native_tools")))
        self.constrain_json_check.setChecked(bool(c.get("constrain_json")))
        self.guard_secrets_check.setChecked(bool(c.get("guard_secrets")))
        self.egress_check.setChecked(bool(c.get("confirm_network_egress")))
        self.custom_instructions.setPlainText(c.get("custom_instructions") or "")

        mode_index = {MODE_ASK: 0, MODE_AUTO: 1, MODE_CHAT: 2}.get(c.get("agent_mode"), 0)
        self.mode_box.setCurrentIndex(mode_index)
        self._update_mode_desc()
        lang_index = self.lang_box.findData(c.get("output_language"))
        self.lang_box.setCurrentIndex(lang_index if lang_index >= 0 else 0)
        ui_index = self.ui_lang_box.findData(c.get("ui_language"))
        self.ui_lang_box.setCurrentIndex(ui_index if ui_index >= 0 else 0)
        self.steps_spin.setValue(int(c.get("max_steps")))
        self.timeout_spin.setValue(int(c.get("command_timeout")))
        # Reflect the actual autostart file state (the source of truth).
        self.autostart_check.setChecked(autostart.is_enabled())
        self.tray_check.setChecked(bool(c.get("close_to_tray")))
        self.greet_check.setChecked(bool(c.get("greet_on_start")))

    def _save(self) -> None:
        if self._pull_running():
            self.pull_status.setText(tr("Please wait for the download to finish…"))
            return
        self.config.update({
            "backend": self.backend_box.currentData(),
            "ollama_host": self.ollama_host.text().strip() or "http://localhost:11434",
            "ollama_model": self.ollama_model.currentText().strip() or "llama3.1",
            "ollama_num_ctx": (
                "auto" if self.ctx_auto.isChecked() else int(self.ollama_ctx.value())
            ),
            "ollama_keep_alive": self.keep_alive_box.currentData(),
            "ollama_preload": self.preload_check.isChecked(),
            "ollama_think": self.think_check.isChecked(),
            "ollama_num_gpu": int(self.num_gpu.value()),
            "tool_groups": [
                g for g, box in self.group_boxes.items() if box.isChecked()
            ],
            "native_tools": self.native_tools_check.isChecked(),
            "constrain_json": self.constrain_json_check.isChecked(),
            "gemini_api_key": self.gemini_key.text().strip(),
            "gemini_model": self.gemini_model.currentText().strip() or "gemini-2.5-flash",
            "openai_api_key": self.openai_key.text().strip(),
            "openai_base_url": self._current_openai_base() or "https://api.openai.com/v1",
            "openai_model": self.openai_model.currentText().strip() or "gpt-4o-mini",
            "agent_mode": self.mode_box.currentData(),
            "output_language": self.lang_box.currentData(),
            "ui_language": self.ui_lang_box.currentData(),
            "max_steps": self.steps_spin.value(),
            "command_timeout": self.timeout_spin.value(),
            "stream_responses": self.stream_check.isChecked(),
            "block_dangerous_commands": self.block_danger_check.isChecked(),
            "auto_approve_readonly": self.auto_readonly_check.isChecked(),
            "guard_secrets": self.guard_secrets_check.isChecked(),
            "confirm_network_egress": self.egress_check.isChecked(),
            "custom_instructions": self.custom_instructions.toPlainText().strip(),
            "autostart": self.autostart_check.isChecked(),
            "close_to_tray": self.tray_check.isChecked(),
            "greet_on_start": self.greet_check.isChecked(),
        })
        # Write/remove the per-user autostart entry to match the toggle.
        autostart.set_enabled(self.autostart_check.isChecked())
        self.config.save()
        self.accept()

    # ── guard against closing mid-download ───────────────────────────────
    def _pull_running(self) -> bool:
        w = getattr(self, "_pull_worker", None)
        return bool(w and w.isRunning())

    def reject(self) -> None:
        if self._pull_running():
            self.pull_status.setText(tr("Please wait for the download to finish…"))
            return
        super().reject()

    # ── frameless drag ───────────────────────────────────────────────────
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        handle = self.windowHandle()
        if handle is not None and handle.startSystemMove():
            return
        self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = None
