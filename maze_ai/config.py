"""Persistent configuration for Maze-AI.

Settings live in ~/.config/maze-ai/config.json. The file is created with
0600 permissions because it may contain a Gemini API key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "maze-ai"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Agent execution policies.
MODE_CHAT = "chat"        # never runs tools, pure conversation
MODE_ASK = "ask"          # asks before running any command / launching apps
MODE_AUTO = "auto"        # runs tools without asking (use with care)

# Output languages offered in Settings. "auto" mirrors the user's own language.
LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Auto  ·  match my message"),
    ("en", "English"),
    ("tr", "Türkçe"),
    ("de", "Deutsch"),
    ("fr", "Français"),
    ("es", "Español"),
    ("it", "Italiano"),
    ("pt", "Português"),
    ("ru", "Русский"),
    ("ar", "العربية"),
    ("zh", "中文"),
    ("ja", "日本語"),
]

DEFAULTS: dict[str, Any] = {
    "backend": "ollama",              # "ollama" | "gemini" | "openai"
    "agent_mode": MODE_ASK,           # "chat" | "ask" | "auto"
    "output_language": "auto",        # code from LANGUAGES
    "ui_language": "auto",            # interface language: auto | en | tr
    # Ollama — the primary backend
    "ollama_host": "http://localhost:11434",
    "ollama_model": "llama3.1",
    # Context window. "auto" derives it from the model's own limit and the
    # machine's RAM; a number pins it. It must fit the system prompt, the
    # replayed history and any attached image.
    "ollama_num_ctx": "auto",
    # How long the server keeps the model resident after a reply. Keeping it
    # warm costs RAM but saves a 5-15s reload before every single message.
    "ollama_keep_alive": "30m",
    "ollama_preload": True,           # load the model at startup, not on the
    #                                   first question the user asks
    "ollama_think": False,            # let thinking models reason first (slower)
    "ollama_num_gpu": 0,              # layers forced onto the GPU (0 = let
    #                                   Ollama decide; raise it when its own
    #                                   estimate leaves VRAM unused)
    "native_tools": True,             # use the model's own function calling
    #                                   when it supports it (far more reliable
    #                                   than the prompt-based JSON protocol)
    "constrain_json": True,           # force schema-valid JSON on models that
    #                                   have no native tool calling
    # Tool groups the agent may use. Trimming these shrinks the definition
    # block a local model has to carry in its context on every single turn.
    "tool_groups": ["shell", "files", "web", "desktop", "reminders"],
    # Gemini
    "gemini_api_key": "",
    "gemini_model": "gemini-2.5-flash",
    # OpenAI-compatible (OpenAI, OpenRouter, Groq, LM Studio, llama.cpp, …)
    "openai_api_key": "",
    "openai_model": "gpt-4o-mini",
    "openai_base_url": "https://api.openai.com/v1",
    # Agent
    "max_steps": 12,
    "command_timeout": 120,
    "context_char_budget": 24000,     # cap on replayed history fed to the model
    "custom_instructions": "",        # extra persistent system guidance
    "stream_responses": True,         # stream tokens into the UI as they arrive
    # Safety
    "block_dangerous_commands": True,  # force approval for destructive commands
    "auto_approve_readonly": True,     # skip approval for safe read-only commands
    "always_allow": [],               # remembered per-command approvals (auto add)
    "guard_secrets": True,            # always confirm access to keys/tokens/history
    "confirm_network_egress": True,   # confirm fetches that carry data outwards
    # UI
    "autostart": True,                # launch on login (into the tray)
    "start_hidden": False,            # start minimized to tray
    "close_to_tray": True,
    "sidebar_visible": True,          # chat history rail shown (Ctrl+B)
    "greet_on_start": True,           # send a greeting notification on launch
    "onboarded": False,               # first-run wizard has been completed
}


class Config:
    """Small JSON-backed settings store with attribute-style access helpers."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = dict(DEFAULTS)
        self.load()

    # ── persistence ──────────────────────────────────────────────────────
    def load(self) -> None:
        try:
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Only keep known keys, fall back to defaults for the rest.
                for key in DEFAULTS:
                    if key in raw:
                        self._data[key] = raw[key]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

    def save(self) -> None:
        """Write the settings, never leaving the API key world-readable.

        The temporary file is opened 0600 from the start rather than written
        and chmod-ed afterwards. os.replace preserves the *temporary* file's
        mode, so a tmp file created at the default umask (0644) handed that
        mode straight to config.json, and the key sat readable by every local
        account for the whole write plus the gap before the chmod. Widening a
        file and narrowing it again is not the same as never widening it.

        The directory is tightened too: 0600 on the file is no help while the
        directory it lives in lets anyone list and open what is in it. Same
        reasoning, and the same fix, as haze's storage/settings.py.
        """
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(CONFIG_DIR, 0o700)
        except OSError:
            pass

        tmp = CONFIG_FILE.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        # O_CREAT leaves an EXISTING tmp file's mode alone, so pin it here too
        # for anyone upgrading from a version that wrote it world-readable.
        os.chmod(tmp, 0o600)
        os.replace(tmp, CONFIG_FILE)

    # ── access ───────────────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, values: dict[str, Any]) -> None:
        self._data.update(values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)
