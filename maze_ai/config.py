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
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, CONFIG_FILE)
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except OSError:
            pass

    # ── access ───────────────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, values: dict[str, Any]) -> None:
        self._data.update(values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)
