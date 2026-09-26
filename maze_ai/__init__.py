"""Maze-AI — agentic AI assistant for Maze Linux.

Works with local Ollama models and the Google Gemini API through a single,
model-agnostic agent protocol. Ships a monochrome PySide6 UI with a custom
title bar, animated aurora background and a system-tray presence.
"""

# Kept in step with pyproject.toml. Reading it back from the installed
# distribution metadata sounds tidier, but a stale egg-info in the source tree
# then reports an old number — the literal is the honest answer.
__version__ = "1.18.0"

__app_name__ = "Maze AI"
