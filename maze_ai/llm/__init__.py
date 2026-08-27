"""LLM backends for Maze-AI (Ollama + Gemini) behind a single interface."""

from __future__ import annotations

from ..config import Config
from .base import LLMBackend, LLMError
from .gemini_backend import GeminiBackend
from .ollama_backend import OllamaBackend
from .openai_backend import OpenAIBackend

__all__ = [
    "LLMBackend",
    "LLMError",
    "GeminiBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "build_backend",
    "resolve_ollama_model",
]


def resolve_ollama_model(config: Config) -> str | None:
    """Ensure the configured Ollama model is actually installed.

    If the selected model isn't present but others are, switch to the first
    installed one and persist it. Returns the resolved model name, or ``None``
    if nothing could be resolved (server down / no models). Never changes
    anything when the current model is already installed.
    """
    if config.get("backend") != "ollama":
        return config.get("ollama_model")
    backend = OllamaBackend(host=config.get("ollama_host"))
    installed = backend.available_models()
    if not installed:
        return None
    current = config.get("ollama_model")
    # Accept both "llama3.2" and "llama3.2:latest" style matches.
    def _base(name: str) -> str:
        return name.split(":", 1)[0]

    if current in installed or any(_base(m) == _base(current) for m in installed):
        return current
    # Prefer a model that can actually call tools — the agent is far more
    # capable with one, and picking alphabetically would be a coin toss.
    chosen = installed[0]
    for name in installed:
        backend.model = name
        if backend.supports_native_tools:
            chosen = name
            break
    backend.model = chosen
    config.set("ollama_model", chosen)
    config.save()
    return chosen


def build_backend(config: Config) -> LLMBackend:
    """Instantiate the backend selected in ``config``."""
    backend = config.get("backend")
    if backend == "gemini":
        return GeminiBackend(
            api_key=config.get("gemini_api_key"),
            model=config.get("gemini_model"),
        )
    if backend == "openai":
        return OpenAIBackend(
            api_key=config.get("openai_api_key"),
            model=config.get("openai_model"),
            base_url=config.get("openai_base_url"),
        )
    return OllamaBackend(
        host=config.get("ollama_host"),
        model=config.get("ollama_model"),
        num_ctx=config.get("ollama_num_ctx") or "auto",
        keep_alive=config.get("ollama_keep_alive") or "30m",
        think=bool(config.get("ollama_think")),
        num_gpu=int(config.get("ollama_num_gpu") or 0),
    )
