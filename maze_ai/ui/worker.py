"""Background worker that runs the agent off the UI thread."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QThread, Signal

from ..agent import Agent, AgentEvent, ApprovalRequest
from ..config import Config
from ..llm import OllamaBackend, resolve_ollama_model
from ..llm.base import LLMError

log = logging.getLogger(__name__)


class ModelResolveWorker(QThread):
    """Checks (off the UI thread) that the configured Ollama model exists.

    This is an HTTP call to a server that may not be running, so doing it
    inline at startup made the window wait on a socket timeout before it could
    appear. Emits the resolved model name, or "" when nothing could be found.
    """

    resolved = Signal(str)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            self.resolved.emit(resolve_ollama_model(self.config) or "")
        except Exception as exc:  # noqa: BLE001 - never take the app down for this
            log.warning("could not resolve the Ollama model: %s", exc)
            self.resolved.emit("")


class OllamaPullWorker(QThread):
    """Downloads an Ollama model, streaming progress to the UI."""

    progress = Signal(str, int, int)   # status, completed bytes, total bytes
    finished_ok = Signal(str)          # model name
    failed = Signal(str)               # error message

    def __init__(self, host: str, model: str) -> None:
        super().__init__()
        self.host = host
        self.model = model

    def run(self) -> None:
        backend = OllamaBackend(self.host)
        try:
            for _ in backend.pull_model(self.model, self._on_progress):
                pass
        except LLMError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(self.model)

    def _on_progress(self, status: str, completed: int, total: int) -> None:
        self.progress.emit(status, completed, total)


class BenchmarkWorker(QThread):
    """Times a short generation so the user can see what their box does."""

    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, backend: OllamaBackend) -> None:
        super().__init__()
        self.backend = backend

    def run(self) -> None:
        try:
            self.finished_ok.emit(self.backend.benchmark())
        except LLMError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            log.warning("benchmark failed: %s", exc)
            self.failed.emit(str(exc))


class AgentWorker(QThread):
    event = Signal(object)              # AgentEvent
    approval_needed = Signal(object)    # ApprovalRequest
    done = Signal(str)                  # final answer text

    def __init__(self, agent: Agent, message: str, images: list[str] | None = None) -> None:
        super().__init__()
        self.agent = agent
        self.message = message
        self.images = images or []
        self._approval_gate = threading.Event()
        self._approval_result = False

    # Called from the worker thread by the agent; blocks until the UI answers.
    def _approve(self, request: ApprovalRequest) -> bool:
        self._approval_gate.clear()
        self._approval_result = False
        self.approval_needed.emit(request)
        self._approval_gate.wait()
        return self._approval_result

    # Called from the UI thread once the user has decided.
    def provide_approval(self, approved: bool) -> None:
        self._approval_result = approved
        self._approval_gate.set()

    # Called from the UI thread to stop the current turn as soon as possible.
    def cancel(self) -> None:
        self.agent.request_cancel()
        # Unblock the approval gate if we're waiting on the user, denying it.
        self._approval_result = False
        self._approval_gate.set()

    def _emit(self, ev: AgentEvent) -> None:
        self.event.emit(ev)

    def run(self) -> None:
        try:
            answer = self.agent.run(self.message, self._emit, self._approve, self.images)
        except Exception as exc:  # noqa: BLE001 - never crash the thread silently
            self.event.emit(AgentEvent("error", text=str(exc), ok=False))
            answer = ""
        self.done.emit(answer)
