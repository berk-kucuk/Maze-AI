"""Shared backend interface.

Every backend takes a list of ``{"role", "content"}`` messages (roles:
``system`` | ``user`` | ``assistant`` | ``tool``) and returns the assistant's
reply. Keeping the contract small lets the agent loop drive any model — local
or hosted — the same way.

Two levels exist on purpose:

``chat`` / ``chat_stream``
    Plain text in, plain text out. Every backend supports this, and the
    prompt-based JSON tool protocol runs on top of it.
``chat_ex``
    The richer call: native tool calling, schema-constrained output, separated
    "thinking" tokens and generation metrics. Backends that can do more (Ollama)
    override it; the default implementation degrades to ``chat``/``chat_stream``
    so nothing has to special-case support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

Message = dict  # {"role": str, "content": str}

# Called with each new chunk of streamed text as it arrives.
StreamCallback = Callable[[str], None]


class LLMError(RuntimeError):
    """Raised when a backend cannot produce a completion."""


@dataclass
class ToolCall:
    """A tool the model asked to run, in native function-calling form."""

    name: str
    arguments: dict = field(default_factory=dict)
    id: str = ""


@dataclass
class LLMReply:
    """One assistant turn: prose, tool calls, thinking and timings."""

    text: str = ""
    #: Reasoning tokens, kept apart from the answer by servers that separate
    #: them (Ollama's ``thinking`` field). Shown in the activity trail, never
    #: as the answer.
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Raw generation stats: eval_count, eval_duration, load_duration, …
    metrics: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Generation speed, or 0.0 when the backend doesn't report it."""
        count = self.metrics.get("eval_count") or 0
        duration = self.metrics.get("eval_duration") or 0
        if count and duration:
            return count / (duration / 1e9)
        return 0.0

    @property
    def prompt_tokens_per_second(self) -> float:
        """Prefill speed — how fast the model read the prompt.

        Reported separately from generation because they have different
        bottlenecks: prefill is compute-bound, generation is memory-bound. A
        model that spilled to the CPU shows it here first.
        """
        count = self.metrics.get("prompt_eval_count") or 0
        duration = self.metrics.get("prompt_eval_duration") or 0
        if count and duration:
            return count / (duration / 1e9)
        return 0.0

    @property
    def prompt_tokens(self) -> int:
        return int(self.metrics.get("prompt_eval_count") or 0)

    @property
    def load_seconds(self) -> float:
        """How long the server spent loading the model for this reply."""
        return (self.metrics.get("load_duration") or 0) / 1e9


class LLMBackend(ABC):
    name: str = "llm"
    #: Whether this backend can accept image parts in a message. Overridden by
    #: multimodal backends (Gemini, OpenAI-vision, llava).
    supports_vision: bool = False
    #: Whether the server can run native function calling for the active model.
    #: When False the agent falls back to the prompt-based JSON protocol.
    supports_native_tools: bool = False
    #: Whether the server can constrain output to a JSON schema. This is what
    #: makes small local models reliable on the JSON protocol.
    supports_schema: bool = False

    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        """Return the assistant reply for ``messages`` (blocking)."""

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield the reply in chunks as they arrive.

        Backends that support server-side streaming override this. The default
        falls back to a single ``chat`` call so every backend "streams" (as one
        chunk) and callers never need to special-case support.
        """
        yield self.chat(messages)

    def chat_ex(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        schema: dict | None = None,
        stream: bool = False,
        on_text: StreamCallback | None = None,
        on_thinking: StreamCallback | None = None,
    ) -> LLMReply:
        """Full-featured completion. See the module docstring.

        The default ignores ``tools``/``schema`` — a backend advertising
        neither capability is never asked for them — and streams text through
        ``on_text`` when asked.
        """
        if stream and on_text is not None:
            buf: list[str] = []
            for chunk in self.chat_stream(messages):
                buf.append(chunk)
                on_text(chunk)
            return LLMReply(text="".join(buf))
        return LLMReply(text=self.chat(messages))

    @abstractmethod
    def available_models(self) -> list[str]:
        """Return a list of model ids the backend can serve (may be empty)."""

    def describe(self) -> str:
        return self.name
