"""Ollama backend — the primary target: local models over HTTP.

This backend does more than shuttle messages. Local models are small, slow to
load and tight on context, so it also:

* asks the server what the active model can actually do (tools / vision /
  thinking) instead of guessing from its name,
* sizes the context window from the model's own limit and the machine's RAM,
* keeps the model resident between messages (``keep_alive``) and can preload it
  so the first question doesn't pay a ten-second load,
* uses native function calling when the model supports it, and
  schema-constrained JSON when it doesn't — the two things that make a 4B model
  usable as an agent,
* reports generation speed so the user can see what their hardware is doing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime

import requests

from .base import LLMBackend, LLMError, LLMReply, Message, StreamCallback, ToolCall
from .hardware import (
    FitReport,
    context_that_fits,
    detect_gpus,
    estimate_fit,
    system_ram_gb,
    total_vram_bytes,
)
from .http import request_with_retry
from .images import encode_image

log = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"

# Curated list of popular models from the Ollama library, shown in the
# "download a model" picker. Names are valid `ollama pull` tags.
POPULAR_MODELS: list[tuple[str, str]] = [
    ("qwen2.5", "Qwen 2.5 · 7B · tool calling, strong agent"),
    ("qwen2.5-coder", "Qwen 2.5 Coder · 7B · coding focused"),
    ("llama3.2", "Meta Llama 3.2 · 3B · fast, tool calling"),
    ("llama3.1", "Meta Llama 3.1 · 8B · tool calling, general"),
    ("mistral-nemo", "Mistral Nemo · 12B · tools, larger context"),
    ("mistral", "Mistral · 7B · fast and capable"),
    ("gemma3", "Google Gemma 3 · 4B · vision, small"),
    ("gemma2", "Google Gemma 2 · 9B · balanced"),
    ("phi4", "Microsoft Phi-4 · 14B · reasoning"),
    ("deepseek-r1", "DeepSeek R1 · 7B · reasoning model"),
    ("granite3.3", "IBM Granite 3.3 · 8B · tools"),
    ("llava", "LLaVA · 7B · vision + text"),
]

# Fallback only: modern Ollama reports capabilities per model, so these name
# markers are just for servers too old to answer /api/show.
_VISION_MARKERS = (
    "llava", "bakllava", "vision", "moondream", "minicpm-v", "gemma3",
    "qwen2-vl", "qwen2.5vl", "qwen2.5-vl", "qwen3-vl", "pixtral",
    "cogvlm", "internvl", "granite3.2-vision", "mistral-small3", "llama4",
)
_TOOL_MARKERS = (
    "llama3.1", "llama3.2", "llama3.3", "qwen2.5", "qwen3", "mistral",
    "mistral-nemo", "command-r", "firefunction", "granite3", "hermes3",
    "phi4", "smollm2", "athene",
)

# Context sizes we are willing to ask for by default. A model may advertise
# 262144 tokens; allocating that on a laptop would swap the machine to death.
_CTX_FLOOR = 4096
_CTX_CEILING = 32768


def _system_ram_gb() -> float:
    """Total RAM in GiB (0.0 when it can't be determined)."""
    return system_ram_gb()


def auto_context(
    model_limit: int,
    ram_gb: float | None = None,
    *,
    weights_bytes: int = 0,
    vram_bytes: int | None = None,
) -> int:
    """Pick a context window that keeps the model on the GPU if it can.

    The KV cache grows linearly with the window and lives beside the weights,
    so an over-generous context is what pushes a model off the GPU and makes it
    five times slower. Three limits apply, smallest wins: the model's own
    maximum, what the system RAM can carry, and what is left of VRAM after the
    weights. When the weights don't fit in VRAM at all the GPU limit is
    dropped — the model is running from RAM anyway, and a bigger window there
    costs nothing but memory.
    """
    ram = system_ram_gb() if ram_gb is None else ram_gb
    if ram >= 32:
        budget = _CTX_CEILING
    elif ram >= 16:
        budget = 16384
    elif ram >= 8:
        budget = 8192
    else:
        budget = _CTX_FLOOR

    vram = total_vram_bytes() if vram_bytes is None else vram_bytes
    if weights_bytes and vram:
        weights_fit = estimate_fit(
            weights_bytes, _CTX_FLOOR, total_bytes=vram
        ).fits
        if weights_fit:
            budget = min(budget, context_that_fits(weights_bytes, total_bytes=vram))

    limit = int(model_limit or 0)
    if limit <= 0:
        return max(_CTX_FLOOR, budget)
    return max(_CTX_FLOOR, min(limit, budget))


@dataclass
class ModelRuntime:
    """Where a loaded model actually lives, from ``/api/ps``.

    ``size`` is what the model occupies in total and ``size_vram`` how much of
    that is on the GPU. The ratio is the number that decides whether answers
    take three seconds or thirty.
    """

    name: str = ""
    loaded: bool = False
    size: int = 0
    size_vram: int = 0
    context_length: int = 0
    expires_at: str = ""

    @property
    def gpu_fraction(self) -> float:
        if not self.size:
            return 0.0
        return max(0.0, min(1.0, self.size_vram / self.size))

    @property
    def on_gpu(self) -> bool:
        """True when essentially all of the model sits in VRAM."""
        return self.loaded and self.gpu_fraction >= 0.99

    @property
    def spilled(self) -> bool:
        """True when part of the model had to fall back to system RAM."""
        return self.loaded and 0.0 < self.gpu_fraction < 0.99

    def processor(self) -> str:
        """The split, in Ollama's own phrasing: '100% GPU', '40%/60% CPU/GPU'."""
        if not self.loaded:
            return ""
        gpu = round(self.gpu_fraction * 100)
        if gpu >= 99:
            return "100% GPU"
        if gpu <= 1:
            return "100% CPU"
        return f"{100 - gpu}%/{gpu}% CPU/GPU"

    def expires_in(self) -> float:
        """Seconds until the server unloads the model (0 when unknown)."""
        if not self.expires_at:
            return 0.0
        try:
            when = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return 0.0
        now = datetime.now(when.tzinfo)
        return max(0.0, (when - now).total_seconds())


class OllamaBackend(LLMBackend):
    name = "ollama"
    supports_schema = True   # every Ollama version we target accepts `format`

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = "llama3.1",
        num_ctx: int | str = "auto",
        keep_alive: str = "30m",
        think: bool = False,
        num_gpu: int = 0,
    ) -> None:
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.model = model or "llama3.1"
        #: Either a token count or "auto" (derive from the model + machine).
        self.num_ctx = num_ctx
        #: How long the server keeps the model in memory after a reply. The
        #: default costs RAM but saves a full reload on every message.
        self.keep_alive = keep_alive or "30m"
        #: Whether to let thinking models reason before answering.
        self.think = think
        #: How many layers to force onto the GPU. 0 lets Ollama decide, which
        #: is usually right; a manual value is the escape hatch when its
        #: estimate leaves VRAM on the table or overcommits it.
        self.num_gpu = int(num_gpu or 0)
        #: /api/show results, which are authoritative about capabilities.
        self._info_cache: dict[str, dict] = {}
        #: (timestamp, usable_bytes, other_usage_bytes) — VRAM accounting is
        #: two subprocess/HTTP calls, so it is cached for a few seconds.
        self._vram_cache: tuple[float, int, int] | None = None
        #: /api/tags metadata (size, family) for the model picker. Kept apart
        #: because the listing under-reports capabilities on some servers —
        #: granite, for one, lists only "completion" there but "tools" here.
        self._tags_cache: dict[str, dict] = {}

    # ── model capabilities ───────────────────────────────────────────────
    def model_info(self, model: str = "") -> dict:
        """What the server knows about a model: capabilities and context size.

        Cached per model — the UI asks for this on every status refresh, and it
        is a network round trip.
        """
        model = model or self.model
        cached = self._info_cache.get(model)
        if cached is not None:
            return cached
        info: dict = {}
        try:
            resp = requests.post(f"{self.host}/api/show", json={"model": model}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                details = data.get("details") or {}
                model_info = data.get("model_info") or {}
                context = details.get("context_length") or 0
                if not context:
                    # Older servers put it under "<family>.context_length".
                    for key, value in model_info.items():
                        if key.endswith(".context_length") and isinstance(value, int):
                            context = value
                            break
                info = {
                    "capabilities": list(data.get("capabilities") or []),
                    "context_length": int(context or 0),
                    "parameter_size": details.get("parameter_size", ""),
                    "family": details.get("family", ""),
                    "quantization": details.get("quantization_level", ""),
                }
        except (requests.RequestException, ValueError) as exc:
            log.debug("could not read model info for %s: %s", model, exc)
        if info:
            self._info_cache[model] = info
        return info

    def _capabilities(self) -> list[str]:
        return self.model_info().get("capabilities") or []

    def _guess(self, markers: tuple[str, ...]) -> bool:
        name = (self.model or "").lower()
        return any(marker in name for marker in markers)

    @property
    def supports_vision(self) -> bool:
        caps = self._capabilities()
        return "vision" in caps if caps else self._guess(_VISION_MARKERS)

    @property
    def supports_native_tools(self) -> bool:
        caps = self._capabilities()
        return "tools" in caps if caps else self._guess(_TOOL_MARKERS)

    @property
    def supports_thinking(self) -> bool:
        return "thinking" in self._capabilities()

    def context_limit(self) -> int:
        """The model's own maximum context, or 0 if unknown."""
        return int(self.model_info().get("context_length") or 0)

    # ── how much VRAM is really available ────────────────────────────────
    def usable_vram(self) -> tuple[int, int]:
        """``(usable_bytes, other_usage_bytes)`` for model loading.

        Total VRAM is the wrong number to plan with: a desktop session, a
        browser and a compositor can hold two gigabytes of it. What Ollama can
        use is the card's total minus whatever *isn't* Ollama — which we can
        work out by comparing the GPU's reported usage with the VRAM the
        currently loaded model occupies.
        """
        now = time.monotonic()
        cached = self._vram_cache
        if cached and now - cached[0] < 15.0:
            return cached[1], cached[2]

        total = total_vram_bytes()
        if not total:
            self._vram_cache = (now, 0, 0)
            return 0, 0
        used = sum(gpu.used_mb for gpu in detect_gpus()) * 1024 * 1024
        ours = sum(
            int(entry.get("size_vram") or 0) for entry in self._loaded_entries()
        )
        other = max(0, used - ours)
        usable = max(0, total - other)
        self._vram_cache = (now, usable, other)
        return usable, other

    def _loaded_entries(self) -> list[dict]:
        try:
            resp = requests.get(f"{self.host}/api/ps", timeout=3)
            resp.raise_for_status()
            return list(resp.json().get("models", []))
        except (requests.RequestException, ValueError):
            return []

    def weights_bytes(self, model: str = "") -> int:
        """Size of the model on disk — the floor of what it needs in memory."""
        model = model or self.model
        cached = self._tags_cache.get(model)
        if cached is None:
            self.available_models()          # fills the listing cache
            cached = self._tags_cache.get(model)
        return int((cached or {}).get("size") or 0)

    def resolved_ctx(self) -> int:
        """The window we will actually ask for."""
        if isinstance(self.num_ctx, str) or not self.num_ctx:
            usable, _ = self.usable_vram()
            return auto_context(
                self.context_limit(),
                weights_bytes=self.weights_bytes(),
                vram_bytes=usable,
            )
        return max(2048, int(self.num_ctx))

    # ── where the model is running ───────────────────────────────────────
    def runtime(self, model: str = "") -> ModelRuntime:
        """Live placement of a model: loaded, and how much of it is on the GPU."""
        model = model or self.model
        try:
            resp = requests.get(f"{self.host}/api/ps", timeout=3)
            resp.raise_for_status()
            entries = resp.json().get("models", [])
        except (requests.RequestException, ValueError):
            return ModelRuntime(name=model)
        for entry in entries:
            if entry.get("name") == model or entry.get("model") == model:
                return ModelRuntime(
                    name=model,
                    loaded=True,
                    size=int(entry.get("size") or 0),
                    size_vram=int(entry.get("size_vram") or 0),
                    context_length=int(entry.get("context_length") or 0),
                    expires_at=str(entry.get("expires_at") or ""),
                )
        return ModelRuntime(name=model)

    def fit(self, model: str = "", context: int = 0) -> FitReport:
        """Predict whether a model will run on the GPU at a given context."""
        model = model or self.model
        context = context or self.resolved_ctx()
        usable, _ = self.usable_vram()
        return estimate_fit(
            self.weights_bytes(model), context,
            total_bytes=usable, free_bytes=usable,
        )

    def best_context(self, model: str = "") -> int:
        """The largest context expected to keep this model on the GPU.

        For a model whose weights don't fit in VRAM at all this is meaningless
        — it runs from RAM either way — so the RAM-based figure is returned
        instead of a pointlessly tiny window.
        """
        model = model or self.model
        weights = self.weights_bytes(model)
        limit = int(self.model_info(model).get("context_length") or 0)
        if not weights:
            return self.resolved_ctx()
        usable, _ = self.usable_vram()
        return auto_context(limit, weights_bytes=weights, vram_bytes=usable)

    def models_that_fit(self, context: int = 0) -> list[str]:
        """Installed models expected to run fully on the GPU, largest first.

        Answers the question a spilled model raises: "fine — which of the
        models I already have would actually fit?"
        """
        usable, _ = self.usable_vram()
        if not usable:
            return []
        context = context or 8192
        fitting: list[tuple[int, str]] = []
        for entry in self.installed_models():
            size = int(entry.get("size") or 0)
            if not size:
                continue
            report = estimate_fit(size, context, total_bytes=usable, free_bytes=usable)
            if report.fits:
                fitting.append((size, entry["name"]))
        # Biggest first: within what fits, more parameters is usually better.
        return [name for _, name in sorted(fitting, reverse=True)]

    # ── measuring a model on this machine ────────────────────────────────
    def benchmark(self, tokens: int = 64) -> dict:
        """Time a short generation and report what the hardware managed.

        Returns load time, prefill and generation speed, and where the model
        ended up running — the four numbers that decide whether a model is
        usable locally. Raises :class:`LLMError` if the server refuses.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content":
                          "Count from one to twenty in words, comma separated."}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.0, "num_ctx": self.resolved_ctx(),
                        "num_predict": int(tokens)},
        }
        started = time.monotonic()
        try:
            resp = request_with_retry(
                "POST", f"{self.host}/api/chat", json=payload, timeout=600
            )
        except requests.RequestException as exc:
            raise self._unreachable(exc) from exc
        self._raise_for_status(resp)
        data = resp.json()
        reply = LLMReply(metrics={k: data.get(k) for k in
                                  ("eval_count", "eval_duration", "load_duration",
                                   "prompt_eval_count", "prompt_eval_duration",
                                   "total_duration")})
        runtime = self.runtime()
        return {
            "model": self.model,
            "context": self.resolved_ctx(),
            "wall_seconds": time.monotonic() - started,
            "load_seconds": reply.load_seconds,
            "prefill_tps": reply.prompt_tokens_per_second,
            "generate_tps": reply.tokens_per_second,
            "tokens": reply.metrics.get("eval_count") or 0,
            "processor": runtime.processor(),
            "gpu_fraction": runtime.gpu_fraction,
            "spilled": runtime.spilled,
        }

    # ── request shaping ──────────────────────────────────────────────────
    def _options(self) -> dict:
        options = {"temperature": 0.4, "num_ctx": self.resolved_ctx()}
        if self.num_gpu > 0:
            options["num_gpu"] = self.num_gpu
        return options

    def _payload(
        self,
        messages: list[Message],
        *,
        stream: bool,
        tools: list[dict] | None = None,
        schema: dict | None = None,
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": self._prepare(messages),
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": self._options(),
        }
        if tools:
            payload["tools"] = tools
        if schema:
            # `format` constrains generation to the schema, so a weak model
            # cannot emit truncated or fenced JSON in the first place.
            payload["format"] = schema
        if self.supports_thinking:
            payload["think"] = bool(self.think)
        return payload

    @staticmethod
    def _prepare(messages: list[Message]) -> list[dict]:
        """Convert our messages to Ollama's wire format.

        Handles the three shapes the agent produces: plain text turns, user
        turns carrying image paths, and tool results (role ``tool``).
        """
        out: list[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            m: dict = {"role": role, "content": msg.get("content", "")}
            if role == "tool" and msg.get("tool_name"):
                m["tool_name"] = msg["tool_name"]
            if msg.get("tool_calls"):
                m["tool_calls"] = [
                    {"function": {"name": call["name"], "arguments": call["arguments"]}}
                    for call in msg["tool_calls"]
                ]
            paths = msg.get("images")
            if paths:
                encoded = []
                for p in paths:
                    try:
                        encoded.append(encode_image(p)[1])
                    except OSError:
                        continue
                if encoded:
                    m["images"] = encoded
            out.append(m)
        return out

    def _raise_for_status(self, resp) -> None:
        if resp.status_code == 404:
            raise LLMError(
                f"Ollama model '{self.model}' not found. Pull it with "
                f"`ollama pull {self.model}`."
            )
        if resp.status_code != 200:
            detail = resp.text[:300]
            if "context" in detail and "exceed" in detail:
                raise LLMError(
                    f"'{self.model}' ran out of context ({self.resolved_ctx()} tokens). "
                    "Lower the context window in Settings, or start a new chat."
                )
            raise LLMError(f"Ollama error {resp.status_code}: {detail}")

    def _unreachable(self, exc: Exception) -> LLMError:
        return LLMError(
            f"Could not reach Ollama at {self.host}. Is it running? "
            f"Start it with `systemctl --user start ollama` or `ollama serve`. ({exc})"
        )

    @staticmethod
    def _tool_calls(message: dict) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            name = fn.get("name") or ""
            if not name:
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=name, arguments=args or {}, id=raw.get("id", "")))
        return calls

    # ── the full call ────────────────────────────────────────────────────
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
        payload = self._payload(messages, stream=stream, tools=tools, schema=schema)
        url = f"{self.host}/api/chat"
        if not stream:
            try:
                resp = request_with_retry("POST", url, json=payload, timeout=600)
            except requests.RequestException as exc:
                raise self._unreachable(exc) from exc
            self._raise_for_status(resp)
            data = resp.json()
            message = data.get("message") or {}
            reply = LLMReply(
                text=message.get("content", "") or "",
                thinking=message.get("thinking", "") or "",
                tool_calls=self._tool_calls(message),
                metrics={k: data.get(k) for k in
                         ("eval_count", "eval_duration", "load_duration",
                          "prompt_eval_count", "prompt_eval_duration",
                          "total_duration")},
            )
            if not (reply.text or reply.tool_calls or reply.thinking):
                raise LLMError("Ollama returned an empty response.")
            return reply

        # Streaming: text and thinking arrive token by token, tool calls in a
        # chunk of their own near the end.
        try:
            resp = request_with_retry(
                "POST", url, json=payload, stream=True, timeout=(10, 600)
            )
        except requests.RequestException as exc:
            raise self._unreachable(exc) from exc
        self._raise_for_status(resp)

        text: list[str] = []
        thinking: list[str] = []
        calls: list[ToolCall] = []
        metrics: dict = {}
        # `with` so an abandoned generation (the user pressed Stop) releases the
        # socket instead of leaving the server generating into a dead pipe.
        # `with` so an abandoned generation (the user pressed Stop) releases
        # the socket instead of leaving the server generating into a dead pipe.
        with resp:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    update = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if update.get("error"):
                    raise LLMError(str(update["error"]))
                message = update.get("message") or {}
                chunk = message.get("content") or ""
                if chunk:
                    text.append(chunk)
                    if on_text is not None:
                        on_text(chunk)
                thought = message.get("thinking") or ""
                if thought:
                    thinking.append(thought)
                    if on_thinking is not None:
                        on_thinking(thought)
                calls.extend(self._tool_calls(message))
                if update.get("done"):
                    metrics = {k: update.get(k) for k in
                               ("eval_count", "eval_duration", "load_duration",
                                "prompt_eval_count", "prompt_eval_duration",
                                "total_duration")}
                    break
        return LLMReply(
            text="".join(text), thinking="".join(thinking),
            tool_calls=calls, metrics=metrics,
        )

    # ── the simple calls, on top of chat_ex ──────────────────────────────
    def chat(self, messages: list[Message]) -> str:
        return self.chat_ex(messages).text

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        """Compatibility shim: collects the stream, then yields it.

        Everything inside Maze AI calls :meth:`chat_ex`, which delivers tokens
        the moment they arrive. This exists so external callers of the old
        interface keep working — it is NOT live.
        """
        chunks: list[str] = []
        self.chat_ex(messages, stream=True, on_text=chunks.append)
        yield from chunks

    # ── server / model management ────────────────────────────────────────
    def available_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
        except (requests.RequestException, ValueError):
            return []
        # /api/tags already carries capabilities and context length — cache
        # them so the UI doesn't need a /api/show per model.
        for entry in models:
            name = entry.get("name")
            if not name:
                continue
            details = entry.get("details") or {}
            self._tags_cache[name] = {
                "capabilities": list(entry.get("capabilities") or []),
                "context_length": int(details.get("context_length") or 0),
                "parameter_size": details.get("parameter_size", ""),
                "family": details.get("family", ""),
                "quantization": details.get("quantization_level", ""),
                "size": int(entry.get("size") or 0),
            }
        return sorted(m.get("name", "") for m in models if m.get("name"))

    def installed_models(self) -> list[dict]:
        """Installed models with their size and capabilities, for the picker."""
        out = []
        for name in self.available_models():
            info = dict(self._tags_cache.get(name) or {})
            # Prefer the authoritative capability list when we already have it.
            shown = self._info_cache.get(name)
            if shown:
                info.update({k: v for k, v in shown.items() if v})
            info["name"] = name
            out.append(info)
        return out

    def loaded_models(self) -> list[str]:
        """Models currently resident in memory (``/api/ps``)."""
        try:
            resp = requests.get(f"{self.host}/api/ps", timeout=3)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except (requests.RequestException, ValueError):
            return []

    def is_reachable(self) -> bool:
        try:
            requests.get(f"{self.host}/api/version", timeout=3).raise_for_status()
            return True
        except requests.RequestException:
            return False

    def server_version(self) -> str:
        try:
            resp = requests.get(f"{self.host}/api/version", timeout=3)
            resp.raise_for_status()
            return str(resp.json().get("version", ""))
        except (requests.RequestException, ValueError):
            return ""

    def preload(self) -> bool:
        """Load the model into memory now, so the first message is instant.

        A cold local model spends five to fifteen seconds loading before it
        emits a single token; doing that while the user is still typing hides
        the whole wait.
        """
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": self.keep_alive},
                timeout=300,
            )
            return resp.status_code == 200
        except requests.RequestException as exc:
            log.debug("preload failed: %s", exc)
            return False

    def unload(self) -> bool:
        """Drop the model from memory (``keep_alive: 0``) to free RAM/VRAM."""
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def delete_model(self, name: str) -> bool:
        """Remove an installed model from disk."""
        name = (name or "").strip()
        if not name:
            return False
        try:
            resp = requests.delete(
                f"{self.host}/api/delete", json={"model": name}, timeout=30
            )
        except requests.RequestException:
            return False
        if resp.status_code == 200:
            self._info_cache.pop(name, None)
            self._tags_cache.pop(name, None)
            return True
        return False

    def pull_model(
        self, name: str, on_progress: Callable[[str, int, int], None] | None = None
    ) -> Iterable[None]:
        """Download a model via /api/pull, reporting progress.

        ``on_progress(status, completed, total)`` is called for each streamed
        update; ``completed``/``total`` are bytes (0 when unknown). Raises
        :class:`LLMError` on failure.
        """
        name = (name or "").strip()
        if not name:
            raise LLMError("No model name given.")
        url = f"{self.host}/api/pull"
        try:
            resp = requests.post(url, json={"model": name, "stream": True},
                                stream=True, timeout=(10, None))
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach Ollama at {self.host} ({exc}).") from exc
        if resp.status_code != 200:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")

        # A pull streams progress per layer (keyed by digest), each with its own
        # total. Aggregate across all seen layers so the reported percentage is
        # coherent (0–100%) instead of resetting or overshooting per layer.
        layers: dict[str, tuple[int, int]] = {}
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                update = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in update:
                raise LLMError(str(update["error"]))

            status = update.get("status", "")
            digest = update.get("digest")
            total = int(update.get("total", 0) or 0)
            completed = int(update.get("completed", 0) or 0)

            if digest and total > 0:
                # Clamp completed to total to avoid transient >100% readings.
                layers[digest] = (min(completed, total), total)

            # Aggregate over every layer seen so far. We do NOT reset to 0 on
            # status-only lines ("verifying sha256", "writing manifest",
            # "success") — those arrive between/after the byte updates, and
            # zeroing here would make a moving bar snap back to an
            # indeterminate state right as the download completes.
            overall_completed = sum(c for c, _ in layers.values())
            overall_total = sum(t for _, t in layers.values())

            if on_progress:
                on_progress(status, overall_completed, overall_total)
            yield None
        self._info_cache.pop(name, None)
        self._tags_cache.pop(name, None)

    def describe(self) -> str:
        return f"Ollama · {self.model}"


def short_size(byte_count: int) -> str:
    """Human-readable model size, e.g. '5.3 GB'."""
    value = float(byte_count or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit in ("GB", "TB") else f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def parse_size_hint(text: str) -> float:
    """Parameter count in billions from a label like '8.8B' (0.0 if unknown)."""
    match = re.match(r"\s*([\d.]+)\s*([BbMm])", text or "")
    if not match:
        return 0.0
    value = float(match.group(1))
    return value / 1000 if match.group(2).lower() == "m" else value
