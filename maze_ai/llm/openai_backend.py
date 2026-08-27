"""OpenAI-compatible backend.

One backend, many providers: anything that speaks the OpenAI
``/v1/chat/completions`` API works here — OpenAI itself, OpenRouter, Groq,
Together, LM Studio, llama.cpp's server, vLLM, LocalAI, etc. The user just
points ``base_url`` at the provider and supplies a key (blank for most local
servers).
"""

from __future__ import annotations

import json
from typing import Iterator

from .base import LLMBackend, LLMError, Message
from .http import request_with_retry
from .images import data_uri

# A few well-known endpoints, shown as presets in Settings.
PRESETS: list[tuple[str, str]] = [
    ("OpenAI", "https://api.openai.com/v1"),
    ("OpenRouter", "https://openrouter.ai/api/v1"),
    ("Groq", "https://api.groq.com/openai/v1"),
    ("Together", "https://api.together.xyz/v1"),
    ("LM Studio (local)", "http://localhost:1234/v1"),
    ("llama.cpp (local)", "http://localhost:8080/v1"),
]

# Sensible fallbacks for the model picker before the API is queried.
KNOWN_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
]


class OpenAIBackend(LLMBackend):
    name = "openai"
    supports_vision = True

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key or ""
        self.model = model or "gpt-4o-mini"
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    # ── helpers ──────────────────────────────────────────────────────────
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _prepare(messages: list[Message]) -> list[dict]:
        """Expand any ``images`` on a message into OpenAI content parts."""
        out: list[dict] = []
        for msg in messages:
            images = msg.get("images")
            if not images:
                out.append({"role": msg["role"], "content": msg.get("content", "")})
                continue
            parts: list[dict] = [{"type": "text", "text": msg.get("content", "")}]
            for path in images:
                try:
                    parts.append({"type": "image_url", "image_url": {"url": data_uri(path)}})
                except OSError:
                    continue
            out.append({"role": msg["role"], "content": parts})
        return out

    def _raise_for_status(self, resp) -> None:
        if resp.status_code == 401:
            raise LLMError("The API rejected the key (401). Check it in Settings.")
        if resp.status_code == 404:
            raise LLMError(
                f"Model '{self.model}' or endpoint not found (404). "
                "Check the base URL and model in Settings."
            )
        if resp.status_code != 200:
            raise LLMError(f"API error {resp.status_code}: {resp.text[:300]}")

    # ── API ──────────────────────────────────────────────────────────────
    def chat(self, messages: list[Message]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": self._prepare(messages),
            "temperature": 0.4,
            "stream": False,
        }
        try:
            resp = request_with_retry(
                "POST", url, headers=self._headers(), json=payload, timeout=600
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Could not reach the API at {self.base_url} ({exc}).") from exc
        self._raise_for_status(resp)
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("The API returned no choices.")
        content = (choices[0].get("message") or {}).get("content", "")
        if not content:
            raise LLMError("The API returned an empty response.")
        return content

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": self._prepare(messages),
            "temperature": 0.4,
            "stream": True,
        }
        try:
            resp = request_with_retry(
                "POST", url, headers=self._headers(), json=payload,
                stream=True, timeout=(10, 600),
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Could not reach the API at {self.base_url} ({exc}).") from exc
        self._raise_for_status(resp)

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta

    def available_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        try:
            resp = request_with_retry("GET", url, headers=self._headers(), timeout=8)
            if resp.status_code != 200:
                return list(KNOWN_MODELS)
            data = resp.json().get("data", [])
            ids = sorted(m.get("id", "") for m in data if m.get("id"))
            return ids or list(KNOWN_MODELS)
        except Exception:  # noqa: BLE001
            return list(KNOWN_MODELS)

    def describe(self) -> str:
        return f"OpenAI-API · {self.model}"
