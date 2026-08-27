"""Gemini backend — Google Generative Language REST API (no SDK dependency)."""

from __future__ import annotations

import json
from typing import Iterator

import requests

from .base import LLMBackend, LLMError, Message
from .http import request_with_retry
from .images import encode_image

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# A sensible, curated default list of current, valid model ids. The picker
# also merges whatever the API advertises at runtime (authoritative).
KNOWN_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]


class GeminiBackend(LLMBackend):
    name = "gemini"
    supports_vision = True

    def __init__(self, api_key: str = "", model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key or ""
        self.model = model or "gemini-2.5-flash"

    # ── message conversion ───────────────────────────────────────────────
    @staticmethod
    def _to_gemini(messages: list[Message]) -> tuple[dict | None, list[dict]]:
        system_parts: list[str] = []
        contents: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            text = msg.get("content", "")
            if role == "system":
                system_parts.append(text)
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:  # user / tool results are fed back as user turns
                parts: list[dict] = [{"text": text}]
                for path in msg.get("images") or []:
                    try:
                        mime, b64 = encode_image(path)
                        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
                    except OSError:
                        continue
                contents.append({"role": "user", "parts": parts})
        system = None
        if system_parts:
            system = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return system, contents

    # ── API ──────────────────────────────────────────────────────────────
    def chat(self, messages: list[Message]) -> str:
        if not self.api_key:
            raise LLMError("No Gemini API key set. Add one in Settings.")
        system, contents = self._to_gemini(messages)
        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": 0.4},
        }
        if system:
            payload["systemInstruction"] = system

        url = f"{API_ROOT}/models/{self.model}:generateContent"
        try:
            resp = request_with_retry(
                "POST", url, params={"key": self.api_key}, json=payload, timeout=600
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach the Gemini API ({exc}).") from exc

        self._raise_for_status(resp)

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(f"Gemini returned no answer ({block or 'empty response'}).")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            raise LLMError("Gemini returned an empty response.")
        return text

    def _raise_for_status(self, resp) -> None:
        if resp.status_code == 400 and "API_KEY_INVALID" in resp.text:
            raise LLMError("Gemini rejected the API key. Check it in Settings.")
        if resp.status_code == 404:
            raise LLMError(f"Gemini model '{self.model}' is not available.")
        if resp.status_code != 200:
            raise LLMError(f"Gemini error {resp.status_code}: {resp.text[:300]}")

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        if not self.api_key:
            raise LLMError("No Gemini API key set. Add one in Settings.")
        system, contents = self._to_gemini(messages)
        payload: dict = {"contents": contents, "generationConfig": {"temperature": 0.4}}
        if system:
            payload["systemInstruction"] = system

        url = f"{API_ROOT}/models/{self.model}:streamGenerateContent"
        try:
            resp = request_with_retry(
                "POST", url, params={"key": self.api_key, "alt": "sse"}, json=payload,
                stream=True, timeout=(10, 600),
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach the Gemini API ({exc}).") from exc
        self._raise_for_status(resp)

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            candidates = obj.get("candidates") or []
            if not candidates:
                continue
            parts = (candidates[0].get("content") or {}).get("parts") or []
            chunk = "".join(p.get("text", "") for p in parts)
            if chunk:
                yield chunk

    def available_models(self) -> list[str]:
        if not self.api_key:
            return list(KNOWN_MODELS)
        try:
            resp = requests.get(
                f"{API_ROOT}/models", params={"key": self.api_key}, timeout=5
            )
            resp.raise_for_status()
            names = []
            for m in resp.json().get("models", []):
                mid = (m.get("name") or "").removeprefix("models/")
                methods = m.get("supportedGenerationMethods", [])
                if mid and "generateContent" in methods:
                    names.append(mid)
            merged = sorted(set(names) | set(KNOWN_MODELS))
            return merged or list(KNOWN_MODELS)
        except requests.RequestException:
            return list(KNOWN_MODELS)

    def describe(self) -> str:
        return f"Gemini · {self.model}"
