"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_wiki_network(monkeypatch):
    """The Arch Wiki tools must never reach the internet from a test.

    Agent tests ask Arch-shaped questions now and then, which triggers the
    automatic lookup; patched at the tool's own seams so tests that need the
    real HTTP stack (the live Ollama ones) are unaffected.
    """
    from maze_ai.agent import docs, news, tools

    def offline(*_a, **_k):
        raise docs.requests.ConnectionError("no network in tests")

    monkeypatch.setattr(tools, "ddg_results", lambda query: ([], "offline in tests"))
    monkeypatch.setattr(docs, "_api_search", offline)
    monkeypatch.setattr(docs, "_online_page", offline)
    monkeypatch.setattr(news, "fetch_news", offline)
