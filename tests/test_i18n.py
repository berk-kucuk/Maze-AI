"""Tests for the interface translation layer."""

from __future__ import annotations

import pytest

from maze_ai import i18n
from maze_ai.agent.agent import (
    REASON_CHANGES,
    REASON_COMMAND,
    REASON_DESTRUCTIVE,
    REASON_EGRESS_SAVE,
    REASON_EGRESS_URL,
    REASON_SENSITIVE,
)


@pytest.fixture(autouse=True)
def restore_language():
    yield
    i18n.set_language("en")


def test_english_is_the_identity():
    i18n.set_language("en")
    assert i18n.tr("Send") == "Send"


def test_turkish_translates():
    i18n.set_language("tr")
    assert i18n.tr("Send") == "Gönder"
    assert i18n.tr("Deny") == "Reddet"


def test_missing_string_falls_back_to_english():
    i18n.set_language("tr")
    assert i18n.tr("Some string nobody translated") == "Some string nobody translated"


def test_unknown_language_falls_back_to_english():
    assert i18n.set_language("kl") == "en"
    assert i18n.tr("Send") == "Send"


def test_auto_follows_the_desktop_locale(monkeypatch):
    monkeypatch.setenv("LC_ALL", "tr_TR.UTF-8")
    assert i18n.set_language("auto") == "tr"
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")   # no catalogue → English
    assert i18n.set_language("auto") == "en"


def test_auto_ignores_the_c_locale(monkeypatch):
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    monkeypatch.delenv("LANGUAGE", raising=False)
    assert i18n.set_language("auto") == "en"


# ── catalogue integrity ────────────────────────────────────────────────────
@pytest.mark.parametrize("reason", [
    REASON_EGRESS_SAVE, REASON_EGRESS_URL, REASON_SENSITIVE,
    REASON_DESTRUCTIVE, REASON_COMMAND, REASON_CHANGES,
])
def test_every_approval_reason_is_translated(reason):
    # These are shown in the dialog that guards the user's machine — an
    # untranslated one would appear as raw English mid-sentence.
    assert reason in i18n.TURKISH


def test_the_welcome_message_is_translated():
    from maze_ai.ui.main_window import WELCOME  # noqa: PLC0415 - avoids a Qt import at module load

    assert WELCOME in i18n.TURKISH


def test_no_translation_is_left_empty():
    empty = [key for key, value in i18n.TURKISH.items() if not value.strip()]
    assert empty == []


def test_placeholders_survive_translation():
    # A dropped {placeholder} would raise at format() time, in front of the user.
    for key, value in i18n.TURKISH.items():
        for token in ("{detail}", "{reason}", "{tool}", "{path}", "{error}",
                      "{count}", "{model}"):
            if token in key:
                assert token in value, f"{token} missing from the Turkish {key!r}"
