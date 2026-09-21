"""Tests for image attachment / message conversion across backends."""

from __future__ import annotations

import base64

import pytest

from maze_ai.agent import tools as tools_mod
from maze_ai.agent.tools import _tesseract_langs, ocr_image
from maze_ai.llm.gemini_backend import GeminiBackend
from maze_ai.llm.images import data_uri, encode_image, mime_for
from maze_ai.llm.ollama_backend import OllamaBackend
from maze_ai.llm.openai_backend import OpenAIBackend

# A 1x1 PNG.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(_PNG)
    return str(p)


def test_mime_for():
    assert mime_for("a.png") == "image/png"
    assert mime_for("a.JPG") == "image/jpeg"
    assert mime_for("a.unknown") == "image/png"


def test_encode_image(img):
    mime, b64 = encode_image(img)
    assert mime == "image/png" and base64.b64decode(b64) == _PNG


def test_data_uri(img):
    assert data_uri(img).startswith("data:image/png;base64,")


def test_openai_prepare_with_image(img):
    msgs = [{"role": "user", "content": "what is this", "images": [img]}]
    out = OpenAIBackend._prepare(msgs)
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_prepare_without_image():
    out = OpenAIBackend._prepare([{"role": "user", "content": "hi"}])
    assert out == [{"role": "user", "content": "hi"}]


def test_ollama_prepare_with_image(img):
    out = OllamaBackend._prepare([{"role": "user", "content": "x", "images": [img]}])
    assert isinstance(out[0]["images"], list) and out[0]["images"]


def test_gemini_to_gemini_with_image(img):
    msgs = [{"role": "user", "content": "look", "images": [img]}]
    _system, contents = GeminiBackend._to_gemini(msgs)
    parts = contents[0]["parts"]
    assert parts[0] == {"text": "look"}
    assert "inlineData" in parts[1] and parts[1]["inlineData"]["mimeType"] == "image/png"


def test_missing_image_is_skipped_gracefully():
    # Non-existent path should be dropped, not crash.
    out = OpenAIBackend._prepare([{"role": "user", "content": "x", "images": ["/no/such.png"]}])
    assert out[0]["content"][0]["text"] == "x"
    assert len(out[0]["content"]) == 1  # image skipped


# ── OCR (ocr_image) ────────────────────────────────────────────────────────
def test_ocr_no_path():
    assert ocr_image("").ok is False


def test_ocr_missing_file(tmp_path):
    res = ocr_image(str(tmp_path / "nope.png"))
    assert res.ok is False and "No such image" in res.output


def test_ocr_reports_when_tesseract_missing(img, monkeypatch):
    monkeypatch.setattr(tools_mod.shutil, "which", lambda _name: None)
    res = ocr_image(img)
    assert res.ok is False and "not installed" in res.output.lower()


def test_ocr_extracts_stdout(img, monkeypatch):
    monkeypatch.setattr(tools_mod.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(tools_mod, "_tesseract_langs", lambda: {"eng"})

    class _Proc:
        returncode = 0
        stdout = "Hello Maze\n"
        stderr = ""

    monkeypatch.setattr(tools_mod.subprocess, "run", lambda *a, **k: _Proc())
    res = ocr_image(img)
    assert res.ok is True and res.output == "Hello Maze"


def test_tesseract_langs_parsing(monkeypatch):
    class _Proc:
        stdout = "List of available languages (2):\neng\ntur\n"
        stderr = ""

    monkeypatch.setattr(tools_mod.subprocess, "run", lambda *a, **k: _Proc())
    assert _tesseract_langs() == {"eng", "tur"}


# ── OCR language substitution ───────────────────────────────────────────────
def _ok_proc(text):
    class _Proc:
        returncode = 0
        stdout = text
        stderr = ""
    return _Proc


def test_a_substituted_language_is_announced_in_the_output(img, monkeypatch):
    # Reading Turkish with the wrong language data does not fail — it returns
    # confidently wrong text with every ğ, ı, ş and ç quietly flattened, and
    # nothing downstream can tell that from a clean read.
    monkeypatch.setattr(tools_mod.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(tools_mod, "_tesseract_langs", lambda: {"afr", "osd"})
    monkeypatch.setattr(tools_mod.subprocess, "run",
                        lambda *a, **k: _ok_proc("Degisiklikler kaydedilmedi")())

    res = tools_mod.ocr_image(img)

    assert res.ok
    assert "Warning" in res.output
    assert "afr" in res.output
    assert "tesseract-data-tur" in res.output, "it should say what to install"
    assert "Degisiklikler kaydedilmedi" in res.output, "the text still comes through"


def test_no_warning_when_the_right_language_is_available(img, monkeypatch):
    monkeypatch.setattr(tools_mod.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(tools_mod, "_tesseract_langs", lambda: {"tur", "eng"})
    monkeypatch.setattr(tools_mod.subprocess, "run",
                        lambda *a, **k: _ok_proc("Değişiklikler kaydedilmedi")())

    res = tools_mod.ocr_image(img)

    assert res.output == "Değişiklikler kaydedilmedi"


def test_english_alone_is_not_treated_as_a_substitution(img, monkeypatch):
    # eng is one of the two the code actually wants — a real choice, not a
    # fallback, even when Turkish is missing.
    monkeypatch.setattr(tools_mod.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(tools_mod, "_tesseract_langs", lambda: {"eng", "osd"})
    monkeypatch.setattr(tools_mod.subprocess, "run", lambda *a, **k: _ok_proc("Hello")())

    assert "Warning" not in tools_mod.ocr_image(img).output


def test_an_explicit_language_is_never_second_guessed(img, monkeypatch):
    monkeypatch.setattr(tools_mod.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(tools_mod, "_tesseract_langs", lambda: {"afr", "osd"})
    monkeypatch.setattr(tools_mod.subprocess, "run", lambda *a, **k: _ok_proc("text")())

    assert "Warning" not in tools_mod.ocr_image(img, lang="afr").output
