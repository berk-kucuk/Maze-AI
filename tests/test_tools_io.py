"""Tests for the file/command tools: clipping, paging, backups, undo."""

from __future__ import annotations

import pytest

from maze_ai.agent import tools


@pytest.fixture
def backups(tmp_path, monkeypatch):
    """Point the backup store at a temp dir so tests never touch real data."""
    directory = tmp_path / "backups"
    monkeypatch.setattr(tools, "_BACKUP_DIR", directory)
    monkeypatch.setattr(tools, "_BACKUP_INDEX", directory / "index.json")
    return directory


# ── output clipping ────────────────────────────────────────────────────────
def test_clip_keeps_both_ends():
    text = "START" + ("x" * (tools.MAX_OUTPUT * 2)) + "END-OF-BUILD-ERROR"
    out = tools._clip(text)
    assert out.startswith("START")
    # The tail is what usually carries the error — it must survive.
    assert out.endswith("END-OF-BUILD-ERROR")
    assert "omitted from the middle" in out
    assert len(out) < len(text)


def test_clip_leaves_short_text_alone():
    assert tools._clip("hello") == "hello"


# ── read_file paging ───────────────────────────────────────────────────────
def test_read_file_slice(tmp_path):
    f = tmp_path / "many.txt"
    f.write_text("\n".join(f"line {i}" for i in range(1, 101)))
    res = tools.read_file(path=str(f), offset=10, limit=3)
    assert res.ok
    assert "line 10\nline 11\nline 12" in res.output
    assert "line 13" not in res.output
    assert "of 100" in res.output


def test_read_file_whole_file_still_works(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert tools.read_file(path=str(f)).output == "hello"


def test_read_file_missing(tmp_path):
    assert not tools.read_file(path=str(tmp_path / "nope")).ok


# ── backups and undo ───────────────────────────────────────────────────────
def test_write_file_backs_up_the_previous_version(tmp_path, backups):
    f = tmp_path / "conf.txt"
    f.write_text("original")
    res = tools.write_file(path=str(f), content="replaced")
    assert res.ok and f.read_text() == "replaced"
    assert "backed up" in res.output

    undo = tools.undo_file_change(path=str(f))
    assert undo.ok, undo.output
    assert f.read_text() == "original"


def test_edit_file_is_undoable(tmp_path, backups):
    f = tmp_path / "app.py"
    f.write_text("debug = True\n")
    tools.edit_file(path=str(f), old="True", new="False")
    assert f.read_text() == "debug = False\n"
    tools.undo_file_change(path=str(f))
    assert f.read_text() == "debug = True\n"


def test_deleted_file_can_be_restored(tmp_path, backups):
    f = tmp_path / "notes.txt"
    f.write_text("important")
    assert tools.delete_path(path=str(f)).ok
    assert not f.exists()
    assert tools.undo_file_change(path=str(f)).ok
    assert f.read_text() == "important"


def test_undo_without_any_backup(tmp_path, backups):
    assert not tools.undo_file_change(path=str(tmp_path / "never.txt")).ok


def test_new_file_write_has_nothing_to_back_up(tmp_path, backups):
    f = tmp_path / "fresh.txt"
    res = tools.write_file(path=str(f), content="hi")
    assert res.ok and "backed up" not in res.output


# ── run_command ────────────────────────────────────────────────────────────
def test_run_command_reports_its_directory(tmp_path):
    res = tools.run_command(command="pwd", cwd=str(tmp_path))
    assert res.ok
    assert res.cwd == str(tmp_path)


def test_run_command_tracks_a_cd(tmp_path):
    sub = tmp_path / "inner"
    sub.mkdir()
    res = tools.run_command(command=f"cd {sub} && pwd", cwd=str(tmp_path))
    assert res.cwd == str(sub)
    assert "working directory is now" in res.output


def test_run_command_marker_is_not_leaked(tmp_path):
    res = tools.run_command(command="echo hello", cwd=str(tmp_path))
    assert res.output == "hello"


def test_failing_command_keeps_its_exit_code(tmp_path):
    res = tools.run_command(command="exit 3", cwd=str(tmp_path))
    assert not res.ok and "exit code 3" in res.output


# ── network guards ─────────────────────────────────────────────────────────
def test_fetch_url_refuses_local_addresses():
    res = tools.fetch_url(url="http://127.0.0.1:11434/api/tags")
    assert not res.ok and "private" in res.output.lower()


def test_fetch_url_refuses_non_http_schemes():
    res = tools.fetch_url(url="file:///etc/passwd")
    assert not res.ok


# ── streamed download decoding ─────────────────────────────────────────────
def _fake_response(body: bytes, content_type: str = "text/plain", encoding=None):
    """A real requests.Response over an in-memory body, read as a stream."""
    import io

    import requests
    from urllib3 import HTTPResponse

    resp = requests.Response()
    resp.status_code = 200
    resp.headers.update({"Content-Type": content_type})
    resp.raw = HTTPResponse(body=io.BytesIO(body), preload_content=False, status=200)
    resp.encoding = encoding
    return resp


def test_capped_read_without_a_declared_charset():
    # requests' apparent_encoding raises once the stream is consumed, so the
    # reader must not reach for it.
    text, truncated = tools._read_capped(_fake_response("merhaba dünya".encode()))
    assert text == "merhaba dünya"
    assert not truncated


def test_capped_read_honours_the_declared_charset():
    body = "iğne".encode("iso-8859-9")
    text, _ = tools._read_capped(_fake_response(body, encoding="iso-8859-9"))
    assert text == "iğne"


def test_capped_read_stops_at_the_limit():
    text, truncated = tools._read_capped(_fake_response(b"x" * 5000), limit=1000)
    assert truncated and len(text) <= 1000


def test_capped_read_survives_an_unknown_charset():
    text, _ = tools._read_capped(_fake_response(b"hi", encoding="not-a-charset"))
    assert text == "hi"


def test_read_file_offset_past_the_end(tmp_path):
    f = tmp_path / "short.txt"
    f.write_text("one\ntwo\n")
    res = tools.read_file(path=str(f), offset=99, limit=5)
    assert not res.ok and "past the end" in res.output
