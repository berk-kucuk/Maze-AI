"""Tests for Arch news, command explanations, the vision stand-in and folder chat."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from maze_ai.agent import explain, news
from maze_ai.agent import tools as tl
from maze_ai.agent.agent import Agent
from maze_ai.config import MODE_AUTO, MODE_CHAT
from maze_ai.folder_index import FolderIndex, chunk_text
from maze_ai.llm.base import LLMBackend

FEED = """<?xml version="1.0"?><rss><channel>
<item><title>Foo &gt;= 2 requires manual intervention</title>
<link>https://archlinux.org/news/foo/</link>
<pubDate>Tue, 22 Sep 2026 09:09:27 +0000</pubDate>
<description>&lt;p&gt;Run &lt;code&gt;pacman -Syu --overwrite x&lt;/code&gt;&lt;/p&gt;</description></item>
<item><title>Election results</title><link>https://archlinux.org/news/vote/</link>
<pubDate>Thu, 04 Jun 2026 10:00:00 +0000</pubDate><description>Hi</description></item>
</channel></rss>"""


# ── Arch news ───────────────────────────────────────────────────────────────
def test_feed_is_parsed_and_action_items_flagged():
    items = news.parse_feed(FEED)
    assert [i.title for i in items] == ["Foo >= 2 requires manual intervention",
                                        "Election results"]
    assert items[0].needs_action and not items[1].needs_action
    assert "pacman -Syu --overwrite x" in items[0].summary


def test_last_upgrade_comes_from_pacman_log(tmp_path):
    log = tmp_path / "pacman.log"
    log.write_text("[2026-08-01T10:00:00+0300] [PACMAN] starting full system upgrade\n"
                   "[2026-09-01T10:00:00+0300] [ALPM] upgraded foo\n"
                   "[2026-09-10T12:30:00+0300] [PACMAN] starting full system upgrade\n")
    assert news.last_upgrade(log).isoformat() == "2026-09-10T12:30:00+03:00"
    assert news.last_upgrade(tmp_path / "missing") is None


def test_report_warns_about_news_since_the_last_upgrade(monkeypatch):
    monkeypatch.setattr(news, "fetch_news", lambda: news.parse_feed(FEED))
    monkeypatch.setattr(news, "last_upgrade",
                        lambda: datetime(2026, 9, 1, tzinfo=timezone.utc))
    ok, text = news.news_report()
    assert ok and "MANUAL INTERVENTION — Foo" in text and "WARN the user" in text
    assert "Election" not in text            # older than the last upgrade


def test_report_after_a_fresh_upgrade_is_calm(monkeypatch):
    monkeypatch.setattr(news, "fetch_news", lambda: news.parse_feed(FEED))
    monkeypatch.setattr(news, "last_upgrade",
                        lambda: datetime(2026, 9, 30, tzinfo=timezone.utc))
    ok, text = news.news_report()
    assert ok and "No Arch news since the last upgrade" in text and "WARN" not in text


@pytest.mark.parametrize("message,expected", [
    ("sistemi güncelle", True), ("sistemimi güncellemek istiyorum", True),
    ("pacman -Syu yapayım mı", True), ("yay -Syu", True), ("update my system", True),
    ("güncelleme var mı", True), ("bu dosyayı güncelle", False),
    ("firefox güncellemesi", False), ("upgrade python package in venv", False),
])
def test_upgrade_requests(message, expected):
    assert news.is_upgrade_request(message) is expected


class _Scripted(LLMBackend):
    name = "scripted"

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.systems: list[str] = []
        self.seen_images: list = []

    def chat(self, messages):
        self.systems.append(messages[0]["content"])
        self.seen_images.append([m.get("images") for m in messages if m.get("images")])
        return self.replies.pop(0) if self.replies else json.dumps(
            {"thought": "", "action": "final_answer", "action_input": {"answer": "ok"}})

    def available_models(self):
        return []


def test_agent_checks_news_before_an_upgrade(monkeypatch):
    monkeypatch.setattr(news, "fetch_news", lambda: news.parse_feed(FEED))
    monkeypatch.setattr(news, "last_upgrade",
                        lambda: datetime(2026, 9, 1, tzinfo=timezone.utc))
    backend = _Scripted()
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    events = []
    agent.run("sistemi güncellemek istiyorum", events.append, lambda r: False)
    assert "MANUAL INTERVENTION" in backend.systems[0]
    assert any(e.kind == "tool_call" and e.tool == "arch_news" for e in events)


# ── command explanations ────────────────────────────────────────────────────
PACMAN_MAN = """NAME
     pacman - package manager utility

OPERATIONS
     -Q, --query
         Query the package database.

     -R, --remove
         Remove package(s) from the system.

QUERY OPTIONS (APPLY TO -Q)
     -d, --deps
         Restrict output to packages installed as dependencies.

     -n, --native
         Restrict output to packages found in the sync database.

REMOVE OPTIONS (APPLY TO -R)
     -n, --nosave
         Instructs pacman to ignore file backup designations.

     -s, --recursive
         Remove each target including all of their dependencies.
"""


@pytest.fixture
def manuals(monkeypatch):
    pages = {"pacman": PACMAN_MAN}
    explain._manual.cache_clear()
    monkeypatch.setattr(explain, "_manual", lambda name: pages.get(name, ""))
    return pages


def test_bundled_flags_use_the_operations_own_section(manuals):
    parts = {p.text: p.meaning for p in explain.explain_command("pacman -Rns foo")}
    assert parts["pacman"] == "package manager utility"
    assert parts["-R"] == "Remove package(s) from the system."
    assert parts["-n"] == "Instructs pacman to ignore file backup designations."
    assert parts["-s"].startswith("Remove each target")


def test_the_same_flag_under_another_operation(manuals):
    parts = {p.text: p.meaning for p in explain.explain_command("pacman -Qn")}
    assert parts["-n"] == "Restrict output to packages found in the sync database."


def test_shell_syntax_builtins_and_signals_are_explained(manuals):
    parts = {p.text: p.meaning for p in explain.explain_command(
        "cd /tmp && kill -9 42 | pacman -Q > out.txt 2>/dev/null")}
    assert "change the current directory" in parts["cd"]
    assert "succeeded" in parts["&&"]
    assert "force-stop" in parts["-9"]
    assert "replacing" in parts[">"]
    assert "hide error" in parts["2>/dev/null"]


def test_chmod_modes(manuals):
    parts = {p.text: p.meaning for p in explain.explain_command("chmod 754 a.sh")}
    assert "owner: rwx" in parts["754"] and "group: rx" in parts["754"]


def test_unknown_programs_explain_nothing_rather_than_guess(manuals):
    assert explain.explain_command("frobnicate --wibble") == []


# ── vision stand-in ─────────────────────────────────────────────────────────
class _Ollamaish(_Scripted):
    supports_vision = False

    def __init__(self, model="blind:3b"):
        super().__init__()
        self.model = model

    def installed_models(self):
        return [{"name": "blind:3b", "size": 2, "capabilities": ["tools"]},
                {"name": "eyes:4b", "size": 4, "capabilities": ["vision", "tools"]}]

    def model_info(self, name=""):
        return {}

    def usable_vram(self):
        return (0, 0)


def test_an_image_is_answered_by_a_model_that_can_see(tmp_path):
    shot = tmp_path / "x.png"
    shot.write_bytes(b"\x89PNG\r\n")
    used = []

    class Recording(_Ollamaish):
        def chat(self, messages):
            used.append(self.model)
            return super().chat(messages)

    backend = Recording()
    agent = Agent(backend, mode=MODE_CHAT, stream_responses=False)
    events = []
    agent.run("what is this?", events.append, lambda r: False, images=[str(shot)])
    assert used == ["eyes:4b"]
    assert agent.backend is backend and backend.model == "blind:3b"   # restored
    notice = next(e for e in events if e.kind == "notice")
    assert notice.args == {"model": "eyes:4b", "current": "blind:3b"}
    assert agent.can_see()


def test_no_stand_in_without_an_image(tmp_path):
    backend = _Ollamaish()
    agent = Agent(backend, mode=MODE_CHAT, stream_responses=False)
    events = []
    agent.run("hello", events.append, lambda r: False)
    assert not any(e.kind == "notice" for e in events)


# ── folder chat ─────────────────────────────────────────────────────────────
@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("maze_ai.folder_index.CACHE_DIR", tmp_path / "cache")
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "db.py").write_text(
        "import sqlite3\n\n\ndef connect(path):\n    return sqlite3.connect(path)\n")
    (root / "README.md").write_text("# Demo\nA tiny demo that stores notes in SQLite.\n")
    (root / ".env").write_text("API_KEY=secret123\n")
    (root / "node_modules" / "x").mkdir(parents=True)
    (root / "node_modules" / "x" / "i.js").write_text("connect()")
    (root / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")
    return root


def test_index_reads_text_files_and_skips_secrets_and_junk(project):
    index = FolderIndex(project)
    stats = index.update()
    assert stats.files == 2
    paths = {r[0] for r in index.db.execute("SELECT DISTINCT path FROM chunks")}
    assert paths == {"src/db.py", "README.md"}
    hits = index.search("database connect sqlite")
    assert hits and hits[0].path in {"src/db.py", "README.md"}
    assert "secret123" not in "".join(h.text for h in index.search("API_KEY secret"))
    assert oct(index.db_path.stat().st_mode & 0o777) == "0o600"


def test_index_only_rereads_what_changed(project):
    index = FolderIndex(project)
    index.update()
    (project / "README.md").write_text("# Demo\nNow with a changelog parser.\n")
    (project / "src" / "db.py").unlink()
    stats = index.update()
    assert stats.files == 1
    assert index.search("changelog")[0].path == "README.md"
    assert index.search("sqlite3") == [] or index.search("sqlite3")[0].path == "README.md"


def test_vectors_are_fused_with_keywords(project):
    def embed(texts):
        # "storage" questions sit next to db.py in this toy space.
        return [[1.0, 0.0] if ("sqlite3" in t or "storage" in t) else [0.0, 1.0] for t in texts]

    index = FolderIndex(project)
    stats = index.update(embed=embed, embed_model="toy")
    assert stats.embedded
    assert index.search("where is storage handled", embed=embed)[0].path == "src/db.py"


def test_chunks_follow_line_boundaries():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(100))
    chunks = chunk_text(text, size=500)
    assert chunks[0][0] == 1 and chunks[-1][1] == 100
    assert all(start <= end for start, end, _ in chunks)


def test_agent_answers_from_the_attached_folder(project):
    backend = _Scripted()
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    agent.context["folder"] = str(project)
    events = []
    agent.run("how does the app connect to the database?", events.append, lambda r: False)
    system = backend.systems[0]
    assert "# Working folder" in system and "src/db.py:" in system
    assert "[Result of search_folder" in system
    assert "search_folder" in agent.enabled_tools()
    assert any(e.kind == "tool_call" and e.tool == "search_folder" for e in events)


def test_search_folder_tool_goes_to_the_agents_index(project):
    reply = json.dumps({"thought": "", "action": "search_folder",
                        "action_input": {"query": "changelog notes"}})
    backend = _Scripted([reply])
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    agent.context["folder"] = str(project)
    events = []
    agent.run("anything about notes?", events.append, lambda r: False)
    results = [e for e in events if e.kind == "tool_result" and e.tool == "search_folder"]
    assert len(results) == 2 and "README.md" in results[-1].text


def test_no_folder_no_folder_tools():
    agent = Agent(_Scripted(), mode=MODE_AUTO)
    assert "search_folder" not in agent.enabled_tools()
    assert not tl.search_folder("x").ok


# ── an attached folder is context, not a cage ───────────────────────────────
def test_empty_folder_doesnt_inject_a_dead_end(tmp_path, monkeypatch):
    monkeypatch.setattr("maze_ai.folder_index.CACHE_DIR", tmp_path / "cache")
    empty = tmp_path / "Downloads"
    empty.mkdir()
    backend = _Scripted()
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    agent.context["folder"] = str(empty)
    events = []
    agent.run("2. dünya savaşını araştırıp bu klasöre bir txt yaz", events.append,
              lambda r: False)
    system = backend.systems[0]
    assert "working folder" in system and "does not limit you" in system
    assert "search_folder" not in agent.enabled_tools()
    assert "No passage" not in system
    assert not any(e.tool == "search_folder" for e in events)
    assert agent.cwd == str(empty)                 # new files land in the folder


def test_echoed_markers_never_reach_the_user():
    leaky = ("<<<TOOL_OUTPUT\nNo passage in /x matches 'q'.\nTOOL_OUTPUT>>> "
             "Here is the answer.")
    backend = _Scripted([leaky])
    agent = Agent(backend, mode=MODE_CHAT, stream_responses=False)
    events = []
    answer = agent.run("hi", events.append, lambda r: False)
    final = next(e.text for e in events if e.kind == "final")
    for text in (answer, final, agent.history[-1]["content"]):
        assert "TOOL_OUTPUT" not in text and "Here is the answer." in text
