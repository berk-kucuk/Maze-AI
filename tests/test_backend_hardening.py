"""Regression tests for the backend fixes: VRAM-aware sizing, tool robustness,
the native-tools fallback and context budgeting."""

from __future__ import annotations

import json
import time

import pytest

from maze_ai.agent import tools as tl
from maze_ai.agent.agent import Agent
from maze_ai.config import MODE_AUTO, MODE_CHAT
from maze_ai.llm import ollama_backend as ob
from maze_ai.llm.base import LLMBackend, LLMError, LLMReply, ToolCall
from maze_ai.llm.hardware import GB, kv_bytes_per_token
from maze_ai.llm.openai_backend import OpenAIBackend


# ── KV cache sizing from model metadata ────────────────────────────────────
def test_kv_per_token_for_a_plain_transformer():
    # llama3.1-8b: 32 layers, 8 KV heads of 128 dims → 128 kB a token.
    info = {
        "general.architecture": "llama",
        "llama.block_count": 32,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.embedding_length": 4096,
    }
    assert kv_bytes_per_token(info) == 32 * 8 * (128 + 128) * 2


def test_kv_per_token_counts_only_attention_layers_of_a_hybrid():
    # qwen3.5: linear attention everywhere except every fourth layer.
    info = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 32,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.embedding_length": 4096,
        "qwen35.full_attention_interval": 4,
    }
    assert kv_bytes_per_token(info) == 8 * 4 * 512 * 2


def test_kv_per_token_skips_sliding_window_and_shared_layers():
    pattern = [True, True, True, True, False] * 7          # 35 layers
    info = {
        "general.architecture": "gemma4",
        "gemma4.block_count": 35,
        "gemma4.attention.head_count": 8,
        "gemma4.attention.head_count_kv": 1,
        "gemma4.attention.key_length": 512,
        "gemma4.attention.value_length": 512,
        "gemma4.attention.sliding_window": 512,
        "gemma4.attention.sliding_window_pattern": pattern,
        "gemma4.attention.shared_kv_layers": 20,
        "gemma4.embedding_length": 1536,
    }
    # Only the first 15 layers own a cache; of those, 3 are global.
    assert kv_bytes_per_token(info) == 3 * 1 * 1024 * 2


def test_kv_per_token_unknown_without_metadata():
    assert kv_bytes_per_token({}) == 0
    assert kv_bytes_per_token({"general.architecture": "x"}) == 0


def test_a_heavier_cache_means_a_smaller_context():
    light = ob.auto_context(131072, ram_gb=64, weights_bytes=5 * GB,
                            vram_bytes=8 * GB, per_token=16 * 1024)
    heavy = ob.auto_context(131072, ram_gb=64, weights_bytes=5 * GB,
                            vram_bytes=8 * GB, per_token=160 * 1024)
    assert heavy < light


def test_auto_context_is_pinned_once_known(monkeypatch):
    backend = ob.OllamaBackend(model="m", num_ctx="auto")
    backend._info_cache["m"] = {"context_length": 131072, "capabilities": []}
    usable = {"bytes": 8 * GB}
    monkeypatch.setattr(backend, "usable_vram", lambda: (usable["bytes"], 0))
    monkeypatch.setattr(backend, "weights_bytes", lambda model="": 5 * GB)
    first = backend.resolved_ctx()
    # Another app grabs VRAM: re-sizing now would force Ollama to reload.
    usable["bytes"] = 6 * GB
    assert backend.resolved_ctx() == first


# ── tool argument types ────────────────────────────────────────────────────
def test_string_booleans_are_not_truthy():
    args = tl.coerce_args("screenshot", {"region": "false", "save_path": "/tmp/x.png"})
    assert args == {"region": False, "save_path": "/tmp/x.png"}
    assert tl.coerce_args("edit_file", {"all": "true"})["all"] is True


def test_numbers_and_junk_are_normalised():
    assert tl.coerce_args("read_file", {"path": "a", "offset": "10", "limit": 5.0}) == {
        "path": "a", "offset": 10, "limit": 5,
    }
    assert tl.coerce_args("read_file", {"path": "a", "limit": "lots"}) == {"path": "a"}
    assert tl.coerce_args("list_dir", {"path": None, "bogus": 1}) == {}
    assert tl.coerce_args("write_file", {"path": "a", "content": {"k": 1}})["content"] == '{"k": 1}'


# ── run_command ─────────────────────────────────────────────────────────────
def test_timeout_kills_background_children(tmp_path):
    started = time.monotonic()
    # The background sleep keeps the output pipe open; before, subprocess.run
    # killed only the shell and then waited on the pipe forever.
    result = tl.run_command("sleep 30 & sleep 30", timeout=1, cwd=str(tmp_path))
    assert not result.ok
    assert "timed out" in result.output
    assert time.monotonic() - started < 15


def test_commands_never_wait_for_input(tmp_path):
    result = tl.run_command("read line; echo got:$line", timeout=10, cwd=str(tmp_path))
    assert result.ok
    assert "got:" in result.output


def test_command_reports_exit_code_and_stderr(tmp_path):
    result = tl.run_command("echo out; echo err >&2; exit 3", cwd=str(tmp_path))
    assert not result.ok
    assert "exit code 3" in result.output
    assert "out" in result.output and "err" in result.output


# ── files ───────────────────────────────────────────────────────────────────
def test_binary_files_are_refused(tmp_path):
    blob = tmp_path / "x.bin"
    blob.write_bytes(b"\x7fELF\x00\x01\x02")
    result = tl.read_file(str(blob))
    assert not result.ok and "binary" in result.output


def test_huge_files_are_paged_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "MAX_READ_BYTES", 100)
    big = tmp_path / "big.log"
    big.write_text("".join(f"line {i}\n" for i in range(1, 500)))
    result = tl.read_file(str(big), offset=10, limit=3)
    assert result.ok
    assert "line 10\nline 11\nline 12" in result.output
    assert "line 13" not in result.output


def test_copy_over_a_file_keeps_a_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(tl, "_BACKUP_INDEX", tmp_path / "backups" / "index.json")
    src, dst = tmp_path / "a.txt", tmp_path / "b.txt"
    src.write_text("new")
    dst.write_text("precious")
    assert tl.copy_path(str(src), str(dst)).ok
    assert dst.read_text() == "new"
    assert tl.undo_file_change(str(dst)).ok
    assert dst.read_text() == "precious"


def test_copy_into_a_directory_keeps_the_name(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("x")
    (tmp_path / "d").mkdir()
    assert tl.copy_path(str(src), str(tmp_path / "d")).ok
    assert (tmp_path / "d" / "a.txt").read_text() == "x"


def test_search_skips_ignored_directories(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("needle")
    (tmp_path / "src.py").write_text("needle")
    result = tl.search_files(str(tmp_path), "needle")
    assert "src.py" in result.output and "node_modules" not in result.output


def test_search_gives_up_after_its_time_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "SEARCH_SECONDS", -1)
    (tmp_path / "a.txt").write_text("needle")
    result = tl.search_files(str(tmp_path), "needle")
    assert "Stopped after" in result.output


# ── agent: paths, arguments and budgets ─────────────────────────────────────
class Native(LLMBackend):
    name = "native"
    supports_native_tools = True
    model = "m"

    def __init__(self, replies, error=""):
        self.replies = list(replies)
        self.error = error
        self.calls = []

    def chat(self, messages):
        self.calls.append(("chat", messages))
        return self.replies.pop(0) if self.replies else "done"

    def chat_ex(self, messages, *, tools=None, schema=None, stream=False,
                on_text=None, on_thinking=None):
        self.calls.append(("chat_ex", messages, tools))
        if self.error and tools:
            raise LLMError(self.error)
        reply = self.replies.pop(0) if self.replies else LLMReply(text="done")
        if isinstance(reply, str):
            reply = LLMReply(text=reply)
        return reply

    def available_models(self):
        return []


def _run(agent, text="go"):
    events = []
    answer = agent.run(text, events.append, lambda _r: True)
    return answer, events


def test_relative_paths_follow_the_session_directory(tmp_path):
    (tmp_path / "notes.txt").write_text("hello from cwd")
    backend = Native([
        LLMReply(tool_calls=[ToolCall("read_file", {"path": "notes.txt"})]),
        LLMReply(text="ok"),
    ])
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    agent.cwd = str(tmp_path)
    _, events = _run(agent)
    result = next(e for e in events if e.kind == "tool_result")
    assert result.ok and "hello from cwd" in result.text


def test_relative_secret_paths_are_still_guarded(tmp_path, monkeypatch):
    agent = Agent(Native([]), mode=MODE_AUTO)
    agent.cwd = str(tmp_path)
    args = agent._prepare_args("read_file", {"path": ".ssh/id_rsa"})
    reason, _ = agent._approval_reason("read_file", args)
    assert reason is not None


def test_a_refused_native_call_falls_back_to_the_json_protocol():
    backend = Native([json.dumps({"thought": "", "action": "final_answer",
                                  "action_input": {"answer": "via protocol"}})],
                     error='registry.ollama.ai/library/x does not support tools')
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    answer, events = _run(agent)
    assert answer == "via protocol"
    assert not any(e.kind == "error" for e in events)
    assert not agent.uses_native_tools()


def test_cancel_keeps_the_text_already_streamed():
    class Streaming(Native):
        def chat_ex(self, messages, *, tools=None, schema=None, stream=False,
                    on_text=None, on_thinking=None):
            on_text("First half of the answer. ")
            agent.request_cancel()
            on_text("never shown")
            return LLMReply(text="unreachable")

    agent = Agent(Streaming([]), mode=MODE_AUTO, stream_responses=True)
    answer, _ = _run(agent)
    assert answer == "First half of the answer."


class SmallWindow(Native):
    supports_native_tools = False

    def resolved_ctx(self):
        return 4096


def test_history_is_trimmed_to_fit_a_small_window():
    backend = SmallWindow(["fine"])
    agent = Agent(backend, mode=MODE_CHAT, stream_responses=False,
                  context_char_budget=24000)
    for i in range(40):
        agent.history.append({"role": "user", "content": f"q{i} " + "x" * 500})
        agent.history.append({"role": "assistant", "content": f"a{i} " + "y" * 500})
    _run(agent, "latest question")
    sent = backend.calls[-1][1]
    chars = sum(len(m["content"]) for m in sent)
    assert chars // 3 < 4096          # fits the window, with room to answer
    assert sent[-1]["content"] == "latest question"


def test_tool_output_is_clipped_to_the_window(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("z" * 50_000)
    backend = SmallWindow([
        json.dumps({"thought": "", "action": "read_file",
                    "action_input": {"path": str(big)}}),
        json.dumps({"thought": "", "action": "final_answer",
                    "action_input": {"answer": "ok"}}),
    ])
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    _run(agent)
    observation = backend.calls[-1][1][-1]["content"]
    assert len(observation) < 4096 * 3 // 3 + 500


def test_protocol_schema_lists_only_enabled_tools():
    class Schema(Native):
        supports_schema = True

    agent = Agent(Schema([]), tool_groups=["web"])
    enum = agent._schema()["properties"]["action"]["enum"]
    assert set(enum) == {"fetch_url", "web_search", "final_answer"}


# ── OpenAI-compatible: reasoning models reject temperature ──────────────────
class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def test_openai_retries_without_temperature(monkeypatch):
    sent = []

    def fake(method, url, **kwargs):
        sent.append(kwargs["json"])
        if "temperature" in kwargs["json"]:
            return _Resp(400, {"error": {"message": "Unsupported value: 'temperature' "
                                         "does not support 0.4", "param": "temperature"}})
        return _Resp(200, {"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr("maze_ai.llm.openai_backend.request_with_retry", fake)
    backend = OpenAIBackend(model="gpt-5")
    assert backend.chat([{"role": "user", "content": "x"}]) == "hi"
    assert backend.chat([{"role": "user", "content": "x"}]) == "hi"
    # One refusal, then the parameter is no longer sent at all.
    assert ["temperature" in p for p in sent] == [True, False, False]


@pytest.fixture(autouse=True)
def _no_real_ollama(monkeypatch):
    """Nothing here may talk to a real server."""
    def refuse(*_a, **_k):
        raise ob.requests.ConnectionError("offline in tests")
    monkeypatch.setattr(ob.requests, "get", refuse)
    monkeypatch.setattr(ob.requests, "post", refuse)
