"""Tests for the native function-calling loop and the tool-group budget."""

from __future__ import annotations

from maze_ai.agent.agent import Agent, AgentEvent
from maze_ai.agent.tools import TOOL_GROUPS, tool_schemas, tools_for_groups
from maze_ai.config import MODE_ASK, MODE_AUTO
from maze_ai.llm.base import LLMBackend, LLMReply, ToolCall


class NativeBackend(LLMBackend):
    """A backend that answers with scripted LLMReply objects."""

    name = "native-fake"
    supports_native_tools = True
    supports_schema = True

    def __init__(self, replies: list[LLMReply]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages):  # pragma: no cover - the loop uses chat_ex
        return ""

    def chat_ex(self, messages, *, tools=None, schema=None, stream=False,
                on_text=None, on_thinking=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        reply = self.replies.pop(0) if self.replies else LLMReply(text="done")
        if stream and on_text and reply.text:
            on_text(reply.text)
        if stream and on_thinking and reply.thinking:
            on_thinking(reply.thinking)
        return reply

    def available_models(self):
        return ["native-fake"]


def call(tool: str, **args) -> LLMReply:
    return LLMReply(tool_calls=[ToolCall(name=tool, arguments=args)])


class Recorder:
    def __init__(self, approve: bool = True) -> None:
        self.events: list[AgentEvent] = []
        self.approvals: list = []
        self._approve = approve

    def emit(self, ev):
        self.events.append(ev)

    def approve(self, request):
        self.approvals.append(request)
        return self._approve

    def kinds(self):
        return [e.kind for e in self.events]

    def of(self, kind):
        return [e for e in self.events if e.kind == kind]


def make_agent(replies, **kwargs):
    kwargs.setdefault("mode", MODE_AUTO)
    kwargs.setdefault("stream_responses", False)
    return Agent(NativeBackend(replies), **kwargs)


# ── the loop ───────────────────────────────────────────────────────────────
def test_native_path_is_chosen_when_supported():
    agent = make_agent([LLMReply(text="hi")])
    assert agent.uses_native_tools()


def test_native_tools_can_be_turned_off():
    agent = make_agent([LLMReply(text="hi")], native_tools=False)
    assert not agent.uses_native_tools()


def test_tool_call_then_answer(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello")
    agent = make_agent([call("read_file", path=str(target)),
                        LLMReply(text="It says hello.")])
    rec = Recorder()
    answer = agent.run("read it", rec.emit, rec.approve)

    assert answer == "It says hello."
    assert rec.of("tool_result")[0].text == "hello"
    assert [m["role"] for m in agent.history] == ["user", "assistant"]
    # The result went back as a tool message, fenced as untrusted data.
    tool_msg = agent.backend.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_name"] == "read_file"
    assert "untrusted DATA" in tool_msg["content"]


def test_tool_definitions_are_sent(tmp_path):
    agent = make_agent([LLMReply(text="hi")])
    agent.run("hello", Recorder().emit, lambda r: True)
    tools = agent.backend.calls[0]["tools"]
    names = {t["function"]["name"] for t in tools}
    assert "run_command" in names
    assert all("parameters" in t["function"] for t in tools)


def test_thinking_becomes_a_trail_entry():
    agent = make_agent([LLMReply(text="answer", thinking="  weighing options  ")])
    rec = Recorder()
    agent.run("q", rec.emit, rec.approve)
    assert rec.of("thought")[0].text == "weighing options"


def test_narration_before_a_tool_closes_the_bubble(tmp_path):
    agent = make_agent([
        LLMReply(text="Let me check that.",
                 tool_calls=[ToolCall("list_dir", {"path": str(tmp_path)})]),
        LLMReply(text="Empty."),
    ])
    rec = Recorder()
    agent.run("look", rec.emit, rec.approve)
    assert rec.of("stream_end")[0].text == "Let me check that."


def test_denied_tool_is_reported_to_the_model(tmp_path):
    target = tmp_path / "x.txt"
    agent = make_agent([call("write_file", path=str(target), content="no"),
                        LLMReply(text="Understood.")], mode=MODE_ASK)
    rec = Recorder(approve=False)
    agent.run("write", rec.emit, rec.approve)
    assert not target.exists()
    assert "denied" in rec.kinds()
    assert "denied this action" in agent.backend.calls[1]["messages"][-1]["content"]


def test_sensitive_read_still_asks_in_autonomous_mode():
    agent = make_agent([call("read_file", path="~/.ssh/id_rsa"), LLMReply(text="no")])
    rec = Recorder(approve=False)
    agent.run("read my key", rec.emit, rec.approve)
    assert rec.approvals and "sensitive" in rec.approvals[0].reason


def test_repeated_identical_call_runs_once(tmp_path):
    same = call("list_dir", path=str(tmp_path))
    agent = make_agent([same, call("list_dir", path=str(tmp_path)),
                        LLMReply(text="ok")])
    rec = Recorder()
    agent.run("look twice", rec.emit, rec.approve)
    assert len(rec.of("tool_result")) == 1


def test_step_limit_wraps_up(tmp_path):
    replies = [call("list_dir", path=str(tmp_path / str(i))) for i in range(3)]
    replies.append(LLMReply(text="Out of steps, here's what I found."))
    agent = make_agent(replies, max_steps=3)
    rec = Recorder()
    answer = agent.run("explore", rec.emit, rec.approve)
    assert answer == "Out of steps, here's what I found."
    assert [m["role"] for m in agent.history] == ["user", "assistant"]


def test_cancel_mid_turn_records_the_turn(tmp_path):
    agent = make_agent([call("list_dir", path=str(tmp_path)), LLMReply(text="x")])
    original = agent.backend.chat_ex

    def cancelling(*args, **kwargs):
        agent.request_cancel()
        return original(*args, **kwargs)

    agent.backend.chat_ex = cancelling
    rec = Recorder()
    assert agent.run("stop", rec.emit, rec.approve) == "⏹ Stopped."
    assert [m["role"] for m in agent.history] == ["user", "assistant"]


def test_metrics_reach_the_ui():
    agent = make_agent([LLMReply(text="hi", metrics={
        "eval_count": 20, "eval_duration": 2_000_000_000, "load_duration": 5_000_000_000,
    })])
    rec = Recorder()
    agent.run("q", rec.emit, rec.approve)
    metrics = rec.of("metrics")[0].args
    assert metrics["tokens_per_second"] == 10.0
    assert metrics["load_seconds"] == 5.0


# ── tool groups ────────────────────────────────────────────────────────────
def test_disabled_group_is_not_offered():
    agent = make_agent([LLMReply(text="hi")], tool_groups=["shell"])
    assert set(agent.enabled_tools()) == set(TOOL_GROUPS["shell"])
    agent.run("hello", Recorder().emit, lambda r: True)
    names = {t["function"]["name"] for t in agent.backend.calls[0]["tools"]}
    assert names == set(TOOL_GROUPS["shell"])
    assert "read_file" not in names


def test_calling_a_disabled_tool_is_refused(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("secret plans")
    agent = make_agent([call("read_file", path=str(target)), LLMReply(text="cannot")],
                       tool_groups=["shell"])
    rec = Recorder()
    agent.run("read it", rec.emit, rec.approve)
    assert not rec.of("tool_result")
    assert "Unknown tool" in agent.backend.calls[1]["messages"][-1]["content"]


def test_every_tool_belongs_to_exactly_one_group():
    from maze_ai.agent.tools import TOOLS

    seen: list[str] = []
    for names in TOOL_GROUPS.values():
        seen.extend(names)
    assert sorted(seen) == sorted(TOOLS)
    assert len(seen) == len(set(seen))


def test_trimming_groups_shrinks_the_definition_block():
    import json

    everything = len(json.dumps(tool_schemas(tools_for_groups(None))))
    shell_only = len(json.dumps(tool_schemas(tools_for_groups(["shell"]))))
    assert shell_only < everything / 2


def test_compact_schemas_are_smaller():
    import json

    full = len(json.dumps(tool_schemas()))
    compact = len(json.dumps(tool_schemas(compact=True)))
    assert compact < full
