"""Tests for live answer streaming, including out of schema-constrained JSON."""

from __future__ import annotations

from maze_ai.agent.agent import Agent, AgentEvent, AnswerStreamer
from maze_ai.config import MODE_AUTO
from maze_ai.llm.base import LLMBackend, LLMReply


# ── the incremental extractor ──────────────────────────────────────────────
def feed_all(chunks: list[str]) -> tuple[list[str], str]:
    streamer = AnswerStreamer()
    buf, deltas = "", []
    for chunk in chunks:
        buf += chunk
        delta = streamer.feed(buf)
        if delta:
            deltas.append(delta)
    return deltas, streamer.text


def test_answer_is_extracted_as_it_arrives():
    deltas, text = feed_all(['{"action": "final_ans', 'wer", "action_input": ',
                             '{"answer": "Merhaba', ' dünya"}}'])
    assert deltas == ["Merhaba", " dünya"]
    assert text == "Merhaba dünya"


def test_escapes_split_across_chunks_are_handled():
    deltas, text = feed_all(['{"answer": "line', '\\', 'n2 \\u00e7', 'ay"}'])
    assert text == "line\n2 çay"
    assert "".join(deltas) == text


def test_nothing_is_emitted_before_the_answer_field():
    deltas, text = feed_all(['{"thought": "I should check the disk"'])
    assert deltas == [] and text == ""


def test_a_tool_call_object_yields_nothing():
    deltas, _ = feed_all(['{"action": "run_command", "action_input": '
                          '{"command": "rm -rf /tmp/x"}}'])
    assert deltas == []


# ── the agent's streaming decision ─────────────────────────────────────────
class ScriptedStream(LLMBackend):
    """Streams a canned reply one character-group at a time."""

    name = "scripted"
    supports_schema = True

    def __init__(self, reply: str, chunk: int = 12) -> None:
        self.reply = reply
        self.chunk = chunk
        self.schema_seen = None

    def chat(self, messages):
        return self.reply

    def chat_ex(self, messages, *, tools=None, schema=None, stream=False,
                on_text=None, on_thinking=None):
        self.schema_seen = schema
        if stream and on_text:
            for i in range(0, len(self.reply), self.chunk):
                on_text(self.reply[i:i + self.chunk])
        return LLMReply(text=self.reply)

    def available_models(self):
        return ["scripted"]


def run_agent(reply: str, **kwargs):
    agent = Agent(ScriptedStream(reply), mode=MODE_AUTO, stream_responses=True, **kwargs)
    events: list[AgentEvent] = []
    answer = agent.run("q", events.append, lambda r: True)
    streamed = "".join(e.text for e in events if e.kind == "stream")
    return agent, answer, streamed


def test_json_final_answer_streams_the_answer_text():
    agent, answer, streamed = run_agent(
        '{"thought": "easy", "action": "final_answer", '
        '"action_input": {"answer": "Root disk is 66% full."}}'
    )
    assert streamed == "Root disk is 66% full."
    assert answer == "Root disk is 66% full."
    # …and never the raw JSON.
    assert "action_input" not in streamed


def test_tool_call_json_streams_nothing(tmp_path):
    _, _, streamed = run_agent(
        '{"thought": "look", "action": "list_dir", '
        f'"action_input": {{"path": "{tmp_path}"}}}}'
    )
    assert streamed == ""


def test_plain_prose_still_streams():
    _, answer, streamed = run_agent("Just a friendly hello.")
    assert streamed == "Just a friendly hello."
    assert answer == "Just a friendly hello."


def test_schema_is_sent_when_the_backend_supports_it():
    agent, _, _ = run_agent('{"action": "final_answer", "action_input": {"answer": "hi"}}')
    assert agent.backend.schema_seen is not None
    assert agent.backend.schema_seen["properties"]["action"]["enum"]


def test_schema_can_be_switched_off():
    agent, _, _ = run_agent(
        '{"action": "final_answer", "action_input": {"answer": "hi"}}',
        constrain_json=False,
    )
    assert agent.backend.schema_seen is None
