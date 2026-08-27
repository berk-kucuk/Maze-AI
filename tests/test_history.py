"""Tests for conversation persistence."""

from __future__ import annotations

import json

import pytest

from maze_ai import history
from maze_ai.history import ChatStore, Conversation, export_markdown


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "CHATS_DIR", tmp_path / "chats")
    return ChatStore()


def test_saved_messages_never_carry_attachments(store):
    conv = Conversation()
    conv.messages.append({"role": "user", "content": "look", "images": ["/tmp/a.png"]})
    conv.messages.append({"role": "assistant", "content": "nice photo"})
    store.save(conv)

    on_disk = json.loads((history.CHATS_DIR / f"{conv.id}.json").read_text())
    assert "images" not in on_disk["messages"][0]


def test_legacy_attachments_are_dropped_on_load(store):
    conv = Conversation()
    conv.messages.append({"role": "user", "content": "hi"})
    store.save(conv)
    # Simulate a file written by an older version.
    path = history.CHATS_DIR / f"{conv.id}.json"
    data = json.loads(path.read_text())
    data["messages"][0]["images"] = ["/tmp/old.png"]
    path.write_text(json.dumps(data))

    loaded = store.load(conv.id)
    assert "images" not in loaded.messages[0]


def test_listing_reuses_parsed_files(store):
    conv = Conversation(title="Kept")
    conv.messages.append({"role": "user", "content": "hi"})
    store.save(conv)
    first = store.list_conversations()
    second = store.list_conversations()
    assert first[0] is second[0], "unchanged files should not be re-parsed"


def test_listing_notices_a_changed_file(store):
    conv = Conversation()
    conv.messages.append({"role": "user", "content": "hi"})
    store.save(conv)
    store.list_conversations()
    store.rename(conv.id, "Renamed")
    assert store.list_conversations()[0].title == "Renamed"


def test_deleted_chat_leaves_the_listing(store):
    conv = Conversation()
    conv.messages.append({"role": "user", "content": "hi"})
    store.save(conv)
    store.delete(conv.id)
    assert store.list_conversations() == []


def test_export_markdown_round_trip(store):
    conv = Conversation(title="Notes")
    conv.messages += [
        {"role": "user", "content": "merhaba"},
        {"role": "assistant", "content": "selam"},
    ]
    text = export_markdown(conv)
    assert "# Notes" in text and "merhaba" in text and "selam" in text
