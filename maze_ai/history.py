"""Persistent chat history.

Each conversation is stored as a JSON file under
``~/.local/share/maze-ai/chats/<id>.json``. A conversation holds the clean
``user``/``assistant`` message list (the same list the agent replays as
context), plus a title and timestamps. The activity trail (thoughts / tool
calls) is intentionally not persisted — only the messages the model needs to
remember and that make sense to re-read.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .private import private_dir, tighten, write_private

DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "maze-ai"
CHATS_DIR = DATA_DIR / "chats"

#: What a conversation id may look like. Ids come from files on disk and from
#: UI signals; anything else (``../x``) must never become a path.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str = "New chat"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    messages: list[dict] = field(default_factory=list)  # [{"role","content"}]

    # ── derived ──────────────────────────────────────────────────────────
    @property
    def is_empty(self) -> bool:
        return not self.messages

    def touch(self) -> None:
        self.updated = time.time()

    def ensure_title(self) -> None:
        """Derive a title from the first user message if still default."""
        if self.title and self.title != "New chat":
            return
        for msg in self.messages:
            if msg.get("role") == "user":
                text = " ".join((msg.get("content") or "").split())
                if text:
                    self.title = text[:48] + ("…" if len(text) > 48 else "")
                return

    # ── serialisation ────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            # Attachments are turn-local: the paths are not worth persisting and
            # replaying them would re-send the images on every later turn.
            "messages": [
                {k: v for k, v in msg.items() if k != "images"}
                for msg in self.messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        messages = []
        for msg in data.get("messages") or []:
            if isinstance(msg, dict):
                # Older files stored the attached image paths on the message.
                # Replaying those would re-send (and re-bill) every picture on
                # every later turn, so drop them on load.
                messages.append({k: v for k, v in msg.items() if k != "images"})
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            title=str(data.get("title") or "New chat"),
            created=float(data.get("created") or time.time()),
            updated=float(data.get("updated") or time.time()),
            messages=messages,
        )


class ChatStore:
    """File-backed store for conversations."""

    def __init__(self) -> None:
        private_dir(CHATS_DIR.parent)
        private_dir(CHATS_DIR)
        # Chats saved by older versions were world-readable; fix them once.
        for path in CHATS_DIR.glob("*.json"):
            try:
                if path.stat().st_mode & 0o077:
                    tighten(path)
            except OSError:
                continue
        # path -> (mtime, size, Conversation). The sidebar re-lists after every
        # turn and on every search keystroke; without this, each refresh parsed
        # every chat file on disk again.
        self._cache: dict[str, tuple[float, int, Conversation]] = {}

    def _path(self, conv_id: str) -> Path:
        if not _ID_RE.match(conv_id or ""):
            raise ValueError(f"invalid conversation id: {conv_id!r}")
        return CHATS_DIR / f"{conv_id}.json"

    # ── queries ──────────────────────────────────────────────────────────
    def list_conversations(self) -> list[Conversation]:
        """All saved conversations, most-recently-updated first."""
        convs: list[Conversation] = []
        seen: set[str] = set()
        for path in CHATS_DIR.glob("*.json"):
            key = str(path)
            seen.add(key)
            try:
                stat = path.stat()
            except OSError:
                continue
            cached = self._cache.get(key)
            if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
                convs.append(cached[2])
                continue
            try:
                conv = Conversation.from_dict(json.loads(path.read_text("utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
            self._cache[key] = (stat.st_mtime, stat.st_size, conv)
            convs.append(conv)
        # Forget entries for files that are gone.
        for key in set(self._cache) - seen:
            self._cache.pop(key, None)
        convs.sort(key=lambda c: c.updated, reverse=True)
        return convs

    def load(self, conv_id: str) -> Conversation | None:
        try:
            return Conversation.from_dict(
                json.loads(self._path(conv_id).read_text("utf-8"))
            )
        except (json.JSONDecodeError, OSError, ValueError):
            return None

    # ── mutations ────────────────────────────────────────────────────────
    def save(self, conv: Conversation) -> None:
        """Persist a conversation (skips empty ones — nothing to remember).

        Note: this does *not* bump ``updated``. That timestamp tracks the last
        message sent, not the last save — otherwise merely opening a chat (which
        saves the one being left) would reshuffle the sidebar. Callers touch the
        conversation when a real new turn begins.
        """
        if conv.is_empty:
            return
        conv.ensure_title()
        if not _ID_RE.match(conv.id or ""):
            conv.id = uuid.uuid4().hex
        path = self._path(conv.id)
        self._cache.pop(str(path), None)
        write_private(path, json.dumps(conv.to_dict(), ensure_ascii=False, indent=2))

    def delete(self, conv_id: str) -> None:
        try:
            path = self._path(conv_id)
        except ValueError:
            return
        self._cache.pop(str(path), None)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def rename(self, conv_id: str, title: str) -> bool:
        """Give a conversation a custom title. Returns True on success."""
        conv = self.load(conv_id)
        if conv is None:
            return False
        conv.title = (title or "").strip()[:80] or conv.title
        self.save(conv)
        return True


def export_markdown(conv: Conversation) -> str:
    """Render a conversation as a portable Markdown transcript."""
    import time as _time

    lines = [
        f"# {conv.title}",
        "",
        f"*Exported from Maze AI · {_time.strftime('%Y-%m-%d %H:%M', _time.localtime(conv.updated))}*",
        "",
    ]
    for msg in conv.messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        who = "You" if role == "user" else "Maze AI"
        lines.append(f"### {who}")
        lines.append("")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)
