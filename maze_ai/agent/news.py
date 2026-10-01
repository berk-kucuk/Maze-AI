"""Arch Linux news, checked before a system upgrade.

Most broken Arch upgrades come from a news item the user never read: "X
requires manual intervention". This reads the official feed, works out which
items appeared since the user's last full upgrade (from pacman's own log), and
flags the ones that need action — so Maze AI can warn *before* the user runs
``pacman -Syu``, not after.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

FEED_URL = "https://archlinux.org/feeds/news/"
PACMAN_LOG = Path(os.environ.get("MAZE_AI_PACMAN_LOG", "/var/log/pacman.log"))
CACHE_FILE = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "maze-ai" / "arch-news.json"
)
CACHE_SECONDS = 3600
_HEADERS = {"User-Agent": "MazeAI/1.0 (Maze Linux assistant; news check)"}
_ACTION_RE = re.compile(r"manual intervention|requires? (?:action|attention)|"
                        r"must be|need to|breaking", re.IGNORECASE)


@dataclass
class NewsItem:
    title: str
    link: str
    published: datetime
    summary: str

    @property
    def needs_action(self) -> bool:
        return bool(_ACTION_RE.search(self.title))


def _text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(raw or ""))
    return " ".join(text.split())


def parse_feed(xml_text: str) -> list[NewsItem]:
    items: list[NewsItem] = []
    root = ET.fromstring(xml_text)
    for node in root.iter("item"):
        try:
            published = parsedate_to_datetime(node.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        items.append(NewsItem(
            title=_text(node.findtext("title") or ""),
            link=(node.findtext("link") or "").strip(),
            published=published,
            summary=_text(node.findtext("description") or "")[:4000],
        ))
    items.sort(key=lambda item: item.published, reverse=True)
    return items


def fetch_news() -> list[NewsItem]:
    """The feed, cached for an hour so a chat about updating doesn't refetch."""
    try:
        cached = json.loads(CACHE_FILE.read_text("utf-8"))
        if time.time() - float(cached.get("time", 0)) < CACHE_SECONDS:
            return parse_feed(cached["xml"])
    except (OSError, ValueError, KeyError, ET.ParseError):
        pass
    resp = requests.get(FEED_URL, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    items = parse_feed(resp.text)
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({"time": time.time(), "xml": resp.text}), "utf-8")
    except OSError:
        pass
    return items


def last_upgrade(log: Path | None = None) -> datetime | None:
    """When the last full system upgrade started, from pacman's log."""
    path = log or PACMAN_LOG
    try:
        with path.open("rb") as fh:
            # The log can be megabytes; the answer is near the end.
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 2_000_000))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    stamps = re.findall(r"^\[([^\]]+)\] \[PACMAN\] starting full system upgrade", tail, re.M)
    if not stamps:
        return None
    try:
        when = datetime.fromisoformat(stamps[-1])
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def news_report(count: int = 6) -> tuple[bool, str]:
    """News since the last upgrade (or the latest few), action items first."""
    try:
        items = fetch_news()
    except (requests.RequestException, ET.ParseError) as exc:
        return False, (f"Could not read Arch news ({exc}). Ask the user to check "
                       f"https://archlinux.org/news/ before upgrading.")
    since = last_upgrade()
    fresh = [i for i in items if since is None or i.published > since]
    shown = fresh or items[: max(1, count)]
    lines = []
    if since:
        lines.append(f"Last full system upgrade: {since:%Y-%m-%d %H:%M}.")
    if fresh:
        lines.append(f"{len(fresh)} news item(s) published since then:")
    else:
        lines.append("No Arch news since the last upgrade. Latest items, for reference:")
    for item in sorted(shown, key=lambda i: (not i.needs_action, -i.published.timestamp())):
        flag = "MANUAL INTERVENTION — " if item.needs_action and item in fresh else ""
        lines.append(f"\n- {flag}{item.title} ({item.published:%Y-%m-%d}) {item.link}")
        if item in fresh:
            # Action items in full: the steps are the point, and a cut-off
            # summary is what leads a model to fill the gap with invented flags.
            limit = 3500 if item.needs_action else 600
            lines.append(f"  {item.summary[:limit]}")
    action = [i for i in fresh if i.needs_action]
    if action:
        lines.append(
            "\nWARN the user before they upgrade: the item(s) marked MANUAL "
            "INTERVENTION need steps around this upgrade. Give the steps exactly as "
            "the news text states them — same commands, same options. If the text "
            "doesn't spell a step out, say so and point to the linked page; never "
            "invent a command or flag."
        )
    return True, "\n".join(lines)


_UPGRADE_RE = re.compile(
    r"-S\w*y\w*u|\bsyu\b|\b(?:system|full)\s+(?:update|upgrade)|"
    r"\bupgrade\s+(?:my\s+|the\s+)?(?:system|arch|packages)\b|"
    r"\bupdate\s+(?:my\s+)?(?:system|packages|arch)\b|"
    r"(?:sistem\w*|paket\w*|arch\w*|bilgisayar\w*)\s+(?:\w+\s+)?güncelle|"
    r"(?:sistem\w*|paket\w*|arch\w*|bilgisayar\w*)\s+(?:\w+\s+)?guncelle|"
    r"güncelleme\w*\s+(?:yap|var|kur|çalıştır|calistir)|guncelleme\w*\s+(?:yap|var|kur)",
    re.IGNORECASE,
)


def is_upgrade_request(message: str) -> bool:
    """Is the user about to update the system ("sistemi güncelle", "-Syu")?"""
    return bool(_UPGRADE_RE.search(message or ""))
