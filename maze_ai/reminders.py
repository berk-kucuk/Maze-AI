"""Reminders / to-do store with a small natural-time parser.

Reminders are saved as JSON in ``~/.local/share/maze-ai/reminders.json``. The
UI polls :meth:`ReminderStore.due` on a timer and fires a desktop notification
when one is ready.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "maze-ai"
REMINDERS_FILE = DATA_DIR / "reminders.json"


@dataclass
class Reminder:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str = ""
    due: float = 0.0
    created: float = field(default_factory=time.time)
    fired: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "due": self.due,
                "created": self.created, "fired": self.fired}

    @classmethod
    def from_dict(cls, d: dict) -> "Reminder":
        return cls(
            id=str(d.get("id") or uuid.uuid4().hex[:8]),
            text=str(d.get("text") or ""),
            due=float(d.get("due") or 0.0),
            created=float(d.get("created") or time.time()),
            fired=bool(d.get("fired")),
        )

    def when_str(self) -> str:
        return datetime.fromtimestamp(self.due).strftime("%Y-%m-%d %H:%M")


_REL_UNITS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1, "saniye": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60, "dakika": 60, "dk": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "saat": 3600,
    "day": 86400, "days": 86400, "gun": 86400, "gün": 86400,
    "week": 604800, "weeks": 604800, "hafta": 604800,
}


def local_tz():
    """The user's timezone as a real zone (not a frozen UTC offset).

    Wall-clock reminders must survive a DST change: "every day at 09:00" is
    09:00 in Istanbul, not "09:00 as the offset happened to be when it was
    scheduled". A fixed offset from ``astimezone()`` would drift by an hour.
    """
    name = os.environ.get("TZ") or ""
    if not name:
        try:
            link = os.readlink("/etc/localtime")
            # …/zoneinfo/Europe/Istanbul → Europe/Istanbul
            if "zoneinfo/" in link:
                name = link.split("zoneinfo/", 1)[1]
        except OSError:
            name = ""
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            pass
    return datetime.now().astimezone().tzinfo


def parse_when(text: str, now: float | None = None) -> float | None:
    """Parse a human time expression into a unix timestamp.

    Supports: ``in 10 minutes`` / ``10 dakika sonra`` / ``2 saat``,
    ``HH:MM`` (today or tomorrow), ``tomorrow 9:00`` / ``yarin 09:00``,
    ``YYYY-MM-DD HH:MM`` and ISO ``YYYY-MM-DDTHH:MM``.

    Clock times are interpreted as local wall-clock in the user's timezone;
    relative offsets ("in 2 hours") are real elapsed time.
    """
    if not text:
        return None
    text = text.strip().lower()
    tz = local_tz()
    now_ts = now if now is not None else time.time()
    base = datetime.fromtimestamp(now_ts, tz)

    # Absolute: YYYY-MM-DD[ T]HH:MM
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[ t](\d{1,2}):(\d{2})", text)
    if m:
        y, mo, d, h, mi = map(int, m.groups())
        try:
            return datetime(y, mo, d, h, mi, tzinfo=tz).timestamp()
        except ValueError:
            return None

    # Relative: "in N unit" / "N unit sonra" / "N unit"
    m = re.search(r"(\d+)\s*(second|seconds|sec|secs|saniye|minute|minutes|min|mins|"
                  r"dakika|dk|hour|hours|hr|hrs|saat|day|days|gun|gün|week|weeks|hafta)",
                  text)
    if m and ("in " in text or "sonra" in text or "içinde" in text
              or text.startswith(m.group(0)) or re.fullmatch(r"\d+\s*\w+", text)):
        n = int(m.group(1))
        unit = _REL_UNITS.get(m.group(2))
        if unit:
            # Elapsed time, not wall-clock: "in 2 hours" is 7200 seconds even
            # across a daylight-saving jump.
            return now_ts + n * unit

    # Clock time HH:MM, optionally with tomorrow/yarin
    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if not (0 <= h < 24 and 0 <= mi < 60):
            return None
        tomorrow = "tomorrow" in text or "yarin" in text or "yarın" in text
        day = base.date() + timedelta(days=1) if tomorrow else base.date()
        target = datetime(day.year, day.month, day.day, h, mi, tzinfo=tz)
        # A time that already passed today means tomorrow — unless the user
        # said "tomorrow", in which case it is already on the right day.
        if not tomorrow and target.timestamp() <= now_ts:
            day = day + timedelta(days=1)
            target = datetime(day.year, day.month, day.day, h, mi, tzinfo=tz)
        return target.timestamp()

    return None


class ReminderStore:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._items: list[Reminder] = []
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(REMINDERS_FILE.read_text("utf-8"))
            self._items = [Reminder.from_dict(d) for d in raw]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._items = []

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = REMINDERS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps([r.to_dict() for r in self._items],
                                  ensure_ascii=False, indent=2), "utf-8")
        os.replace(tmp, REMINDERS_FILE)

    def add(self, text: str, due: float) -> Reminder:
        r = Reminder(text=text, due=due)
        self._items.append(r)
        self.save()
        return r

    def remove(self, ident: str) -> bool:
        """Remove one reminder, by id or by text. Returns True if it went.

        Matching is deliberately narrow and ordered — id, then exact text, then
        a substring that identifies exactly ONE reminder. The old behaviour
        deleted every reminder containing the string, so ``remove("a")`` wiped
        most of the list; a to-do you can lose by accident is worse than one
        you have to name precisely.
        """
        ident = (ident or "").strip()
        if not ident:
            return False
        ident_l = ident.lower()

        by_id = [r for r in self._items if r.id == ident]
        exact = [r for r in self._items if r.text.strip().lower() == ident_l]
        partial = [r for r in self._items if ident_l in r.text.lower()]
        for candidates in (by_id, exact, partial):
            if len(candidates) == 1:
                self._items.remove(candidates[0])
                self.save()
                return True
            if len(candidates) > 1:
                # Ambiguous: refuse rather than guess which one they meant.
                return False
        return False

    def matches(self, ident: str) -> list[Reminder]:
        """Reminders a remove() call would consider — used to explain a miss."""
        ident_l = (ident or "").strip().lower()
        if not ident_l:
            return []
        return [r for r in self._items
                if r.id == ident_l or ident_l in r.text.lower()]

    def pending(self) -> list[Reminder]:
        return sorted((r for r in self._items if not r.fired), key=lambda r: r.due)

    def due(self, now: float | None = None) -> list[Reminder]:
        """Return not-yet-fired reminders whose time has arrived, marking them fired."""
        now = now if now is not None else time.time()
        ready = [r for r in self._items if not r.fired and r.due <= now]
        if ready:
            for r in ready:
                r.fired = True
            self.save()
        return ready
