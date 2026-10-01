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

from .private import private_dir, tighten, write_private

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
    "seconds": 1, "second": 1, "secs": 1, "sec": 1, "saniye": 1, "sn": 1,
    "minutes": 60, "minute": 60, "mins": 60, "min": 60, "dakika": 60, "dk": 60,
    "hours": 3600, "hour": 3600, "hrs": 3600, "hr": 3600, "saat": 3600,
    "days": 86400, "day": 86400, "gün": 86400, "gun": 86400,
    "weeks": 604800, "week": 604800, "hafta": 604800,
}
_UNIT_RE = "|".join(sorted(_REL_UNITS, key=len, reverse=True))
# "1 saat 30 dakika", "an hour", "yarım saat", "1.5 hours"
_PAIR_RE = re.compile(
    rf"(\d+(?:[.,]\d+)?|yarım|yarim|half an|half a|an|a|bir)\s*({_UNIT_RE})(?![a-zçğıöşü])"
)
_REL_WORDS = re.compile(r"\b(in|sonra|içinde|icinde|later|from now|ve|and)\b")
# Parts of the day that turn "8" into 20:00 ("akşam 8", "8 pm").
_PM_WORDS = ("pm", "akşam", "aksam", "gece", "öğleden sonra", "ogleden sonra", "evening",
             "tonight")


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


def _relative(text: str) -> float | None:
    """Seconds from now for "in 1 hour 30 minutes" / "yarım saat sonra"."""
    pairs = _PAIR_RE.findall(text)
    if not pairs:
        return None
    rest = _REL_WORDS.sub(" ", _PAIR_RE.sub(" ", text))
    if not (_REL_WORDS.search(text) or not rest.strip(" ,.")):
        return None
    total = 0.0
    for amount, unit in pairs:
        if amount in ("yarım", "yarim", "half an", "half a"):
            value = 0.5
        elif amount in ("an", "a", "bir"):
            value = 1.0
        else:
            value = float(amount.replace(",", "."))
        total += value * _REL_UNITS[unit]
    return total or None


def _clock(text: str) -> tuple[int, int] | None:
    """The time of day in a phrase, as (hour, minute)."""
    m = re.search(r"(?<![\d.])(\d{1,2})[:.](\d{2})(?![\d.])", text)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        m = (
            re.search(r"(?:saat|at|@)\s*(\d{1,2})(?!\d)", text)
            or re.search(r"(?<!\d)(\d{1,2})\s*(?:am|pm)\b", text)
            or re.search(r"(?<!\d)(\d{1,2})\s*['’]?\s*(?:de|da|te|ta)\b", text)
            or re.search(r"(?:sabah|akşam|aksam|gece|öğlen|oglen|morning|evening|tonight)"
                         r"\s*(\d{1,2})(?!\d)", text)
        )
        if not m:
            return None
        hour, minute = int(m.group(1)), 0
    if hour < 12 and any(word in text for word in _PM_WORDS):
        if not (hour == 12 or ("gece" in text and hour <= 4)):
            hour += 12
    if re.search(r"(?<!\d)12\s*am\b", text):
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    return hour, minute


def parse_when(text: str, now: float | None = None) -> float | None:
    """Parse a human time expression into a unix timestamp.

    Supports, in English and Turkish: ``in 10 minutes`` / ``10 dakika sonra``
    / ``1 saat 30 dakika sonra`` / ``yarım saat sonra``; clock times
    ``18:30``, ``18.30``, ``saat 18``, ``9pm``, ``akşam 8``, ``8'de`` (today, or
    tomorrow once passed); ``tomorrow 9:00`` / ``yarın 09:00``; and dates as
    ``YYYY-MM-DD HH:MM`` or ``DD.MM.YYYY HH:MM`` (09:00 when no time is given).

    Clock times are interpreted as local wall-clock in the user's timezone;
    relative offsets ("in 2 hours") are real elapsed time.
    """
    if not text:
        return None
    text = " ".join(text.strip().lower().split())
    tz = local_tz()
    now_ts = now if now is not None else time.time()
    base = datetime.fromtimestamp(now_ts, tz)

    # Absolute dates: ISO, or the day-first form Turkish speakers write.
    date = None
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        date = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        m = re.search(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)", text)
        if m:
            date = (int(m.group(3)), int(m.group(2)), int(m.group(1)))
    if date:
        # "…T09:30" (ISO) leaves a "t" glued to the time.
        rest = text[:m.start()] + " " + re.sub(r"^t", " ", text[m.end():])
        hour, minute = _clock(rest) or (9, 0)
        try:
            return datetime(*date, hour, minute, tzinfo=tz).timestamp()
        except ValueError:
            return None

    offset = _relative(text)
    if offset is not None:
        # Elapsed time, not wall-clock: "in 2 hours" is 7200 seconds even
        # across a daylight-saving jump.
        return now_ts + offset

    days = 0
    if re.search(r"day after tomorrow|öbür gün|obur gun|ertesi gün", text):
        days = 2
    elif re.search(r"tomorrow|yarın|yarin", text):
        days = 1
    clock = _clock(text)
    if clock is None:
        if not days:
            return None
        clock = (9, 0)              # "tomorrow" alone: morning
    day = base.date() + timedelta(days=days)
    target = datetime(day.year, day.month, day.day, *clock, tzinfo=tz)
    # A time that already passed today means tomorrow — unless the user named
    # the day, in which case it is already on the right one.
    if not days and target.timestamp() <= now_ts:
        day = day + timedelta(days=1)
        target = datetime(day.year, day.month, day.day, *clock, tzinfo=tz)
    return target.timestamp()


class ReminderStore:
    def __init__(self) -> None:
        private_dir(DATA_DIR)
        if REMINDERS_FILE.exists():
            tighten(REMINDERS_FILE)
        self._items: list[Reminder] = []
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(REMINDERS_FILE.read_text("utf-8"))
            self._items = [Reminder.from_dict(d) for d in raw]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._items = []

    def save(self) -> None:
        write_private(REMINDERS_FILE, json.dumps([r.to_dict() for r in self._items],
                                                  ensure_ascii=False, indent=2))

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
