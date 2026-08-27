"""Tests for the natural-time reminder parser."""

from __future__ import annotations

from datetime import datetime

import pytest

from maze_ai import reminders
from maze_ai.reminders import parse_when

# A fixed reference point so relative parsing is deterministic.
NOW = datetime(2026, 7, 22, 12, 0).timestamp()


def test_relative_minutes():
    assert parse_when("in 10 minutes", NOW) == NOW + 600


def test_relative_turkish():
    assert parse_when("30 dakika sonra", NOW) == NOW + 1800


def test_relative_hours_bare():
    assert parse_when("2 saat", NOW) == NOW + 7200


def test_absolute_datetime():
    assert parse_when("2026-08-01 09:30", NOW) == datetime(2026, 8, 1, 9, 30).timestamp()


def test_clock_time_rolls_to_tomorrow_when_past():
    # 08:00 is before the 12:00 reference, so it should be tomorrow.
    got = parse_when("08:00", NOW)
    assert got == datetime(2026, 7, 23, 8, 0).timestamp()


def test_clock_time_today_when_future():
    got = parse_when("18:30", NOW)
    assert got == datetime(2026, 7, 22, 18, 30).timestamp()


def test_tomorrow_keyword():
    got = parse_when("yarin 09:00", NOW)
    assert got == datetime(2026, 7, 23, 9, 0).timestamp()


def test_unparseable_returns_none():
    assert parse_when("whenever", NOW) is None
    assert parse_when("", NOW) is None


# ── removal ────────────────────────────────────────────────────────────────
@pytest.fixture
def store(tmp_path, monkeypatch):
    """A ReminderStore backed by a temp file."""
    monkeypatch.setattr(reminders, "DATA_DIR", tmp_path)
    monkeypatch.setattr(reminders, "REMINDERS_FILE", tmp_path / "reminders.json")
    s = reminders.ReminderStore()
    s.add("Take a break", NOW + 600)
    s.add("Call Ayşe", NOW + 1200)
    s.add("Buy bread", NOW + 1800)
    return s


def test_remove_by_id(store):
    target = store.pending()[0]
    assert store.remove(target.id)
    assert all(r.id != target.id for r in store.pending())


def test_remove_by_exact_text(store):
    assert store.remove("Buy bread")
    assert len(store.pending()) == 2


def test_remove_by_unique_substring(store):
    assert store.remove("Ayşe")
    assert len(store.pending()) == 2


def test_ambiguous_removal_deletes_nothing(store):
    # "a" appears in every reminder — the old behaviour wiped the whole list.
    assert not store.remove("a")
    assert len(store.pending()) == 3


def test_removing_nothing_is_a_no_op(store):
    assert not store.remove("")
    assert not store.remove("nonexistent")
    assert len(store.pending()) == 3


def test_matches_lists_the_candidates(store):
    assert len(store.matches("a")) == 3


# ── daylight saving ────────────────────────────────────────────────────────
def test_relative_time_is_elapsed_not_wall_clock():
    # "in 3 hours" is 3 real hours even if the clocks move in between.
    assert parse_when("in 3 hours", NOW) == NOW + 3 * 3600
