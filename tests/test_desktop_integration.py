"""Tests for the file-manager right-click integration."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from maze_ai import desktop_integration as di


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point every integration at a throwaway home directory."""
    monkeypatch.setattr(di, "_DATA_HOME", tmp_path / "data")
    monkeypatch.setattr(di, "_CONFIG_HOME", tmp_path / "config")
    return tmp_path


@pytest.fixture
def all_managers(monkeypatch):
    """Pretend every supported file manager is installed."""
    monkeypatch.setattr(di.shutil, "which", lambda name: f"/usr/bin/{name}")


# ── Dolphin ────────────────────────────────────────────────────────────────
def test_dolphin_menu_is_installed_and_executable(home, all_managers):
    assert di.install(["dolphin"]) == {"dolphin": ""}
    path = home / "data" / "kio" / "servicemenus" / "maze-ai.desktop"
    assert path.exists()
    # KDE ignores service menus that are not executable.
    assert path.stat().st_mode & 0o111
    body = path.read_text()
    assert "maze-ai --file %F" in body
    assert "maze-ai --file-action explain --file %F" in body
    assert di.MARKER in body
    # The format KIO has wanted since Frameworks 5.85 — the old
    # Type=Service/KonqPopupMenu shape is ignored by Plasma 6.
    assert "Type=Application" in body
    assert "X-KDE-Submenu=" in body
    assert "ServiceTypes" not in body
    # It is a menu entry, not something to show in the application launcher.
    assert "NoDisplay=true" in body


def test_uninstall_removes_the_menu(home, all_managers):
    di.install(["dolphin"])
    assert di.uninstall(["dolphin"]) == ["dolphin"]
    assert not (home / "data" / "kio" / "servicemenus" / "maze-ai.desktop").exists()


def test_uninstalling_twice_is_harmless(home, all_managers):
    di.install(["dolphin"])
    di.uninstall(["dolphin"])
    assert di.uninstall(["dolphin"]) == []


def test_status_follows_reality(home, all_managers):
    assert di.status()["dolphin"] is False
    di.install(["dolphin"])
    assert di.status()["dolphin"] is True


# ── Nautilus ───────────────────────────────────────────────────────────────
def test_nautilus_script_is_runnable(home, all_managers):
    di.install(["nautilus"])
    scripts = list((home / "data" / "nautilus" / "scripts").iterdir())
    assert len(scripts) == 1
    script = scripts[0]
    assert script.stat().st_mode & 0o111
    body = script.read_text()
    assert "NAUTILUS_SCRIPT_SELECTED_FILE_PATHS" in body
    assert "maze-ai --file" in body


# ── Nemo ───────────────────────────────────────────────────────────────────
def test_nemo_action_file(home, all_managers):
    di.install(["nemo"])
    body = (home / "data" / "nemo" / "actions" / "maze-ai.nemo_action").read_text()
    assert "[Nemo Action]" in body and "maze-ai --file %F" in body


# ── Thunar shares one XML file with the user's own actions ────────────────
def test_thunar_action_is_added(home, all_managers):
    di.install(["thunar"])
    path = home / "config" / "Thunar" / "uca.xml"
    root = ET.parse(path).getroot()
    commands = [a.findtext("command") for a in root.findall("action")]
    assert "maze-ai --file %F" in commands


def test_thunar_keeps_existing_actions(home, all_managers):
    path = home / "config" / "Thunar" / "uca.xml"
    path.parent.mkdir(parents=True)
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<actions><action><name>Open Terminal</name>"
        "<command>xfce4-terminal</command></action></actions>"
    )
    di.install(["thunar"])
    root = ET.parse(path).getroot()
    commands = [a.findtext("command") for a in root.findall("action")]
    assert "xfce4-terminal" in commands and "maze-ai --file %F" in commands

    di.uninstall(["thunar"])
    commands = [a.findtext("command") for a in ET.parse(path).getroot().findall("action")]
    assert commands == ["xfce4-terminal"], "only our own action may be removed"


def test_thunar_install_is_idempotent(home, all_managers):
    di.install(["thunar"])
    di.install(["thunar"])
    root = ET.parse(home / "config" / "Thunar" / "uca.xml").getroot()
    assert len(root.findall("action")) == 1


def test_broken_thunar_config_is_replaced_not_crashed(home, all_managers):
    path = home / "config" / "Thunar" / "uca.xml"
    path.parent.mkdir(parents=True)
    path.write_text("this is not xml at all")
    assert di.install(["thunar"]) == {"thunar": ""}
    assert "maze-ai" in path.read_text()


# ── discovery ──────────────────────────────────────────────────────────────
def test_only_installed_managers_are_offered(home, monkeypatch):
    monkeypatch.setattr(di.shutil, "which",
                        lambda name: "/usr/bin/dolphin" if name == "dolphin" else None)
    assert [item.key for item in di.available()] == ["dolphin"]


def test_install_without_arguments_skips_missing_managers(home, monkeypatch):
    monkeypatch.setattr(di.shutil, "which",
                        lambda name: "/usr/bin/dolphin" if name == "dolphin" else None)
    assert list(di.install()) == ["dolphin"]


def test_describe_marks_what_is_installed(home, monkeypatch):
    monkeypatch.setattr(di.shutil, "which",
                        lambda name: "/usr/bin/dolphin" if name == "dolphin" else None)
    assert "—" in di.describe()
    di.install(["dolphin"])
    assert "✓" in di.describe()


def test_describe_without_any_file_manager(home, monkeypatch):
    monkeypatch.setattr(di.shutil, "which", lambda name: None)
    assert "No supported file manager" in di.describe()


def test_write_failure_is_reported_not_raised(home, all_managers, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(di.Path, "write_text", boom)
    result = di.install(["dolphin"])
    assert "read-only" in result["dolphin"]
