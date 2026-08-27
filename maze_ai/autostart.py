"""Login autostart, managed by the app itself.

Instead of shipping a system-wide ``/etc/xdg/autostart`` entry (which duplicates
the app across every user and can't be toggled off), Maze AI writes its own
per-user autostart file on demand. The entry launches with ``--hidden`` so the
app comes up in the system tray rather than as a window.
"""

from __future__ import annotations

import os
from pathlib import Path

_AUTOSTART_DIR = Path(
    os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
) / "autostart"
_AUTOSTART_FILE = _AUTOSTART_DIR / "maze-ai.desktop"

_ENTRY = """\
[Desktop Entry]
Type=Application
Name=Maze AI
GenericName=AI Assistant
Comment=Start Maze AI in the system tray at login
Exec=maze-ai --hidden
Icon=maze-ai
Terminal=false
Categories=Utility;
StartupNotify=false
StartupWMClass=maze-ai
X-GNOME-Autostart-enabled=true
X-KDE-autostart-after=panel
"""


def is_enabled() -> bool:
    return _AUTOSTART_FILE.exists()


def set_enabled(enabled: bool) -> None:
    """Create or remove the per-user autostart entry."""
    if enabled:
        try:
            _AUTOSTART_DIR.mkdir(parents=True, exist_ok=True)
            _AUTOSTART_FILE.write_text(_ENTRY, encoding="utf-8")
        except OSError:
            pass
    else:
        try:
            _AUTOSTART_FILE.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def sync(enabled: bool) -> None:
    """Make the autostart file match the desired state (no-op if already so)."""
    if bool(enabled) != is_enabled():
        set_enabled(enabled)
