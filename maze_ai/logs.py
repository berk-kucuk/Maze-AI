"""Application logging.

Without a log there is nothing to look at when a user reports "it just stopped
answering": the agent loop, the tools and the backends all fail in ways the UI
deliberately softens. Everything goes to a rotating file under the XDG state
directory; ``--debug`` also mirrors it to the console and raises the level.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from .private import private_dir

STATE_DIR = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "maze-ai"
)
LOG_FILE = STATE_DIR / "maze-ai.log"

_MAX_BYTES = 1_000_000
_BACKUPS = 3
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_configured = False


def setup(debug: bool = False) -> Path | None:
    """Configure root logging once. Returns the log file path (or None)."""
    global _configured
    if _configured:
        return LOG_FILE if LOG_FILE.exists() else None
    _configured = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    path: Path | None = None
    try:
        private_dir(STATE_DIR)
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        path = LOG_FILE
    except OSError:
        # A read-only home shouldn't stop the app from starting.
        pass

    if debug:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)

    # Requests' connection chatter is noise at DEBUG level.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return path
