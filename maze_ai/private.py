"""Owner-only storage for everything Maze AI keeps on disk.

Chats hold whatever the assistant read for you — file contents, command
output, sometimes a secret. The undo backups are verbatim copies of files it
changed or deleted, ``~/.ssh`` included if it came to that. Screenshots and
pasted images show your screen. With the default umask all of that was
world-readable (0644 files in 0755 directories), i.e. readable by every other
account on the machine. Everything here is created 0700 / 0600 instead, and
directories that already exist are tightened on first use.
"""

from __future__ import annotations

import os
from pathlib import Path

DIR_MODE = 0o700
FILE_MODE = 0o600


def private_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) and make sure only the owner can enter it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)
    try:
        if path.stat().st_mode & 0o077:
            os.chmod(path, DIR_MODE)
    except OSError:
        pass  # not ours to change (e.g. a shared parent): leave it be
    return path


def write_private(path: Path | str, data: str | bytes, encoding: str = "utf-8") -> None:
    """Atomically write ``data`` to ``path`` as an owner-only (0600) file.

    The temporary file is created 0600 *before* any byte is written, so there
    is no window in which the content sits in a readable file, and the rename
    keeps those permissions.
    """
    path = Path(path)
    private_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    payload = data.encode(encoding) if isinstance(data, str) else data
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.chmod(tmp, FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def tighten(path: Path | str) -> None:
    """Make an existing file owner-only (best effort)."""
    try:
        os.chmod(path, FILE_MODE)
    except OSError:
        pass
