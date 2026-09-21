"""The settings file holds an API key, so it must never be world-readable.

config.json is on the agent's own sensitive-path list (safety.py), which is the
project saying it counts as a secret. This pins the file mode so a future edit
to Config.save cannot quietly go back to writing it at the default umask.
"""

from __future__ import annotations

import importlib
import json
import os
import stat


def _config_module(tmp_path, monkeypatch):
    """Reimport maze_ai.config with XDG_CONFIG_HOME pointed at a temp dir.

    CONFIG_DIR is computed at import time, so the environment has to be set
    before the module object exists.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import maze_ai.config as config
    return importlib.reload(config)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_saved_config_is_owner_only(tmp_path, monkeypatch):
    config = _config_module(tmp_path, monkeypatch)
    cfg = config.Config()
    cfg.set("gemini_api_key", "AIzaSy-not-a-real-key")
    cfg.save()

    assert _mode(config.CONFIG_FILE) == 0o600
    # A 0600 file inside a 0755 directory is still listed by everyone.
    assert _mode(config.CONFIG_DIR) == 0o700


def test_second_save_does_not_widen_the_file(tmp_path, monkeypatch):
    """The rewrite path is where this kind of bug hides.

    save() writes a temp file and renames it. rename carries the TEMP file's
    mode onto the destination, so a temp file created at the default umask
    silently re-widened an already-correct config on every save after the
    first — the same shape as the maze-cloak ownership bug.
    """
    config = _config_module(tmp_path, monkeypatch)
    cfg = config.Config()
    cfg.set("gemini_api_key", "first")
    cfg.save()
    cfg.set("gemini_api_key", "second")
    cfg.save()

    assert _mode(config.CONFIG_FILE) == 0o600
    assert json.loads(config.CONFIG_FILE.read_text())["gemini_api_key"] == "second"
    assert not config.CONFIG_FILE.with_suffix(".tmp").exists()


def test_upgrading_from_a_world_readable_config_is_repaired(tmp_path, monkeypatch):
    """Someone running an older version already has a 0644 config on disk."""
    config = _config_module(tmp_path, monkeypatch)
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config.CONFIG_FILE.write_text(json.dumps({"gemini_api_key": "old"}))
    os.chmod(config.CONFIG_DIR, 0o755)
    os.chmod(config.CONFIG_FILE, 0o644)

    cfg = config.Config()
    cfg.save()

    assert _mode(config.CONFIG_FILE) == 0o600
    assert _mode(config.CONFIG_DIR) == 0o700
