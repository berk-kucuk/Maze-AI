"""A recursive search is judged by what it would read (audit MA1), and the
secret list covers wallets, key stores and Maze's own app data (MA2).

`grep -r BEGIN ~` named no secret on its command line, so it passed as a
read-only inspect and ran without a prompt — while reading ~/.ssh/id_ed25519.
Each test builds its own home directory; nothing real is read.
"""
from __future__ import annotations

import os

import pytest

from maze_ai.agent.agent import Agent, REASON_SENSITIVE
from maze_ai.agent.rules import classify
from maze_ai.agent.safety import is_readonly_command, is_sensitive_path, touches_sensitive_path
from maze_ai.config import MODE_ASK, MODE_AUTO


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / ".ssh").mkdir(parents=True)
    (h / ".ssh" / "id_ed25519").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    (h / ".bitcoin").mkdir()
    (h / ".bitcoin" / "wallet.dat").write_text("w")
    proj = h / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "a.py").write_text("print('TODO')\n")
    (proj / ".git" / "hooks").mkdir(parents=True)
    (proj / ".git" / "config").write_text("[core]\n")          # every repo has one
    (h / "envproj").mkdir()
    (h / "envproj" / ".env").write_text("API_KEY=1\n")
    (h / "linked").mkdir()
    os.symlink(h / ".ssh", h / "linked" / "keys")
    monkeypatch.setenv("HOME", str(h))
    return h


@pytest.mark.parametrize("command, where", [
    ("grep -r BEGIN ~", "proj"),
    ("grep -rI 'PRIVATE KEY' ~/", "proj"),
    ("grep -R x .", ""),                                # run in the home directory
    ("rg BEGIN ~", "proj"),
    ("ag BEGIN", ""),
    ("grep -d recurse x ~", "proj"),
    ("grep --directories=recurse x ~", "proj"),
    ("grep -rn -e x -- ~", "proj"),
    ("diff -r ~/.bitcoin ~/proj", "proj"),
    ("cd ~ && grep -r x .", "proj"),                   # the cd moves the search
    ("grep -r x ../linked", "proj"),                   # a symlink to ~/.ssh inside
    ("grep -r x ~/envproj", "proj"),                   # a .env inside
    ("echo ~ | xargs grep -r x", "proj"),              # paths decided at run time
    ("find ~ -exec grep -r x {} +", "proj"),
    ("sudo grep -r x /root", "proj"),                  # unreadable to us, not to sudo
    ("grep -r x $(echo ~)", "proj"),
])
def test_recursive_search_is_judged_by_what_it_walks(home, command, where):
    cwd = str(home / where) if where else str(home)
    assert touches_sensitive_path(command, cwd), command
    assert not is_readonly_command(command, cwd), command
    assert classify(command, cwd=cwd).tier != "safe", command


@pytest.mark.parametrize("command", [
    "grep -rn TODO .", "grep -rn TODO src", "rg TODO", "ag TODO", "grep -rn -e a -e b .",
    "grep -n TODO src/a.py", "grep -A3 TODO src/a.py", "diff src/a.py src/a.py",
    "grep -r --include=*.py TODO .", "ls -la ~", "tree -L 2",
])
def test_searching_a_project_stays_prompt_free(home, command):
    """A repository's .git/config is not a secret: ordinary searches in a
    project must not start asking."""
    cwd = str(home / "proj")
    assert not touches_sensitive_path(command, cwd), command
    assert is_readonly_command(command, cwd), command


@pytest.mark.parametrize("mode", [MODE_ASK, MODE_AUTO])
def test_agent_asks_before_the_attack_even_in_auto_mode(home, mode):
    agent = Agent(backend=object(), mode=mode)
    agent.cwd = str(home / "proj")
    reason, detail = agent._approval_reason("run_command", {"command": "grep -r BEGIN ~"})
    assert reason == REASON_SENSITIVE
    assert detail


def test_agent_uses_its_own_directory(home):
    agent = Agent(backend=object(), mode=MODE_ASK)
    agent.cwd = str(home / "proj")
    assert agent._needs_approval("run_command", {"command": "grep -rn TODO ."}) is False
    agent.cwd = str(home)
    assert agent._needs_approval("run_command", {"command": "grep -rn TODO ."}) is True


@pytest.mark.parametrize("path", [
    "~/.bitcoin/wallet.dat", "~/.bitcoin/wallets/default", "~/.electrum/wallets/default_wallet",
    "~/Monero/wallets/main.keys", "~/.ethereum/keystore/UTC--x", "~/.sparrow/wallets/w",
    "~/.config/Ledger Live/app.json", "~/.config/Exodus/exodus.wallet/x",
    "~/.config/sops/age/keys.txt", "~/Documents/passwords.kdbx", "~/.local/share/kwalletd/kdewallet.kwl",
    "~/.config/Signal/sql/db.sqlite", "~/.local/share/TelegramDesktop/tdata/x", "~/.thunderbird/p/Mail/x",
    "~/.local/share/maze-vault/accounts/1/vault.json", "~/.config/maze-vault/settings.json",
    "~/.local/share/maze-capsule/capsules/1/key-backup/gocryptfs.conf", "/media/usb/vault/gocryptfs.conf",
    "~/.config/haze/settings.json", "~/.config/haze/vault/1.hzv",
    "~/.local/share/hazedrop/history.db", "~/.local/share/mazeconnect/identity.key",
    "~/.local/share/maze-crypto/portfolio.db",
])
def test_new_secret_locations(path):
    assert is_sensitive_path(path), path
    assert touches_sensitive_path(f"cat {path!r}"), path
