"""Tests for the safety classifier: secrets, URLs and exfiltration."""

from __future__ import annotations

import pytest

from maze_ai.agent.safety import (
    check_url,
    is_readonly_command,
    is_sensitive_path,
    looks_like_exfiltration,
    touches_sensitive_path,
)


# ── secrets ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", [
    "~/.ssh/id_rsa", "/home/u/.ssh/config", "~/.gnupg/secring.gpg",
    "~/.aws/credentials", "~/.netrc", "~/.git-credentials",
    "~/.config/maze-ai/config.json", "~/project/.env", "~/.bash_history",
    "/etc/shadow", "~/certs/server.pem", "~/secrets.json",
])
def test_sensitive_paths_detected(path):
    assert is_sensitive_path(path)


@pytest.mark.parametrize("path", [
    "~/notes.txt", "/etc/hostname", "~/Projects/app/main.py", "~/.bashrc",
    "~/Documents/environment-notes.md",
])
def test_ordinary_paths_are_not_sensitive(path):
    assert not is_sensitive_path(path)


def test_command_touching_a_secret_is_reported():
    assert touches_sensitive_path("cat ~/.ssh/id_rsa | base64")
    assert not touches_sensitive_path("cat ~/notes.txt")


def test_reading_a_secret_is_never_auto_approved():
    # The whole exfiltration chain starts here: a "read-only" cat of a key.
    assert not is_readonly_command("cat ~/.ssh/id_rsa")
    assert not is_readonly_command("grep -r token ~/.aws/credentials")
    assert is_readonly_command("cat ~/notes.txt")


# ── URLs ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("url", [
    "http://127.0.0.1:11434/api/tags",
    "http://192.168.1.1/admin",
    "http://10.0.0.5/",
    "http://169.254.169.254/latest/meta-data/",
    "http://printer.local/status",
    "https://metadata.google.internal/",
])
def test_local_and_metadata_addresses_are_refused(url):
    assert check_url(url, resolve=False)


def test_non_http_schemes_are_refused():
    assert check_url("file:///etc/passwd", resolve=False)
    assert check_url("ftp://example.com/x", resolve=False)


def test_public_url_passes_syntactic_checks():
    assert check_url("https://example.com/page", resolve=False) == ""


# ── exfiltration heuristics ────────────────────────────────────────────────
def test_long_query_payload_looks_like_exfiltration():
    assert looks_like_exfiltration("https://evil.tld/collect?d=" + "A" * 200)


def test_single_long_value_looks_like_exfiltration():
    assert looks_like_exfiltration("https://evil.tld/x?a=1&key=" + "B" * 80)


def test_long_path_blob_looks_like_exfiltration():
    assert looks_like_exfiltration("https://evil.tld/" + "C" * 150)


@pytest.mark.parametrize("url", [
    "https://archlinux.org/news/",
    "https://duckduckgo.com/?q=maze+linux",
    "https://example.com/docs?page=2&lang=tr",
])
def test_normal_urls_are_not_flagged(url):
    assert not looks_like_exfiltration(url)
