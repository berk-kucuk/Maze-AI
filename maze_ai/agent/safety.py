"""Safety classification for agent actions.

Everything the agent might do is judged here before it runs: whether a shell
command is destructive, whether it is a harmless read-only inspect, whether it
touches secrets, and whether a URL is safe to fetch. Keeping the rules in one
module (instead of scattered through the tool implementations) means the agent
loop, the tools and the tests all agree on the same definitions.
"""

from __future__ import annotations

import ipaddress
import os
import re
import shlex
import socket
from pathlib import Path
from urllib.parse import urlparse

# ── destructive commands ──────────────────────────────────────────────────
# A match forces explicit approval even in autonomous mode (when
# "block_dangerous_commands" is on) — the agent should never silently wipe a
# disk, format a filesystem or fork-bomb the machine.
_DANGEROUS_PATTERNS = [
    r"\brm\s+(?:-\w*\s+)*-\w*r\w*f|\brm\s+(?:-\w*\s+)*-\w*f\w*r",  # rm -rf / -fr
    r"\brm\s+-[rf]\w*\s+(?:-[rf]\w*\s+)+",                          # rm -r -f …
    # rm with recursive AND force in any order / long form (--recursive --force)
    r"\brm\b(?=[^\n|;&]*(?:\s-\w*r|\s--recursive))(?=[^\n|;&]*(?:\s-\w*f|\s--force))",
    r"\bmkfs(\.\w+)?\b",                                            # format fs
    r"\bwipefs\b",
    r"\bdd\b[^\n]*\bof=/dev/",                                      # raw disk write
    r">\s*/dev/(sd|nvme|vd|hd|mmcblk)",                             # clobber a disk
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",                    # fork bomb
    r"\b(shutdown|reboot|poweroff|halt)\b",
    r"\bchmod\s+-R\s+0*777\s+/",
    r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|dash)\b",     # pipe to shell
    r"\b(userdel|groupdel)\b",
    r"\bmv\b[^\n]*\s+/dev/null\b",
]
_DANGEROUS_RE = re.compile("|".join(_DANGEROUS_PATTERNS), re.IGNORECASE)


def is_dangerous_command(command: str) -> bool:
    """True if the command matches a known destructive pattern."""
    return bool(_DANGEROUS_RE.search(command or ""))


# ── read-only commands ────────────────────────────────────────────────────
# Commands whose whole purpose is to READ/inspect — safe to run without a
# prompt in "ask" mode (when "auto_approve_readonly" is on).
_READONLY_CMDS = {
    "ls", "cat", "pwd", "whoami", "id", "df", "du", "free", "uptime", "date",
    "head", "tail", "wc", "stat", "file", "uname", "lscpu", "lsblk", "lsusb",
    "lspci", "ps", "which", "whereis", "printenv", "cal", "hostname",
    "echo", "printf", "tree", "realpath", "dirname", "basename", "nproc",
    "arch", "groups", "locale", "getent", "neofetch", "fastfetch", "sensors",
    "true", "test",
    # read-only text filters (common in pipelines)
    "grep", "egrep", "fgrep", "rg", "ag", "sort", "uniq", "cut", "column",
    "tr", "comm", "diff", "cmp", "tac", "rev", "fold", "nl", "paste", "look",
    "expand", "unexpand",
}
# NOT in the list, on purpose: `env`. It reads the environment when called with
# no arguments, but its actual job is running a program — `env <anything>` is
# arbitrary execution wearing a read-only name, which made it a way past this
# whole function. `printenv` covers the legitimate use.

# Flags that turn one of the commands above into something that WRITES a file
# or RUNS code. Deliberately per-command rather than one global list: `-o` is a
# file to overwrite for `sort` and merely "only-matching" for `grep`, so a
# blanket ban would be both wrong and annoying. Every entry was checked against
# that program's own manual.
_UNSAFE_FLAGS: dict[str, set[str]] = {
    "sort":      {"-o", "--output",         # writes its result to a file
                  "--compress-program"},    # RUNS the named program on temp files
    "tree":      {"-o"},                    # same
    "date":      {"-s", "--set"},           # sets the system clock
    "hostname":  {"-b", "--boot", "-F", "--file"},
    "file":      {"-m", "--magic-file", "-C", "--compile"},
    # These source a *shell script* as their config, so pointing them at an
    # attacker-chosen file is straightforward code execution.
    "neofetch":  {"--config"},
    "fastfetch": {"-c", "--config"},
    # Search tools that RUN a helper program of the caller's choosing.
    "rg":        {"--pre"},                 # preprocessor, executed per file
    "ag":        {"--pager"},               # output piped through a program
    # git: --ext-diff runs the external diff program from config; --output
    # writes a file; --textconv (explicit) runs conversion drivers. None of
    # them is needed for reading.
    "git":       {"--ext-diff", "--output", "--textconv"},
}

# git subcommands that LIST by default but CHANGE the repository once given an
# argument (`git branch -D x`, `git tag -d v1`, `git remote add …`). They count
# as read-only only with nothing after them but these listing flags.
_GIT_LIST_ONLY: dict[str, set[str]] = {
    "branch": {"-a", "-r", "-v", "-vv", "--all", "--remotes", "--verbose",
               "--list", "--show-current"},
    "tag":    {"-l", "--list"},
    "remote": {"-v", "--verbose"},
}


def _has_unsafe_flag(base: str, tokens: list[str]) -> bool:
    """True if a read-only command carries a flag that makes it write or run."""
    unsafe = _UNSAFE_FLAGS.get(base)
    if not unsafe:
        return False
    for token in tokens:
        # Both spellings: `--output FILE` and `--output=FILE`.
        if token in unsafe or token.split("=", 1)[0] in unsafe:
            return True
    return False


# base command -> set of read-only subcommands (None = always read-only).
_READONLY_SUBCMD: dict[str, set[str] | None] = {
    "git": {"status", "log", "diff", "branch", "show", "remote", "tag",
            "describe", "rev-parse", "ls-files", "blame"},
    "pacman": {"-Q", "-Qi", "-Qs", "-Ql", "-Qe", "-Qm", "-Si", "-Ss", "-Sg", "-Sl"},
    "systemctl": {"status", "list-units", "list-unit-files", "is-active",
                  "is-enabled", "show", "cat"},
    "journalctl": None,
    "docker": {"ps", "images", "logs", "version", "info", "inspect"},
    "flatpak": {"list", "info"},
    "pip": {"list", "show", "freeze"},
    "npm": {"list", "ls", "view", "outdated"},
}
# Commands whose operation is spelled as a FLAG rather than a word, so the
# "first non-flag token is the subcommand" rule below cannot apply to them.
_FLAG_SUBCMD = {"pacman"}

# Every way the shell starts a NEW command inside one string. A newline belongs
# here as much as `;` does: leaving it out meant "ls\n<anything>" was judged on
# the `ls` alone and everything after the line break ran unexamined — the whole
# rest of the command line was invisible to this function.
_SHELL_SPLIT = re.compile(r"\|\||&&|\||;|&|\n|\r")

# Substrings that hand a piece of the command line back to the shell to execute
# as a command of its own. `$(` and a backtick are the familiar pair; `<(` and
# `>(` (process substitution) do the same thing and were missing, so
# `grep foo <(curl …)` passed as a read-only grep.
_SHELL_EXEC_MARKERS = ("$(", "`", "<(", ">(")


def is_readonly_command(command: str) -> bool:
    """True if every segment of the command line is a safe, read-only inspect.

    Conservative by design: anything with output redirection, ``sudo`` or an
    unrecognised command is treated as NOT read-only, so it still goes through
    the normal approval path. A read-only command that reads a *secret* is not
    considered safe either — see :func:`touches_sensitive_path`.

    The whole command line has to survive this, not just its first word. Every
    hole this function has had was the same shape: something that starts a
    second command — a newline, a process substitution, ``env`` — sitting in a
    line whose *first* token looked harmless.
    """
    cmd = (command or "").strip()
    if not cmd or ">" in cmd:
        return False
    if any(marker in cmd for marker in _SHELL_EXEC_MARKERS):
        return False
    if touches_sensitive_path(cmd):
        return False
    segments = [s.strip() for s in _SHELL_SPLIT.split(cmd) if s.strip()]
    if not segments:
        return False
    for seg in segments:
        try:
            tokens = shlex.split(seg)
        except ValueError:
            return False
        if not tokens or tokens[0] == "sudo":
            return False
        base = tokens[0]
        if base in _READONLY_CMDS:
            if _has_unsafe_flag(base, tokens[1:]):
                return False
            continue
        if base in _READONLY_SUBCMD:
            allowed = _READONLY_SUBCMD[base]
            if allowed is None:
                continue
            if base in _FLAG_SUBCMD:
                # pacman-style: the operation IS a flag. Require that at least
                # one is a listed read-only operation and that no OTHER flag
                # rides along, since an unlisted one could be anything.
                flags = [t for t in tokens[1:] if t.startswith("-")]
                if flags and all(f in allowed for f in flags):
                    continue
                return False
            # Word-style subcommand: judge the FIRST non-flag token, not "any
            # token that happens to appear in the list". `any()` let
            # `git checkout branch` through on the word "branch", and let
            # `git -c core.pager=<command> log` through on the word "log" —
            # which runs that command through git's pager.
            rest = tokens[1:]
            if rest and any(t.startswith("-") for t in rest[:1]):
                # A flag BEFORE the subcommand is how these tools are told to
                # load config or an alternate helper (`git -c`, `git
                # --exec-path`). Nothing read-only needs one, so refusing costs
                # a prompt and closes the category rather than one flag of it.
                return False
            subcmd = next((t for t in rest if not t.startswith("-")), "")
            if subcmd in allowed:
                if _has_unsafe_flag(base, rest):
                    return False
                listing = _GIT_LIST_ONLY.get(subcmd) if base == "git" else None
                if listing is not None:
                    after = rest[rest.index(subcmd) + 1:]
                    if any(t not in listing for t in after):
                        return False
                continue
        return False
    return True


# ── secrets ───────────────────────────────────────────────────────────────
# Paths that hold credentials, keys or private history. Reading one is never
# auto-approved: a poisoned web page could otherwise talk the model into
# dumping the user's SSH key without a single prompt.
_SENSITIVE_PATTERNS = [
    r"(^|/)\.ssh(/|$)",
    r"(^|/)\.gnupg(/|$)",
    r"(^|/)\.aws(/|$)",
    r"(^|/)\.azure(/|$)",
    r"(^|/)\.kube(/|$)",
    r"(^|/)\.docker/config\.json$",
    r"(^|/)\.netrc$",
    r"(^|/)\.pgpass$",
    r"(^|/)\.npmrc$",
    r"(^|/)\.pypirc$",
    r"(^|/)\.git-credentials$",
    r"(^|/)\.config/maze-ai/config\.json$",
    r"(^|/)\.local/share/keyrings(/|$)",
    r"(^|/)\.mozilla/.*(cookies|logins|key\d)",
    r"(^|/)\.config/(google-chrome|chromium|BraveSoftware)/.*(Login Data|Cookies)",
    r"(^|/)\.password-store(/|$)",
    r"(^|/)id_(rsa|dsa|ecdsa|ed25519)",
    r"(^|/)\.(bash|zsh|python|mysql|psql)_history$",
    r"(^|/)\.env(\.[\w.-]+)?$",
    r"/etc/(shadow|gshadow|sudoers)",
    r"\.(pem|p12|pfx|jks|keystore)$",
    r"(secret|credential|passwd|password|api[_-]?key|token)s?\.(json|ya?ml|txt|ini|conf|env)$",
    # Persistence. Nothing here holds a secret — these are the files that decide
    # what runs automatically, at every login or every shell. A line appended to
    # .bashrc or a .desktop dropped in autostart is code execution from then on,
    # long after the conversation that wrote it is forgotten, so it gets the same
    # "always ask, never remember the approval" treatment as a private key.
    r"(^|/)\.(bashrc|zshrc|profile|bash_profile|bash_login|zprofile|zshenv)$",
    r"(^|/)\.config/fish/config\.fish$",
    r"(^|/)\.config/autostart(/|$)",
    r"(^|/)\.config/systemd/user(/|$)",
    r"(^|/)\.config/plasma-workspace/env(/|$)",
    # The backup store is the undo history itself. Editing its index is how you
    # would make a deletion unrecoverable without anyone noticing.
    r"(^|/)\.local/share/maze-ai/backups(/|$)",
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)


def is_sensitive_path(path: str | os.PathLike[str]) -> bool:
    """True if the path looks like it holds credentials, keys or private data."""
    text = str(path or "")
    if not text:
        return False
    expanded = os.path.expanduser(os.path.expandvars(text))
    return bool(_SENSITIVE_RE.search(expanded) or _SENSITIVE_RE.search(text))


def touches_sensitive_path(command: str) -> str:
    """Return the sensitive fragment a command references, or "" if none.

    Works on the raw command line (rather than parsed arguments) so it also
    catches paths hidden inside pipelines, quotes and globs.
    """
    text = (command or "")
    if not text:
        return ""
    expanded = os.path.expanduser(text)
    match = _SENSITIVE_RE.search(expanded) or _SENSITIVE_RE.search(text)
    return match.group(0) if match else ""


# ── URLs ──────────────────────────────────────────────────────────────────
# The agent picks URLs from model output, which may itself have been steered by
# a web page it just read. Refuse anything that points back into the machine or
# the local network: Ollama's own API, printers, routers, cloud metadata.
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localdomain")
_METADATA_HOSTS = {"metadata.google.internal", "instance-data"}


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def check_url(url: str, *, resolve: bool = True) -> str:
    """Return an error message if the URL is unsafe to fetch, else "".

    ``resolve=False`` skips the DNS lookup (used by tests and by callers that
    only want the cheap syntactic checks).
    """
    raw = (url or "").strip()
    if not raw:
        return "No URL provided."
    parsed = urlparse(raw if "://" in raw else "https://" + raw)
    if parsed.scheme not in ("http", "https"):
        return f"Refusing to fetch a '{parsed.scheme}' URL — only http/https are allowed."
    host = (parsed.hostname or "").lower()
    if not host:
        return "That URL has no host."
    if host in _METADATA_HOSTS or host.endswith(_BLOCKED_HOST_SUFFIXES):
        return f"Refusing to fetch '{host}' — it is a local/internal address."
    # A literal IP can be judged without DNS.
    if _is_private_ip(host.strip("[]")):
        return f"Refusing to fetch '{host}' — it is a private/loopback address."
    if not resolve:
        return ""
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        return f"Could not resolve '{host}' ({exc})."
    for info in infos:
        ip = info[4][0]
        if _is_private_ip(ip):
            return (
                f"Refusing to fetch '{host}' — it resolves to the private address "
                f"{ip}. Local services are not reachable through this tool."
            )
    return ""


# How much data a URL may carry before a fetch counts as "sending something
# out" and needs the user's explicit blessing.
_EXFIL_QUERY_CHARS = 96
_EXFIL_VALUE_CHARS = 64


def looks_like_exfiltration(url: str) -> bool:
    """True if the URL appears to carry a payload out rather than just fetch.

    The classic prompt-injection ending is ``fetch_url`` with the secret pasted
    into the query string. A long query/fragment, or one long opaque value, is
    treated as data leaving the machine and must be confirmed.
    """
    raw = (url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else "https://" + raw)
    carried = f"{parsed.query}{parsed.fragment}"
    if len(carried) > _EXFIL_QUERY_CHARS:
        return True
    for chunk in re.split(r"[&;]", carried):
        _, _, value = chunk.partition("=")
        if len(value) > _EXFIL_VALUE_CHARS:
            return True
    # A very long opaque path segment (base64 blob) counts too.
    return any(len(seg) > 120 for seg in parsed.path.split("/"))


# ── protected paths ───────────────────────────────────────────────────────
# Paths we refuse to delete or overwrite in bulk, to avoid catastrophe.
PROTECTED_PATHS = {
    Path("/"), Path.home(), Path("/etc"), Path("/usr"), Path("/var"),
    Path("/boot"), Path("/bin"), Path("/lib"), Path("/root"), Path("/opt"),
    Path.home() / ".config", Path.home() / ".local",
}
