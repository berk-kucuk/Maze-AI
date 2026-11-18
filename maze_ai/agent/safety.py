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
from urllib.parse import unquote, urlparse

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
    """True if the command matches a known destructive pattern.

    The full rule set lives in rules.py; anything it blocks or always
    confirms counts as dangerous here too.
    """
    if _DANGEROUS_RE.search(command or ""):
        return True
    from .rules import blocked_rule, confirm_rule

    return bool(blocked_rule(command) or confirm_rule(command))


# ── read-only commands ────────────────────────────────────────────────────
# Commands whose whole purpose is to READ/inspect — safe to run without a
# prompt in "ask" mode (when "auto_approve_readonly" is on).
_READONLY_CMDS = {
    "ls", "cat", "pwd", "whoami", "id", "df", "du", "free", "uptime", "date",
    "head", "tail", "wc", "stat", "file", "uname", "lscpu", "lsblk", "lsusb",
    "lspci", "ps", "which", "whereis", "cal", "hostname",
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
# whole function. `printenv` is not here either: the environment is where API
# keys live, so dumping it is a secret read and goes through approval.

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
    # journalctl reads logs, except for the maintenance flags that delete,
    # rotate, move or re-key them.
    "journalctl": {"--vacuum-size", "--vacuum-time", "--vacuum-files", "--rotate",
                   "--flush", "--sync", "--relinquish-var", "--smart-relinquish-var",
                   "--setup-keys", "--update-catalog"},
}

# Commands that read with flags alone but WRITE once given a plain word:
# `uniq IN OUT` overwrites OUT, `hostname NAME` renames the machine, `date
# MMDDhhmm` sets the clock. Maps each to (flags that take a value, the most
# plain words allowed). Words starting with "+" are date formats, not times.
_POSITIONAL_LIMIT: dict[str, tuple[set[str], int]] = {
    "uniq":     ({"-f", "-s", "-w", "--skip-fields", "--skip-chars", "--check-chars"}, 1),
    "hostname": (set(), 0),
    "date":     ({"-d", "--date", "-r", "--reference", "-f", "--file"}, 0),
}


def _too_many_positionals(base: str, tokens: list[str]) -> bool:
    """True if a command in :data:`_POSITIONAL_LIMIT` got a word that writes."""
    spec = _POSITIONAL_LIMIT.get(base)
    if spec is None:
        return False
    takes_value, limit = spec
    count, skip = 0, False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token == "--":
            continue
        if token.startswith("-") and token != "-":
            skip = token in takes_value
            continue
        if base == "date" and token.startswith("+"):
            continue
        count += 1
    return count > limit

# git subcommands that LIST by default but CHANGE the repository once given an
# argument (`git branch -D x`, `git tag -d v1`, `git remote add …`). They count
# as read-only only with nothing after them but these listing flags.
_GIT_LIST_ONLY: dict[str, set[str]] = {
    "branch": {"-a", "-r", "-v", "-vv", "--all", "--remotes", "--verbose",
               "--list", "--show-current"},
    "tag":    {"-l", "--list"},
    "remote": {"-v", "--verbose"},
}

# `ip addr` lists, `ip addr add …` changes the network: only a bare listing,
# optionally with `show`/`list` and a device, is read-only.
_IP_LIST_ONLY = {"show", "list", "dev"}

# Flags allowed in front of the subcommand because they only pick WHICH
# manager to ask: `systemctl --user status x`.
_SCOPE_FLAGS = {"systemctl": {"--user", "--system", "--no-pager"}}


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
    # -Fy/-Sy/-S/-R/-U are the writing ones; none of them is here.
    "pacman": {"-Q", "-Qi", "-Qii", "-Qs", "-Ql", "-Qe", "-Qm", "-Qo", "-Qq",
               "-Qqe", "-Qqm", "-Qdt", "-Qdtq", "-Qtd", "-Qtdq", "-Qu", "-Qqu",
               "-Qk", "-Qkk", "-Si", "-Sii", "-Ss", "-Ssq", "-Sg", "-Sl",
               "-F", "-Fl", "-Fx", "-Fq"},
    "systemctl": {"status", "list-units", "list-unit-files", "list-timers",
                  "is-active", "is-enabled", "is-failed", "show", "cat"},
    "ip": {"a", "addr", "address", "r", "route", "l", "link", "n", "neigh"},
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


def is_readonly_command(command: str, cwd: str | None = None) -> bool:
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
    if touches_sensitive_path(cmd, cwd):
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
            if _has_unsafe_flag(base, tokens[1:]) or _too_many_positionals(base, tokens[1:]):
                return False
            continue
        if base in _READONLY_SUBCMD:
            allowed = _READONLY_SUBCMD[base]
            if allowed is None:
                if _has_unsafe_flag(base, tokens[1:]):
                    return False
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
            scope = _SCOPE_FLAGS.get(base, set())
            while rest and rest[0] in scope:
                rest = rest[1:]
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
                if base == "ip":
                    after = rest[rest.index(subcmd) + 1:]
                    # Words after it: show/list/dev and at most an interface
                    # name — never an action like add/del/set/flush.
                    names = [t for t in after if t not in _IP_LIST_ONLY]
                    if len(names) > 1 or (names and not re.fullmatch(r"[\w.@-]+", names[0])) \
                            or any(t in ("add", "del", "delete", "set", "flush", "change",
                                         "replace", "append") for t in after):
                        return False
                continue
        return False
    return True


# ── secrets ───────────────────────────────────────────────────────────────
# Paths that hold credentials, keys or private history. Reading one is never
# auto-approved: a poisoned web page could otherwise talk the model into
# dumping the user's SSH key without a single prompt.
_SECRET_PATTERNS = [
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
    r"(^|/)(\.mozilla|\.thunderbird|\.config/mozilla)/.*(cookies|logins|key\d)",
    r"(^|/)\.config/(google-chrome|chromium|BraveSoftware|vivaldi|microsoft-edge|opera)/"
    r".*(Login Data|Cookies)",
    r"(^|/)\.config/(gh|hub)(/|$)",
    r"(^|/)\.config/rclone/rclone\.conf$",
    r"(^|/)\.config/gcloud(/|$)",
    r"(^|/)\.cargo/credentials(\.toml)?$",
    r"(^|/)\.terraform\.d/credentials",
    r"(^|/)\.vault-token$",
    r"(^|/)NetworkManager/system-connections(/|$)",
    r"(^|/)\.local/share/fish/fish_history$",
    r"(^|/)\.password-store(/|$)",
    r"(^|/)id_(rsa|dsa|ecdsa|ed25519)",
    r"(^|/)\.(bash|zsh|python|mysql|psql)_history$",
    r"(^|/)\.env(\.[\w.-]+)?$",
    r"/etc/(shadow|gshadow|sudoers)",
    r"\.(pem|p12|pfx|jks|keystore)$",
    r"(secret|credential|passwd|password|api[_-]?key|token)s?\.(json|ya?ml|txt|ini|conf|env)$",
    # Wallets: a wallet file is the money itself.
    r"(^|/)wallet\.dat$",
    r"(^|/)\.bitcoin/wallets?(/|$)",
    r"(^|/)\.electrum(-ltc)?/wallets(/|$)",
    r"(^|/)Monero/wallets(/|$)",
    r"(^|/)\.ethereum/keystore(/|$)",
    r"(^|/)\.sparrow(/|$)",
    r"(^|/)\.config/(Exodus|Ledger Live|Electrum|Sparrow)(/|$)",
    # Key and password stores.
    r"(^|/)\.config/sops(/|$)",
    r"\.kdbx?$",
    r"(^|/)\.local/share/kwalletd(/|$)",
    # Private conversations and mail.
    r"(^|/)\.config/(Signal|Element|Session)(/|$)",
    r"(^|/)\.local/share/TelegramDesktop(/|$)",
    r"(^|/)\.thunderbird(/|$)",
    # Maze's own apps: the vault, encrypted capsules and their key backups,
    # Haze's settings (lock hash) and saved sessions, HazeDrop's history,
    # Maze Connect's device identity key, Maze Crypto's wallets and portfolio.
    r"(^|/)\.(config|local/share)/maze-vault(/|$)",
    r"(^|/)\.(config|local/share)/maze-capsule(/|$)",
    r"(^|/)gocryptfs\.conf$",
    r"(^|/)\.config/haze(/|$)",
    r"(^|/)\.local/share/hazedrop(/|$)",
    r"(^|/)\.(config|local/share)/mazeconnect(/|$)",
    r"(^|/)\.local/share/maze-crypto(/|$)",
    # The process environment holds API keys as often as any dotfile does.
    r"^/proc/(self|\d+)/environ$",
    # The backup store is the undo history itself, copies of edited files
    # (secrets included) — and editing its index is how you would make a
    # deletion unrecoverable without anyone noticing.
    r"(^|/)\.local/share/maze-ai/backups(/|$)",
]
# Not secrets, but files that decide what runs automatically.
_PERSISTENCE_PATTERNS = [
    # Persistence. Nothing here holds a secret — these are the files that decide
    # what runs automatically, at every login or every shell. A line appended to
    # .bashrc or a .desktop dropped in autostart is code execution from then on,
    # long after the conversation that wrote it is forgotten, so it gets the same
    # "always ask, never remember the approval" treatment as a private key.
    r"(^|/)\.(bashrc|zshrc|profile|bash_profile|bash_login|zprofile|zshenv)$",
    r"(^|/)\.config/fish/config\.fish$",
    r"(^|/)\.config/autostart(/|$)",
    r"(^|/)\.config/systemd/user(/|$)",
    r"(^|/)\.config/plasma-workspace/(env|shutdown)(/|$)",
    r"(^|/)\.config/autostart-scripts(/|$)",
    r"(^|/)\.config/environment\.d(/|$)",
    r"(^|/)\.config/fish/conf\.d(/|$)",
    r"(^|/)\.(zlogin|zlogout|bash_logout|xprofile|xinitrc|xsession|xsessionrc)$",
    r"(^|/)\.local/bin(/|$)",
    r"(^|/)\.local/share/applications(/|$)",
    r"(^|/)\.gitconfig$",
    r"(^|/)\.config/git/config$",
    r"(^|/)\.git/(hooks(/|$)|config$)",
]
_SENSITIVE_PATTERNS = _SECRET_PATTERNS + _PERSISTENCE_PATTERNS
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)
# Secrets only: what a recursive search must not read (see _recursive_secret).
# Persistence files are harmless to READ, and every repository has .git/config.
_SECRET_RE = re.compile("|".join(_SECRET_PATTERNS), re.IGNORECASE)

# Shell glob metacharacters. A path spelled with them (`~/.s[s]h/id_*`) never
# matches the patterns above as text, so it is expanded against the real disk.
_GLOB_CHARS = re.compile(r"[*?\[]")
# Bound the work a single check may do when expanding a glob.
_GLOB_LIMIT = 256

_BRACE_SEQ = re.compile(r"^(-?\d+|[A-Za-z])\.\.(-?\d+|[A-Za-z])(?:\.\.-?\d+)?$")


def _expand_braces(text: str, limit: int = _GLOB_LIMIT) -> list[str]:
    """Every word bash makes of ``text`` by brace expansion, as far as matters
    here: ``~/.{ssh,x}`` -> ``~/.ssh``, ``~/.x``; ``~/.s{r..t}h`` -> ``.srh``…

    Quoting is ignored on purpose — expanding a quoted brace that bash would
    leave alone only means one extra spelling is checked. Bounded by ``limit``.
    """
    out: list[str] = []
    todo = [text]
    while todo and len(out) < limit:
        word = todo.pop()
        start = word.find("{")
        while start != -1:
            depth, commas, end = 0, [], -1
            for i in range(start, len(word)):
                ch = word[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
                elif ch == "," and depth == 1:
                    commas.append(i)
            if end == -1:
                start = -1
                break
            head, body, tail = word[:start], word[start + 1:end], word[end + 1:]
            parts: list[str] = []
            if commas:
                bounds = [start, *commas, end]
                parts = [word[a + 1:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
            else:
                seq = _BRACE_SEQ.match(body)
                if seq:
                    lo, hi = seq.group(1), seq.group(2)
                    if lo.lstrip("-").isdigit() and hi.lstrip("-").isdigit():
                        a, b = int(lo), int(hi)
                        step = 1 if b >= a else -1
                        parts = [str(n) for n in range(a, b + step, step)[:limit]]
                    elif lo.isalpha() and hi.isalpha():
                        a, b = ord(lo), ord(hi)
                        step = 1 if b >= a else -1
                        parts = [chr(n) for n in range(a, b + step, step)]
            if parts:
                todo.extend(head + part + tail for part in parts)
                break
            start = word.find("{", start + 1)
        if start == -1:
            out.append(word)
    return out or [text]


# Shell syntax that runs or splices a command into the line. Removing it leaves
# the words the inner command would print in place: `cat $(echo ~/.ssh)/k`
# collapses to `cat echo ~/.ssh/k`, which the patterns can see. Best effort —
# a substitution can compute any string — but it covers the plain spellings.
_SPLICE = re.compile(r"\$\(|<\(|>\(|\)|`|['\"\\]")


def _collapse_shell(text: str) -> str:
    return _SPLICE.sub("", text)


def _path_forms(text: str) -> list[str]:
    """Every spelling of a path worth matching: as written, with ~ and $VARS
    expanded, normalised (``./`` and ``../`` folded), symlinks resolved, and —
    when it contains glob characters — each file the glob expands to.

    The sensitivity regexes are lexical. Without this, ``~/.config/./autostart``
    or ``~/.config/maze-ai/../maze-ai/config.json`` or a symlink pointing at
    ``~/.ssh/id_ed25519`` all reached the same file under a name no pattern
    recognised.
    """
    import glob as _glob

    forms = [text]
    candidates: list[str] = []
    for variant in _expand_braces(text):
        expanded = os.path.expanduser(os.path.expandvars(variant))
        forms.append(expanded)
        candidates.append(expanded)
        if _GLOB_CHARS.search(expanded):
            try:
                matches = _glob.glob(expanded)
            except (OSError, ValueError, re.error):
                matches = []
            candidates.extend(matches[:_GLOB_LIMIT])
    for cand in candidates:
        norm = os.path.normpath(cand)
        forms.append(norm)
        try:
            forms.append(os.path.realpath(norm))
        except (OSError, ValueError):
            pass
    return forms


def is_sensitive_path(path: str | os.PathLike[str]) -> bool:
    """True if the path looks like it holds credentials, keys or private data."""
    text = str(path or "")
    if not text:
        return False
    return any(_SENSITIVE_RE.search(form) for form in _path_forms(text))



# ── recursive reads ───────────────────────────────────────────────────────
# A search that walks a directory reads every file under it, named or not.
# `grep -r BEGIN ~` read ~/.ssh/id_ed25519 although the command line named no
# secret, so the path patterns above never saw it — and grep is a read-only
# command, so it ran without a prompt. Here the directories a recursive
# search would walk are looked into before it runs: any secret under them,
# or anything that cannot be looked into, makes the command a sensitive read.

# Programs that read file CONTENTS while walking directories. (`ls -R`,
# `tree`, `du` and `find` list names, not contents.)
_ALWAYS_RECURSIVE = {"rg", "ag"}
_RECURSIVE_WITH_FLAG = {"grep", "egrep", "fgrep", "diff"}
# Options that take the NEXT word as their value. Only options that really
# do: wrongly treating a path as an option's value would hide that path from
# the check, so anything not listed is assumed to take no value.
_VALUE_FLAGS: dict[str, set[str]] = {
    "grep": {"-e", "-f", "-m", "-A", "-B", "-C", "-d", "-D", "--regexp", "--file",
             "--max-count", "--after-context", "--before-context", "--context",
             "--directories", "--devices", "--include", "--exclude", "--exclude-dir",
             "--exclude-from", "--label", "--binary-files", "--group-separator"},
    "rg": {"-e", "-f", "-g", "-t", "-T", "-m", "-A", "-B", "-C", "-j", "-M", "-r", "-E",
           "-d", "--regexp", "--file", "--glob", "--iglob", "--type", "--type-not",
           "--max-count", "--after-context", "--before-context", "--context", "--threads",
           "--max-columns", "--replace", "--encoding", "--max-depth", "--max-filesize",
           "--path-separator", "--pre", "--pre-glob", "--sort", "--sortr", "--type-add",
           "--type-clear", "--context-separator", "--field-match-separator",
           "--field-context-separator", "--colors", "--color", "--ignore-file", "--engine",
           "--dfa-size-limit", "--regex-size-limit", "--hostname-bin", "--hyperlink-format"},
    "ag": {"-A", "-B", "-C", "-G", "-g", "-m", "-p", "--after", "--before", "--context",
           "--file-search-regex", "--max-count", "--depth", "--ignore", "--ignore-dir",
           "--path-to-ignore", "--workers", "--pager"},
    "diff": {"-I", "-x", "-X", "-S", "-W", "-F", "-D", "--ignore-matching-lines",
             "--exclude", "--exclude-from", "--starting-file", "--width",
             "--show-function-line", "--label", "--ifdef", "--line-format",
             "--horizon-lines", "--palette", "--from-file", "--to-file"},
}
_VALUE_FLAGS["egrep"] = _VALUE_FLAGS["fgrep"] = _VALUE_FLAGS["grep"]
# Options that give the PATTERN, so no positional word is the pattern.
_PATTERN_FLAGS = {"-e", "-f", "--regexp", "--file"}
# Words that stand for paths decided at run time: unknown, so not safe.
_RUNTIME_PATHS = {"{}", "-"}
# How many entries a walk may visit before giving up. A tree that large (a
# home directory, /) is not searched without asking anyway.
_WALK_LIMIT = 10_000


def _parse_reader(base: str, args: list[str]) -> tuple[bool, list[str]]:
    """(walks directories?, the paths it is pointed at) for one invocation."""
    values = _VALUE_FLAGS.get(base, set())
    recursive = base in _ALWAYS_RECURSIVE
    positionals: list[str] = []
    explicit_pattern = False
    files_mode = False
    i = 0
    while i < len(args):
        tok = args[i]
        i += 1
        if tok == "--":
            positionals.extend(args[i:])
            break
        if tok.startswith("--"):
            name, eq, val = tok.partition("=")
            if name in ("--recursive", "--dereference-recursive"):
                recursive = True
            if name == "--directories" and (val if eq else (args[i] if i < len(args) else "")) == "recurse":
                recursive = True
            if name in _PATTERN_FLAGS:
                explicit_pattern = True
            if name == "--files":
                files_mode = True
            if name in ("--from-file", "--to-file"):
                positionals.append(val if eq else (args[i] if i < len(args) else ""))
            if name in values and not eq:
                i += 1
            continue
        if tok.startswith("-") and tok != "-":
            # A cluster of short options: -rn, -iRl, -A3, -e PATTERN.
            for j, ch in enumerate(tok[1:], start=1):
                flag = "-" + ch
                if ch in "rR" and base != "rg":
                    recursive = True
                if flag in _PATTERN_FLAGS:
                    explicit_pattern = True
                if flag in values:
                    rest = tok[j + 1:]
                    if not rest:
                        if flag == "-d" and i < len(args) and args[i] == "recurse":
                            recursive = True
                        i += 1                  # the value is the next word
                    elif flag == "-d" and rest.lstrip("=") == "recurse":
                        recursive = True
                    break                       # the rest of the cluster is the value
            continue
        positionals.append(tok)
    if base in ("grep", "egrep", "fgrep", "rg", "ag") and not explicit_pattern and not files_mode:
        positionals = positionals[1:]           # the first word is the pattern
    if not positionals and base != "diff":
        positionals = ["."]                     # these search the working directory
    return recursive, positionals


def _tree_secret(root: str, budget: list[int]) -> str:
    """The first secret found at or under ``root`` (the path as a reason), or
    a reason it could not be looked into, or ""."""
    if _SECRET_RE.search(root):
        return root
    try:
        real = os.path.realpath(root)
    except (OSError, ValueError):
        return f"{root} (cannot be resolved)"
    if _SECRET_RE.search(real):
        return root
    if not os.path.isdir(real):
        return ""
    blocked: list[str] = []
    for dirpath, dirnames, filenames in os.walk(real, followlinks=False,
                                                onerror=lambda e: blocked.append(str(e.filename))):
        if blocked:
            # Something under it is unreadable to us — but maybe not to the
            # command (sudo), so it cannot be vouched for.
            return f"{root} ({blocked[0]} cannot be inspected)"
        for name in (*dirnames, *filenames):
            budget[0] -= 1
            if budget[0] < 0:
                return f"{root} (too large to inspect)"
            path = os.path.join(dirpath, name)
            if _SECRET_RE.search(path):
                return path
            if os.path.islink(path):
                try:
                    if _SECRET_RE.search(os.path.realpath(path)):
                        return path
                except (OSError, ValueError):
                    return path
    if blocked:
        return f"{root} ({blocked[0]} cannot be inspected)"
    return ""


def _recursive_secret(command: str, cwd: str | None) -> str:
    """A secret a recursive search in ``command`` would read, or ""."""
    if not any(name in command for name in (*_ALWAYS_RECURSIVE, *_RECURSIVE_WITH_FLAG)):
        return ""
    here: str | None = cwd or str(Path.home())
    budget = [_WALK_LIMIT]
    for seg in _SHELL_SPLIT.split(command):
        seg = seg.strip()
        if not seg:
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if not tokens:
            continue
        # Directory changes earlier on the line move where relative paths go.
        if tokens[0] in ("cd", "pushd"):
            target = tokens[1] if len(tokens) > 1 else "~"
            if target == "-" or any(m in target for m in _SHELL_EXEC_MARKERS):
                here = None
            else:
                target = os.path.expanduser(os.path.expandvars(target))
                here = os.path.normpath(os.path.join(here, target)) if here is not None \
                    else (target if os.path.isabs(target) else None)
            continue
        # The reader may sit behind sudo, env, timeout, xargs, find -exec…
        for k, word in enumerate(tokens):
            base = os.path.basename(word)
            if base not in _ALWAYS_RECURSIVE and base not in _RECURSIVE_WITH_FLAG:
                continue
            recursive, roots = _parse_reader(base, tokens[k + 1:])
            if not recursive:
                break
            if "xargs" in (os.path.basename(t) for t in tokens[:k]):
                return f"{base} on paths read from input"
            for root in roots:
                if root in _RUNTIME_PATHS or "$" in root or any(m in root for m in _SHELL_EXEC_MARKERS):
                    return f"{base} on a path decided at run time ({root})"
                expanded = os.path.expanduser(os.path.expandvars(root))
                if not os.path.isabs(expanded):
                    if here is None:
                        return f"{base} in an unknown directory"
                    expanded = os.path.join(here, expanded)
                targets = [expanded]
                if _GLOB_CHARS.search(expanded):
                    import glob as _glob
                    targets = _glob.glob(expanded)[:_GLOB_LIMIT]
                for target in targets:
                    hit = _tree_secret(os.path.normpath(target), budget)
                    if hit:
                        return hit
            break
    return ""

def touches_sensitive_path(command: str, cwd: str | None = None) -> str:
    """Return the sensitive fragment a command references, or "" if none.

    Works on the raw command line and on every word in it, each canonicalised
    by :func:`_path_forms`, so quoting (``~/.s''sh``), globs, braces, ``$HOME``,
    ``./``/``../`` segments and symlinks cannot hide the file being read.
    A recursive search is judged by what it would walk (:func:`_recursive_secret`),
    relative paths against ``cwd`` — where the command runs (home by default,
    as for run_command).
    """
    text = (command or "")
    if not text:
        return ""
    expanded = os.path.expanduser(text)
    match = _SENSITIVE_RE.search(expanded) or _SENSITIVE_RE.search(text)
    if match:
        return match.group(0)
    try:
        words = shlex.split(text, comments=False, posix=True)
    except ValueError:
        words = text.split()
    # The same line with command substitution spliced flat, so a path built by
    # `$(echo ~/.ssh)` or `` `printf ~/.s`sh `` is judged as the path it makes.
    collapsed = _collapse_shell(text)
    if collapsed != text:
        words += collapsed.split()
    for word in words:
        # Strip redirection/assignment prefixes so `<file` and `X=file` count.
        word = word.lstrip("<>").split("=", 1)[-1] if "=" in word else word.lstrip("<>")
        if not word or word.startswith("-"):
            continue
        for form in _path_forms(word):
            hit = _SENSITIVE_RE.search(form)
            if hit:
                return hit.group(0)
    return _recursive_secret(text, cwd)


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
# Splitting a payload over many short path segments or subdomain labels slipped
# under the per-part limits, so the totals are bounded too.
_EXFIL_PATH_CHARS = 200
_EXFIL_PATH_SEGMENTS = 12
_EXFIL_HOST_CHARS = 80
_EXFIL_LABEL_CHARS = 40


def egress_chars(url: str) -> int:
    """How much data a URL can carry beyond naming a page.

    Counted: the query, the fragment, the path past its first segment and any
    subdomain labels — every place a chunk of a secret can ride. A URL that
    stays under looks_like_exfiltration's per-URL limits can still carry a
    piece; the agent adds these up across a turn (see Agent._egress_used).
    """
    raw = (url or "").strip()
    if not raw:
        return 0
    parsed = urlparse(raw if "://" in raw else "https://" + raw)
    labels = (parsed.hostname or "").split(".")
    subdomain = ".".join(labels[:-2]) if len(labels) > 2 else ""
    segments = [s for s in parsed.path.split("/") if s]
    return (len(parsed.query) + len(parsed.fragment)
            + sum(len(s) for s in segments[1:]) + len(subdomain))


def outbound_text(url_or_query: str, is_url: bool = True) -> tuple[str, str]:
    """The data-carrying parts of a URL (or a search text), URL-decoded, as
    (case-preserving text, host name). The host is apart because DNS has no
    case: it can only be compared case-insensitively."""
    raw = (url_or_query or "").strip()
    if not is_url:
        return raw, ""
    parsed = urlparse(raw if "://" in raw else "https://" + raw)
    segments = [s for s in parsed.path.split("/") if s]
    return (unquote(" ".join([parsed.query, parsed.fragment, *segments])),
            (parsed.hostname or "").replace(".", " "))


# "=" and "/" separate: a key in "API_KEY=…" and in "?k=…" must be the same token.
_TOKEN = re.compile(r"[A-Za-z0-9_+-]{12,}")
_HEX = frozenset("0123456789abcdefABCDEF")


def secret_like_tokens(text: str, case_blind: bool = False) -> set[str]:
    """Runs of characters that look like key material rather than words:
    12+ characters mixing upper case, lower case and digits (API keys,
    tokens, base64), or 16+ hex digits — with high entropy. Words, versions,
    paths and error messages do not qualify, so searching an error a command
    printed stays free. ``case_blind`` (host names) drops the mixed-case
    requirement: letters and digits are enough there."""
    import math
    out = set()
    for tok in _TOKEN.findall(text or ""):
        upper = any(c.isupper() for c in tok)
        lower = any(c.islower() for c in tok)
        digit = any(c.isdigit() for c in tok)
        hexlike = (len(tok) >= 16 and digit and any(c.isalpha() for c in tok)
                   and all(c in _HEX for c in tok))
        mixed = (upper and lower and digit) or (case_blind and digit and (upper or lower))
        if not (mixed or hexlike):
            continue
        counts = {c: tok.count(c) for c in set(tok)}
        entropy = -sum(n / len(tok) * math.log2(n / len(tok)) for n in counts.values())
        if entropy >= 3.0:
            out.add(tok)
    return out


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
    if any(len(seg) > 120 for seg in parsed.path.split("/")):
        return True
    # ...and so does the same blob cut into many short pieces, in the path or
    # in the host name (one DNS label per chunk).
    segments = [s for s in parsed.path.split("/") if s]
    if len(parsed.path) > _EXFIL_PATH_CHARS or len(segments) > _EXFIL_PATH_SEGMENTS:
        return True
    host = parsed.hostname or ""
    if len(host) > _EXFIL_HOST_CHARS:
        return True
    return any(len(label) > _EXFIL_LABEL_CHARS for label in host.split("."))


# ── protected paths ───────────────────────────────────────────────────────
# Paths we refuse to delete or overwrite in bulk, to avoid catastrophe.
PROTECTED_PATHS = {
    Path("/"), Path.home(), Path("/etc"), Path("/usr"), Path("/var"),
    Path("/boot"), Path("/bin"), Path("/lib"), Path("/root"), Path("/opt"),
    Path.home() / ".config", Path.home() / ".local",
}
