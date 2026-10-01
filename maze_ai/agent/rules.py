"""The command rule set: what the agent may never run, what it must always ask
about, and what it may run without asking.

Three tiers, checked in this order for every shell command (and for anything
``launch_app`` would start):

``BLOCKED``
    Never runs — not in autonomous mode, not after an approval click. These are
    the commands with no legitimate place in an assistant's hands: root
    escalation, wiping a disk or the home folder, piping a download into a
    shell, opening a reverse shell, killing every process. The model is told
    the rule that stopped it; the user can still run the command themselves.
``CONFIRM``
    Always asks, in every mode, and an approval is never remembered: recursive
    deletes, history-rewriting git commands, killing processes, powering off.
``SAFE``
    Read-only inspections (``ls``, ``cd``, ``cat``, ``git status``,
    ``pacman -Q``…) run without a prompt in Ask mode.

Users extend BLOCKED and SAFE in Settings. Their entries are command prefixes
("git push" matches ``git push origin main``) or, prefixed with ``re:``, regular
expressions. A user's SAFE entry never outranks a BLOCKED or CONFIRM rule.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from .safety import _SHELL_SPLIT, is_readonly_command, touches_sensitive_path


@dataclass(frozen=True)
class Rule:
    id: str
    #: English description; the UI translates it, the model reads it as is.
    label: str
    pattern: re.Pattern


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# A word in command position: start of the line, after a separator, a quote or
# a subshell, or after a wrapper that runs its argument (`nohup sudo …`).
_CMD = (r"(?:^|[;&|(`'\"\n]|\$\(|\b(?:exec|nohup|env|xargs|time|nice|ionice|"
        r"command|watch|setsid|stdbuf|timeout\s+\S+)\s+|-c\s+)\s*")

# Places whose recursive removal (or move) takes the system or the whole home
# folder with it.
_CRITICAL = (r"(?:/|/\*|~|~/|~/\*|\$HOME|\$HOME/|\$HOME/\*|\"\$HOME\"|\$\{HOME\}|"
             r"/(?:home|etc|usr|var|boot|bin|sbin|lib|lib64|opt|root|srv|sys|proc|dev|"
             r"mnt|media|run)/?\*?)")
_END = r"(?=\s|$|[;&|)\"'])"

BLOCKED_RULES: list[Rule] = [
    Rule("root", "Gaining root (sudo, su, doas, pkexec, run0)",
         _rx(_CMD + r"(?:sudo|su|doas|pkexec|run0)\b")),
    Rule("wipe-critical", "Deleting the whole system or home folder",
         _rx(_CMD + r"rm\b[^;&|\n]*\s(?:-\w*[rR]\w*|--recursive)\b[^;&|\n]*\s" + _CRITICAL + _END
             + r"|--no-preserve-root")),
    Rule("move-critical", "Moving the system or home folder away",
         _rx(r"\bmv\s+(?:-\S+\s+)*" + _CRITICAL + r"\s")),
    Rule("find-delete-critical", "Mass-deleting from / or the home folder with find",
         _rx(r"\bfind\s+" + _CRITICAL + r"\s[^;&|\n]*(?:-delete|-exec\s+rm\b)")),
    Rule("disk", "Formatting, partitioning or overwriting a disk",
         _rx(r"\b(?:mkfs(?:\.\w+)?|wipefs|fdisk|sfdisk|cfdisk|gdisk|sgdisk|parted|"
             r"blkdiscard|mkswap)\b|\bdd\b[^\n]*\bof=/dev/|>\s*/dev/(?:sd|nvme|vd|hd|mmcblk)"
             r"|\bshred\b[^\n]*/dev/")),
    Rule("fork-bomb", "Fork bombs",
         _rx(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")),
    Rule("pipe-to-shell", "Running a downloaded script straight in a shell",
         _rx(r"\b(?:curl|wget|fetch)\b[^\n]*\|\s*(?:\S*/)?(?:sh|bash|zsh|dash|fish|ksh|"
             r"python\d?|perl|ruby|node)\b"
             r"|(?:sh|bash|zsh|source|\.)\s+<\(\s*(?:curl|wget)\b"
             r"|(?:sh|bash|zsh)\s+-c\s+[\"']?\$\(\s*(?:curl|wget)\b")),
    Rule("reverse-shell", "Opening a remote shell (reverse shells, nc -e)",
         _rx(r"/dev/(?:tcp|udp)/|\b(?:nc|ncat|netcat)\b[^\n]*\s-[a-z]*[ec]\b"
             r"|\bsocat\b[^\n]*\bexec:|\bbash\s+-i\s*>&")),
    Rule("kill-all", "Killing every process of the session",
         _rx(r"\bkill\s+(?:-\S+\s+)*-1\b|\b(?:pkill|killall)\s+(?:-\S+\s+)*-u\b"
             r"|\bloginctl\s+(?:terminate|kill)-(?:session|user)\b")),
    Rule("chmod-critical", "Changing permissions or owner of everything under / or home",
         _rx(r"\b(?:chmod|chown|chgrp)\s+[^;&|\n]*(?:-R|--recursive)\b[^;&|\n]*\s"
             + _CRITICAL + _END)),
]

CONFIRM_RULES: list[Rule] = [
    Rule("rm-recursive", "Recursive or forced delete",
         _rx(_CMD + r"rm\b[^;&|\n]*\s(?:-\w*[rRf]\w*|--recursive|--force)\b")),
    Rule("rm-glob", "Deleting with a wildcard",
         _rx(_CMD + r"rm\b[^;&|\n]*\*")),
    Rule("find-delete", "Deleting files with find",
         _rx(r"\bfind\b[^;&|\n]*(?:-delete|-exec(?:dir)?\s+(?:rm|shred|mv)\b)")),
    Rule("shred", "Shredding or truncating files",
         _rx(r"\bshred\b|\btruncate\b[^\n]*-s\s*0")),
    Rule("git-history", "Discarding work or rewriting history in git",
         _rx(r"\bgit\s+(?:reset\s+--hard|clean\s+-\w*f|push\b[^\n]*(?:\s-f\b|--force)|"
             r"checkout\s+--\s|restore\s|branch\s+-D|stash\s+(?:drop|clear)|"
             r"filter-branch|reflog\s+expire|gc\s+--prune)")),
    Rule("kill", "Stopping processes", _rx(_CMD + r"(?:kill|pkill|killall|xkill)\b")),
    Rule("services", "Stopping or disabling services",
         _rx(r"\bsystemctl\s+(?:--user\s+)?(?:stop|disable|mask|kill|reset-failed|isolate)\b")),
    Rule("power", "Shutting down, rebooting or logging out",
         _rx(r"\b(?:shutdown|reboot|poweroff|halt)\b|\bsystemctl\s+(?:poweroff|reboot|halt|"
             r"suspend|hibernate|kexec)\b|\bloginctl\s+(?:terminate|kill)|\bqdbus\b[^\n]*logout")),
    Rule("chmod-recursive", "Recursive permission or owner changes",
         _rx(r"\b(?:chmod|chown|chgrp)\b[^;&|\n]*(?:-R|--recursive)\b|\bchmod\s+[0-7]*777\b")),
    Rule("crontab", "Replacing or removing scheduled jobs",
         _rx(r"\bcrontab\s+(?:-r|-\w*r)\b|\bcrontab\s+(?!-)\S+\s*$")),
    Rule("uninstall", "Uninstalling software or deleting models and containers",
         _rx(r"\b(?:pip|pip3|pipx|npm|pnpm|yarn|cargo|flatpak|snap)\s+(?:uninstall|remove|rm)\b"
             r"|\bollama\s+rm\b|\b(?:docker|podman)\s+(?:rm|rmi|system\s+prune|volume\s+rm)\b")),
    Rule("accounts", "Removing users or groups", _rx(r"\b(?:userdel|groupdel)\b")),
    Rule("devnull-move", "Moving files into /dev/null", _rx(r"\bmv\b[^\n]*\s/dev/null\b")),
]

# Programs launch_app must never start: with them, "launching an app" is just
# running arbitrary code with none of run_command's checks.
LAUNCH_FORBIDDEN = {
    "sh", "bash", "zsh", "dash", "fish", "ksh", "csh", "tcsh", "env", "xargs",
    "python", "python3", "python2", "perl", "ruby", "node", "php", "lua",
    "sudo", "su", "doas", "pkexec", "run0", "nohup", "setsid", "exec", "eval",
    "rm", "dd", "mkfs", "shred", "kill", "pkill", "killall",
}

# Read-only extras on top of safety.is_readonly_command: moving around, and
# asking a toolchain for its version.
_NAVIGATION = {"cd", "pushd", "popd", "dirs", "clear"}
_VERSION_FLAGS = {"--version", "-V", "-v", "version", "--help", "-h"}
_VERSIONED_TOOLS = {
    "python", "python3", "node", "npm", "gcc", "g++", "clang", "rustc", "cargo",
    "go", "java", "javac", "ruby", "perl", "php", "git", "pacman", "yay", "paru",
    "ollama", "docker", "podman", "make", "cmake", "ffmpeg", "bash", "zsh",
    "fish", "kwin_wayland", "plasmashell", "tesseract", "pip", "pip3",
}
# `find` is read-only unless it is told to act on what it finds.
_FIND_ACTIONS = ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint",
                 "-fprint0", "-fprintf", "-fls")
_EXTRA_READONLY = {"lsof", "lsmod", "findmnt", "inxi", "expac", "nvidia-smi"}
# Word-style subcommands that only read.
_SAFE_SUBCOMMANDS = {
    "ollama": {"list", "ls", "ps", "show", "--version", "-v"},
    "flatpak": {"list", "info", "search"},
    "yay": {"-Q", "-Qi", "-Ss", "-Si", "-Qu", "-Qua"},
    "paru": {"-Q", "-Qi", "-Ss", "-Si", "-Qu", "-Qua"},
}

#: What the Settings page lists as built-in safe commands.
SAFE_SUMMARY = (
    "ls, cd, pwd, cat, head, tail, grep, rg, find, tree, "
    "stat, file, wc, du, df, free, uptime, ps, lscpu, lsblk, lspci, ip a, "
    "git status/log/diff/show, pacman -Q…/-Si/-Ss, systemctl status, journalctl, "
    "--version of common tools"
)


@dataclass
class Verdict:
    """What the rule set says about one command."""

    tier: str            # "blocked" | "confirm" | "safe" | "normal"
    rule: str = ""       # label of the rule that decided it ("" for safe/normal)


def _compile_user(patterns) -> list[tuple[str, re.Pattern | list[str]]]:
    """User entries: ``re:<regex>`` or a command prefix (token-wise)."""
    out = []
    for raw in patterns or []:
        line = str(raw).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("re:"):
            try:
                out.append((line, re.compile(line[3:].strip(), re.IGNORECASE)))
            except re.error:
                continue
        else:
            try:
                tokens = shlex.split(line)
            except ValueError:
                tokens = line.split()
            if tokens:
                out.append((line, tokens))
    return out


def validate_user_pattern(line: str) -> str:
    """An error message for a malformed Settings entry, or ""."""
    line = (line or "").strip()
    if line.startswith("re:"):
        try:
            re.compile(line[3:].strip())
        except re.error as exc:
            return f"{line}: {exc}"
    return ""


def _segments(command: str) -> list[list[str]]:
    out = []
    for seg in _SHELL_SPLIT.split(command or ""):
        seg = seg.strip()
        if not seg:
            continue
        try:
            out.append(shlex.split(seg))
        except ValueError:
            out.append(seg.split())
    return out


def _prefix_match(tokens: list[str], prefix: list[str]) -> bool:
    return len(tokens) >= len(prefix) and tokens[:len(prefix)] == prefix


def _user_hit(command: str, compiled) -> str:
    for line, matcher in compiled:
        if isinstance(matcher, re.Pattern):
            if matcher.search(command):
                return line
        elif any(_prefix_match(tokens, matcher) for tokens in _segments(command)):
            return line
    return ""


def blocked_rule(command: str, user_blocked=()) -> str:
    """Label of the BLOCKED rule this command breaks, or ""."""
    text = command or ""
    for rule in BLOCKED_RULES:
        if rule.pattern.search(text):
            return rule.label
    hit = _user_hit(text, _compile_user(user_blocked))
    return f"Blocked in your settings: {hit}" if hit else ""


def confirm_rule(command: str) -> str:
    """Label of the CONFIRM rule this command matches, or ""."""
    text = command or ""
    for rule in CONFIRM_RULES:
        if rule.pattern.search(text):
            return rule.label
    return ""


def _segment_safe(tokens: list[str], user_safe) -> bool:
    if not tokens:
        return True
    base = tokens[0]
    if base in _NAVIGATION:
        return True
    if base in _VERSIONED_TOOLS and len(tokens) == 2 and tokens[1] in _VERSION_FLAGS:
        return True
    if base == "find" and not any(t in _FIND_ACTIONS for t in tokens):
        return True
    if base in _SAFE_SUBCOMMANDS and len(tokens) >= 2 and tokens[1] in _SAFE_SUBCOMMANDS[base] \
            and all(t in _SAFE_SUBCOMMANDS[base] or not t.startswith("-") for t in tokens[1:]):
        return True
    if base in _EXTRA_READONLY and not any(t.startswith(("-K", "--kill", "-r", "--gpu-reset",
                                                         "-pl", "--power-limit", "-ac",
                                                         "--applications-clocks", "-pm",
                                                         "--persistence-mode", "-c",
                                                         "--compute-mode"))
                                           for t in tokens[1:]):
        return True
    if is_readonly_command(shlex.join(tokens)):
        return True
    return any(_prefix_match(tokens, prefix) for _line, prefix in user_safe
               if isinstance(prefix, list))


def is_safe_command(command: str, user_safe=()) -> bool:
    """True if every part of the command line may run without asking.

    The structural guards come first and apply to user entries too: no output
    redirection, no command substitution, nothing that touches secrets.
    """
    cmd = (command or "").strip()
    if not cmd or ">" in cmd:
        return False
    if any(marker in cmd for marker in ("$(", "`", "<(", ">(")):
        return False
    if touches_sensitive_path(cmd):
        return False
    compiled = _compile_user(user_safe)
    segments = _segments(cmd)
    if not segments:
        return False
    if all(_segment_safe(tokens, compiled) for tokens in segments):
        return True
    # A user regex may vouch for the whole line.
    return any(isinstance(m, re.Pattern) and m.fullmatch(cmd) for _l, m in compiled)


def classify(command: str, user_blocked=(), user_safe=()) -> Verdict:
    """The tier a command falls in. Order matters: blocked beats everything,
    and a user's "safe" entry never outranks a confirm rule."""
    rule = blocked_rule(command, user_blocked)
    if rule:
        return Verdict("blocked", rule)
    rule = confirm_rule(command)
    if rule:
        return Verdict("confirm", rule)
    if is_safe_command(command, user_safe):
        return Verdict("safe")
    return Verdict("normal")


def launch_blocked(app: str, args: str = "", user_blocked=()) -> str:
    """Why launch_app must not start this, or ""."""
    name = (app or "").strip().rsplit("/", 1)[-1]
    if name in LAUNCH_FORBIDDEN or re.fullmatch(r"python\d[\d.]*|mkfs\.\w+", name):
        return f"launch_app cannot start '{name}' (use run_command, which is checked)"
    return blocked_rule(f"{app} {args}".strip(), user_blocked)
