"""Explain a shell command, part by part, from the installed manual pages.

The approval dialog shows this under the command, so "approve" means approving
something understood: `pacman -Rns foo` reads as "remove packages · also
remove configuration files · also remove dependencies nothing else needs".

The words come from the man page of the program actually installed, never from
the model — an explanation the model invented would be exactly as trustworthy
as the command it is meant to vet.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from functools import lru_cache

from . import docs

# Shell syntax and builtins have no man page of their own worth quoting.
SHELL_PARTS = {
    "|": "pipe: feed the output of the previous command into the next one",
    "||": "run the next command only if the previous one failed",
    "&&": "run the next command only if the previous one succeeded",
    ";": "then run the next command",
    "&": "run the previous command in the background",
    ">": "write the output to a file, replacing what is in it",
    ">>": "append the output to the end of a file",
    "<": "read input from a file",
    "2>/dev/null": "hide error messages",
    "2>&1": "send error messages to the same place as normal output",
    "&>/dev/null": "hide all output",
}
BUILTINS = {
    "cd": "change the current directory",
    "export": "set an environment variable for the commands that follow",
    "source": "run the commands in a file in the current shell",
    ".": "run the commands in a file in the current shell",
    "echo": "print text",
    "printf": "print formatted text",
    "alias": "define or show a command alias",
    "unset": "remove a variable",
    "test": "check a condition (file exists, strings equal, …)",
    "[": "check a condition (file exists, strings equal, …)",
    "exit": "leave the shell",
    "true": "do nothing, successfully",
    "nohup": "keep running after the terminal closes",
    "time": "measure how long the command takes",
}
_SPLIT = re.compile(r"(\|\||&&|2>&1|&>/dev/null|2>/dev/null|>>|[|;&<>])")
_MAX_PARTS = 24


@dataclass
class Part:
    text: str          # the piece of the command, e.g. "-n" or "pacman"
    meaning: str       # what it does ("" when unknown)
    source: str = ""   # "man pacman", "shell", …


@lru_cache(maxsize=64)
def _manual(program: str) -> str:
    """The plain-text man page (cached: the dialog may ask twice)."""
    try:
        return docs._man_text(program, "")
    except Exception:  # noqa: BLE001 - an explanation is optional
        return ""


def _summary(page: str) -> str:
    """'pacman - package manager utility' → 'package manager utility'."""
    match = re.search(r"NAME\s*\n\s*(.+?)\n\s*\n", page, re.S)
    if not match:
        return ""
    line = " ".join(match.group(1).split())
    return line.split(" - ", 1)[1].strip() if " - " in line else line


def _sections(page: str) -> list[tuple[str, str]]:
    """(header, body) for each top-level section (headers are flush left)."""
    parts = re.split(r"\n(?=[A-Z][A-Z0-9 ,()/'-]{2,}\n)", page)
    out = []
    for part in parts:
        header, _, body = part.partition("\n")
        out.append((header.strip(), body))
    return out


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(r"(\w)‐ (\w)", r"\1\2", text)          # re-join hyphenated breaks
    match = re.match(r"(.+?[.;:])(\s|$)", text)
    sentence = match.group(1) if match else text
    return sentence[:220].rstrip(";:")


def _option_meaning(page: str, option: str, operation: str = "") -> str:
    """The description of ``option`` (e.g. "-n" or "--force") in the page.

    When an option is defined in several sections (pacman's -n means one thing
    for -R and another elsewhere), the section named after the operation
    ("REMOVE OPTIONS" for -R/--remove) wins.
    """
    pattern = re.compile(
        # The option may follow its short alias on the same line: "-f, --force".
        rf"(?m)^(?P<indent>[ \t]+)(?:-[^\s,]+,\s+)*{re.escape(option)}(?:[ ,=\[<|]|$)[^\n]*\n"
        r"(?P<body>(?:[ \t]+[^\n]*\n?)+)"
    )
    hits: list[tuple[int, str, str]] = []
    for header, body in _sections(page):
        for match in pattern.finditer(body):
            indent = len(match.group("indent").expandtabs())
            text = match.group("body")
            first = text.splitlines()[0] if text else ""
            # A definition's description sits deeper than its name; a line
            # that merely starts with the option inside a paragraph doesn't.
            if len(first) - len(first.lstrip()) <= indent:
                continue
            hits.append((indent, header, text))
    if hits:
        shallowest = min(h[0] for h in hits)
        hits = [h for h in hits if h[0] == shallowest]
    # Options with several names list them on lines of their own
    # ("-f\n--force\n--no-force-with-lease…"): skip to the description.
    hits_by_section = [
        (header, "\n".join(_drop_name_lines(text.splitlines())))
        for _i, header, text in hits
    ]
    hits = hits_by_section
    if not hits:
        # Single-line style: "  -n, --nosave   Do not save …"
        inline = re.search(rf"(?m)^[ \t]+{re.escape(option)}\b[^\n]*?\s{{2,}}(\S[^\n]+)", page)
        return _first_sentence(inline.group(1)) if inline else ""
    if operation and len(hits) > 1:
        for header, body in hits:
            if operation.upper() in header.upper():
                return _first_sentence(body)
    return _first_sentence(hits[0][1])


def _drop_name_lines(lines: list[str]) -> list[str]:
    while lines and lines[0].strip().startswith("-"):
        lines = lines[1:]
    return lines


def _operation_name(page: str, flag: str) -> str:
    """'-R' → 'remove' (the long name pacman-style tools give an operation)."""
    match = re.search(rf"(?m)^[ \t]+{re.escape(flag)}, --([a-z-]+)", page)
    return match.group(1) if match else ""


def _subcommand_meaning(page: str, sub: str) -> str:
    """'restart' in `systemctl restart` → its description in the COMMANDS list.

    Only pages with a COMMANDS section are searched, so a file name after `cp`
    is never mistaken for a subcommand.
    """
    for header, body in _sections(page):
        if "COMMAND" not in header.upper():
            continue
        match = re.search(
            rf"(?m)^(?P<indent>[ \t]+){re.escape(sub)}\b[^\n]*\n"
            r"(?P<body>(?:[ \t]+[^\n]*\n?)+)",
            body,
        )
        if match:
            return _first_sentence(match.group("body"))
    return ""


_SIGNALS = {"9": "SIGKILL", "KILL": "SIGKILL", "15": "SIGTERM", "TERM": "SIGTERM",
            "1": "SIGHUP", "HUP": "SIGHUP", "2": "SIGINT", "INT": "SIGINT"}
_SIGNAL_MEANING = {
    "SIGKILL": "force-stop at once: the program cannot save or clean up",
    "SIGTERM": "ask the program to quit cleanly (the default)",
    "SIGHUP": "tell the program to reload or hang up",
    "SIGINT": "interrupt, like pressing Ctrl+C",
}


def _special(name: str, args: list[str]) -> list[Part]:
    """Arguments whose meaning is a convention rather than a documented flag."""
    out: list[Part] = []
    if name in ("kill", "pkill", "killall"):
        for arg in args:
            sig = arg.lstrip("-").upper().removeprefix("SIG") if arg.startswith("-") else ""
            if sig in _SIGNALS:
                out.append(Part(arg, _SIGNAL_MEANING[_SIGNALS[sig]], "signal(7)"))
        return out
    if name == "chmod" and args:
        mode = args[0]
        if re.fullmatch(r"[0-7]{3,4}", mode):
            who = ("owner", "group", "everyone else")
            bits = mode[-3:]
            desc = "; ".join(
                f"{w}: " + ("".join(c for c, b in zip("rwx", (4, 2, 1), strict=False) if int(d) & b) or "nothing")
                for w, d in zip(who, bits, strict=False)
            )
            out.append(Part(mode, f"permissions — {desc} (r=read, w=write, x=run)", "chmod"))
        elif re.fullmatch(r"[ugoa]*[+-=][rwxXst]+", mode):
            verb = {"+": "add", "-": "remove", "=": "set exactly"}[re.search(r"[+-=]", mode).group()]
            out.append(Part(mode, f"{verb} permission '{mode.lstrip('ugoa+-=')}' "
                                  "(x = allow running it as a program)", "chmod"))
        return out
    return out


def _explain_segment(tokens: list[str]) -> list[Part]:
    if not tokens:
        return []
    program = tokens[0]
    name = program.rsplit("/", 1)[-1]
    if name in BUILTINS:
        parts = [Part(program, BUILTINS[name], "shell")]
        return parts + [Part(t, "", "") for t in tokens[1:]]
    page = _manual(name) if re.fullmatch(r"[\w.+-]+", name) else ""
    parts = [Part(program, _summary(page) if page else "", f"man {name}" if page else "")]
    special = _special(name, tokens[1:])
    if special:
        return parts + special
    operation = ""
    rest = tokens[1:]
    # A subcommand (`git reset`, `systemctl restart`) is the first plain word.
    sub = next((t for t in rest if not t.startswith("-")), "")
    if sub and re.fullmatch(r"[a-z][a-z-]+", sub) and all(
        t.startswith("--") and "=" not in t for t in rest[:rest.index(sub)]
    ):
        # Leading long flags (`systemctl --user restart`) are explained too.
        for flag in rest[:rest.index(sub)]:
            parts.append(Part(flag, _option_meaning(page, flag), f"man {name}"))
        rest = rest[rest.index(sub):]
        sub_page = _manual(f"{name}-{sub}") if name in ("git", "docker", "podman") else ""
        if sub_page:
            parts.append(Part(sub, _summary(sub_page), f"man {name}-{sub}"))
            page, name = sub_page, f"{name}-{sub}"
        else:
            parts.append(Part(sub, _subcommand_meaning(page, sub), f"man {name}"))
        rest = rest[1:]
    else:
        sub = ""
    for token in rest:
        if not page or not token.startswith("-") or token == "-":
            parts.append(Part(token, "", ""))
            continue
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            parts.append(Part(token, _option_meaning(page, option, operation), f"man {name}"))
            continue
        # Short options may be bundled: -Rns is -R, -n, -s. The first capital
        # letter of a pacman-style tool names the operation.
        letters = token[1:]
        whole = _option_meaning(page, token, operation)
        if whole or len(letters) == 1:
            parts.append(Part(token, whole, f"man {name}" if whole else ""))
            if letters[:1].isupper() and not operation:
                operation = _operation_name(page, token)
            continue
        for letter in letters:
            flag = f"-{letter}"
            if letter.isupper() and not operation:
                operation = _operation_name(page, flag)
                meaning = _option_meaning(page, flag)
            else:
                meaning = _option_meaning(page, flag, operation)
            parts.append(Part(flag, meaning, f"man {name}" if meaning else ""))
    return parts


def explain_command(command: str) -> list[Part]:
    """Every piece of a command line with what it does, as far as is known."""
    command = (command or "").strip()
    if not command:
        return []
    out: list[Part] = []
    for chunk in _SPLIT.split(command):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk in SHELL_PARTS:
            out.append(Part(chunk, SHELL_PARTS[chunk], "shell"))
            continue
        try:
            tokens = shlex.split(chunk)
        except ValueError:
            tokens = chunk.split()
        out.extend(_explain_segment(tokens))
        if len(out) >= _MAX_PARTS:
            break
    # Plain arguments (file names, package names) need no gloss; keep only the
    # parts with something to say, plus the first word of each command.
    return [p for p in out if p.meaning][:_MAX_PARTS]
