"""Headless entry points — Maze AI from the terminal, without a window.

The star is ``--fix``: your last command failed, you run ``mzfix``, and the
corrected command lands on your prompt ready to press Enter. That only works if
the assistant can be asked a question without spinning up a GUI, which is what
this module is for.
"""

from __future__ import annotations

import logging
import re
import select
import shutil
import subprocess
import sys

from .agent.prompts import guess_language
from .config import Config
from .llm import build_backend
from .llm.base import LLMError

log = logging.getLogger(__name__)

FIX_SYSTEM = """\
You are Maze AI on Maze Linux (Arch-based, pacman/AUR). The user ran a shell \
command that failed or did the wrong thing.

Line 1 of your reply MUST be a runnable shell command and nothing else — no \
prompt marker, no backticks, no quotes around the whole line, no prose. Line 2 \
onwards: one short sentence saying what was wrong.

Example reply:
sudo pacman -Syu --noconfirm
"--noconfrm" was a typo for "--noconfirm".

If the command needs root, include sudo. If the command itself is fine and the \
error is elsewhere (a missing file, a wrong path), give the closest command \
that achieves what the user clearly wanted."""

ASK_SYSTEM = """\
You are Maze AI on Maze Linux (Arch-based). Answer in plain text for a \
terminal: no markdown headings, no code fences. Be brief and concrete."""

_SHELL_SNIPPETS = {
    "zsh": """\
# Maze AI shell integration — add to ~/.zshrc:  eval "$(maze-ai --shell-init zsh)"
mz() { maze-ai --ask "$*" }
mzask() { maze-ai --ask-cli "$*" }
mzfix() {
  local last=${1:-$(fc -ln -1)}
  local fixed
  fixed=$(maze-ai --fix --quiet "$last") || return 1
  [[ -n "$fixed" ]] && print -z -- "$fixed"
}
# Ctrl+X, F fixes the command you are typing (or the last one you ran).
_maze_fix_widget() {
  local target=${BUFFER:-$(fc -ln -1)}
  local fixed
  fixed=$(maze-ai --fix --quiet "$target") || return
  [[ -n "$fixed" ]] && BUFFER="$fixed" && CURSOR=${#BUFFER}
}
zle -N _maze_fix_widget
bindkey '^Xf' _maze_fix_widget
""",
    "bash": """\
# Maze AI shell integration — add to ~/.bashrc:  eval "$(maze-ai --shell-init bash)"
mz() { maze-ai --ask "$*"; }
mzask() { maze-ai --ask-cli "$*"; }
mzfix() {
  local last="${1:-$(fc -ln -1)}"
  local fixed
  fixed=$(maze-ai --fix --quiet "$last") || return 1
  [ -n "$fixed" ] && READLINE_LINE="$fixed" && READLINE_POINT=${#fixed} || true
  [ -n "$fixed" ] && echo "$fixed"
}
""",
    "fish": """\
# Maze AI shell integration — add to ~/.config/fish/config.fish:
#   maze-ai --shell-init fish | source
function mz; maze-ai --ask $argv; end
function mzask; maze-ai --ask-cli $argv; end
function mzfix
    set -l last (test -n "$argv[1]"; and echo $argv; or history --max=1)
    set -l fixed (maze-ai --fix --quiet "$last")
    test -n "$fixed"; and commandline -r -- $fixed
end
""",
}


def shell_init(shell: str) -> int:
    """Print the shell integration snippet for ``shell``."""
    snippet = _SHELL_SNIPPETS.get((shell or "").strip().lower())
    if snippet is None:
        print(
            f"Unknown shell '{shell}'. Available: {', '.join(_SHELL_SNIPPETS)}",
            file=sys.stderr,
        )
        return 2
    print(snippet)
    return 0


def _stdin_text(limit: int = 8000, wait: float = 0.2) -> str:
    """Anything piped in (the failed command's output), if there is any.

    Never blocks: a plain ``maze-ai --fix "…"`` from a script has stdin open
    but empty, and reading it would hang forever waiting for an EOF that isn't
    coming. So we only read when the descriptor says it has something.
    """
    if sys.stdin is None or sys.stdin.closed or sys.stdin.isatty():
        return ""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], wait)
        if not ready:
            return ""
        return sys.stdin.read(limit)
    except (OSError, ValueError):
        return ""


def _copy_to_clipboard(text: str) -> bool:
    for argv in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-ib"]):
        if not shutil.which(argv[0]):
            continue
        try:
            subprocess.run(argv, input=text, text=True, timeout=10, check=True)
            return True
        except (subprocess.SubprocessError, OSError):
            continue
    return False


def _language_note(text: str) -> str:
    """Ask for the user's language when we can tell what it is."""
    code = guess_language(text)
    names = {"tr": "Turkish", "de": "German", "fr": "French", "es": "Spanish",
             "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
             "zh": "Chinese", "ja": "Japanese", "en": "English"}
    name = names.get(code)
    return f"\nWrite the explanation in {name}." if name else ""


def _complete(system: str, user: str, config: Config | None = None) -> str:
    config = config or Config()
    backend = build_backend(config)
    return backend.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]).strip()


def fix_command(command: str, quiet: bool = False) -> int:
    """Correct a failed shell command.

    ``quiet`` prints only the corrected command, which is what the shell
    integration feeds back onto your prompt.
    """
    command = (command or "").strip()
    if not command:
        print("Nothing to fix — pass the command.", file=sys.stderr)
        return 2
    output = _stdin_text()
    question = f"Command:\n{command}"
    if output.strip():
        question += f"\n\nOutput:\n{output.strip()[:4000]}"
    try:
        reply = _complete(FIX_SYSTEM + _language_note(command), question)
    except LLMError as exc:
        print(f"maze-ai: {exc}", file=sys.stderr)
        return 1

    lines = [line for line in reply.splitlines() if line.strip()]
    if not lines:
        print("maze-ai: the model returned nothing.", file=sys.stderr)
        return 1
    fixed = _command_line(lines)
    if quiet:
        # Nothing is printed when the model answered in prose: the shell
        # integration puts this straight onto the prompt, and a sentence there
        # would be worse than silence.
        if fixed:
            print(fixed)
        return 0 if fixed else 1
    print(fixed)
    if len(lines) > 1:
        print()
        print("\n".join(lines[1:]).strip())
    if _copy_to_clipboard(fixed):
        print("\n(copied to the clipboard)", file=sys.stderr)
    return 0


# Shell words that are real commands but never on PATH.
_BUILTINS = {
    "cd", "export", "source", "alias", "unset", "echo", "printf", "read",
    "for", "while", "if", "case", "sudo", "doas", "time", "env", "set",
    "trap", "exec", "eval", "pushd", "popd", "test",
}
# Punctuation that only appears in shell, never in an English sentence.
_SHELL_MARKERS = re.compile(r"(\s-{1,2}\w|\||&&|;|>|<|\$\(|~/|\*\.)")


def _looks_like_command(line: str) -> bool:
    """Is this line something you could press Enter on?

    The reliable signal is the first word: `grep` and `pacman` exist on the
    machine, "The" and "Bu" do not. Lines for a program that isn't installed
    still pass if they are shaped like a command line (flags, pipes, globs).
    """
    words = line.split()
    if not words:
        return False
    head = words[0].strip("\"'")
    if head in _BUILTINS or shutil.which(head):
        return True
    ends_like_prose = line.rstrip().endswith((".", "!", "?"))
    if ends_like_prose:
        return False
    # Not installed, but shaped like a command line (flags, pipes, globs).
    if _SHELL_MARKERS.search(line):
        return True
    # Short, lowercase, ASCII: `npm install`, `yay -S foo` on a box without npm
    # or yay. Prose starts with a capital or carries non-ASCII letters.
    return (
        len(words) <= 5
        and head.isascii()
        and head[:1].islower()
        and all(word.isascii() for word in words)
    )


def _command_line(lines: list[str]) -> str:
    """Pick the runnable command out of a model reply ("" if there isn't one)."""
    cleaned: list[str] = []
    for raw in lines:
        candidate = re.sub(r"^(?:```\w*|\$|#|>)\s*", "", raw.strip()).strip(" `")
        if candidate and not candidate.startswith("```"):
            cleaned.append(candidate)
    for candidate in cleaned:
        if _looks_like_command(candidate):
            return candidate
    return ""


def ask_cli(question: str) -> int:
    """Answer a question straight in the terminal, streaming as it comes."""
    question = (question or "").strip()
    if not question:
        print("Nothing to ask.", file=sys.stderr)
        return 2
    piped = _stdin_text()
    if piped.strip():
        question += f"\n\nContext:\n{piped.strip()[:6000]}"
    config = Config()
    backend = build_backend(config)
    messages = [
        {"role": "system", "content": ASK_SYSTEM + _language_note(question)},
        {"role": "user", "content": question},
    ]
    try:
        for chunk in backend.chat_stream(messages):
            sys.stdout.write(chunk)
            sys.stdout.flush()
    except LLMError as exc:
        print(f"\nmaze-ai: {exc}", file=sys.stderr)
        return 1
    print()
    return 0
