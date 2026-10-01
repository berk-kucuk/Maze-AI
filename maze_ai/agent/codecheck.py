"""Static checks for code the agent writes — without ever running it.

Maze AI may write code but never execute it (see rules.py): code a model wrote
can do anything once it runs. So every check here only *reads* the code:

* before a file is saved, its syntax is validated in-process (Python's own
  parser, json, tomllib) and a broken file is refused with the error line;
* after it is saved, an installed linter looks at it — ruff, shellcheck,
  ``bash -n``, ``node --check``, ``gcc -fsyntax-only`` — and the findings go
  back to the model so it can fix them.

None of these execute the program: ``compile()`` builds bytecode without
running it, ``bash -n`` and ``node --check`` only parse, ``-fsyntax-only``
stops before code generation. Tools that would run code (``cargo check`` runs
build scripts, test runners run tests) are deliberately absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import tomllib

_TIMEOUT = 15
_MAX_REPORT = 2500

_PYTHON = {".py", ".pyw"}
_SHELL = {".sh", ".bash"}
_JS = {".js", ".mjs", ".cjs"}
_C = {".c", ".h"}
_CPP = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}


def _is_shell(path: Path, content: str) -> bool:
    if path.suffix in _SHELL:
        return True
    first = content.split("\n", 1)[0] if content else ""
    return first.startswith("#!") and any(sh in first for sh in ("/sh", "bash", "/zsh"))


def validate_content(path: str | Path, content: str) -> str:
    """A syntax error in ``content`` for this kind of file, or "" if it parses.

    Runs in-process and never executes anything; used before a write so a
    broken file is not saved at all.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in _PYTHON:
            # compile() parses and builds bytecode; it does not run the code.
            compile(content, str(path), "exec", dont_inherit=True)
        elif suffix == ".json":
            json.loads(content)
        elif suffix == ".toml":
            tomllib.loads(content)
    except SyntaxError as exc:
        line = (exc.text or "").rstrip("\n")
        pointer = f"\n    {line}\n    {' ' * max(0, (exc.offset or 1) - 1)}^" if line else ""
        return f"Python syntax error at line {exc.lineno}: {exc.msg}{pointer}"
    except json.JSONDecodeError as exc:
        return f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    except tomllib.TOMLDecodeError as exc:
        return f"Invalid TOML: {exc}"
    except ValueError as exc:          # e.g. a NUL byte in Python source
        return f"Invalid source: {exc}"
    except (RecursionError, MemoryError):
        return "The code is nested too deeply to parse."
    return ""


def _run(argv: list[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=_TIMEOUT, cwd=str(cwd),
            stdin=subprocess.DEVNULL, errors="replace", start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return 0, ""
    except OSError:
        return 0, ""
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def lint_file(path: str | Path) -> tuple[str, str]:
    """``(checker, findings)`` from a static linter ("" findings means clean).

    ``checker`` is "" when no linter for this file type is installed.
    """
    path = Path(path)
    if not path.is_file():
        return "", ""
    suffix = path.suffix.lower()
    try:
        content = path.read_text("utf-8", errors="replace")
    except OSError:
        return "", ""
    cwd = path.parent
    if suffix in _PYTHON:
        if shutil.which("ruff"):
            # Errors only (E9/F: syntax, undefined names, unused imports …):
            # style opinions would drown the real problems.
            code, out = _run(["ruff", "check", "--no-cache", "--output-format=concise",
                              "--select", "E9,F", "--", str(path)], cwd)
            return "ruff", "" if code == 0 else out
        error = validate_content(path, content)
        return "python", error
    if _is_shell(path, content):
        if shutil.which("shellcheck"):
            code, out = _run(["shellcheck", "--format=gcc", "--severity=warning", "--",
                              str(path)], cwd)
            return "shellcheck", "" if code == 0 else out
        shell = "bash" if "zsh" not in content.split("\n", 1)[0] else "zsh"
        if shutil.which(shell):
            code, out = _run([shell, "-n", str(path)], cwd)      # parse only
            return f"{shell} -n", "" if code == 0 else out
        return "", ""
    if suffix in _JS and shutil.which("node"):
        code, out = _run(["node", "--check", str(path)], cwd)   # parse only
        return "node --check", "" if code == 0 else out
    if suffix in _C | _CPP:
        compiler = "gcc" if suffix in _C else "g++"
        if shutil.which(compiler):
            code, out = _run([compiler, "-fsyntax-only", "-Wall", str(path)], cwd)
            return f"{compiler} -fsyntax-only", "" if code == 0 and not out else out
        return "", ""
    if suffix in (".json", ".toml"):
        return suffix[1:], validate_content(path, content)
    return "", ""


def check_report(path: str | Path) -> str:
    """One line (or a few) for the tool result: what the static check found."""
    checker, findings = lint_file(path)
    if not checker:
        return ""
    if not findings:
        return f"\nStatic check ({checker}): no problems found."
    if len(findings) > _MAX_REPORT:
        findings = findings[:_MAX_REPORT] + "\n…"
    return (f"\nStatic check ({checker}) found problems — fix them before you say "
            f"you are done:\n{findings}")
