"""Tool registry for the agent.

Each tool is a plain callable returning a :class:`ToolResult`. Tools are
described to the model in the system prompt (see ``prompts.py``) so the exact
same set works for Gemini and any Ollama model — no native function-calling
required.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, unquote, urlparse

import requests

from ..reminders import ReminderStore, parse_when
from .safety import (
    PROTECTED_PATHS,
    check_url,
    is_dangerous_command,
    is_readonly_command,
    is_sensitive_path,
    looks_like_exfiltration,
    touches_sensitive_path,
)

log = logging.getLogger(__name__)

__all__ = [
    "TOOLS", "ToolResult", "ToolSpec", "SIDE_EFFECT_TOOLS", "EGRESS_TOOLS",
    "SENSITIVE_TOOLS",
    "tool_schemas", "PROTOCOL_SCHEMA", "TOOL_GROUPS", "tools_for_groups",
    "is_dangerous_command", "is_readonly_command", "is_sensitive_path",
    "looks_like_exfiltration", "touches_sensitive_path", "check_url",
    "set_reminder_store",
]

# Tools that change the system or run code. In "ask" mode these require the
# user's approval before they run.
SIDE_EFFECT_TOOLS = {
    "run_command", "launch_app", "write_file", "edit_file", "append_file",
    "create_dir", "delete_path", "move_path", "copy_path", "clipboard_copy",
    "undo_file_change",
}


# Tools that read private data without naming a path in their arguments, so the
# sensitive-path check can't see it. They are confirmed like any other access to
# secrets while "guard_secrets" is on.
SENSITIVE_TOOLS = {"recent_commands"}

# Tools that reach out to the network with a model-chosen destination. These
# are the exfiltration channel a prompt-injected page would try to use, so they
# get their own approval rule in the agent loop.
EGRESS_TOOLS = {"fetch_url"}

# Maximum bytes fetch_url will pull down before giving up — a model-chosen URL
# must never be able to pull a multi-gigabyte file into memory.
MAX_FETCH_BYTES = 5_000_000

# Shared reminder store — the UI injects its instance so tool writes and the
# UI's due-checker see the same data. Falls back to a lazily-created one.
_REMINDER_STORE: ReminderStore | None = None


def set_reminder_store(store: ReminderStore) -> None:
    global _REMINDER_STORE
    _REMINDER_STORE = store


def _reminders() -> ReminderStore:
    global _REMINDER_STORE
    if _REMINDER_STORE is None:
        _REMINDER_STORE = ReminderStore()
    return _REMINDER_STORE

MAX_OUTPUT = 12_000  # chars of command output fed back to the model


@dataclass
class ToolResult:
    ok: bool
    output: str
    #: Where the shell ended up, when a tool can change the working directory.
    cwd: str = ""
    #: Image paths the model should look at, when it has eyes. The text output
    #: always stands on its own, so a model without vision loses nothing.
    images: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        status = "OK" if self.ok else "ERROR"
        return f"[{status}] {self.output}"


@dataclass
class ToolSpec:
    name: str
    description: str
    args: dict[str, str]           # arg name -> human description
    run: Callable[..., ToolResult]
    example: str = ""
    side_effect: bool = field(default=False)


def _clip(text: str) -> str:
    """Trim long output to fit the model's context — keeping BOTH ends.

    Head-only truncation loses exactly the part that usually matters: a build
    log's error summary, a command's last line, a traceback's final frame. So
    keep two thirds from the start and one third from the end, with an explicit
    marker for what was dropped.
    """
    text = text or ""
    if len(text) <= MAX_OUTPUT:
        return text
    head = (MAX_OUTPUT * 2) // 3
    tail = MAX_OUTPUT - head
    dropped = len(text) - MAX_OUTPUT
    return (
        text[:head]
        + f"\n\n… [{dropped} chars omitted from the middle] …\n\n"
        + text[-tail:]
    )


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path))).resolve()


# ── backups (so a bad edit is never final) ───────────────────────────────
_BACKUP_DIR = (
    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "maze-ai" / "backups"
)
_BACKUP_INDEX = _BACKUP_DIR / "index.json"
MAX_BACKUP_BYTES = 20_000_000   # don't copy huge files just to allow an undo
MAX_BACKUPS = 200               # keep the index (and the folder) bounded


def _read_backup_index() -> list[dict]:
    try:
        data = json.loads(_BACKUP_INDEX.read_text("utf-8"))
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _backup(path: Path) -> str:
    """Snapshot a file before it is overwritten or deleted.

    Returns a short note for the tool output ("" when nothing was copied).
    Failures are never fatal: a missing backup must not block the edit the user
    asked for, it just means there is nothing to undo.
    """
    try:
        if not path.is_file():
            return ""
        size = path.stat().st_size
        if size > MAX_BACKUP_BYTES:
            return " (too large to back up)"
        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dest = _BACKUP_DIR / f"{stamp}-{path.name}"
        shutil.copy2(path, dest)
        index = _read_backup_index()
        index.append({
            "src": str(path),
            "backup": str(dest),
            "time": time.time(),
            "size": size,
        })
        # Trim the oldest entries (and their files) so the folder can't grow
        # without bound over months of use.
        while len(index) > MAX_BACKUPS:
            old = index.pop(0)
            try:
                Path(old.get("backup", "")).unlink()
            except OSError:
                pass
        tmp = _BACKUP_INDEX.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1), "utf-8")
        os.replace(tmp, _BACKUP_INDEX)
        return " (previous version backed up)"
    except OSError as exc:
        log.warning("could not back up %s: %s", path, exc)
        return ""


# ── implementations ──────────────────────────────────────────────────────
# Appended to every command so the shell reports where it ended up. Without
# this a `cd` would silently evaporate between calls, and the agent would have
# to re-state the full path in every single command.
_CWD_MARKER = "\x1eMAZE_CWD:"  # ASCII record separator: never in real output
_CWD_PROBE = f"\n__maze_rc=$?; printf '{_CWD_MARKER}%s' \"$PWD\"; exit $__maze_rc"


def run_command(command: str = "", timeout: int = 120, cwd: str = "", **_) -> ToolResult:
    """Run a shell command, returning its output and the directory it ended in.

    ``cwd`` is where the command starts (the agent threads its session
    directory through here). If the command itself changes directory, the new
    one is reported back on :attr:`ToolResult.cwd` so the next call continues
    from there.
    """
    command = (command or "").strip()
    if not command:
        return ToolResult(False, "No command provided.")
    workdir = Path.home()
    if cwd:
        candidate = _expand(cwd)
        if candidate.is_dir():
            workdir = candidate
    try:
        proc = subprocess.run(
            command + _CWD_PROBE,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workdir),
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"Command timed out after {timeout}s.")
    except Exception as exc:  # noqa: BLE001 - surface anything to the model
        return ToolResult(False, f"Failed to run command: {exc}")

    stdout, ended_in = proc.stdout, ""
    if _CWD_MARKER in stdout:
        stdout, _, ended_in = stdout.rpartition(_CWD_MARKER)
        ended_in = ended_in.strip()

    body = ""
    if stdout:
        body += stdout
    if proc.stderr:
        body += ("\n[stderr]\n" if body else "") + proc.stderr
    body = body.strip() or "(no output)"
    ok = proc.returncode == 0
    prefix = "" if ok else f"exit code {proc.returncode}\n"
    if ended_in and ended_in != str(workdir):
        body += f"\n[working directory is now {ended_in}]"
    return ToolResult(ok, _clip(prefix + body), cwd=ended_in)


def launch_app(app: str = "", args: str = "", **_) -> ToolResult:
    app = (app or "").strip()
    if not app:
        return ToolResult(False, "No application specified.")
    try:
        argv = [app] + (shlex.split(args) if args else [])
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        return ToolResult(False, f"Application '{app}' not found in PATH.")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Failed to launch '{app}': {exc}")
    return ToolResult(True, f"Launched '{app}'.")


def read_file(path: str = "", offset: int = 0, limit: int = 0, **_) -> ToolResult:
    """Read a text file, optionally a slice of it.

    ``offset``/``limit`` are 1-based line numbers, so a file larger than the
    output budget can be paged through instead of being silently cut off.
    """
    try:
        p = _expand(path)
        if not p.exists():
            return ToolResult(False, f"No such file: {p}")
        if p.is_dir():
            return ToolResult(False, f"{p} is a directory (use list_dir).")
        try:
            offset = max(0, int(offset or 0))
            limit = max(0, int(limit or 0))
        except (TypeError, ValueError):
            offset, limit = 0, 0

        text = p.read_text(encoding="utf-8", errors="replace")
        if not offset and not limit:
            body = _clip(text)
            lines = text.count("\n") + 1
            if len(text) > MAX_OUTPUT:
                body += (
                    f"\n[This file has ~{lines} lines. Re-read with offset/limit "
                    "to see a specific part.]"
                )
            return ToolResult(True, body)

        all_lines = text.splitlines()
        start = max(0, offset - 1) if offset else 0
        if start >= len(all_lines):
            return ToolResult(
                False,
                f"{p} has only {len(all_lines)} lines; offset {offset} is past the end.",
            )
        end = start + limit if limit else len(all_lines)
        chunk = all_lines[start:end]
        header = f"{p} · lines {start + 1}-{start + len(chunk)} of {len(all_lines)}\n"
        return ToolResult(True, _clip(header + "\n".join(chunk)))
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not read file: {exc}")


def write_file(path: str = "", content: str = "", **_) -> ToolResult:
    try:
        p = _expand(path)
        if p.is_dir():
            return ToolResult(False, f"{p} is a directory.")
        note = _backup(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content or "", encoding="utf-8")
        return ToolResult(True, f"Wrote {len(content or '')} bytes to {p}.{note}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not write file: {exc}")


def list_dir(path: str = ".", **_) -> ToolResult:
    try:
        p = _expand(path)
        if not p.is_dir():
            return ToolResult(False, f"Not a directory: {p}")
        entries = []
        for child in sorted(p.iterdir()):
            marker = "/" if child.is_dir() else ""
            entries.append(child.name + marker)
        listing = "\n".join(entries) or "(empty)"
        return ToolResult(True, _clip(f"{p}:\n{listing}"))
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not list directory: {exc}")


class _HTMLToText(HTMLParser):
    """Very small HTML → readable-text converter (no external deps)."""

    _SKIP = {"script", "style", "noscript", "svg", "template"}
    _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "header", "footer", "ul", "ol", "table", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._out: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK:
            self._out.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK:
            self._out.append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            # Capture the title for the summary, but keep it out of the body
            # text so it isn't duplicated with the page's <h1>.
            if not self.title:
                self.title = text
            return
        self._out.append(text + " ")

    def text(self) -> str:
        raw = "".join(self._out)
        # Collapse runs of blank lines and trailing spaces.
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _html_to_text(html: str) -> tuple[str, str]:
    parser = _HTMLToText()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML shouldn't crash the tool
        pass
    return parser.text(), parser.title


def _read_capped(resp: requests.Response, limit: int = MAX_FETCH_BYTES) -> tuple[str, bool]:
    """Read a streamed response up to ``limit`` bytes. Returns (text, truncated)."""
    chunks: list[bytes] = []
    size = 0
    truncated = False
    for chunk in resp.iter_content(64 * 1024):
        if not chunk:
            continue
        chunks.append(chunk)
        size += len(chunk)
        if size >= limit:
            truncated = True
            break
    raw = b"".join(chunks)[:limit]
    # Only the declared charset: requests' apparent_encoding re-reads
    # ``resp.content``, which raises once the stream has been consumed — which
    # it always has by this point. Undeclared content is treated as UTF-8.
    encoding = resp.encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace"), truncated
    except LookupError:
        return raw.decode("utf-8", errors="replace"), truncated


def fetch_url(url: str = "", save_path: str = "", raw: bool = False, **_) -> ToolResult:
    url = (url or "").strip()
    if not url:
        return ToolResult(False, "No URL provided.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # The destination comes from model output, which may have been steered by a
    # page the model just read. Refuse anything pointing back at this machine or
    # the local network before a single byte is sent.
    problem = check_url(url)
    if problem:
        log.warning("blocked fetch_url: %s", problem)
        return ToolResult(False, problem)

    try:
        resp = requests.get(
            url,
            timeout=30,
            stream=True,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) MazeAI/1.0"},
        )
    except requests.RequestException as exc:
        return ToolResult(False, f"Could not fetch {url}: {exc}")
    # A redirect can land somewhere the first check would have refused.
    if resp.url != url:
        problem = check_url(resp.url)
        if problem:
            resp.close()
            return ToolResult(False, f"Redirected to a blocked address: {problem}")
    if resp.status_code != 200:
        resp.close()
        return ToolResult(False, f"HTTP {resp.status_code} fetching {url}.")

    ctype = resp.headers.get("Content-Type", "").lower()
    with resp:
        body, truncated = _read_capped(resp)
    title = ""
    is_html = "html" in ctype or body.lstrip()[:1] == "<"
    if is_html and not raw:
        body, title = _html_to_text(body)

    if truncated:
        body += f"\n… (download stopped at {MAX_FETCH_BYTES // 1000} kB)"

    if save_path:
        try:
            p = _expand(save_path)
            _backup(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"Fetched {url} but could not save: {exc}")
        head = (title + " — ") if title else ""
        return ToolResult(True, f"{head}Fetched {url} and wrote {len(body)} chars to {p}.")

    prefix = f"[{title}]\n" if title else ""
    return ToolResult(True, _clip(prefix + body))


def _ddg_clean(href: str) -> str:
    """Resolve a DuckDuckGo result href to the real destination URL."""
    href = (href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urlparse(href)
        # DDG wraps results as /l/?uddg=<url-encoded real url>&rut=…
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            uddg = parse_qs(parsed.query).get("uddg")
            if uddg:
                return unquote(uddg[0])
    except ValueError:
        pass
    return href


class _DDGResults(HTMLParser):
    """Extract result title/URL/snippet from a DuckDuckGo HTML results page.

    Handles both endpoints: ``html.duckduckgo.com`` (``result__a`` /
    ``result__snippet``) and the lighter ``lite.duckduckgo.com`` (``result-link``,
    no snippet class).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._cur: dict | None = None
        self._mode: str | None = None   # "title" | "snippet"
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        d = dict(attrs)
        cls = d.get("class", "") or ""
        if "result__a" in cls or "result-link" in cls:
            self._flush()
            self._cur = {"title": "", "url": _ddg_clean(d.get("href", "")), "snippet": ""}
            self._mode, self._buf = "title", []
        elif "result__snippet" in cls and self._cur is not None:
            self._mode, self._buf = "snippet", []

    def handle_endtag(self, tag):
        if tag == "a" and self._mode and self._cur is not None:
            text = " ".join("".join(self._buf).split())
            self._cur[self._mode] = text
            self._mode, self._buf = None, []

    def handle_data(self, data):
        if self._mode:
            self._buf.append(data)

    def _flush(self) -> None:
        if self._cur is not None and self._cur.get("title") and self._cur.get("url"):
            self.results.append(self._cur)
        self._cur = None

    def finish(self) -> list[dict]:
        self._flush()
        return self.results


def web_search(query: str = "", max_results: int = 5, **_) -> ToolResult:
    """Search the web via DuckDuckGo and return ranked title/URL/snippet results.

    Far more reliable than guessing a search-engine URL and fetch_url-ing it: it
    hits DDG's server-rendered HTML endpoints and parses the real result links.
    """
    query = (query or "").strip()
    if not query:
        return ToolResult(False, "No search query provided.")
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 5
    max_results = max(1, min(max_results, 10))

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    results: list[dict] = []
    last_err = "no results"
    for endpoint in ("https://html.duckduckgo.com/html/",
                     "https://lite.duckduckgo.com/lite/"):
        try:
            resp = requests.post(endpoint, data={"q": query}, timeout=30, headers=headers)
        except requests.RequestException as exc:
            last_err = str(exc)
            continue
        if resp.status_code != 200:
            last_err = f"HTTP {resp.status_code}"
            continue
        parser = _DDGResults()
        try:
            parser.feed(resp.text)
        except Exception:  # noqa: BLE001 - malformed HTML shouldn't crash the tool
            pass
        results = parser.finish()
        if results:
            break

    if not results:
        return ToolResult(False, f"No search results for '{query}' ({last_err}).")

    lines = [f"Web search results for '{query}':", ""]
    for i, r in enumerate(results[:max_results], 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['url']}")
        if r["snippet"]:
            lines.append(f"   {r['snippet']}")
        lines.append("")
    lines.append(
        "Next step: SYNTHESISE an answer for the user now — read the titles and "
        "snippets above and write a direct final_answer that summarises the key "
        "findings in the user's language, citing the relevant sources (title + "
        "URL). Only call fetch_url on a result first if the snippets don't contain "
        "enough detail to answer. Do not stop without giving the user a summary."
    )
    return ToolResult(True, _clip("\n".join(lines)))


# ── file / folder management ─────────────────────────────────────────────
_PROTECTED = PROTECTED_PATHS


def create_dir(path: str = "", **_) -> ToolResult:
    try:
        p = _expand(path)
        p.mkdir(parents=True, exist_ok=True)
        return ToolResult(True, f"Created directory {p}.")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not create directory: {exc}")


def delete_path(path: str = "", **_) -> ToolResult:
    try:
        p = _expand(path)
        if p in _PROTECTED:
            return ToolResult(False, f"Refusing to delete protected path {p}.")
        if not p.exists():
            return ToolResult(False, f"No such path: {p}")
        if p.is_dir():
            shutil.rmtree(p)
            return ToolResult(True, f"Deleted directory {p} (and its contents).")
        note = _backup(p)
        p.unlink()
        return ToolResult(True, f"Deleted file {p}.{note}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not delete: {exc}")


def move_path(src: str = "", dst: str = "", **_) -> ToolResult:
    try:
        s, d = _expand(src), _expand(dst)
        if not s.exists():
            return ToolResult(False, f"Source does not exist: {s}")
        if s in _PROTECTED:
            return ToolResult(False, f"Refusing to move protected path {s}.")
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return ToolResult(True, f"Moved {s} → {d}.")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not move/rename: {exc}")


def copy_path(src: str = "", dst: str = "", **_) -> ToolResult:
    try:
        s, d = _expand(src), _expand(dst)
        if not s.exists():
            return ToolResult(False, f"Source does not exist: {s}")
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
        return ToolResult(True, f"Copied {s} → {d}.")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not copy: {exc}")


def undo_file_change(path: str = "", **_) -> ToolResult:
    """Restore a file from the snapshot taken before the last write/delete."""
    entries = _read_backup_index()
    if not entries:
        return ToolResult(False, "No backups recorded yet.")
    if path:
        target = str(_expand(path))
        matches = [e for e in entries if e.get("src") == target]
        if not matches:
            return ToolResult(False, f"No backup recorded for {target}.")
    else:
        matches = entries  # newest overall
    entry = matches[-1]
    src, backup = entry.get("src", ""), Path(entry.get("backup", ""))
    if not backup.is_file():
        return ToolResult(False, f"The backup file for {src} is gone.")
    try:
        dest = Path(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot the current state too, so an undo can itself be undone.
        _backup(dest)
        shutil.copy2(backup, dest)
    except OSError as exc:
        return ToolResult(False, f"Could not restore {src}: {exc}")
    when = datetime.fromtimestamp(entry.get("time", 0)).strftime("%H:%M:%S")
    return ToolResult(True, f"Restored {src} from the {when} backup.")


# ── targeted file editing ─────────────────────────────────────────────────
def edit_file(path: str = "", old: str = "", new: str = "", all: bool = False, **_) -> ToolResult:
    """Replace an exact substring in a file — cheaper/safer than a full rewrite."""
    try:
        p = _expand(path)
        if not p.exists() or p.is_dir():
            return ToolResult(False, f"No such file: {p}")
        if not old:
            return ToolResult(False, "No 'old' text given to replace.")
        text = p.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(old)
        if occurrences == 0:
            return ToolResult(False, "The 'old' text was not found in the file.")
        if occurrences > 1 and not all:
            return ToolResult(
                False,
                f"'old' appears {occurrences} times; pass all=true to replace every "
                "occurrence, or include more surrounding context to make it unique.",
            )
        note = _backup(p)
        updated = text.replace(old, new) if all else text.replace(old, new, 1)
        p.write_text(updated, encoding="utf-8")
        n = occurrences if all else 1
        return ToolResult(True, f"Replaced {n} occurrence(s) in {p}.{note}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not edit file: {exc}")


def append_file(path: str = "", content: str = "", **_) -> ToolResult:
    """Append text to a file (creating it if needed)."""
    try:
        p = _expand(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(content or "")
        return ToolResult(True, f"Appended {len(content or '')} bytes to {p}.")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not append to file: {exc}")


def search_files(path: str = ".", pattern: str = "", max_results: int = 50, **_) -> ToolResult:
    """Recursively search text files for a regex, returning file:line matches."""
    pattern = (pattern or "").strip()
    if not pattern:
        return ToolResult(False, "No search pattern given.")
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return ToolResult(False, f"Invalid regex: {exc}")
    try:
        max_results = max(1, min(int(max_results), 500))
    except (TypeError, ValueError):
        max_results = 50

    root = _expand(path)
    if not root.exists():
        return ToolResult(False, f"No such path: {root}")
    _SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache"}
    hits: list[str] = []
    files = [root] if root.is_file() else root.rglob("*")
    for fp in files:
        if len(hits) >= max_results:
            break
        if not fp.is_file() or any(part in _SKIP_DIRS for part in fp.parts):
            continue
        try:
            with fp.open("r", encoding="utf-8", errors="strict") as fh:
                for lineno, line in enumerate(fh, 1):
                    if rx.search(line):
                        hits.append(f"{fp}:{lineno}: {line.strip()[:200]}")
                        if len(hits) >= max_results:
                            break
        except (UnicodeDecodeError, OSError):
            continue  # skip binaries / unreadable files
    if not hits:
        return ToolResult(True, f"No matches for /{pattern}/ under {root}.")
    body = "\n".join(hits)
    return ToolResult(True, _clip(f"{len(hits)} match(es) for /{pattern}/:\n{body}"))


# ── clipboard & screenshots ────────────────────────────────────────────────
def clipboard_copy(text: str = "", **_) -> ToolResult:
    """Copy text to the system clipboard (Wayland wl-copy or X11 xclip/xsel)."""
    if not text:
        return ToolResult(False, "No text to copy.")
    for argv in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-ib"]):
        if not shutil.which(argv[0]):
            continue
        try:
            subprocess.run(argv, input=text, text=True, timeout=10, check=True)
            return ToolResult(True, f"Copied {len(text)} chars to the clipboard.")
        except (subprocess.SubprocessError, OSError):
            continue
    return ToolResult(False, "No clipboard tool found (install wl-clipboard or xclip).")


def _screenshot_path(save_path: str = "") -> Path:
    if save_path:
        out = _expand(save_path)
    else:
        out = _expand(f"~/Pictures/maze-ai-{datetime.now():%Y%m%d-%H%M%S}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _run_capture(argv: list[str], out: Path, timeout: int = 120) -> bool:
    if not shutil.which(argv[0]):
        return False
    try:
        subprocess.run(argv, timeout=timeout, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError):
        return False
    return out.exists() and out.stat().st_size > 0


CANCELLED = (
    "The user cancelled the screen selection. Do not try again — ask them what "
    "they wanted to show you, or offer to capture the whole screen instead."
)


def capture_region(save_path: str = "") -> ToolResult:
    """Let the user drag a rectangle, and save that area to a PNG.

    Exactly ONE selector is ever started. Falling through to the next tool when
    the first one "failed" would be wrong here: the usual reason it failed is
    that the user pressed Escape, and answering that by popping up a second
    crosshair — or ImageMagick's `import`, which then sits waiting for a click —
    is the worst thing this could do.
    """
    out = _screenshot_path(save_path)

    # First selector that exists for this desktop wins.
    if shutil.which("spectacle"):
        argv = ["spectacle", "-r", "-b", "-n", "-o", str(out)]
    elif shutil.which("grim") and shutil.which("slurp"):
        argv = []          # handled below: slurp picks, grim grabs
    elif shutil.which("maim"):
        argv = ["maim", "-s", str(out)]
    elif shutil.which("scrot"):
        argv = ["scrot", "-s", str(out)]
    elif shutil.which("import"):
        argv = ["import", str(out)]
    else:
        return ToolResult(
            False,
            "No region-screenshot tool found. Install spectacle (KDE), "
            "grim + slurp (Wayland) or maim (X11).",
        )

    if argv:
        if _run_capture(argv, out):
            return ToolResult(True, str(out))
        return ToolResult(False, CANCELLED)

    # wlroots: slurp returns the geometry, grim takes the shot.
    try:
        picked = subprocess.run(["slurp"], capture_output=True, text=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as exc:
        return ToolResult(False, f"Could not start the area selector: {exc}")
    geometry = picked.stdout.strip()
    if not geometry:
        return ToolResult(False, CANCELLED)
    if _run_capture(["grim", "-g", geometry, str(out)], out):
        return ToolResult(True, str(out))
    return ToolResult(False, CANCELLED)


def screenshot(save_path: str = "", region: bool = False, **_) -> ToolResult:
    """Capture the screen to a PNG. Uses grim/spectacle (Wayland) or maim/scrot/import."""
    if region:
        result = capture_region(save_path)
        if result.ok:
            return ToolResult(True, f"Saved the selected area to {result.output}.")
        return result

    out = _screenshot_path(save_path)
    for argv in (["grim", str(out)],
                 ["spectacle", "-b", "-n", "-o", str(out)],
                 ["maim", str(out)],
                 ["scrot", str(out)],
                 ["import", "-window", "root", str(out)]):
        if _run_capture(argv, out, timeout=30):
            return ToolResult(True, f"Saved a screenshot to {out}.")
    return ToolResult(False, "No screenshot tool found (install grim, maim or scrot).")


# ── OCR (read text from an image) ──────────────────────────────────────────
def _tesseract_langs() -> set[str]:
    """The OCR languages installed for Tesseract (e.g. {'eng', 'tur'})."""
    try:
        proc = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return set()
    langs: set[str] = set()
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        line = line.strip()
        # Skip the "List of available languages (N):" header and any noise;
        # real language codes are single bare tokens like "eng" or "tur".
        if line and " " not in line and ":" not in line:
            langs.add(line)
    return langs


def ocr_image(path: str = "", lang: str = "", **_) -> ToolResult:
    """Extract text from an image file with Tesseract OCR.

    Deterministic, model-independent text extraction — use this to read the
    words in a screenshot, document, receipt or photo, especially when a vision
    model would misread them (or the model has no vision at all).
    """
    if not path:
        return ToolResult(False, "No image path given.")
    p = _expand(path)
    if not p.exists() or p.is_dir():
        return ToolResult(False, f"No such image file: {p}")
    if not shutil.which("tesseract"):
        return ToolResult(
            False,
            "Tesseract OCR is not installed. Install it with "
            "`sudo pacman -S tesseract tesseract-data-eng tesseract-data-tur`.",
        )

    langs = (lang or "").strip()
    if not langs:
        available = _tesseract_langs()
        preferred = [code for code in ("tur", "eng") if code in available]
        if not preferred:
            # "osd" is orientation detection, not a language. With no real
            # language data installed, tesseract fails with a wall of text
            # about TESSDATA_PREFIX — say what to install instead.
            real = sorted(available - {"osd"})
            if not real:
                return ToolResult(
                    False,
                    "Tesseract has no language data installed, so it cannot read "
                    "text. Install it with `sudo pacman -S tesseract-data-eng "
                    "tesseract-data-tur`.",
                )
            preferred = real[:1]
        langs = "+".join(preferred)

    def _run(with_lang: bool) -> subprocess.CompletedProcess:
        argv = ["tesseract", str(p), "stdout"]
        if with_lang and langs:
            argv += ["-l", langs]
        return subprocess.run(argv, capture_output=True, text=True, timeout=60)

    try:
        proc = _run(True)
        # If the requested language pack is missing, retry with the default.
        if proc.returncode != 0 and langs:
            proc = _run(False)
    except subprocess.TimeoutExpired:
        return ToolResult(False, "OCR timed out.")
    except (subprocess.SubprocessError, OSError) as exc:
        return ToolResult(False, f"OCR failed: {exc}")

    if proc.returncode != 0:
        error = proc.stderr.strip()
        if "Error opening data file" in error or "Failed loading language" in error:
            return ToolResult(
                False,
                "Tesseract is missing the language data it needs. Install it with "
                "`sudo pacman -S tesseract-data-eng tesseract-data-tur`.",
            )
        return ToolResult(False, f"OCR failed: {error or 'unknown error'}")
    text = proc.stdout.strip()
    if not text:
        return ToolResult(True, "(no text detected in the image)")
    return ToolResult(True, _clip(text))


# ── seeing the screen ─────────────────────────────────────────────────────
def _active_window_title() -> str:
    """The focused window's title, when the desktop will tell us."""
    for argv in (
        ["kdotool", "getactivewindow", "getwindowname"],
        ["xdotool", "getactivewindow", "getwindowname"],
    ):
        if not shutil.which(argv[0]):
            continue
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        except (subprocess.SubprocessError, OSError):
            continue
        title = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        if title:
            return title
    return ""


def read_screen(region: bool = False, save_path: str = "", lang: str = "", **_) -> ToolResult:
    """Capture the screen and read what is on it.

    Deliberately text-first: the picture is run through OCR and the *words* are
    returned, so this works with every model, not just the multimodal ones. The
    image path comes back as well, so a model that can see gets both — and the
    OCR text is more reliable for exact strings (error codes, paths) than
    eyesight anyway.
    """
    shot = capture_region(save_path) if region else screenshot(save_path)
    if not shot.ok:
        return shot
    # screenshot() reports a sentence; capture_region() reports the path.
    path = shot.output if region else shot.output.rsplit(" ", 1)[-1].rstrip(".")

    text = ocr_image(path=path, lang=lang)
    title = _active_window_title()
    header = f"Screen capture: {path}"
    if title:
        header += f"\nActive window: {title}"
    if text.ok and text.output.strip() and "no text detected" not in text.output:
        body = f"{header}\n\nText on screen (OCR):\n{text.output}"
    elif not text.ok:
        # OCR is unavailable (no tesseract, no language data). A vision model
        # can still work from the picture; a text-only one needs to say so.
        body = (
            f"{header}\n\nThe screen could not be read as text: {text.output}\n"
            "If you can see images, describe the attached capture instead."
        )
    else:
        body = (
            f"{header}\n\n(No text could be read from the screen. If you can see "
            "images, look at the attached capture; otherwise ask the user what "
            "they are looking at.)"
        )
    return ToolResult(True, _clip(body), images=[path])


def _focus_window(name: str) -> str:
    """Bring a window matching ``name`` to the front. Returns "" or an error.

    Wayland deliberately forbids apps from poking at each other's windows, so
    this leans on the compositor's own helper (kdotool on KDE, xdotool/wmctrl
    on X11). Without one of those installed there is no supported way to do it,
    and the caller falls back to the whole screen.
    """
    name = (name or "").strip()
    if not name:
        return "No window name given."
    attempts = [
        ["kdotool", "search", "--name", name, "windowactivate"],
        ["xdotool", "search", "--name", name, "windowactivate"],
        ["wmctrl", "-a", name],
    ]
    available = [argv for argv in attempts if shutil.which(argv[0])]
    if not available:
        return (
            "No window-control tool is installed, so a single window cannot be "
            "focused. Install `kdotool` (Wayland/KDE) or `xdotool` (X11) for "
            "per-window capture."
        )
    for argv in available:
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        except (subprocess.SubprocessError, OSError):
            continue
        if proc.returncode == 0:
            time.sleep(0.4)   # let the compositor finish raising it
            return ""
    return f"No window matching '{name}' was found."


def _capture_active_window(out: Path) -> bool:
    """Grab just the focused window, if the desktop offers a way."""
    if _run_capture(["spectacle", "-a", "-b", "-n", "-o", str(out)], out, timeout=30):
        return True
    if shutil.which("xdotool") and shutil.which("maim"):
        try:
            found = subprocess.run(["xdotool", "getactivewindow"],
                                   capture_output=True, text=True, timeout=10)
            window = found.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            window = ""
        if window and _run_capture(["maim", "-i", window, str(out)], out, timeout=30):
            return True
    return False


def read_window(name: str = "", lang: str = "", save_path: str = "", **_) -> ToolResult:
    """Read what is inside one application's window.

    Use this when the user talks about a specific app ("what's in my browser",
    "what does the terminal say") instead of the screen as a whole. Falls back
    to the whole screen when the window cannot be isolated, and says so, rather
    than pretending it looked at the right thing.
    """
    out = _screenshot_path(save_path)
    focus_error = _focus_window(name)
    note = ""
    captured = False
    if not focus_error:
        captured = _capture_active_window(out)
        if not captured:
            note = f"(Could not capture just the '{name}' window; showing the screen.)"
    else:
        note = f"({focus_error} Showing the whole screen instead.)"

    if not captured:
        whole = screenshot(str(out))
        if not whole.ok:
            return whole

    text = ocr_image(path=str(out), lang=lang)
    header = f"Window capture: {out}"
    if name:
        header += f"\nRequested window: {name}"
    if note:
        header += f"\n{note}"
    if text.ok and text.output.strip() and "no text detected" not in text.output:
        body = f"{header}\n\nText in the capture (OCR):\n{text.output}"
    else:
        body = (
            f"{header}\n\nNo text could be read from it: {text.output}\n"
            "If you can see images, describe the attached capture."
        )
    return ToolResult(True, _clip(body), images=[str(out)])


# ── shell history ─────────────────────────────────────────────────────────
_HISTORY_FILES = (
    ("~/.zsh_history", "zsh"),
    ("~/.bash_history", "bash"),
    ("~/.local/share/fish/fish_history", "fish"),
)


def _parse_history(text: str, shell: str) -> list[str]:
    commands: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if shell == "zsh" and line.startswith(":"):
            # Extended history: ": <timestamp>:<elapsed>;<command>"
            _, _, command = line.partition(";")
            line = command.strip()
        elif shell == "fish":
            if not line.startswith("- cmd:"):
                continue
            line = line[len("- cmd:"):].strip()
        if line:
            commands.append(line)
    return commands


def recent_commands(count: int = 15, **_) -> ToolResult:
    """The last commands the user ran in their terminal.

    "What did I just do in the terminal?" is a history question, not a
    screenshot question — the answer is exact here, and OCR of a scrolled-off
    terminal would not have it at all. Shell history is a sensitive file, so
    this always goes through the approval gate.
    """
    try:
        count = max(1, min(int(count), 200))
    except (TypeError, ValueError):
        count = 15
    for raw, shell in _HISTORY_FILES:
        path = _expand(raw)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(False, f"Could not read {path}: {exc}")
        commands = _parse_history(text, shell)
        if not commands:
            continue
        recent = commands[-count:]
        listed = "\n".join(f"{i}. {cmd}" for i, cmd in enumerate(recent, 1))
        return ToolResult(
            True,
            _clip(f"Last {len(recent)} commands from {shell} history ({path}):\n{listed}"),
        )
    return ToolResult(False, "No shell history file was found.")


# ── notifications ────────────────────────────────────────────────────────
def notify(title: str = "Maze AI", message: str = "", **_) -> ToolResult:
    message = (message or "").strip()
    if not message:
        return ToolResult(False, "No message to notify.")
    try:
        subprocess.Popen(
            ["notify-send", "-a", "Maze AI", title or "Maze AI", message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return ToolResult(False, "notify-send not available on this system.")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not send notification: {exc}")
    return ToolResult(True, f"Notification sent: {message}")


# ── reminders / to-do ────────────────────────────────────────────────────
def add_reminder(text: str = "", when: str = "", **_) -> ToolResult:
    text = (text or "").strip()
    if not text:
        return ToolResult(False, "No reminder text given.")
    due = parse_when(when)
    if due is None:
        return ToolResult(
            False,
            f"Couldn't understand the time '{when}'. Try 'in 10 minutes', "
            "'18:30', 'tomorrow 09:00' or '2026-07-16 14:00'.",
        )
    r = _reminders().add(text, due)
    return ToolResult(True, f"Reminder set for {r.when_str()}: {text}")


def list_reminders(**_) -> ToolResult:
    pending = _reminders().pending()
    if not pending:
        return ToolResult(True, "No pending reminders.")
    lines = [f"- [{r.id}] {r.when_str()} — {r.text}" for r in pending]
    return ToolResult(True, "Pending reminders:\n" + "\n".join(lines))


def remove_reminder(which: str = "", **_) -> ToolResult:
    which = (which or "").strip()
    store = _reminders()
    if store.remove(which):
        return ToolResult(True, f"Removed the reminder matching '{which}'.")
    candidates = store.matches(which)
    if len(candidates) > 1:
        listed = "\n".join(f"- [{r.id}] {r.when_str()} — {r.text}" for r in candidates)
        return ToolResult(
            False,
            f"'{which}' matches {len(candidates)} reminders — nothing was removed. "
            f"Ask the user which one, or pass its id:\n{listed}",
        )
    return ToolResult(False, f"No reminder matched '{which}'.")


TOOLS: dict[str, ToolSpec] = {
    "run_command": ToolSpec(
        name="run_command",
        description="Run a shell command on the user's machine and read its output. "
        "Use for inspecting the system, installing packages, git, etc. "
        "Do NOT use sudo — ask the user to run privileged commands themselves.",
        args={
            "command": "the shell command to execute",
            "cwd": "optional directory to run it in (the session's current "
                   "directory is used automatically; `cd` inside a command "
                   "carries over to the next one)",
        },
        run=run_command,
        example='{"action":"run_command","action_input":{"command":"ls -la ~"}}',
        side_effect=True,
    ),
    "launch_app": ToolSpec(
        name="launch_app",
        description="Launch a GUI or desktop application (detached).",
        args={"app": "executable name, e.g. firefox", "args": "optional argument string"},
        run=launch_app,
        example='{"action":"launch_app","action_input":{"app":"firefox","args":"https://archlinux.org"}}',
        side_effect=True,
    ),
    "read_file": ToolSpec(
        name="read_file",
        description="Read the contents of a text file. For long files, page "
        "through them with offset/limit instead of re-reading the whole thing.",
        args={
            "path": "path to the file",
            "offset": "optional first line to read (1-based)",
            "limit": "optional number of lines to read",
        },
        run=read_file,
        example='{"action":"read_file","action_input":{"path":"~/.bashrc"}}',
    ),
    "write_file": ToolSpec(
        name="write_file",
        description="Create or overwrite a text file with the given content.",
        args={"path": "path to the file", "content": "full file content"},
        run=write_file,
        example='{"action":"write_file","action_input":{"path":"~/notes.txt","content":"hi"}}',
        side_effect=True,
    ),
    "edit_file": ToolSpec(
        name="edit_file",
        description="Replace an exact piece of text in an existing file. Prefer this "
        "over write_file for small changes — it doesn't require rewriting the whole "
        "file. 'old' must match exactly; set all=true to replace every occurrence.",
        args={
            "path": "path to the file",
            "old": "exact text to find",
            "new": "replacement text",
            "all": "optional true to replace all occurrences (default first only)",
        },
        run=edit_file,
        example='{"action":"edit_file","action_input":'
        '{"path":"~/app.py","old":"debug = True","new":"debug = False"}}',
        side_effect=True,
    ),
    "append_file": ToolSpec(
        name="append_file",
        description="Append text to the end of a file (creates it if missing).",
        args={"path": "path to the file", "content": "text to append"},
        run=append_file,
        example='{"action":"append_file","action_input":'
        '{"path":"~/notes.txt","content":"\\n- new note"}}',
        side_effect=True,
    ),
    "search_files": ToolSpec(
        name="search_files",
        description="Recursively search text files under a path for a regular "
        "expression, returning matching file:line results. Great for finding where "
        "something is defined or used.",
        args={
            "path": "directory or file to search (default '.')",
            "pattern": "the regex to search for",
            "max_results": "optional cap on matches (default 50)",
        },
        run=search_files,
        example='{"action":"search_files","action_input":'
        '{"path":"~/project","pattern":"def main"}}',
    ),
    "list_dir": ToolSpec(
        name="list_dir",
        description="List the entries of a directory.",
        args={"path": "directory path (default '.')"},
        run=list_dir,
        example='{"action":"list_dir","action_input":{"path":"~/Projects"}}',
    ),
    "fetch_url": ToolSpec(
        name="fetch_url",
        description="Download a web page or URL and get its readable text "
        "(HTML is stripped to plain text automatically). Optionally save the "
        "result straight to a file with 'save_path'. Use this to read web "
        "content or save a page's text — not launch_app.",
        args={
            "url": "the page/URL to fetch",
            "save_path": "optional file path to write the content to",
            "raw": "optional true to keep raw HTML instead of extracted text",
        },
        run=fetch_url,
        example='{"action":"fetch_url","action_input":'
        '{"url":"https://example.com","save_path":"~/example.txt"}}',
    ),
    "web_search": ToolSpec(
        name="web_search",
        description="Search the web (DuckDuckGo) and get a ranked list of result "
        "titles, URLs and snippets. Use this to find current information or the "
        "right page — then fetch_url a result to read it. Prefer this over "
        "guessing a URL or building a search-engine link yourself.",
        args={
            "query": "what to search for",
            "max_results": "optional number of results (1-10, default 5)",
        },
        run=web_search,
        example='{"action":"web_search","action_input":'
        '{"query":"latest Linux kernel version","max_results":5}}',
    ),
    "create_dir": ToolSpec(
        name="create_dir",
        description="Create a directory (including parent directories).",
        args={"path": "directory path to create"},
        run=create_dir,
        example='{"action":"create_dir","action_input":{"path":"~/dev/project"}}',
        side_effect=True,
    ),
    "delete_path": ToolSpec(
        name="delete_path",
        description="Delete a file or directory (recursive). Refuses protected "
        "system/home paths.",
        args={"path": "file or directory to delete"},
        run=delete_path,
        example='{"action":"delete_path","action_input":{"path":"~/old.txt"}}',
        side_effect=True,
    ),
    "move_path": ToolSpec(
        name="move_path",
        description="Move or rename a file/directory.",
        args={"src": "source path", "dst": "destination path"},
        run=move_path,
        example='{"action":"move_path","action_input":'
        '{"src":"~/a.txt","dst":"~/b.txt"}}',
        side_effect=True,
    ),
    "copy_path": ToolSpec(
        name="copy_path",
        description="Copy a file or directory to a new location.",
        args={"src": "source path", "dst": "destination path"},
        run=copy_path,
        example='{"action":"copy_path","action_input":'
        '{"src":"~/a.txt","dst":"~/backup/a.txt"}}',
        side_effect=True,
    ),
    "clipboard_copy": ToolSpec(
        name="clipboard_copy",
        description="Copy text to the system clipboard so the user can paste it.",
        args={"text": "the text to copy"},
        run=clipboard_copy,
        example='{"action":"clipboard_copy","action_input":{"text":"hello"}}',
        side_effect=True,
    ),
    "screenshot": ToolSpec(
        name="screenshot",
        description="Capture the screen to a PNG file (defaults to ~/Pictures). "
        "Use when the user asks for a screenshot. Captures the whole screen "
        "unless region=true, which makes the user drag a rectangle first — only "
        "do that when they asked to choose an area.",
        args={
            "save_path": "optional path for the PNG (default ~/Pictures/…)",
            "region": "optional true ONLY when the user asked to select an area",
        },
        run=screenshot,
        example='{"action":"screenshot","action_input":{}}',
    ),
    "read_screen": ToolSpec(
        name="read_screen",
        description="Look at the user's screen: takes a screenshot and reads the "
        "text on it with OCR. Use when they ask about what is on screen, an "
        "error they can see, or 'this window'. By DEFAULT it captures the whole "
        "screen instantly and needs nothing from the user — that is what you "
        "want almost always. Only pass region=true if the user explicitly asked "
        "to pick or select an area, because it stops and waits for them to drag "
        "a box. The text is returned for every model; if you can see images, the "
        "capture is attached too.",
        args={
            "region": "optional true ONLY when the user asked to select an area "
                      "themselves; leave it out to grab the whole screen",
            "save_path": "optional path for the capture",
            "lang": "optional OCR language(s), e.g. 'tur+eng'",
        },
        run=read_screen,
        example='{"action":"read_screen","action_input":{"region":true}}',
    ),
    "read_window": ToolSpec(
        name="read_window",
        description="Read the contents of ONE application's window: brings it to "
        "the front, captures it and reads the text with OCR. Use this whenever "
        "the user asks about a named app or window ('what's in Konsole', 'what "
        "does the browser say') instead of read_screen, which grabs everything. "
        "If the window can't be isolated it falls back to the whole screen and "
        "tells you so.",
        args={
            "name": "part of the window title or app name, e.g. 'konsole'",
            "lang": "optional OCR language(s)",
            "save_path": "optional path for the capture",
        },
        run=read_window,
        example='{"action":"read_window","action_input":{"name":"konsole"}}',
    ),
    "recent_commands": ToolSpec(
        name="recent_commands",
        description="List the last commands the user ran in their shell. This is "
        "the right tool for 'what did I just run', 'what did I do in the "
        "terminal' or 'repeat my last command' — far more exact than reading a "
        "terminal window off the screen. Reads the shell history file, so the "
        "user is asked to approve it.",
        args={"count": "how many commands to list (default 15)"},
        run=recent_commands,
        example='{"action":"recent_commands","action_input":{"count":10}}',
    ),
    "ocr_image": ToolSpec(
        name="ocr_image",
        description="Read the text out of an image file using OCR (Tesseract). "
        "Use this to accurately extract the words in a screenshot, document, "
        "receipt or photo — it is far more reliable for TEXT than eyeballing the "
        "picture, and works even with non-vision models. Give it the image path "
        "(for an attached image, its path is listed in the message).",
        args={
            "path": "path to the image file",
            "lang": "optional Tesseract language(s), e.g. 'eng', 'tur', 'tur+eng' "
            "(defaults to installed tur+eng)",
        },
        run=ocr_image,
        example='{"action":"ocr_image","action_input":{"path":"~/Pictures/receipt.jpg"}}',
    ),
    "undo_file_change": ToolSpec(
        name="undo_file_change",
        description="Restore a file to the version Maze AI saved before its "
        "last write, edit or delete. Use when the user says an edit was wrong "
        "or asks to undo/revert what you just changed.",
        args={"path": "the file to restore (omit for the most recent change)"},
        run=undo_file_change,
        example='{"action":"undo_file_change","action_input":{"path":"~/notes.txt"}}',
        side_effect=True,
    ),
    "notify": ToolSpec(
        name="notify",
        description="Send a desktop notification to the user. Use to get their "
        "attention, confirm something is done, or greet them.",
        args={"title": "short title", "message": "the notification text"},
        run=notify,
        example='{"action":"notify","action_input":'
        '{"title":"Done","message":"Your backup finished."}}',
    ),
    "add_reminder": ToolSpec(
        name="add_reminder",
        description="Set a reminder / to-do that notifies the user at a given "
        "time. Accepts times like 'in 10 minutes', '18:30', 'tomorrow 09:00' "
        "or '2026-07-16 14:00'.",
        args={"text": "what to remind about", "when": "when to remind"},
        run=add_reminder,
        example='{"action":"add_reminder","action_input":'
        '{"text":"Take a break","when":"in 30 minutes"}}',
    ),
    "list_reminders": ToolSpec(
        name="list_reminders",
        description="List the user's pending reminders / to-dos.",
        args={},
        run=list_reminders,
        example='{"action":"list_reminders","action_input":{}}',
    ),
    "remove_reminder": ToolSpec(
        name="remove_reminder",
        description="Remove a pending reminder by its id or matching text.",
        args={"which": "reminder id or text to match"},
        run=remove_reminder,
        example='{"action":"remove_reminder","action_input":{"which":"Take a break"}}',
    ),
}

# ── native function-calling schemas ──────────────────────────────────────
# Ollama (and any OpenAI-compatible server) takes JSON-Schema tool definitions
# instead of a prompt catalogue. Argument types and required fields live in one
# table so the shape of every tool is visible at a glance — the descriptions
# still come from the ToolSpec, so there is only one place to edit wording.
_SCHEMA_HINTS: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    "run_command": ({"command": "string", "cwd": "string"}, ("command",)),
    "launch_app": ({"app": "string", "args": "string"}, ("app",)),
    "read_file": ({"path": "string", "offset": "integer", "limit": "integer"}, ("path",)),
    "write_file": ({"path": "string", "content": "string"}, ("path", "content")),
    "edit_file": (
        {"path": "string", "old": "string", "new": "string", "all": "boolean"},
        ("path", "old", "new"),
    ),
    "append_file": ({"path": "string", "content": "string"}, ("path", "content")),
    "search_files": (
        {"path": "string", "pattern": "string", "max_results": "integer"},
        ("pattern",),
    ),
    "list_dir": ({"path": "string"}, ()),
    "fetch_url": ({"url": "string", "save_path": "string", "raw": "boolean"}, ("url",)),
    "web_search": ({"query": "string", "max_results": "integer"}, ("query",)),
    "create_dir": ({"path": "string"}, ("path",)),
    "delete_path": ({"path": "string"}, ("path",)),
    "move_path": ({"src": "string", "dst": "string"}, ("src", "dst")),
    "copy_path": ({"src": "string", "dst": "string"}, ("src", "dst")),
    "clipboard_copy": ({"text": "string"}, ("text",)),
    "screenshot": ({"save_path": "string", "region": "boolean"}, ()),
    "read_screen": (
        {"region": "boolean", "save_path": "string", "lang": "string"}, (),
    ),
    "read_window": (
        {"name": "string", "lang": "string", "save_path": "string"}, ("name",),
    ),
    "recent_commands": ({"count": "integer"}, ()),
    "ocr_image": ({"path": "string", "lang": "string"}, ("path",)),
    "undo_file_change": ({"path": "string"}, ()),
    "notify": ({"title": "string", "message": "string"}, ("message",)),
    "add_reminder": ({"text": "string", "when": "string"}, ("text", "when")),
    "list_reminders": ({}, ()),
    "remove_reminder": ({"which": "string"}, ("which",)),
}


# Tools grouped by what they touch, so a user on a small local model can turn
# off what they don't need. Every definition costs context: the full set is
# ~2300 tokens, which is a quarter of an 8k window before the conversation even
# starts.
TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "shell": ("run_command", "launch_app", "recent_commands"),
    "files": (
        "read_file", "write_file", "edit_file", "append_file", "list_dir",
        "search_files", "create_dir", "delete_path", "move_path", "copy_path",
        "undo_file_change",
    ),
    "web": ("fetch_url", "web_search"),
    "desktop": ("screenshot", "read_screen", "read_window", "ocr_image",
                "clipboard_copy", "notify"),
    "reminders": ("add_reminder", "list_reminders", "remove_reminder"),
}
DEFAULT_GROUPS: tuple[str, ...] = tuple(TOOL_GROUPS)


def tools_for_groups(groups: list[str] | tuple[str, ...] | None) -> list[str]:
    """Tool names belonging to the enabled groups, in registry order."""
    if not groups:
        groups = DEFAULT_GROUPS
    allowed = {
        name for group in groups for name in TOOL_GROUPS.get(group, ())
    }
    return [name for name in TOOLS if name in allowed]


def _first_sentence(text: str, limit: int = 140) -> str:
    """The gist of a description — enough for a model choosing a tool."""
    text = " ".join((text or "").split())
    cut = text.find(". ")
    if 0 < cut < limit:
        return text[: cut + 1]
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def tool_schemas(names: list[str] | None = None, compact: bool = False) -> list[dict]:
    """The tool registry as JSON-Schema function definitions.

    Used for native tool calling, where the server injects the definitions into
    the model's template — which keeps the several-thousand-token prompt
    catalogue out of the context window entirely. That matters most on local
    models, where the window is small and every token costs latency.

    ``compact`` trims each description to its first sentence, roughly halving
    the definition block for models with a small window.
    """
    schemas: list[dict] = []
    for name in names or list(TOOLS):
        spec = TOOLS.get(name)
        if spec is None:
            continue
        types, required = _SCHEMA_HINTS.get(name, ({}, ()))
        properties = {
            arg: {
                "type": types.get(arg, "string"),
                "description": _first_sentence(description, 70) if compact
                else description,
            }
            for arg, description in spec.args.items()
        }
        schemas.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": _first_sentence(spec.description) if compact
                else spec.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(required),
                },
            },
        })
    return schemas


# The shape of one protocol reply, for servers that can constrain generation to
# a JSON schema. A small local model that would otherwise emit truncated or
# fenced JSON physically cannot, once this is in force.
PROTOCOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": [*TOOLS, "final_answer"]},
        "action_input": {"type": "object"},
    },
    "required": ["thought", "action", "action_input"],
}
