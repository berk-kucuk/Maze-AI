"""Chat with a folder: a local index of its text files.

The user attaches a project or a folder of notes; Maze AI indexes the text in
it and, on every question, pulls the most relevant passages into the model's
context, so answers come from the files (with ``path:line`` citations) rather
than guesses.

Search is SQLite FTS5 (BM25) — built into Python, instant, no download. When an
Ollama embedding model is installed, chunks are also embedded and the two
rankings are fused, which finds passages that use different words than the
question (a Turkish question about English code, say).

Everything stays on this machine: the index lives owner-only in
``~/.cache/maze-ai/folders/``. Secrets (keys, ``.env``, shell history…) and
binary or huge files are never indexed.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import time
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .agent.safety import is_sensitive_path
from .private import private_dir

CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "maze-ai" / "folders"
)

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    "build", "dist", "target", ".cache", ".idea", ".vscode", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", ".gradle", "vendor", "site-packages",
    ".next", ".nuxt", "out", "bin", "obj",
}
MAX_FILE_BYTES = 1_000_000
MAX_FILES = 5000
MAX_TOTAL_BYTES = 150_000_000
CHUNK_CHARS = 1500
EMBED_BATCH = 32
_SCHEMA_VERSION = 2


@dataclass
class Hit:
    path: str          # relative to the folder
    start: int         # first line (1-based)
    end: int           # last line
    text: str
    score: float = 0.0

    def cite(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"


@dataclass
class IndexStats:
    files: int
    chunks: int
    embedded: bool
    skipped: int = 0
    truncated: bool = False


def _is_text(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut at the 4 kB boundary is still text.
        return exc.start >= len(head) - 4
    return True


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[tuple[int, int, str]]:
    """Split into ~size-character chunks on line boundaries: (start, end, text)."""
    chunks: list[tuple[int, int, str]] = []
    lines = text.splitlines()
    buf: list[str] = []
    start = 1
    used = 0
    for number, line in enumerate(lines, 1):
        if buf and used + len(line) > size:
            chunks.append((start, number - 1, "\n".join(buf)))
            buf, used, start = [], 0, number
        buf.append(line[:size])
        used += len(line) + 1
    if buf and any(part.strip() for part in buf):
        chunks.append((start, len(lines), "\n".join(buf)))
    return chunks


def _fts_query(text: str) -> str:
    """Words of a question as an OR query FTS5 accepts (quotes neutralise syntax)."""
    words = re.findall(r"[\w.-]{2,}", (text or "").lower())
    seen: list[str] = []
    for word in words:
        word = word.strip(".-")
        if len(word) > 1 and word not in seen:
            seen.append(word)
    return " OR ".join(f'"{w}"' for w in seen[:24])


def _pack(vector: list[float]) -> bytes:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return array("f", (v / norm for v in vector)).tobytes()


def _unpack(blob: bytes) -> array:
    vec = array("f")
    vec.frombytes(blob)
    return vec


class FolderIndex:
    def __init__(self, root: str | Path, cache_dir: Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        cache = cache_dir or CACHE_DIR
        private_dir(cache)
        digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:16]
        self.db_path = cache / f"{digest}.sqlite"
        self.db = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        os.chmod(self.db_path, 0o600)
        # The indexer and a chat turn may use the index at the same time.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=10000")
        self._setup()

    # ── storage ──────────────────────────────────────────────────────────
    def _setup(self) -> None:
        db = self.db
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version != _SCHEMA_VERSION:
            db.executescript("""
                DROP TABLE IF EXISTS files; DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS fts; DROP TABLE IF EXISTS vectors;
                DROP TABLE IF EXISTS meta;
            """)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime REAL, size INTEGER);
            CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, path TEXT,
                                              start INTEGER, end INTEGER, text TEXT);
            CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text);
            CREATE TABLE IF NOT EXISTS vectors(id INTEGER PRIMARY KEY, vec BLOB);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        """)
        db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        db.commit()

    def _meta(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def _set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def close(self) -> None:
        self.db.close()

    # ── walking the folder ───────────────────────────────────────────────
    def _candidates(self) -> tuple[dict[str, tuple[float, int]], int, bool]:
        """{relative path: (mtime, size)} of indexable files, skipped count, truncated."""
        found: dict[str, tuple[float, int]] = {}
        skipped = 0
        total = 0
        truncated = False
        for base, dirs, names in os.walk(self.root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(names):
                path = Path(base) / name
                if name.startswith(".") and name not in (".bashrc", ".zshrc"):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if (not path.is_file() or stat.st_size == 0 or stat.st_size > MAX_FILE_BYTES
                        or is_sensitive_path(path)):
                    skipped += 1
                    continue
                if len(found) >= MAX_FILES or total + stat.st_size > MAX_TOTAL_BYTES:
                    truncated = True
                    break
                rel = str(path.relative_to(self.root))
                found[rel] = (stat.st_mtime, stat.st_size)
                total += stat.st_size
            if truncated:
                break
        return found, skipped, truncated

    def update(
        self,
        embed: Callable[[list[str]], list[list[float]]] | None = None,
        embed_model: str = "",
        progress: Callable[[str, int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> IndexStats:
        """Bring the index up to date: only new or changed files are read.

        ``embed`` turns texts into vectors (Ollama); without it the index is
        keyword-only. ``progress(phase, done, total)`` reports as it goes.
        """
        db = self.db
        found, skipped, truncated = self._candidates()
        known = dict(db.execute("SELECT path, mtime || ':' || size FROM files").fetchall())
        changed = [p for p, (m, s) in found.items() if known.get(p) != f"{m}:{s}"]
        removed = [p for p in known if p not in found]
        # A different embedding model makes every stored vector meaningless.
        if embed and self._meta("embed_model") != embed_model:
            db.execute("DELETE FROM vectors")
            self._set_meta("embed_model", embed_model)
        for rel in removed + changed:
            ids = [r[0] for r in db.execute("SELECT id FROM chunks WHERE path = ?", (rel,))]
            for cid in ids:
                db.execute("DELETE FROM fts WHERE rowid = ?", (cid,))
                db.execute("DELETE FROM vectors WHERE id = ?", (cid,))
            db.execute("DELETE FROM chunks WHERE path = ?", (rel,))
            db.execute("DELETE FROM files WHERE path = ?", (rel,))
        for number, rel in enumerate(changed, 1):
            if cancelled and cancelled():
                break
            path = self.root / rel
            if not _is_text(path):
                skipped += 1
                db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?)", (rel, *found[rel]))
                continue
            try:
                text = path.read_text("utf-8", errors="replace")
            except OSError:
                continue
            for start, end, body in chunk_text(text):
                cur = db.execute("INSERT INTO chunks(path, start, end, text) VALUES (?,?,?,?)",
                                 (rel, start, end, body))
                db.execute("INSERT INTO fts(rowid, text) VALUES (?, ?)",
                           (cur.lastrowid, f"{rel}\n{body}"))
            db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?)", (rel, *found[rel]))
            if progress and (number % 25 == 0 or number == len(changed)):
                progress("reading", number, len(changed))
        db.commit()

        embedded = False
        if embed:
            missing = db.execute(
                "SELECT c.id, c.path, c.text FROM chunks c LEFT JOIN vectors v ON v.id = c.id "
                "WHERE v.id IS NULL").fetchall()
            done = 0
            try:
                for i in range(0, len(missing), EMBED_BATCH):
                    if cancelled and cancelled():
                        break
                    batch = missing[i:i + EMBED_BATCH]
                    vectors = embed([f"{p}\n{t}" for _id, p, t in batch])
                    for (cid, _p, _t), vec in zip(batch, vectors, strict=False):
                        db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                                   (cid, _pack(vec)))
                    done += len(batch)
                    db.commit()
                    if progress:
                        progress("embedding", done, len(missing))
            except Exception:  # noqa: BLE001 - keyword search still works without vectors
                db.commit()
            embedded = bool(db.execute("SELECT 1 FROM vectors LIMIT 1").fetchone())
        self._set_meta("updated", str(time.time()))
        db.commit()
        stats = self.stats()
        stats.skipped, stats.truncated = skipped, truncated
        stats.embedded = embedded or stats.embedded
        return stats

    def stats(self) -> IndexStats:
        files = self.db.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embedded = bool(self.db.execute("SELECT 1 FROM vectors LIMIT 1").fetchone())
        return IndexStats(files, chunks, embedded)

    # ── searching ────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        limit: int = 6,
        embed: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> list[Hit]:
        """Best passages for a question: keyword ranking, fused with vector
        similarity when the index has embeddings."""
        ranked: dict[int, float] = {}
        match = _fts_query(query)
        if match:
            try:
                rows = self.db.execute(
                    "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT 60",
                    (match,)).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for rank, (cid,) in enumerate(rows):
                ranked[cid] = ranked.get(cid, 0.0) + 1.0 / (60 + rank)
        if embed and self.stats().embedded:
            try:
                qvec = _unpack(_pack(embed([query])[0]))
            except Exception:  # noqa: BLE001 - fall back to keywords alone
                qvec = None
            if qvec is not None:
                for rank, (cid, _score) in enumerate(self._nearest(qvec, 60)):
                    ranked[cid] = ranked.get(cid, 0.0) + 1.0 / (60 + rank)
        best = sorted(ranked.items(), key=lambda item: item[1], reverse=True)[:limit]
        hits: list[Hit] = []
        for cid, score in best:
            row = self.db.execute("SELECT path, start, end, text FROM chunks WHERE id = ?",
                                  (cid,)).fetchone()
            if row:
                hits.append(Hit(row[0], row[1], row[2], row[3], score))
        return hits

    def _nearest(self, qvec: array, limit: int) -> list[tuple[int, float]]:
        rows = self.db.execute("SELECT id, vec FROM vectors").fetchall()
        if not rows:
            return []
        try:
            import numpy as np

            ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
            mat = mat.reshape(len(rows), -1)
            scores = mat @ np.frombuffer(qvec.tobytes(), dtype=np.float32)
            order = np.argsort(-scores)[:limit]
            return [(int(ids[i]), float(scores[i])) for i in order]
        except (ImportError, ValueError):
            scored = []
            for cid, blob in rows:
                vec = _unpack(blob)
                if len(vec) != len(qvec):
                    continue
                scored.append((cid, sum(a * b for a, b in zip(vec, qvec, strict=False))))
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:limit]


def format_hits(hits: list[Hit], budget: int = 6000) -> str:
    """Passages as the model sees them: path:lines, then the text."""
    out: list[str] = []
    used = 0
    for hit in hits:
        block = f"--- {hit.cite()}\n{hit.text}"
        if used + len(block) > budget:
            room = budget - used
            if room > 400:
                out.append(block[:room] + "\n…")
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)



# ── project map (for code folders) ──────────────────────────────────────────
PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", "package.json", "Cargo.toml",
                   "go.mod", "Makefile", "CMakeLists.txt", "meson.build", "PKGBUILD",
                   "pom.xml", "build.gradle", "composer.json", "Gemfile")
_CONVENTION_FILES = (".editorconfig", "pyproject.toml", "setup.cfg", "ruff.toml",
                     ".ruff.toml", ".prettierrc", ".eslintrc.json", "rustfmt.toml",
                     ".clang-format", "CONTRIBUTING.md")
# Every repository option that could make git run a program: the filesystem
# monitor, hooks, external diff/textconv drivers, pagers. A cloned repository
# can set any of them, and `git status` must not become code execution.
_SAFE_GIT = [
    "git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
    "-c", "core.pager=cat", "-c", "diff.external=", "-c", "core.untrackedCache=false",
    "--no-pager",
]


def is_code_project(root: Path) -> bool:
    return (root / ".git").exists() or any((root / m).exists() for m in PROJECT_MARKERS)


def _git(root: Path, *args: str) -> str:
    import subprocess

    if not (root / ".git").exists():
        return ""
    env = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0",
           "GIT_OPTIONAL_LOCKS": "0", "HOME": str(Path.home()), "LANG": "C.UTF-8"}
    try:
        proc = subprocess.run([*_SAFE_GIT, *args], cwd=str(root), capture_output=True,
                              text=True, timeout=10, stdin=subprocess.DEVNULL, env=env,
                              errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _python_outline(path: Path) -> list[str]:
    """Classes and functions of a Python file — parsed, never imported."""
    import ast

    try:
        tree = ast.parse(path.read_text("utf-8", errors="replace"))
    except (SyntaxError, ValueError, OSError, RecursionError):
        return []
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"def {node.name}()")
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and not n.name.startswith("__")][:8]
            out.append(f"class {node.name}" + (f": {', '.join(methods)}" if methods else ""))
    return out


_DEF_RE = re.compile(
    r"^\s*(?:export\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
    r"(?:function|fn|func|class|struct|enum|trait|interface|impl)\s+([A-Za-z_]\w*)", re.M)


def project_map(root: str | Path, files: list[str], budget: int = 3500) -> str:
    """A compact overview of a code project for the model."""
    root = Path(root)
    lines = [f"Project root: {root}"]
    status = _git(root, "status", "--short", "--branch", "--untracked-files=normal")
    if status:
        rows = status.splitlines()
        more = f"\n… {len(rows) - 12} more" if len(rows) > 12 else ""
        lines.append("git status:\n" + "\n".join(rows[:12]) + more)
    conventions = []
    for name in _CONVENTION_FILES:
        path = root / name
        if path.is_file():
            try:
                text = path.read_text("utf-8", errors="replace")
            except OSError:
                continue
            if name == "pyproject.toml":
                # The tool settings say how code here is written; the rest is noise.
                keep = re.findall(r"(?ms)^\[tool\.(?:ruff|black|isort|mypy)[^\]]*\]"
                                  r".*?(?=^\[|\Z)", text)
                deps = re.search(r"(?ms)^dependencies\s*=\s*\[.*?\]", text)
                text = "\n".join(([deps.group(0)] if deps else []) + keep)
            if text.strip():
                conventions.append(f"{name}:\n{text.strip()[:600]}")
    if conventions:
        lines.append("Project conventions (follow them):\n" + "\n\n".join(conventions[:3]))
    outline: list[str] = []
    for rel in sorted(files)[:400]:
        path = root / rel
        if rel.endswith(".py"):
            symbols = _python_outline(path)
        elif path.suffix in (".js", ".ts", ".tsx", ".rs", ".go", ".java", ".kt", ".swift"):
            try:
                symbols = [m.group(1) for m in
                           _DEF_RE.finditer(path.read_text("utf-8", errors="replace"))][:12]
            except OSError:
                symbols = []
        else:
            symbols = []
        outline.append(f"{rel}" + (f" — {'; '.join(symbols)[:220]}" if symbols else ""))
    if outline:
        lines.append("Files and what they define:\n" + "\n".join(outline))
    text = "\n\n".join(lines)
    return text[:budget] + ("\n…" if len(text) > budget else "")
