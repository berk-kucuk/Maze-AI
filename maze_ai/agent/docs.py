"""Documentation tools: the Arch Wiki and man pages.

Small local models answer Arch questions from half-remembered training data
and invent flags with confidence. These two tools let the agent check before it
answers: ``man_page`` reads the documentation of the program actually installed
on this machine, and ``arch_wiki`` reads the Arch Wiki — offline from the
``arch-wiki-docs`` package when it is installed, otherwise from
wiki.archlinux.org.

Both return the relevant part, not the whole document: a man page or a wiki
article is often 30-100k characters, which would swamp an 8k context window.
"""

from __future__ import annotations

import html as html_lib
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import requests

WIKI_ROOT = Path(os.environ.get("MAZE_AI_WIKI_ROOT", "/usr/share/doc/arch-wiki/html"))
WIKI_URL = "https://wiki.archlinux.org"
_API = f"{WIKI_URL}/api.php"
_HEADERS = {"User-Agent": "MazeAI/1.0 (Maze Linux assistant; arch_wiki tool)"}
#: How much of a document goes back to the model.
DOC_BUDGET = 7000

_STOP = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "how", "do", "i",
    "is", "with", "my", "what", "nasıl", "nasil", "ne", "bir", "ve", "ile", "için",
    "icin", "mi", "mı", "nedir",
}


def _terms(query: str) -> list[str]:
    words = re.findall(r"[\w.+-]+", (query or "").lower())
    return [w for w in words if w not in _STOP and len(w) > 1]


# ── HTML → text with headings kept ─────────────────────────────────────────
class _Sections(HTMLParser):
    """Turns an article into Markdown-ish text: headings, paragraphs, code."""

    _SKIP = {"script", "style", "nav", "footer", "noscript", "sup"}
    _BLOCK = {"p", "div", "li", "tr", "dd", "dt", "br", "table", "ul", "ol", "dl"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.pre = 0
        self.heading = ""

    _VOID = {"br", "img", "hr", "input", "meta", "link", "wbr", "col", "source", "area"}

    def handle_starttag(self, tag, attrs):
        if self.skip:
            # Inside something being skipped: just track nesting depth.
            if tag not in self._VOID:
                self.skip += 1
            return
        cls = dict(attrs).get("class", "") or ""
        if tag in self._SKIP or any(c in cls for c in ("mw-editsection", "toc", "navbox",
                                                       "catlinks", "printfooter",
                                                       "archwiki-template-meta")):
            if tag not in self._VOID:
                self.skip = 1
            return
        if tag in ("h1", "h2", "h3", "h4"):
            self.heading = "#" * min(int(tag[1]) + 1, 4)
            self.out.append(f"\n\n{self.heading} ")
        elif tag == "pre":
            self.pre += 1
            self.out.append("\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_startendtag(self, tag, attrs):
        if not self.skip and tag == "br":
            self.out.append("\n")

    def handle_endtag(self, tag):
        if self.skip:
            if tag not in self._VOID:
                self.skip -= 1
            return
        if tag in ("h1", "h2", "h3", "h4"):
            self.out.append("\n")
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self.out.append("\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip:
            return
        self.out.append(data if self.pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        raw = "".join(self.out)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_sections(page: str) -> str:
    # The offline pages carry the whole MediaWiki skin; the article body sits
    # in #bodyContent / .mw-parser-output. Narrowing first keeps menus out.
    match = re.search(r'<div[^>]+class="[^"]*mw-parser-output[^"]*"[^>]*>', page)
    if match:
        page = page[match.start():]
    parser = _Sections()
    try:
        parser.feed(page)
    except Exception:  # noqa: BLE001 - malformed HTML must not break the tool
        pass
    return parser.text()


def relevant_excerpt(text: str, query: str, budget: int = DOC_BUDGET) -> str:
    """The introduction plus the sections that best match the query."""
    parts = re.split(r"\n(?=#{2,4} )", text)
    intro, sections = parts[0], parts[1:]
    terms = _terms(query)
    if not terms or len(text) <= budget:
        return text[:budget] + ("\n…" if len(text) > budget else "")

    def score(section: str) -> float:
        head, _, body = section.partition("\n")
        low_head, low_body = head.lower(), body.lower()
        return sum(3 * low_head.count(t) + min(low_body.count(t), 6) for t in terms)

    ranked = sorted(sections, key=score, reverse=True)
    chosen: list[str] = []
    used = min(len(intro), budget // 4)
    for section in ranked:
        if score(section) <= 0:
            break
        if used + len(section) > budget:
            room = budget - used
            if room > 600:
                chosen.append(section[:room] + "\n…")
            break
        chosen.append(section)
        used += len(section)
    # Keep the article's own order: it reads better than "best first".
    chosen.sort(key=lambda s: text.find(s[:80]))
    body = "\n\n".join(chosen) if chosen else text[used:budget]
    return f"{intro[: budget // 4]}\n\n{body}".strip()


# ── Arch Wiki: offline ─────────────────────────────────────────────────────
def _offline_dir() -> Path | None:
    for lang in ("en", ""):
        root = WIKI_ROOT / lang if lang else WIKI_ROOT
        if root.is_dir() and any(root.glob("*.html")):
            return root
    return None


def _offline_search(query: str, limit: int = 5) -> list[tuple[str, Path]]:
    """Pages whose title best matches the query (titles are the file names)."""
    root = _offline_dir()
    if root is None:
        return []
    terms = _terms(query)
    if not terms:
        return []
    scored: list[tuple[float, str, Path]] = []
    for path in root.glob("*.html"):
        title = path.stem.replace("_", " ")
        low = title.lower()
        words = set(re.findall(r"[\w.+-]+", low))
        hits = sum(1 for t in terms if t in words) + 0.5 * sum(1 for t in terms if t in low)
        if not hits:
            continue
        # Prefer the main article over "Pacman/Tips and tricks", and exact titles.
        exact = 3 if low == " ".join(terms) else 0
        scored.append((hits + exact - 0.1 * low.count("/") - 0.01 * len(low), title, path))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(title, path) for _s, title, path in scored[:limit]]


# ── Arch Wiki: online ──────────────────────────────────────────────────────
def _api_search(text: str, limit: int) -> list[tuple[str, str]]:
    resp = requests.get(_API, params={
        "action": "query", "list": "search", "srsearch": text, "srlimit": limit,
        "format": "json", "srprop": "snippet", "srnamespace": 0,
    }, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    hits = resp.json().get("query", {}).get("search", [])
    return [(h["title"], re.sub(r"<[^>]+>", "", html_lib.unescape(h.get("snippet", ""))))
            for h in hits]


#: Translated copies carry the language in brackets: "Pacman (Español)".
_TRANSLATED = re.compile(r"\([^)]*\)\s*(/|$)")


def _online_search(query: str, limit: int = 5) -> list[tuple[str, str]]:
    """Wiki pages for a query, best first.

    The wiki's own search wants every word on one page and ranks poorly
    ("pacman remove" finds nothing), so the web search is asked first,
    restricted to the wiki; the wiki API is the fallback. User pages and
    translated copies ("Pacman (Español)") are skipped.
    """
    from .tools import ddg_results

    found: list[tuple[str, str]] = []
    results, _err = ddg_results(f"site:wiki.archlinux.org {query}")
    for item in results:
        match = re.match(r"https?://wiki\.archlinux\.org/title/([^?#]+)", item.get("url", ""))
        if not match:
            continue
        title = requests.utils.unquote(match.group(1)).replace("_", " ")
        if title.startswith(("User:", "Talk:", "Special:", "Category:")) \
                or _TRANSLATED.search(title):
            continue
        if title not in (t for t, _s in found):
            found.append((title, item.get("snippet", "")))
    if found:
        return found[:limit]
    hits = []
    for text in (query, " ".join(_terms(query)[:2])):
        try:
            hits = [h for h in _api_search(text, 10) if not _TRANSLATED.search(h[0])]
        except requests.RequestException:
            continue
        if hits:
            break
    return hits[:limit]


def _online_page(title: str) -> str:
    resp = requests.get(_API, params={
        "action": "parse", "page": title, "prop": "text", "format": "json",
        "redirects": 1, "disableeditsection": 1,
    }, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise LookupError(data["error"].get("info", "page not found"))
    return data["parse"]["text"]["*"]


def _page_url(title: str) -> str:
    return f"{WIKI_URL}/title/{title.replace(' ', '_')}"


def arch_wiki(query: str = "", page: str = "", **_) -> tuple[bool, str]:
    """Search the Arch Wiki and return the relevant part of the best article."""
    query = (query or "").strip()
    page = (page or "").strip()
    if not query and not page:
        return False, "Give a search query or a page title."

    offline = _offline_dir() is not None
    try:
        if offline:
            if page:
                candidate = _offline_dir() / f"{page.replace(' ', '_')}.html"
                matches = [(page, candidate)] if candidate.is_file() else _offline_search(page)
            else:
                matches = _offline_search(query)
            if not matches:
                return False, (f"No offline Arch Wiki page matches '{page or query}'. "
                               "Try other words, or the article's exact title.")
            title, path = matches[0]
            text = html_to_sections(path.read_text("utf-8", errors="replace"))
            others = [t for t, _p in matches[1:]]
            source = "Arch Wiki (offline copy)"
        else:
            if page:
                title, others = page, []
            else:
                hits = _online_search(query)
                if not hits:
                    return False, f"The Arch Wiki has no results for '{query}'."
                title, others = hits[0][0], [t for t, _s in hits[1:]]
            text = html_to_sections(_online_page(title))
            source = "Arch Wiki (wiki.archlinux.org)"
    except (requests.RequestException, LookupError, ValueError, OSError) as exc:
        hint = "" if offline else (
            " Install `arch-wiki-docs` for an offline copy that always works."
        )
        return False, f"Could not read the Arch Wiki: {exc}.{hint}"

    excerpt = relevant_excerpt(text, query or page)
    lines = [f"[{source}] {title} — {_page_url(title)}", "", excerpt]
    if others:
        lines += ["", "Other matching pages: " + ", ".join(others[:4])]
    lines += ["", "Cite this page (title + URL) when you use it in your answer."]
    return True, "\n".join(lines)


# ── man pages ──────────────────────────────────────────────────────────────
_NAME_RE = re.compile(r"^[A-Za-z0-9][\w.+-]{0,63}$")


def _man_text(name: str, section: str) -> str:
    argv = ["man", "-P", "cat"]
    if section:
        argv.append(section)
    argv.append(name)
    env = dict(os.environ, MANWIDTH="100", MAN_KEEP_FORMATTING="0", MANPAGER="cat",
               PAGER="cat", GROFF_NO_SGR="1", LC_ALL=os.environ.get("LC_ALL", "C.UTF-8"))
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=15, env=env,
                          stdin=subprocess.DEVNULL, errors="replace")
    if proc.returncode != 0:
        return ""
    # Strip any backspace overstriking groff left in.
    return re.sub(r".\x08", "", proc.stdout)


def _help_text(name: str) -> str:
    """`<name> --help`, only for installed programs the rules call harmless."""
    from .rules import LAUNCH_FORBIDDEN, classify

    path = shutil.which(name)
    if not path or not path.startswith(("/usr/bin/", "/bin/", "/usr/sbin/")):
        return ""
    if name in LAUNCH_FORBIDDEN or classify(f"{name} --help").tier in ("blocked", "confirm"):
        return ""
    try:
        proc = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=5,
                              stdin=subprocess.DEVNULL, errors="replace",
                              start_new_session=True)
    except (subprocess.SubprocessError, OSError):
        return ""
    return (proc.stdout or proc.stderr or "").strip()


def _focus(text: str, query: str, budget: int) -> str:
    """Paragraphs that mention the query (an option like -Rns, or a word)."""
    paragraphs = re.split(r"\n\s*\n", text)
    needles = [q for q in re.split(r"[\s,]+", query.strip()) if q]
    # "-Qdt" is three options: the operation -Q and the modifiers -d and -t.
    # Look each one up where it is defined (at the start of a line).
    patterns: list[re.Pattern] = []
    for needle in needles:
        if re.fullmatch(r"-[A-Za-z]{2,}", needle):
            patterns += [re.compile(rf"(?m)^\s*-{c}\b") for c in needle[1:]]
        else:
            patterns.append(re.compile(re.escape(needle)))
    hits: list[str] = []
    for i, para in enumerate(paragraphs):
        if any(p.search(para) for p in patterns):
            hits.append(para)
            # Some pages put an option's name alone on a line, with its
            # description as the next paragraph.
            if i + 1 < len(paragraphs) and "\n" not in para.strip():
                hits.append(paragraphs[i + 1])
    if not hits:
        return ""
    seen, out, used = set(), [], 0
    for para in hits:
        if para in seen:
            continue
        seen.add(para)
        if used + len(para) > budget:
            break
        out.append(para)
        used += len(para)
    return "\n\n".join(out)


def man_page(command: str = "", query: str = "", section: str = "", **_) -> tuple[bool, str]:
    """The installed documentation for a command, focused on ``query``."""
    name = (command or "").strip().split()[0] if (command or "").strip() else ""
    section = (section or "").strip()
    if not name or not _NAME_RE.match(name):
        return False, "Give a single command name, e.g. 'pacman'."
    if section and not re.fullmatch(r"[0-9][a-z]*", section):
        return False, "The section must look like 1, 5 or 8."
    if not shutil.which("man"):
        text, source = _help_text(name), f"{name} --help"
    else:
        text, source = _man_text(name, section), f"man {section + ' ' if section else ''}{name}"
        if not text:
            text, source = _help_text(name), f"{name} --help"
    if not text:
        return False, (f"No manual page or --help output for '{name}' on this machine. "
                       "It may not be installed — check with `pacman -Qo` or `which`.")
    body = ""
    if query:
        body = _focus(text, query, DOC_BUDGET - 400)
        if not body:
            body = f"('{query}' does not appear in it — showing the start instead.)\n\n"
    if not body or body.startswith("("):
        body += text[: DOC_BUDGET - len(body)]
        if len(text) > DOC_BUDGET:
            body += "\n… (pass `query` with an option or a word to see a specific part)"
    return True, f"[{source} — installed on this machine]\n\n{body}"


# ── Python library documentation ───────────────────────────────────────────
_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
#: Modules whose import does something visible (opens a browser, prints).
_NO_IMPORT = {"antigravity", "this", "__hello__", "__phello__", "turtledemo", "idlelib"}

# Runs in an isolated interpreter (-I: no current directory, no user site
# packages, PYTHONPATH ignored) from "/", and refuses any module that doesn't
# live in the system library directories — so the import can only run code
# pacman installed, never the project's or the model's.
_PYDOC_SCRIPT = r"""
import importlib.util, pydoc, sys, sysconfig
name = sys.argv[1]
spec = importlib.util.find_spec(name.split(".")[0])
if spec is None:
    print("NOT_INSTALLED"); sys.exit(2)
origin = spec.origin or next(iter(spec.submodule_search_locations or []), "")
roots = {sysconfig.get_paths()[k] for k in ("stdlib", "platstdlib", "purelib", "platlib")}
if spec.origin not in ("built-in", "frozen") and not any(
        str(origin).startswith(root + "/") for root in roots):
    print("NOT_SYSTEM", origin); sys.exit(3)
try:
    print(pydoc.render_doc(name, renderer=pydoc.plaintext))
except Exception as exc:
    print("NO_DOC", exc); sys.exit(4)
"""


def python_doc(name: str = "", query: str = "", **_) -> tuple[bool, str]:
    """The documentation of an installed Python module, class or function."""
    name = (name or "").strip()
    if not _MODULE_RE.match(name):
        return False, "Give a dotted Python name, e.g. 'pathlib.Path.glob' or 'requests'."
    if name.split(".")[0] in _NO_IMPORT:
        return False, f"'{name}' is not looked up (importing it has side effects)."
    python = shutil.which("python3") or shutil.which("python")
    if not python:
        return False, "No Python interpreter found."
    try:
        proc = subprocess.run(
            [python, "-I", "-c", _PYDOC_SCRIPT, name], capture_output=True, text=True,
            timeout=20, cwd="/", stdin=subprocess.DEVNULL, errors="replace",
            start_new_session=True, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"Could not read the documentation: {exc}"
    out = (proc.stdout or "").strip()
    if proc.returncode == 2 or out.startswith("NOT_INSTALLED"):
        return False, (f"'{name.split('.')[0]}' is not installed for the system Python. "
                       "Don't use it unless the user installs it (check `pacman -Ss python-…`).")
    if proc.returncode == 3 or out.startswith("NOT_SYSTEM"):
        return False, (f"'{name}' is not a system-installed library (it lives in the "
                       "project or the user's own packages); read its source with "
                       "read_file instead.")
    if proc.returncode != 0 or not out:
        detail = out.removeprefix("NO_DOC").strip() or (proc.stderr or "").strip()[-300:]
        return False, f"No documentation for '{name}': {detail}"
    if query:
        focused = _focus(out, query, DOC_BUDGET - 300)
        if focused:
            out = focused
    if len(out) > DOC_BUDGET:
        out = out[:DOC_BUDGET] + "\n… (pass `query` with a method or word to see a specific part)"
    return True, f"[pydoc {name} — installed on this machine]\n\n{out}"
