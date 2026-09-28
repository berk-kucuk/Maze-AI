"""The Ollama model library, searched live from ollama.com.

The download picker used to offer a dozen hard-coded tags, which were a year
out of date the day they shipped. The library has no JSON API, but its search
and tag pages are plain server-rendered HTML with a stable shape, so we read
those — the same pages ``ollama.com/search`` shows in a browser.

Everything here degrades quietly: offline, or if the markup changes, search
falls back to :data:`FALLBACK_MODELS` and a tag lookup returns nothing (the
user can still type any tag by hand).
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field

import requests

log = logging.getLogger(__name__)

LIBRARY_URL = "https://ollama.com"
_TIMEOUT = 10
_HEADERS = {
    # HX-Request makes the server return the bare result list, which is also
    # what makes ?page=2 and later pages answer at all.
    "HX-Request": "true",
    "User-Agent": "maze-ai (model browser)",
}

#: Search orderings ollama.com understands.
ORDER_POPULAR = "popular"
ORDER_NEWEST = "newest"

#: Capability filters ollama.com understands (its ``c=`` parameter).
CAPABILITIES = ("tools", "vision", "thinking", "embedding", "cloud")

_SIZE_UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


@dataclass
class LibraryModel:
    """One entry in the library search results."""

    name: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)   # parameter sizes: "8b", "27b"
    pulls: str = ""
    tag_count: int = 0
    updated: str = ""

    @property
    def cloud_only(self) -> bool:
        """Runs only on Ollama's servers — nothing to download locally."""
        return "cloud" in self.capabilities and not self.sizes


@dataclass
class LibraryTag:
    """One downloadable tag of a model, e.g. ``qwen3.6:27b``."""

    tag: str
    size_bytes: int = 0
    size_label: str = ""
    context: str = ""
    inputs: str = ""
    cloud: bool = False


# Hand-picked fallback shown when ollama.com is unreachable. Every tag here has
# a local (non-cloud) build and tool calling unless noted.
FALLBACK_MODELS: list[LibraryModel] = [
    LibraryModel("qwen3.8", "Qwen 3.8 · coding, research and long agentic tasks",
                 ["vision", "tools", "thinking"], ["27b"]),
    LibraryModel("qwen3.6", "Qwen 3.6 · agentic coding and thinking",
                 ["vision", "tools", "thinking"], ["27b", "35b"]),
    LibraryModel("qwen3", "Qwen 3 · dense and MoE, strong small agents",
                 ["tools", "thinking"], ["0.6b", "1.7b", "4b", "8b", "14b", "30b", "32b"]),
    LibraryModel("qwen3-vl", "Qwen 3 VL · vision-language",
                 ["vision", "tools", "thinking"], ["2b", "4b", "8b", "30b", "32b"]),
    LibraryModel("qwen3-coder", "Qwen 3 Coder · agentic coding",
                 ["tools"], ["30b"]),
    LibraryModel("gemma4", "Google Gemma 4 · multimodal, phone to workstation sizes",
                 ["vision", "tools", "thinking"], ["e2b", "e4b", "12b", "26b", "31b"]),
    LibraryModel("gemma3", "Google Gemma 3 · vision, runs on a single GPU",
                 ["vision"], ["1b", "4b", "12b", "27b"]),
    LibraryModel("gpt-oss", "OpenAI open-weight reasoning model",
                 ["tools", "thinking"], ["20b", "120b"]),
    LibraryModel("granite4.1", "IBM Granite 4.1 · efficient tool use",
                 ["tools"], ["3b", "8b", "30b"]),
    LibraryModel("llama3.2", "Meta Llama 3.2 · small and fast",
                 ["tools"], ["1b", "3b"]),
    LibraryModel("llama3.1", "Meta Llama 3.1 · general purpose",
                 ["tools"], ["8b", "70b"]),
    LibraryModel("mistral-small3.2", "Mistral Small 3.2 · vision and tools",
                 ["vision", "tools"], ["24b"]),
    LibraryModel("deepseek-r1", "DeepSeek R1 · reasoning",
                 ["tools", "thinking"], ["1.5b", "7b", "8b", "14b", "32b"]),
    LibraryModel("phi4-mini", "Microsoft Phi-4 mini · small reasoning",
                 ["tools"], ["3.8b"]),
]


def _text(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def parse_size(label: str) -> int:
    """'18GB' → bytes (0 when it isn't a size, e.g. a cloud tag)."""
    match = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?B)\s*", label or "", re.IGNORECASE)
    if not match:
        return 0
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2).upper()])


def parse_search(page: str) -> list[LibraryModel]:
    """Models from an ollama.com search page (full page or htmx fragment)."""
    models: list[LibraryModel] = []
    for block in re.split(r"<li\b", page)[1:]:
        link = re.search(r'href="/library/([^"/:?#]+)"', block)
        if not link:
            continue
        name = html.unescape(link.group(1))
        desc = re.search(r'<p class="max-w-lg[^"]*">([\s\S]*?)</p>', block)
        chips = re.findall(
            r'<span[^>]*class="[^"]*rounded-md (bg-[^\s"]+)[^"]*"[^>]*>([^<]+)</span>',
            block,
        )
        caps, sizes = [], []
        for colour, value in chips:
            value = value.strip().lower()
            if not value:
                continue
            # Parameter sizes are the light-blue chips; everything else is a
            # capability ("tools", "vision", "cloud", …).
            if "ddf4ff" in colour or re.fullmatch(r"[\d.]+[bm](-[a-z0-9]+)?|e\d+b", value):
                sizes.append(value)
            else:
                caps.append(value)
        stats = re.findall(r"<span\s*>([^<]+)</span>\s*<span[^>]*>&nbsp;(Pulls|Tags)", block)
        pulls = next((v.strip() for v, kind in stats if kind == "Pulls"), "")
        tags = next((v.strip() for v, kind in stats if kind == "Tags"), "0")
        updated = re.search(r"Updated&nbsp;</span>\s*<span\s*>([^<]+)</span>", block)
        models.append(LibraryModel(
            name=name,
            description=_text(desc.group(1)) if desc else "",
            capabilities=caps,
            sizes=sizes,
            pulls=pulls,
            tag_count=int(tags) if tags.isdigit() else 0,
            updated=updated.group(1).strip() if updated else "",
        ))
    return models


def parse_tags(page: str, name: str) -> list[LibraryTag]:
    """Tags from an ollama.com ``/library/<name>/tags`` page."""
    tags: list[LibraryTag] = []
    seen: set[str] = set()
    pattern = re.compile(
        r'<a href="/library/([^"]+)" class="md:hidden[\s\S]*?'
        r'<div class="flex flex-col text-neutral-500 text-\[13px\]">([\s\S]*?)</div>'
    )
    for tag, body in pattern.findall(page):
        tag = html.unescape(tag)
        if ":" not in tag or tag in seen or not tag.startswith(name + ":"):
            continue
        seen.add(tag)
        # "<digest> • 18GB • 256K context window • Text, Image input • 1 month ago"
        parts = [p.strip() for p in _text(body).split("•")]
        size_label = parts[1] if len(parts) > 1 else ""
        context = parts[2].replace("context window", "").strip() if len(parts) > 2 else ""
        inputs = parts[3] if len(parts) > 3 else ""
        size = parse_size(size_label)
        tags.append(LibraryTag(
            tag=tag,
            size_bytes=size,
            size_label=size_label if size else "",
            context=context,
            inputs=inputs,
            cloud=tag.endswith("cloud") or (not size and "usage" in size_label.lower()),
        ))
    return tags


def search(
    query: str = "",
    *,
    order: str = ORDER_POPULAR,
    capability: str = "",
    page: int = 1,
) -> list[LibraryModel]:
    """Search the live library. Raises :class:`requests.RequestException`."""
    params: dict[str, str | int] = {}
    if query.strip():
        params["q"] = query.strip()
    if order == ORDER_NEWEST:
        params["o"] = "newest"
    if capability in CAPABILITIES:
        params["c"] = capability
    if page > 1:
        params["page"] = page
    resp = requests.get(
        f"{LIBRARY_URL}/search", params=params, headers=_HEADERS, timeout=_TIMEOUT
    )
    resp.raise_for_status()
    return parse_search(resp.text)


def fallback(query: str = "", capability: str = "") -> list[LibraryModel]:
    """The offline list, filtered the way a live search would be."""
    words = query.lower().split()
    out = []
    for model in FALLBACK_MODELS:
        haystack = f"{model.name} {model.description}".lower()
        if words and not all(w in haystack for w in words):
            continue
        if capability and capability not in model.capabilities:
            continue
        out.append(model)
    return out


def tags(name: str) -> list[LibraryTag]:
    """Every tag of a model. Raises :class:`requests.RequestException`."""
    name = name.split(":", 1)[0].strip()
    resp = requests.get(
        f"{LIBRARY_URL}/library/{requests.utils.quote(name, safe='/')}/tags",
        headers={"User-Agent": _HEADERS["User-Agent"]},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return parse_tags(resp.text, name)
