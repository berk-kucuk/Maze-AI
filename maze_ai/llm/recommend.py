"""Which local model suits this machine.

The best model for Maze AI is the biggest one that can call tools *and* stays
entirely in VRAM: a model that spills onto the CPU answers several times
slower, and one without tool calling can barely act as an agent. This picks
from what is installed first, and otherwise from a short list of tool-capable
models that handle Turkish and English well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .hardware import GB, estimate_fit, system_ram_gb

#: (tag, download size in bytes). Every one calls tools and speaks Turkish.
CANDIDATES: list[tuple[str, int]] = [
    ("qwen3:1.7b", int(1.4 * GB)),
    ("qwen3:4b", int(2.5 * GB)),
    ("qwen3:8b", int(5.2 * GB)),
    ("qwen3:14b", int(9.3 * GB)),
    ("qwen3:30b", int(19 * GB)),
    ("qwen3:32b", int(20 * GB)),
]
#: The window a recommendation has to keep on the GPU along with the weights.
CONTEXT = 8192
#: An installed model is preferred over a download unless the download is
#: clearly bigger: re-downloading for a marginal gain isn't worth gigabytes.
_INSTALLED_MARGIN = 0.6


@dataclass
class Recommendation:
    model: str
    size: int
    installed: bool
    #: "gpu" when it stays in VRAM, "cpu" for machines without a usable GPU.
    placement: str
    #: Short English explanation; the UI translates the template.
    reason: str


def _tool_capable(entry: dict) -> bool:
    caps = entry.get("capabilities") or []
    return "tools" in caps if caps else False


def recommend(installed: list[dict], usable_vram: int,
              ram_gb: float | None = None) -> Recommendation | None:
    """The model to suggest, given installed models and usable VRAM (bytes)."""
    ram = system_ram_gb() if ram_gb is None else ram_gb
    tool_models = [m for m in installed if _tool_capable(m) and int(m.get("size") or 0)]

    if usable_vram > 0:
        def fits(size: int) -> bool:
            return estimate_fit(size, CONTEXT, total_bytes=usable_vram,
                                free_bytes=usable_vram).fits

        best_installed = max((m for m in tool_models if fits(int(m["size"]))),
                             key=lambda m: int(m["size"]), default=None)
        best_download = max(((tag, size) for tag, size in CANDIDATES if fits(size)),
                            key=lambda c: c[1], default=None)
        placement = "gpu"
        reason = "the biggest tool-calling model that stays entirely in your VRAM"
    else:
        # No GPU: everything runs on the CPU, where speed falls fast with size.
        limit = 3 * GB if ram >= 16 else int(1.5 * GB)

        def fits(size: int) -> bool:
            return size <= limit

        best_installed = max((m for m in tool_models if fits(int(m["size"]))),
                             key=lambda m: int(m["size"]), default=None)
        best_download = max(((tag, size) for tag, size in CANDIDATES if fits(size)),
                            key=lambda c: c[1], default=None)
        placement = "cpu"
        reason = "small enough to answer at a usable speed without a GPU"

    if best_installed is not None and (
        best_download is None
        or int(best_installed["size"]) >= _INSTALLED_MARGIN * best_download[1]
    ):
        return Recommendation(best_installed["name"], int(best_installed["size"]),
                              True, placement, reason)
    if best_download is not None:
        installed_names = {m.get("name") for m in installed}
        return Recommendation(best_download[0], best_download[1],
                              best_download[0] in installed_names, placement, reason)
    return None


def recommend_for(backend) -> Recommendation | None:
    """Ask a live Ollama backend (network + nvidia-smi: call off the UI thread)."""
    usable, _ = backend.usable_vram()
    entries = backend.installed_models()
    # The listing under-reports capabilities on some servers; ask the model.
    for entry in entries:
        if not entry.get("capabilities"):
            entry["capabilities"] = backend.model_info(entry["name"]).get("capabilities") or []
    return recommend(entries, usable)


#: Coding models (tag, download size). All call tools in Ollama.
CODER_CANDIDATES: list[tuple[str, int]] = [
    ("qwen2.5-coder:1.5b", int(1.0 * GB)),
    ("qwen2.5-coder:3b", int(1.9 * GB)),
    ("qwen2.5-coder:7b", int(4.7 * GB)),
    ("qwen2.5-coder:14b", int(9.0 * GB)),
    ("qwen2.5-coder:32b", int(20 * GB)),
]
_CODER_NAME = re.compile(
    r"coder|codestral|devstral|codellama|starcoder|codegemma|codeqwen|deepseek-coder", re.I)


def is_coder(name: str) -> bool:
    return bool(_CODER_NAME.search(name or ""))


def recommend_coder(installed: list[dict], usable_vram: int,
                    ram_gb: float | None = None) -> Recommendation | None:
    """The coding model to suggest: an installed one that fits, else a download."""
    coders = [m for m in installed if is_coder(m.get("name", ""))]
    ram = system_ram_gb() if ram_gb is None else ram_gb

    def fits(size: int) -> bool:
        if usable_vram > 0:
            return estimate_fit(size, CONTEXT, total_bytes=usable_vram,
                                free_bytes=usable_vram).fits
        return size <= (3 * GB if ram >= 16 else int(1.5 * GB))

    best = max((m for m in coders if fits(int(m.get("size") or 0))),
               key=lambda m: int(m.get("size") or 0), default=None)
    placement = "gpu" if usable_vram > 0 else "cpu"
    reason = "a coding model that fits this machine"
    if best is not None:
        return Recommendation(best["name"], int(best.get("size") or 0), True, placement, reason)
    pick = max(((t, s) for t, s in CODER_CANDIDATES if fits(s)), key=lambda c: c[1],
               default=None)
    if pick is None:
        return None
    return Recommendation(pick[0], pick[1], False, placement, reason)


def recommend_both(backend) -> tuple[Recommendation | None, Recommendation | None]:
    """General and coding recommendations from one look at the server."""
    usable, _ = backend.usable_vram()
    entries = backend.installed_models()
    for entry in entries:
        if not entry.get("capabilities"):
            entry["capabilities"] = backend.model_info(entry["name"]).get("capabilities") or []
    return recommend(entries, usable), recommend_coder(entries, usable)
