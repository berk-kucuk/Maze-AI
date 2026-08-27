"""What this machine can actually run.

A local model is fast when its weights sit in VRAM and slow the moment they
spill to system RAM — often by an order of magnitude. None of that is visible
from the chat window, so this module reads the hardware directly (NVIDIA, AMD,
plain sysfs) and gives the rest of the app numbers to reason with: how much
VRAM is free, whether a given model will fit, and what to do when it won't.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MB = 1024 * 1024
GB = 1024 * MB


@dataclass
class GPU:
    name: str
    total_mb: int
    used_mb: int
    vendor: str = ""

    @property
    def free_mb(self) -> int:
        return max(0, self.total_mb - self.used_mb)

    def describe(self) -> str:
        return (
            f"{self.name} · {self.free_mb / 1024:.1f}/{self.total_mb / 1024:.1f} GB free"
        )


def system_ram_gb() -> float:
    """Total system RAM in GiB (0.0 when it can't be determined)."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / GB
    except (ValueError, OSError, AttributeError):
        return 0.0


def _nvidia_gpus() -> list[GPU]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    gpus: list[GPU] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(GPU(parts[0], int(float(parts[1])), int(float(parts[2])), "nvidia"))
        except ValueError:
            continue
    return gpus


def _amd_gpus() -> list[GPU]:
    """AMD cards, read from sysfs — no rocm-smi needed."""
    gpus: list[GPU] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        device = card / "device"
        total_file = device / "mem_info_vram_total"
        used_file = device / "mem_info_vram_used"
        if not total_file.exists():
            continue
        try:
            total = int(total_file.read_text().strip()) // MB
            used = int(used_file.read_text().strip()) // MB if used_file.exists() else 0
        except (OSError, ValueError):
            continue
        name = "AMD GPU"
        product = device / "product_name"
        try:
            if product.exists():
                name = product.read_text().strip() or name
        except OSError:
            pass
        if total > 0:
            gpus.append(GPU(name, total, used, "amd"))
    return gpus


_CACHE: tuple[float, list[GPU]] = (0.0, [])
_CACHE_TTL = 5.0  # seconds; nvidia-smi is a process spawn, not a syscall


def detect_gpus(force: bool = False) -> list[GPU]:
    """Discovered GPUs with their memory use, cached for a few seconds."""
    global _CACHE
    now = time.monotonic()
    if not force and _CACHE[1] and now - _CACHE[0] < _CACHE_TTL:
        return _CACHE[1]
    gpus = _nvidia_gpus() or _amd_gpus()
    _CACHE = (now, gpus)
    return gpus


def free_vram_bytes() -> int:
    """Free VRAM across all GPUs, in bytes (0 when there is no GPU)."""
    return sum(gpu.free_mb for gpu in detect_gpus()) * MB


def total_vram_bytes() -> int:
    return sum(gpu.total_mb for gpu in detect_gpus()) * MB


def describe_hardware() -> str:
    """One line for the UI: GPU memory and system RAM."""
    gpus = detect_gpus()
    ram = system_ram_gb()
    if not gpus:
        return f"No GPU detected · {ram:.0f} GB RAM (models will run on the CPU)"
    parts = [gpu.describe() for gpu in gpus]
    parts.append(f"{ram:.0f} GB RAM")
    return "  ·  ".join(parts)


# ── fitting a model into memory ──────────────────────────────────────────
# The KV cache grows linearly with the context window. Its exact size depends
# on the architecture (GQA, sliding-window attention, cache quantisation) and
# is not reliably derivable from what Ollama exposes — so this is a deliberate
# approximation, calibrated against observed loads, used only to warn.
_KV_BYTES_PER_TOKEN = 40 * 1024      # ~40 kB/token for a mid-size model
_OVERHEAD = 400 * MB                 # compute buffers, cuda context, etc.


def kv_cache_bytes(context: int, per_token: int = _KV_BYTES_PER_TOKEN) -> int:
    return max(0, int(context)) * per_token


def estimate_need(weights_bytes: int, context: int, per_token: int | None = None) -> int:
    """Roughly how much memory a model needs at a given context size."""
    per_token = per_token or _KV_BYTES_PER_TOKEN
    return int(weights_bytes) + kv_cache_bytes(context, per_token) + _OVERHEAD


@dataclass
class FitReport:
    """Whether a model is expected to run on the GPU, and why."""

    verdict: str            # "gpu" | "tight" | "spill" | "cpu" | "unknown"
    need_bytes: int
    free_bytes: int
    total_bytes: int

    @property
    def fits(self) -> bool:
        return self.verdict in ("gpu", "tight")


def estimate_fit(
    weights_bytes: int,
    context: int,
    *,
    free_bytes: int | None = None,
    total_bytes: int | None = None,
    per_token: int | None = None,
) -> FitReport:
    """Predict where a model will run, before paying to load it.

    Judged against total VRAM rather than what is free right now: Ollama
    unloads the previous model before loading the next one, so "free" is
    misleading the moment a model is already resident.
    """
    total = total_vram_bytes() if total_bytes is None else total_bytes
    free = free_vram_bytes() if free_bytes is None else free_bytes
    need = estimate_need(weights_bytes, context, per_token)
    if not total:
        return FitReport("cpu", need, free, total)
    if not weights_bytes:
        return FitReport("unknown", need, free, total)
    if need <= total * 0.90:
        return FitReport("gpu", need, free, total)
    if need <= total:
        return FitReport("tight", need, free, total)
    if weights_bytes <= total * 0.95:
        # The weights alone would fit; it is the context that pushes it over.
        return FitReport("spill", need, free, total)
    return FitReport("cpu", need, free, total)


def context_that_fits(
    weights_bytes: int,
    ladder: tuple[int, ...] = (32768, 16384, 8192, 4096, 2048),
    *,
    total_bytes: int | None = None,
    per_token: int | None = None,
) -> int:
    """The largest context from ``ladder`` expected to stay on the GPU."""
    for context in ladder:
        if estimate_fit(weights_bytes, context, total_bytes=total_bytes,
                        per_token=per_token).fits:
            return context
    return ladder[-1]


def parse_vram_hint(text: str) -> int:
    """Bytes from a label like '8 GB' or '8151MiB' (0 when unparseable)."""
    match = re.match(r"\s*([\d.]+)\s*([GMK]?)i?B?", (text or "").upper())
    if not match:
        return 0
    value = float(match.group(1))
    unit = match.group(2)
    return int(value * {"G": GB, "M": MB, "K": 1024, "": 1}[unit])
