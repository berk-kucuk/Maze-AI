"""Tests for GPU/VRAM detection and model fit prediction."""

from __future__ import annotations

import pytest

from maze_ai.llm import hardware
from maze_ai.llm.hardware import GB, GPU, MB, context_that_fits, estimate_fit, estimate_need


@pytest.fixture(autouse=True)
def no_cached_gpus():
    hardware._CACHE = (0.0, [])
    yield
    hardware._CACHE = (0.0, [])


# ── detection ──────────────────────────────────────────────────────────────
def test_nvidia_output_is_parsed(monkeypatch):
    class Proc:
        stdout = "NVIDIA GeForce RTX 5060, 8151, 6876\nNVIDIA A100, 40960, 1024\n"

    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: Proc())
    gpus = hardware.detect_gpus(force=True)
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 5060", "NVIDIA A100"]
    assert gpus[0].free_mb == 8151 - 6876
    assert hardware.total_vram_bytes() == (8151 + 40960) * MB


def test_no_gpu_is_not_an_error(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    monkeypatch.setattr(hardware, "_amd_gpus", lambda: [])
    assert hardware.detect_gpus(force=True) == []
    assert hardware.free_vram_bytes() == 0
    assert "No GPU" in hardware.describe_hardware()


def test_broken_nvidia_smi_is_tolerated(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(hardware.subprocess, "run", boom)
    monkeypatch.setattr(hardware, "_amd_gpus", lambda: [])
    assert hardware.detect_gpus(force=True) == []


def test_detection_is_cached(monkeypatch):
    calls = []

    def counting(*a, **k):
        calls.append(1)

        class Proc:
            stdout = "GPU, 8000, 1000\n"

        return Proc()

    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware.subprocess, "run", counting)
    hardware.detect_gpus(force=True)
    hardware.detect_gpus()
    assert len(calls) == 1


# ── fit prediction ─────────────────────────────────────────────────────────
def test_a_small_model_fits():
    report = estimate_fit(2 * GB, 8192, total_bytes=8 * GB, free_bytes=8 * GB)
    assert report.verdict == "gpu" and report.fits


def test_context_pushes_a_borderline_model_out():
    weights = 7 * GB
    small = estimate_fit(weights, 2048, total_bytes=8 * GB, free_bytes=8 * GB)
    large = estimate_fit(weights, 32768, total_bytes=8 * GB, free_bytes=8 * GB)
    assert small.fits
    assert large.verdict == "spill" and not large.fits


def test_weights_alone_too_big_is_cpu():
    report = estimate_fit(12 * GB, 4096, total_bytes=8 * GB, free_bytes=8 * GB)
    assert report.verdict == "cpu"


def test_machine_without_a_gpu_is_cpu():
    assert estimate_fit(2 * GB, 4096, total_bytes=0, free_bytes=0).verdict == "cpu"


def test_need_grows_with_context():
    assert estimate_need(4 * GB, 16384) > estimate_need(4 * GB, 4096)


def test_context_that_fits_picks_the_largest_workable_window():
    roomy = context_that_fits(5 * GB, total_bytes=8 * GB)
    tight = context_that_fits(7 * GB, total_bytes=8 * GB)
    assert estimate_fit(5 * GB, roomy, total_bytes=8 * GB).fits
    assert estimate_fit(7 * GB, tight, total_bytes=8 * GB).fits
    # The bigger model has to give up context to stay on the card.
    assert tight < roomy


def test_context_that_fits_bottoms_out_for_an_oversized_model():
    assert context_that_fits(20 * GB, total_bytes=8 * GB) == 2048


def test_parse_vram_hint():
    assert hardware.parse_vram_hint("8 GB") == 8 * GB
    assert hardware.parse_vram_hint("8151MiB") == 8151 * MB
    assert hardware.parse_vram_hint("nonsense") == 0


def test_gpu_describe_reads_naturally():
    text = GPU("Card", 8151, 6876).describe()
    assert "Card" in text and "GB free" in text
