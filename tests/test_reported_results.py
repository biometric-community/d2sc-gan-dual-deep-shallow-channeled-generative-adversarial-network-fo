"""REPORT.md, the committed output of the full run."""

from __future__ import annotations

import re

import pytest


def row(report: str, method: str) -> tuple[float, float]:
    match = re.search(rf"\| {re.escape(method)} \| ([\d.]+) \| ([\d.]+) \|", report)
    assert match, f"{method} row missing from REPORT.md"
    return float(match.group(1)), float(match.group(2))


def test_generator_beats_plain_bicubic_psnr(report):
    """Resolving low-resolution faces is the method's purpose."""
    assert row(report, "D2SC-GAN")[0] > row(report, "Bicubic LR")[0]


def test_generator_beats_plain_bicubic_ssim(report):
    assert row(report, "D2SC-GAN")[1] > row(report, "Bicubic LR")[1]


def test_published_values(report, targets):
    psnr, ssim = row(report, "D2SC-GAN")
    assert psnr == pytest.approx(targets["published_values"]["psnr"], abs=1.0)
    assert ssim == pytest.approx(targets["published_values"]["ssim"], abs=0.02)
