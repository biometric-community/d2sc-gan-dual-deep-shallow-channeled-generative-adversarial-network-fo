"""PSNR / SSIM helpers for D2SC-GAN eval."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0, 1)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred, target = to01(pred), to01(target)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim_simple(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Lightweight luminance SSIM (C1/C2) — not full windowed SSIM."""
    pred, target = to01(pred), to01(target)
    c1, c2 = 0.01**2, 0.03**2
    mu_x = pred.mean(dim=(1, 2, 3))
    mu_y = target.mean(dim=(1, 2, 3))
    sig_x = pred.var(dim=(1, 2, 3), unbiased=False)
    sig_y = target.var(dim=(1, 2, 3), unbiased=False)
    sig_xy = ((pred - mu_x[:, None, None, None]) * (target - mu_y[:, None, None, None])).mean(dim=(1, 2, 3))
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sig_x + sig_y + c2)
    return num / (den + eps)
