"""MRR (MR_PMSE + NCD) + channel KLD + adversarial / classification (Sec. III)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mr_pmse(pred: torch.Tensor, target: torch.Tensor, scales=(1, 2, 4)) -> torch.Tensor:
    """Multi-resolution patch-wise MSE (approximation of paper MR_PMSE)."""
    loss = pred.new_zeros(())
    for s in scales:
        if s == 1:
            p, t = pred, target
        else:
            p = F.avg_pool2d(pred, s)
            t = F.avg_pool2d(target, s)
        loss = loss + F.mse_loss(p, t)
    return loss / len(scales)


def ncd(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized chi-square distance on flattened intensities."""
    # map [-1,1] → [0,1] for histogram-like distance
    p = ((pred + 1.0) * 0.5).clamp(0, 1).flatten(1)
    t = ((target + 1.0) * 0.5).clamp(0, 1).flatten(1)
    # soft bin via mean/var moments + elementwise chi2
    num = (p - t).pow(2)
    den = p + t + eps
    return (num / den).mean()


def channel_kld(ys: torch.Tensor, yd: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encourage equal participation: maximize KL(softmax(ys)||softmax(yd)) → minimize -KL.

    Paper maximizes KLD so both channels contribute; we return -KL for minimization.
    """
    # spatial mean → channel logits
    ps = F.softmax(ys.mean(dim=(2, 3)), dim=1).clamp_min(eps)
    pd = F.softmax(yd.mean(dim=(2, 3)), dim=1).clamp_min(eps)
    kl = (ps * (ps.log() - pd.log())).sum(dim=1).mean()
    return -kl


def mrr_loss(pred: torch.Tensor, target: torch.Tensor, ys: torch.Tensor, yd: torch.Tensor, lam_ncd: float = 1.0, lam_kld: float = 0.1) -> torch.Tensor:
    return mr_pmse(pred, target) + lam_ncd * ncd(pred, target) + lam_kld * channel_kld(ys, yd)
