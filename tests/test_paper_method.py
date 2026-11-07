"""Generators, losses and metrics, checked on small controlled inputs."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from d2sc.losses import channel_kld, mr_pmse, ncd
from d2sc.metrics import psnr, ssim_simple
from d2sc.models.generators import D2SCGenerator


def test_generated_image_is_the_average_of_both_channels():
    torch.manual_seed(0)
    model = D2SCGenerator().eval()
    with torch.no_grad():
        out, shallow, deep = model(torch.rand(1, 3, 64, 64) * 2 - 1)
    assert torch.allclose(out, 0.5 * (shallow + deep), atol=1e-6)
    assert out.shape == (1, 3, 64, 64)


def test_multi_resolution_error_is_zero_only_for_a_match():
    image = torch.rand(2, 3, 32, 32)
    assert mr_pmse(image, image).item() == 0.0
    assert mr_pmse(image, image + 0.1).item() > 0.0


def test_ncd_is_zero_only_for_a_match():
    image = torch.rand(2, 3, 16, 16) * 2 - 1
    assert ncd(image, image).item() == 0.0
    assert ncd(image, -image).item() > 0.0


def test_kld_term_rewards_channels_that_differ():
    """Minimising the returned value maximises the divergence between the channels."""
    a = torch.randn(2, 3, 8, 8)
    assert channel_kld(a, a).item() == pytest.approx(0.0, abs=1e-5)
    assert channel_kld(a, -a).item() < 0.0


def test_psnr_matches_its_definition():
    target = torch.zeros(1, 3, 8, 8)  # 0.5 after mapping [-1, 1] to [0, 1]
    pred = target + 0.2  # 0.6 after mapping, so the MSE is 0.01
    assert psnr(pred, target).item() == pytest.approx(20.0, abs=0.01)


def windowed_ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 8) -> float:
    """Wang et al. SSIM: local statistics averaged over windows, on [0, 1] images."""
    x, y = (pred + 1) * 0.5, (target + 1) * 0.5
    c1, c2 = 0.01**2, 0.03**2
    pool = lambda t: F.avg_pool2d(t, window, stride=1)
    mu_x, mu_y = pool(x), pool(y)
    var_x, var_y = pool(x * x) - mu_x**2, pool(y * y) - mu_y**2
    cov = pool(x * y) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return ssim.mean().item()


def test_ssim_uses_local_windows():
    """A global statistic hides local detail loss, which is what SSIM exists to measure."""
    torch.manual_seed(1)
    grid = torch.linspace(-0.8, 0.8, 64)
    target = (grid[None, :] + 0.2 * torch.randn(64, 64)).clamp(-1, 1).expand(1, 3, 64, 64)
    blurred = F.avg_pool2d(target, 5, stride=1, padding=2)
    assert ssim_simple(blurred, target).item() == pytest.approx(windowed_ssim(blurred, target), abs=0.05)


def test_identical_images_have_unit_ssim():
    image = torch.rand(1, 3, 16, 16) * 2 - 1
    assert ssim_simple(image, image).item() == pytest.approx(1.0, abs=1e-4)
