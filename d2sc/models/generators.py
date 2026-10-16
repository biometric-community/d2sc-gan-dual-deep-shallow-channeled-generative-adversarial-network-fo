"""Shallow GS and deep GD generators (paper Figs. 2–3; compact reimplementation)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ShallowGenerator(nn.Module):
    """Low-frequency / structure-oriented encoder–decoder (GS)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepGenerator(nn.Module):
    """High-frequency / detail-oriented deeper encoder–decoder (GD)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


class D2SCGenerator(nn.Module):
    """IG = average(GS(IP), GD(IP))."""

    def __init__(self):
        super().__init__()
        self.gs = ShallowGenerator()
        self.gd = DeepGenerator()

    def forward(self, x: torch.Tensor):
        ys = self.gs(x)
        yd = self.gd(x)
        return 0.5 * (ys + yd), ys, yd
