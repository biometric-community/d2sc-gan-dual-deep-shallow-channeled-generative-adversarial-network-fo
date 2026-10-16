"""Auxiliary discriminator: real/fake + identity (paper Fig. 4)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AuxDiscriminator(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.adv = nn.Linear(512, 1)
        self.cls = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor):
        # x in [-1,1]
        h = self.features(x)
        return self.adv(h).squeeze(1), self.cls(h)
