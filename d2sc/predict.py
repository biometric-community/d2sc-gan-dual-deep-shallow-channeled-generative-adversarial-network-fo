"""Save a few LR / SR / HR grids for qualitative inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from d2sc.data.celeba import resolve_celeba_root
from d2sc.data.celeba_lrhr import CelebALRHR
from d2sc.models.generators import D2SCGenerator


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    ds = CelebALRHR(
        root,
        image_size=int(cfg["model"].get("image_size", 64)),
        lr_scale=int(cfg["model"].get("lr_scale", 4)),
        max_n=int(max_n) if max_n is not None else None,
        seed=int(cfg["train"].get("seed", 42)),
        split="test",
    )
    G = D2SCGenerator().to(device)
    G.load_state_dict(
        torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "d2sc_gan.pt", map_location=device, weights_only=False)["G"]
    )
    G.eval()
    n = min(args.limit, len(ds))
    lrs, hrs, srs = [], [], []
    for i in range(n):
        lr, hr, _ = ds[i]
        sr, _, _ = G(lr.unsqueeze(0).to(device))
        lrs.append(lr)
        hrs.append(hr)
        srs.append(sr.cpu().squeeze(0))
    grid = torch.cat([torch.stack(lrs), torch.stack(srs), torch.stack(hrs)], dim=0)
    out = Path(cfg["paths"]["pred_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "grid_lr_sr_hr.png"
    vutils.save_image(grid, path, nrow=n, normalize=True, value_range=(-1, 1))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
