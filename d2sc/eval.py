"""Eval D2SC-GAN: PSNR/SSIM on CelebA test LR→HR (+ bicubic baseline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from d2sc.data.celeba import resolve_celeba_root
from d2sc.data.celeba_lrhr import CelebALRHR
from d2sc.metrics import psnr, ssim_simple
from d2sc.models.generators import D2SCGenerator


@torch.no_grad()
def evaluate(cfg: dict) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith("cuda") else "cpu")
    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    max_eval = cfg["eval"].get("max_images")
    ds = CelebALRHR(
        root,
        image_size=int(cfg["model"].get("image_size", 64)),
        lr_scale=int(cfg["model"].get("lr_scale", 4)),
        max_n=int(max_n) if max_n is not None else None,
        seed=int(cfg["train"].get("seed", 42)),
        split="test",
    )
    if max_eval is not None:
        ds.samples = ds.samples[: int(max_eval)]
    loader = DataLoader(ds, batch_size=int(cfg["eval"].get("batch_size", 32)), shuffle=False, num_workers=2)

    G = D2SCGenerator().to(device)
    ckpt = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "d2sc_gan.pt", map_location=device, weights_only=False)
    G.load_state_dict(ckpt["G"])
    G.eval()

    p_sr, s_sr, p_bic, s_bic = [], [], [], []
    for lr, hr, _ in tqdm(loader, desc="eval"):
        lr, hr = lr.to(device), hr.to(device)
        sr, _, _ = G(lr)
        p_sr.extend(psnr(sr, hr).cpu().tolist())
        s_sr.extend(ssim_simple(sr, hr).cpu().tolist())
        p_bic.extend(psnr(lr, hr).cpu().tolist())
        s_bic.extend(ssim_simple(lr, hr).cpu().tolist())

    results = {
        "n_images": len(p_sr),
        "psnr_bicubic": float(np.mean(p_bic)),
        "ssim_bicubic": float(np.mean(s_bic)),
        "psnr_d2sc": float(np.mean(p_sr)),
        "ssim_d2sc": float(np.mean(s_sr)),
    }
    print(json.dumps(results, indent=2))
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    evaluate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
