"""Train D2SC-GAN (GS+GD average, aux D, MRR+KLD)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from d2sc.data.celeba import resolve_celeba_root
from d2sc.data.celeba_lrhr import CelebALRHR
from d2sc.losses import mrr_loss
from d2sc.models.discriminator import AuxDiscriminator
from d2sc.models.generators import D2SCGenerator


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith("cuda") else "cpu")

    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    ds = CelebALRHR(
        root,
        image_size=int(cfg["model"].get("image_size", 64)),
        lr_scale=int(cfg["model"].get("lr_scale", 4)),
        max_n=int(max_n) if max_n is not None else None,
        seed=int(cfg["train"].get("seed", 42)),
        split="train",
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(f"[data] n={len(ds)} classes={ds.num_classes} device={device}")

    G = D2SCGenerator().to(device)
    D = AuxDiscriminator(ds.num_classes).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=float(cfg["train"].get("lr_g", 2e-4)), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=float(cfg["train"].get("lr_d", 2e-4)), betas=(0.5, 0.999))
    bce = torch.nn.BCEWithLogitsLoss()

    epochs = int(cfg["train"].get("epochs", 5))
    lam_adv = float(cfg["train"].get("lambda_adv", 1.0))
    lam_cls = float(cfg["train"].get("lambda_cls", 1.0))
    lam_mrr = float(cfg["train"].get("lambda_mrr", 10.0))
    hist = []
    t0 = time.time()
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        G.train()
        D.train()
        g_losses, d_losses = [], []
        for lr, hr, y in tqdm(loader, desc=f"ep{ep}/{epochs}", leave=False):
            lr, hr, y = lr.to(device), hr.to(device), y.to(device)
            bsz = lr.size(0)

            # --- D ---
            with torch.no_grad():
                fake, _, _ = G(lr)
            opt_d.zero_grad(set_to_none=True)
            real_logit, real_cls = D(hr)
            fake_logit, _ = D(fake.detach())
            loss_d = (
                bce(real_logit, torch.ones(bsz, device=device))
                + bce(fake_logit, torch.zeros(bsz, device=device))
                + lam_cls * F.cross_entropy(real_cls, y)
            )
            loss_d.backward()
            opt_d.step()

            # --- G ---
            opt_g.zero_grad(set_to_none=True)
            fake, ys, yd = G(lr)
            fake_logit, fake_cls = D(fake)
            loss_adv = bce(fake_logit, torch.ones(bsz, device=device))
            loss_cls = F.cross_entropy(fake_cls, y)
            loss_mrr = mrr_loss(fake, hr, ys, yd)
            loss_g = lam_adv * loss_adv + lam_cls * loss_cls + lam_mrr * loss_mrr
            loss_g.backward()
            opt_g.step()

            g_losses.append(float(loss_g.detach()))
            d_losses.append(float(loss_d.detach()))

        row = {"epoch": ep, "loss_g": float(np.mean(g_losses)), "loss_d": float(np.mean(d_losses))}
        hist.append(row)
        print(f"[train] epoch={ep} G={row['loss_g']:.4f} D={row['loss_d']:.4f}", flush=True)

    torch.save(
        {"G": G.state_dict(), "D": D.state_dict(), "num_classes": ds.num_classes, "cfg": cfg},
        ckpt_dir / "d2sc_gan.pt",
    )
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "n_train": len(ds),
        "num_classes": ds.num_classes,
        "history": hist,
        "seconds": time.time() - t0,
        "checkpoint": str(ckpt_dir / "d2sc_gan.pt"),
        "device": str(device),
    }
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n_train": summary["n_train"], "seconds": summary["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
