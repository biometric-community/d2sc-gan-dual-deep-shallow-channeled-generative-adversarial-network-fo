"""REPORT.md + SVG figures from real train/eval JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from d2sc.plot_style import BAR_COLOR, FIGSIZE, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    eval_path = log_dir / "eval_results.json"
    train_path = log_dir / "train_summary.json"
    if not eval_path.is_file():
        Path("REPORT.md").write_text("# REPORT\n\n**BLOCKED:** run eval first.\n")
        return
    metrics = json.loads(eval_path.read_text())
    train = json.loads(train_path.read_text()) if train_path.is_file() else {}

    names = ["Bicubic LR", "D2SC-GAN"]
    psnr_vals = [metrics["psnr_bicubic"], metrics["psnr_d2sc"]]
    ssim_vals = [metrics["ssim_bicubic"], metrics["ssim_d2sc"]]
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(names, psnr_vals, color=[MULTI_SERIES_COLORS[0], BAR_COLOR], edgecolor="#002080")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("CelebA LR×4 reconstruction PSNR (ours)")
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "psnr.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(names, ssim_vals, color=[MULTI_SERIES_COLORS[1], BAR_COLOR], edgecolor="#002080")
    ax.set_ylabel("SSIM")
    ax.set_title("CelebA LR×4 reconstruction SSIM (ours)")
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "ssim.svg", format="svg")
    plt.close(fig)

    hist = train.get("history") or []
    if hist:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot([h["epoch"] for h in hist], [h["loss_g"] for h in hist], color=MULTI_SERIES_COLORS[0], linewidth=3, label="G")
        ax.plot([h["epoch"] for h in hist], [h["loss_d"] for h in hist], color=MULTI_SERIES_COLORS[1], linewidth=3, label="D")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("D2SC-GAN training (ours)")
        ax.legend()
        apply_style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / "train_loss.svg", format="svg")
        plt.close(fig)

    report = f"""# REPORT — D2SC-GAN

Lab twin of Bhattacharjee & Das, *D2SC-GAN* (IEEE TBIOM 2020). CelebA ×4 bicubic LR substitutes ICFD/FR_SURV (see DEVIATIONS).

## Train

- CelebA train: n={train.get('n_train')}, classes={train.get('num_classes')}
- Wall_s={train.get('seconds')}
- Protocol: {train.get('protocol')}

## Reconstruction (ours)

| Method | PSNR (dB) | SSIM |
|--------|----------:|-----:|
| Bicubic LR | {metrics['psnr_bicubic']:.3f} | {metrics['ssim_bicubic']:.4f} |
| D2SC-GAN | {metrics['psnr_d2sc']:.3f} | {metrics['ssim_d2sc']:.4f} |

N images={metrics['n_images']}.

## Figures

![PSNR]({(fig_dir / 'psnr.svg').as_posix()})

![SSIM]({(fig_dir / 'ssim.svg').as_posix()})

## Regenerate

```bash
bash scripts/report.sh
```
"""
    Path("REPORT.md").write_text(report)
    print(f"Wrote REPORT.md + {fig_dir}")


if __name__ == "__main__":
    main()
