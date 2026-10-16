# REPORT — D2SC-GAN

Lab twin of Bhattacharjee & Das, *D2SC-GAN* (IEEE TBIOM 2020). CelebA ×4 bicubic LR substitutes ICFD/FR_SURV (see DEVIATIONS).

## Train

- CelebA train: n=20000, classes=5765
- Wall_s=248.34582567214966
- Protocol: full

## Reconstruction (ours)

| Method | PSNR (dB) | SSIM |
|--------|----------:|-----:|
| Bicubic LR | 22.716 | 0.9530 |
| D2SC-GAN | 20.916 | 0.9305 |

N images=2000.

## Figures

![PSNR](outputs/figures/full/psnr.svg)

![SSIM](outputs/figures/full/ssim.svg)

## Regenerate

```bash
bash scripts/report.sh
```
