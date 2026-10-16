# Implementation plan — D2SC-GAN

## Paper

- **Title:** D2SC-GAN: Dual Deep-Shallow Channeled Generative Adversarial Network…
- **Slug:** `d2sc-gan-dual-deep-shallow-channeled-generative-adversarial-network-fo`
- **Package:** `d2sc`

## Method

1. Shallow GS + deep GD; IG = average(GS, GD).
2. Auxiliary D: real/fake + identity classification.
3. Losses: adversarial, classification, MRR (MR_PMSE + NCD), channel KLD.
4. Eval: PSNR/SSIM vs bicubic on CelebA test LR×4.

## Datasets

| Dataset | On disk | Use |
|---------|---------|-----|
| CelebA | ~3.1 GiB | Primary train/eval substitute (D1) |
| ICFD / FR_SURV / … | missing | Not used |

**Size gate:** `../../datasets/celeba` → expect &lt; 5 GiB → full train after smoke.

## Figure inventory

| Paper | Ours |
|-------|------|
| Table II PSNR/SSIM | `psnr.svg`, `ssim.svg` |
| Fig. 10 convergence | `train_loss.svg` |
