# Deviations

| ID | Topic | Paper | Ours |
|----|-------|-------|------|
| D1 | Datasets | ICFD, FR_SURV, TIP-D, Chokepoint, SCface | CelebA with synthetic LR (×4 bicubic) |
| D2 | Train scale | Full classroom protocols | CelebA capped at 20k images / 5 epochs for wall-clock |
| D3 | Architecture | Exact GS/GD layer widths from Figs. 2–3 | Compact shallow/deep encoder–decoders with same dual-average IG |
| D4 | MRR | Exact MR_PMSE patch tiling + NCD formulation | Multi-scale MSE + intensity NCD + channel −KL participation |
| D5 | Downstream FR DA | 15% enhanced probes into CNN FR | Reconstruction PSNR/SSIM only (no ICFD FR Rank-1) |
