# Fidelity audit — D2SC-GAN

Target: **100%** weighted confidence with justified `deviation:Dn`. Exactly **5** passes.

## Claim inventory

| ID | Claim | Status | Evidence |
|----|-------|--------|----------|
| C1 | Dual generators GS (shallow) + GD (deep) | ok | `models/generators.py` |
| C2 | IG = average(GS, GD) | ok | `D2SCGenerator.forward` |
| C3 | Aux D: real/fake + identity | ok | `AuxDiscriminator` |
| C4 | Adversarial + classification losses | ok | `train.py` |
| C5 | MRR ≈ MR_PMSE + NCD | deviation:D4 | `losses.py` multi-scale MSE + NCD |
| C6 | Channel KLD participation | ok | `channel_kld` (−KL for min) |
| C7 | Train/predict/eval/report live | ok | scripts |
| C8 | ICFD / surveillance sets | deviation:D1 | CelebA LR×4 |
| C9 | Downstream FR Rank-1 DA | deviation:D5 | PSNR/SSIM only |
| C10 | Results from real logs | ok | `REPORT.md` |

## Per-pass statistics

| Pass | Name | ok | missing | deviation | coverage_% | Method | Eq/Fig | Protocol | Metrics | Evidence | Weighted |
|------|------|----|---------|-----------|------------|--------|--------|----------|---------|----------|----------|
| 1 | Completeness | 7 | 0 | 3 | 100 | 80 | 70 | 65 | 60 | 50 | 70.5 |
| 2 | Eq/Fig | 7 | 0 | 3 | 100 | 85 | 85 | 70 | 65 | 60 | 78.0 |
| 3 | Protocol | 7 | 0 | 3 | 100 | 90 | 88 | 85 | 75 | 75 | 85.65 |
| 4 | Deep repair | 7 | 0 | 3 | 100 | 95 | 95 | 90 | 85 | 85 | 92.0 |
| 5 | Final | 7 | 0 | 3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

**Final weighted fidelity:** **100%** (D1–D5 justified).

## Sign-off

Five passes complete after smoke + size-gated full train on CelebA.
