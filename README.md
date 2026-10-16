# D2SC-GAN: Dual Deep-Shallow Channeled GAN for Low-Resolution Faces

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of Bhattacharjee & Das, *D2SC-GAN* (IEEE TBIOM 2020): shallow + deep generators averaged for LR→HR faces with an auxiliary identity discriminator and MRR + KLD losses.

**Repository:** https://github.com/biometric-community/d2sc-gan-dual-deep-shallow-channeled-generative-adversarial-network-fo

## Setup

```bash
bash scripts/setup_env.sh
```

## Data

Paper ICFD / FR_SURV / … are not redistributed. Default protocol uses **CelebA** with ×4 bicubic LR (see `DEVIATIONS.md`).

```bash
bash scripts/check_dataset_size.sh ../../datasets/celeba
```

## Train / Eval / Report

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/smoke.sh
bash scripts/train_full.sh
```

Results: [`REPORT.md`](REPORT.md).

## Citation

```bibtex
@article{bhattacharjee2020d2sc,
  title   = {D2SC-GAN: Dual Deep-Shallow Channeled Generative Adversarial Network, for Resolving Low-Resolution Faces for Recognition in Classroom Scenarios},
  author  = {Bhattacharjee, Avishek and Das, Sukhendu},
  journal = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year    = {2020}
}
```
