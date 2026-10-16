#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
bash scripts/check_dataset_size.sh ../../datasets/celeba
bash scripts/train.sh configs/default.yaml
bash scripts/eval.sh configs/default.yaml
bash scripts/predict.sh configs/default.yaml --limit 16
bash scripts/report.sh configs/default.yaml
echo FULL_OK
