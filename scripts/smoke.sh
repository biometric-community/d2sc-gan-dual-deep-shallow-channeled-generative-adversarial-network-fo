#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
bash scripts/train.sh configs/smoke.yaml
bash scripts/eval.sh configs/smoke.yaml
bash scripts/predict.sh configs/smoke.yaml --limit 8
bash scripts/report.sh configs/smoke.yaml
echo SMOKE_OK
