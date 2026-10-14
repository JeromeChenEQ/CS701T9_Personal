#!/usr/bin/env bash
# Best setup so far on our 15 % holdout (seed 0, RTX 3060, 9.05 h):
#   TerraMind + MoE-LoRA + Kendall loss weighting, 36 epochs -> macro-F1 91.56, mAP 31.12
# Batch 16 runs as 2 x 8 with gradient accumulation (same update; fits 12 GB). On a GPU with >= 16 GB, pass
# --accum-steps 1 for a true batch of 16 (faster, same training).
# Usage: bash run_best.sh --data /path/to/cs701-sar-course-data [--seed 1] [--out runs/cluster]
# The data folder needs our holdout split: copy splits/classification and splits/detection into it.
set -e
python train.py --backbone terramind --init pretrained --adapt moelora --loss-weighting kendall \
  --accum-steps 2 --epochs 36 --eval-every 6 --workers 4 "$@"
