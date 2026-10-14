#!/usr/bin/env bash
# The best setup so far (run_best.sh: TerraMind + MoE-LoRA + Kendall, 36 epochs; holdout F1 91.56, mAP 31.12)
# plus every improvement since, all switched on:
#   --rot90             random 90-degree rotations (with the flips: all 8 orientations)
#   --intensity-jitter  mild random gamma / contrast / brightness
#   --adapt-mlp         MoE-LoRA also on the MLP layers (trainable 21.1 M -> 23.6 M; limit 40 M)
#   --detail-branch     light conv branch on the raw image -> stride-4 / stride-8 detection features (small objects)
#   --score-thresh      keep detections down to 0.001 (holdout mAP +0.16 on the 36-epoch model, no retraining)
# Batch 16 runs as 4 x 4 (--accum-steps 4): same update as one batch of 16; at 2 x 8 the RTX 3060 runs out of its
# 12 GB (10.7 GiB peak) and slows down. RTX 3060: 6.6 images/s, about 12.5 h. On a GPU with >= 16 GB, pass
# --accum-steps 1 (much faster, same training).
# Usage: bash run_improved.sh --data /path/to/cs701-sar-course-data [--seed 1] [--out runs/improved]
set -e
python train.py --backbone terramind --init pretrained --adapt moelora --loss-weighting kendall \
  --epochs 36 --eval-every 6 --workers 4 --accum-steps 4 \
  --rot90 --intensity-jitter --adapt-mlp --detail-branch --score-thresh 0.001 "$@"

# Ablations for the report (each: the best setup + one option), to see what each option adds on its own:
#   bash run_best.sh --data ... --rot90 --intensity-jitter --out runs/abl
#   bash run_best.sh --data ... --adapt-mlp --out runs/abl
#   bash run_best.sh --data ... --detail-branch --out runs/abl
# Submission (once val/images.json is released): predict with the finished run, upload the zip to Codabench:
#   python predict.py runs/<run folder> --images /path/to/cs701-sar-course-data/val/images.json --out submission_val.zip
