# Submitted runs

The five models submitted to the Codabench Phase 1 (val) leaderboard, with their settings, holdout results
and val leaderboard results. All were trained on one RTX 3060 (12 GB) under WSL2.

**Best so far: SAM2 Hiera-B+ with LoRA rank 125 on all layers, 48 epochs: val Δm +13.24.**

## Results

| # | Run | Zip | Trainable / total | Train time | Holdout F1 | Holdout mAP | Val F1 | Val mAP | Val AP50 | Val accuracy | **Val Δm** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TerraMind + MoE-LoRA + Kendall, 36 ep, seed 0 | `submission.zip` | 21.1 M / 106.5 M | 9.05 h | 91.56 | 31.28 ¹ | 92.16 | 30.87 | 60.31 | 92.50 | +8.44 |
| 2 | same, seed 42 | `submission_val_ep36_seed42.zip` | 21.1 M / 106.5 M | 9.04 h | 89.88 | 31.04 | 92.13 | 30.75 | 61.30 | 92.50 | +8.22 |
| 3 | #1 + detail branch | `submission_val_detail.zip` | 21.6 M / 107.0 M | 11.01 h | 89.76 | 31.75 | 91.63 | 31.07 | 60.94 | 92.01 | +8.55 |
| 4 | #1 + unfreeze last 2 blocks | `submission_val_unfreeze2.zip` | 35.1 M / 106.3 M | 9.92 h | 91.57 | 31.56 | 92.68 | 30.14 | 60.10 | 93.06 | +7.33 |
| 5 | **SAM2 Hiera-B+ + LoRA r125 (all layers) + Kendall, 48 ep** | `submission_val_sam2.zip` | **39.0 M / 107.7 M** | 10.88 h | **94.76** | **33.30** | **93.12** | **33.10** | **63.59** | **93.41** | **+13.24** |

All values in %. Limits: 130 M total, 40 M trainable. Δm is relative to the reference ViT-B/16 + LoRA on val
(macro-F1 93.31, mAP 26.13): `Δm = ½ [(F1 − 93.31) / 93.31 + (mAP − 26.13) / 26.13] × 100 %`.

¹ Run 1 was trained with the default detection score threshold 0.05 (holdout mAP 31.12) and submitted at 0.001;
31.28 is its holdout mAP at 0.001. All other runs used 0.001 throughout.

- **Holdout:** our 15 % stratified split of the train set (1,408 images; `splits/`, made by `prepare_data.py`,
  seed 0). Same split for every run.
- **Seed variation** (runs 1 and 2): val F1 92.15 ± 0.02, mAP 30.81 ± 0.08, Δm +8.33 ± 0.16. The holdout varies
  more (F1 90.72 ± 1.19, mAP 31.16 ± 0.17), mostly from bridge ↔ playground confusions.

## Commands

Shared settings of runs 1–4 are in `run_best.sh`: TerraMind, MoE-LoRA, Kendall loss weighting, 36 epochs, batch
16 as 2 × 8 (`--accum-steps 2`), validation every 6 epochs. `DATA=cs701-sar-course-data` (in this repo)
below, with our split files from `splits/` copied into `$DATA/classification` and `$DATA/detection`.

```bash
# 1  seed 0 (submitted at --score-thresh 0.001)
bash run_best.sh --data $DATA --out runs/ep36
# 2  seed 42
bash run_best.sh --data $DATA --seed 42 --score-thresh 0.001 --out runs/seed42
# 3  + detail branch
bash run_best.sh --data $DATA --detail-branch --score-thresh 0.001 --out runs/abl
# 4  + unfreeze last 2 blocks (trained at --unfrozen-lr 2e-5, the default)
bash run_best.sh --data $DATA --unfreeze-last 2 --score-thresh 0.001 --out runs/abl
# 5  SAM2 Hiera-B+, LoRA rank 125 on all 99 Linear layers, 48 epochs
python train.py --backbone sam2 --init pretrained --adapt lora --lora-targets all --lora-rank 125 \
  --loss-weighting kendall --accum-steps 2 --epochs 48 --eval-every 6 --score-thresh 0.001 --workers 4 \
  --data $DATA --out runs/sam2 --resume
```

Submission zip from a finished run (the `--out` folder plus the run's subfolder name):

```bash
python predict.py runs/sam2/sam2_pretrained_lora_kendall_all_r125_st0.001 \
  --images $DATA/val/images.json --score-thresh 0.001 --out submission_val_sam2.zip
python check_submission.py $DATA/val/images.json submission_val_sam2.zip   # format check before uploading
```

## Findings

- **SAM2 Hiera-B+ (multi-scale, strides 4–32) beats every TerraMind variant** by about +4 F1 and +2 mAP on the
  holdout and +4.7 Δm on val over the best TerraMind submission, above the reference full fine-tuning (+11.8 on val, 106 M trainable). Its
  training-set F1 is 100 % (fully memorised), training mAP 42.9 vs holdout 33.3. Holdout mAP gained only +0.2
  between epochs 42 and 48.
- **Detail branch:** small, real mAP gain (+0.26 val mAP over the seed mean) but −0.5 F1; net +0.2 Δm for 2 h
  more training. Not adopted.
- **Unfreezing the last 2 TerraMind blocks:** +0.4 holdout mAP but −0.7 val mAP; worse on val (+7.33).
  Not adopted.
- **Lower detection score threshold (0.001 vs 0.05):** +0.16 holdout mAP, no retraining needed.

Runs that were not submitted, on the holdout:

| Run | F1 | mAP | Note |
|---|---|---|---|
| ViT-B/16 + LoRA, 24 ep (reference-style) | 91.57 | 24.50 | first baseline |
| TerraMind + MoE-LoRA + Kendall, 24 ep | 90.18 | 27.75 | 36 epochs gave +3.4 mAP |
| TerraMind + MoE-LoRA + label smoothing 0.1 + detection→class fusion, 24 ep | 90.15 | 27.37 | no gain; fusion weights stayed near 0 |
| #1 + 90° rotations + intensity jitter, 36 ep | 92.17 | 26.53 | **90° rotations hurt**: the no-augmentation model scores 12.9 mAP on 90°-rotated holdout images vs 31.3 unrotated (180° and flips: no loss), so SAR orientation carries information |

## Notes

- Not in this repo: the dataset, `runs/` (model files ~430 MB each), the submission zips and the training logs.
- `prepare_data.py` has the data location hard-coded (`cs701-sar-course-data` in this repo); copying
  `splits/` gives the same holdout without running it.
- `train.py --resume` continues an interrupted run from its last checkpoint (saved at every validation). The first
  SAM2 attempt was lost to an overnight Windows Update (NVIDIA driver install + automatic restart).
- Pretrained models used: TerraMind-1.0-base (IBM/ESA, via terratorch), SAM2 Hiera-B+ (Meta, timm
  `sam2_hiera_base_plus.fb_r896`), ViT-B/16 ImageNet-21k (timm `vit_base_patch16_224.augreg_in21k`).
