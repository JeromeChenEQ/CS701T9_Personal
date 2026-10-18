# Submitted runs

The models submitted to the Codabench Phase 1 (val) leaderboard, with their settings, holdout results and val
leaderboard results. All were trained on PC 1 (see [Compute](#compute)), one GPU per run.

**Best so far: DINOv3 ConvNeXt-B with LoRA rank 125 on all layers, GIoU box loss, label smoothing 0.1, EMA, 108
epochs (best validated weights: plain, epoch 100), detection score threshold 0 and NMS 0.8 tuned on the holdout:
val Δm +41.69.**

## Results

| # | Run | Zip | Trainable / total | Train time | Holdout F1 | Holdout mAP | Val F1 | Val mAP | Val AP50 | Val accuracy | **Val Δm** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TerraMind + MoE-LoRA + Kendall, 36 ep, seed 0 | `submission.zip` | 21.1 M / 106.5 M | 9.05 h | 91.56 | 31.28 ¹ | 92.16 | 30.87 | 60.31 | 92.50 | +8.44 |
| 2 | same, seed 42 | `submission_val_ep36_seed42.zip` | 21.1 M / 106.5 M | 9.04 h | 89.88 | 31.04 | 92.13 | 30.75 | 61.30 | 92.50 | +8.22 |
| 3 | #1 + detail branch | `submission_val_detail.zip` | 21.6 M / 107.0 M | 11.01 h | 89.76 | 31.75 | 91.63 | 31.07 | 60.94 | 92.01 | +8.55 |
| 4 | #1 + unfreeze last 2 blocks | `submission_val_unfreeze2.zip` | 35.1 M / 106.3 M | 9.92 h | 91.57 | 31.56 | 92.68 | 30.14 | 60.10 | 93.06 | +7.33 |
| 5 | SAM2 Hiera-B+ + LoRA r125 (all layers) + Kendall, 48 ep | `submission_val_sam2.zip` | 39.0 M / 107.7 M | 10.88 h | 94.76 | 33.30 | 93.12 | 33.10 | 63.59 | 93.41 | +13.24 |
| 6 | #5 + EMA 0.9995, 98 ep (EMA weights of epoch 98, `best.pt`) | `submission_val_sam2_e98_ema.zip` | 39.0 M / 107.7 M | 23.89 h | 94.89 | 39.14 | 93.46 | 36.57 | 67.24 | 93.76 | +20.06 |
| 7 | DINOv3 ConvNeXt-B + LoRA r125 (all layers) + Kendall + EMA 0.9995 + GIoU + label smoothing 0.1, 108 ep: EMA weights of epoch 64, submitted while training ² | `submission_val_dinov3_ep64.zip` | 40.0 M / 127.5 M | ~15.5 h to epoch 64 | 96.13 | 48.69 | 95.61 | 45.94 | 74.52 | 95.86 | +39.13 |
| 8 | #7 finished: `best.pt` = plain weights of epoch 100 of 108 (holdout Δm +46.67; epoch 108: +46.42 plain, +46.53 EMA) | `submission_val_dinov3_final.zip` | 40.0 M / 127.5 M | 26.35 h | 96.01 | 49.77 | 95.79 | 46.14 | 73.46 | 96.00 | +39.61 |
| 9 | **#8 with the detection thresholds tuned on the holdout: score 0 (was 0.001), NMS IoU 0.8 (was 0.5)**; same weights, inference only | `submission_val_dinov3_final_nms08.zip` | **40.0 M / 127.5 M** | 26.35 h | **96.01** | **50.57** | **95.79** | **47.23** | **74.31** | **96.00** | **+41.69** |

All values in %. Limits: 130 M total, 40 M trainable. Δm is relative to the reference ViT-B/16 + LoRA on val
(macro-F1 93.31, mAP 26.13): `Δm = ½ [(F1 − 93.31) / 93.31 + (mAP − 26.13) / 26.13] × 100 %`.

¹ Run 1 was trained with the default detection score threshold 0.05 (holdout mAP 31.12) and submitted at 0.001;
31.28 is its holdout mAP at 0.001. All other runs used 0.001 throughout.

² Run 7's epoch-64 `best.pt` was copied to `best_ep64.pt` and predicted on the CPU in fp32 (`predict.py --device
cpu`) while training continued on the GPU; holdout numbers are from training's bf16 validation. Exact trainable
count 39,968,859 (limit 40 M).

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
# 6  as 5, plus an EMA of the trainable weights (validated next to the plain weights), 98 epochs, validation
#    every 4 epochs (git branch SAM2-LoRA125-EMA-e98)
python train.py --backbone sam2 --init pretrained --adapt lora --lora-targets all --lora-rank 125 \
  --loss-weighting kendall --ema-decay 0.9995 --accum-steps 2 --epochs 98 --eval-every 4 \
  --score-thresh 0.001 --workers 4 --out runs/sam2_ema --resume
# 7  DINOv3 ConvNeXt-B, LoRA rank 125 on all 72 Linear layers, GIoU box loss, label smoothing 0.1, EMA, 108 epochs
#    (git branch DINOv3CNB-LoRA125-EMA-GIoU-LS-e108); submitted: EMA weights of epoch 64, predicted on the CPU
python train.py --backbone dinov3_convnext_b --init pretrained --adapt lora --lora-targets all --lora-rank 125 \
  --loss-weighting kendall --ema-decay 0.9995 --box-loss giou --label-smoothing 0.1 --accum-steps 2 --epochs 108 \
  --eval-every 4 --score-thresh 0.001 --workers 4 --out runs/dinov3 --resume
cp runs/dinov3/<run>/best.pt runs/dinov3/<run>/best_ep64.pt   # right after the epoch-64 validation
python predict.py runs/dinov3/<run> --weights best_ep64.pt --images $DATA/val/images.json --score-thresh 0.001 \
  --device cpu --threads 6 --batch-size 4 --workers 2 --out submission_val_dinov3_ep64.zip
# 8  run 7 finished (26.35 h, peak GPU memory 7.0 GB): best.pt, predicted on the GPU
python predict.py runs/dinov3/<run> --weights best.pt --images $DATA/val/images.json --score-thresh 0.001   --out submission_val_dinov3_final.zip
# 9  as 8, with the thresholds from the holdout sweep (sweep_thresholds.py; results in the run folder's
#    threshold_sweep*.json)
python sweep_thresholds.py runs/dinov3/<run> --weights best.pt                      # score {0.001, 0.0001, 0} x NMS {0.4, 0.5, 0.6}
python sweep_thresholds.py runs/dinov3/<run> --weights best.pt --score-thresh 0 --nms-thresh 0.65 0.7 0.75 0.8 0.85 0.9
python predict.py runs/dinov3/<run> --weights best.pt --images $DATA/val/images.json --score-thresh 0   --nms-thresh 0.8 --out submission_val_dinov3_final_nms08.zip
```

Submission zip from a finished run (the `--out` folder plus the run's subfolder name; `--weights best.pt` is the
validated weights with the best holdout Δm, plain or EMA, for runs from 6 on):

```bash
python predict.py runs/sam2_ema/sam2_pretrained_lora_kendall_all_r125_ema0.9995_st0.001 --weights best.pt \
  --images $DATA/val/images.json --score-thresh 0.001 --out submission_val_sam2_e98_ema.zip
python check_submission.py $DATA/val/images.json submission_val_sam2_e98_ema.zip   # format check before uploading
```

## Findings

- **DINOv3 ConvNeXt-B (run 7) nearly doubles Δm over SAM2 (run 6):** val +39.13 vs +20.06, from both tasks (val
  mAP 45.94 vs 36.57, F1 95.61 vs 93.46), already at epoch 64 of 108. It also changes GIoU and label smoothing, but
  GIoU alone on SAM2 was level with run 6 at epoch 36 (holdout Δm +11.69 vs +12.73, EMA) and label smoothing
  alone mostly moved classification (+0.4 holdout F1), so the backbone carries most of the gain. Holdout − val gap:
  5.56 Δm (run 6: 5.7), so the holdout still predicts val well. Training classification loss sat at label
  smoothing's floor (0.485) from epoch 24 while holdout mAP kept rising (45.8 at epoch 44 → 48.7 at 64).
- **Run 7 finished (submission 8) vs its epoch-64 weights (submission 7):** +0.48 val Δm (mAP 46.14 vs 45.94, F1
  95.79 vs 95.61) for +1.98 holdout Δm, so the holdout − val gap widened from 5.56 to 7.06; holdout Δm was flat
  at +46.4 to +46.7 from epoch 88 on. Val AP50 fell (73.46 vs 74.52) and the final model outputs far fewer boxes
  at score threshold 0.001: 6,892 vs 15,555, with 90 val images without any detection (epoch 64: 20); the
  detection score and NMS thresholds were tuned next (submission 9).
- **Detection thresholds tuned on the holdout (submission 9), no retraining:** with run 7's final weights,
  score threshold 0 and NMS IoU 0.8 raise holdout Δm from +46.67 to +48.21 (mAP 49.77 → 50.57) and val Δm from
  +39.61 to +41.69 (mAP 46.14 → 47.23, AP50 73.46 → 74.31); classification is unchanged. Holdout Δm by NMS at
  score 0: 0.4 +46.70, 0.5 +47.08, 0.6 +47.54, 0.7 +47.85, 0.75 +48.05, **0.8 +48.21**, 0.85 +48.17, 0.9 +47.07;
  at NMS 0.5 by score: 0.001 +46.67, 0.0001 +46.85, 0 +47.08. Score 0 fills every image's 100 boxes (98 holdout
  images had none at 0.001); a looser NMS keeps shifted second boxes that count at COCO's strict IoU thresholds.
  Still one model and one forward pass per image.

- **SAM2 Hiera-B+ (multi-scale, strides 4–32) beats every TerraMind variant** by about +4 F1 and +2 mAP on the
  holdout and +4.7 Δm on val over the best TerraMind submission, above the reference full fine-tuning (+11.8 on val, 106 M trainable). Its
  training-set F1 is 100 % (fully memorised), training mAP 42.9 vs holdout 33.3. Holdout mAP gained only +0.2
  between epochs 42 and 48.
- **EMA + 98 epochs (run 6) over run 5:** +6.8 val Δm, almost all from detection (val mAP 36.57 vs 33.10, F1
  93.46 vs 93.12). The EMA weights beat the plain ones by +3 to +8.5 holdout Δm from epoch 12 to 64 and by +0.6 to
  +2.2 from 68 to 76; from epoch 80, with the learning rate near 0, the two were within 0.4. The plain weights fell
  back between validations three times (by up to 1.0 Δm), the EMA weights never after epoch 4. Holdout Δm rose from +21.5 (epoch 60) to +25.7 (epoch 98), +0.3 over the last 8 epochs. Training
  classification loss reached 0.000 and the Kendall classification weight grew to 180 (detection: 9.3).
  The val Δm is 5.7 below the holdout Δm (+25.75), against 1.3 for run 5.
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
| #5 + random rotation ±180° + EMA 0.9995, 80 ep, **stopped after epoch 47** | 92.61 | 19.80 | best: EMA weights at epoch 44 (holdout Δm −12.49). **Any-angle rotation hurts detection badly**: mAP 19.8 vs 33.1 for #5 at epoch 42 (F1 92.6 vs 94.7). EMA beat the plain weights at every validation from epoch 16 on, 8 of 11 (epoch 44: Δm −12.49 vs −13.52; epoch 16: −24.45 vs −32.68); before that the average still carried the early weights |

## Compute

Two PCs, one GPU each; every run uses one GPU. Train times in the tables are wall-clock hours on that GPU
(validations and checkpoints included), so they are also the run's GPU hours.

| | PC 1 | PC 2 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB (driver 610.88) | NVIDIA GeForce RTX 3060, 12 GB |
| CPU | AMD Ryzen 5 5600, 6 cores / 12 threads | not recorded |
| RAM | 32 GB (16 GB available to WSL2) | 32 GB DDR4 |
| Storage | NVMe SSD (Lexar NM710, 1 TB) | HDD |
| OS | Windows 11, Ubuntu 26.04 under WSL2 | not recorded |
| Runs | runs 1–6, the stopped rotation run, the regularisation run (branch `SAM2-LoRA125-EMA-Reg-e108`) | GIoU run (branch `SAM2-LoRA125-EMA-GIoU-e108`) |

Same software on both (`requirements.txt`: torch 2.9.0, torchvision 0.24.0, timm 1.0.24; CUDA 12.8 wheels) and
the same holdout split (`splits/`). Training reads the images from disk only until the OS caches them (the
9,392 train images are about 0.8 GB), so the HDD of PC 2 should slow mainly the first epoch and the per-epoch
checkpoint writes (~0.9 GB each).

## Notes

- Not in this repo: the dataset, `runs/` (model files ~430 MB each), the submission zips and the training logs.
- `prepare_data.py` has the data location hard-coded (`cs701-sar-course-data` in this repo); copying
  `splits/` gives the same holdout without running it.
- `train.py --resume` continues an interrupted run from its last checkpoint (saved after every epoch since run 6;
  at every validation before). The first
  SAM2 attempt was lost to an overnight Windows Update (NVIDIA driver install + automatic restart).
- Pretrained models used: TerraMind-1.0-base (IBM/ESA, via terratorch), SAM2 Hiera-B+ (Meta, timm
  `sam2_hiera_base_plus.fb_r896`), ViT-B/16 ImageNet-21k (timm `vit_base_patch16_224.augreg_in21k`).
