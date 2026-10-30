#!/usr/bin/env bash
# Phase 2 (test): download the test images once they are released (2 Nov 2026, 00:01 SGT), predict them with a run's
# weights and the inference recipe tuned on our holdout (score 0, NMS 0.8, rescoring by the class head, alpha 4,
# best 100 of 300 boxes), and check the zip before uploading. Only 2 test submissions per team: upload only zips
# that pass the check.
# Usage: ./phase2.sh [RUN_FOLDER] [WEIGHTS] [NAME]
#   default: submission 16's model (val delta_m +46.11), runs/dinov3_full/<run>/model_ema.pt
# Writes submission_test_<NAME>.zip; predicts on the CPU while a training run holds the GPU.
set -eu
cd "$(dirname "$0")"
RUN=${1:-runs/dinov3_full/dinov3_convnext_b_pretrained_lora_kendall_ls0.1_all_r125_ema0.9995_giou10_full_st0.001}
WEIGHTS=${2:-model_ema.pt}
NAME=${3:-dinov3_full_ep96}
DATA=${DATA:-cs701-sar-course-data}
PY=${PY:-python}
HF=${HF:-hf}
ZIP=submission_test_$NAME.zip

# 1. The test images (once)
if [ ! -f "$DATA/test/images.json" ]; then
  echo "downloading test.tar.gz and sample_submission_test.zip"
  if ! $HF download doem1997/cs701-sar-course-data test.tar.gz sample_submission_test.zip --repo-type dataset \
       --local-dir "$DATA" > /dev/null 2>&1 || [ ! -f "$DATA/test.tar.gz" ]; then
    echo "test.tar.gz is not available yet: the test split is released on 2 Nov 2026, 00:01 SGT"; exit 1
  fi
  tar -xzf "$DATA/test.tar.gz" -C "$DATA"
fi
N=$($PY -c "import json; d = json.load(open('$DATA/test/images.json')); print(len(d['images'] if isinstance(d, dict) else d))")
echo "test/images.json: $N images (expected 2,065)"
[ "$N" = 2065 ] || echo "WARNING: not 2,065 test images"
if [ -f "$DATA/sample_submission_test.zip" ]; then
  echo "--- the course's sample submission, as a format reference:"
  $PY check_submission.py "$DATA/test/images.json" "$DATA/sample_submission_test.zip" 2>&1 | head -2
fi

# 2. Predict
[ -f "$RUN/$WEIGHTS" ] || { echo "no weights at $RUN/$WEIGHTS"; exit 1; }
if pgrep -f "python train.py" > /dev/null; then
  echo "a training run holds the GPU: predicting on the CPU (about 1.5 h for 2,065 images)"
  DEVICE="--device cpu --threads 6 --batch-size 4 --workers 2"; export CUDA_VISIBLE_DEVICES=
else
  DEVICE=""
fi
echo "predicting $N test images with $RUN/$WEIGHTS at $(date '+%a %H:%M')"
$PY predict.py "$RUN" --weights "$WEIGHTS" --images "$DATA/test/images.json" --score-thresh 0 --nms-thresh 0.8 \
  --rescore-alpha 4 --candidates 300 $DEVICE --out "$ZIP" 2>&1 | grep -v -i "warning\|_C._set"

# 3. Check before uploading
echo "--- check of $ZIP:"
$PY check_submission.py "$DATA/test/images.json" "$ZIP"
echo "done at $(date '+%a %H:%M'): upload $ZIP to the Phase 2 (test) page if the check shows no errors"
