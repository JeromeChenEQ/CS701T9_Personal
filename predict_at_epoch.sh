#!/usr/bin/env bash
# Make a val submission from a run that is still training, at a given epoch: wait until that epoch is validated
# (training has moved on to the next one, so best.pt is fully written), freeze best.pt as best_ep<EPOCH>.pt, then
# predict val on the CPU (training keeps the GPU) with the thresholds tuned on the holdout (score 0, NMS 0.8).
# Usage: nohup setsid ./predict_at_epoch.sh runs/<out>/<run>/ 64 submission_val_<name>_ep64.zip > predict_<name>_ep64.out 2>&1 &
set -u
RUN=${1%/}; EPOCH=$2; ZIP=$3
PY=${PY:-python}
cd "$(dirname "$0")"
echo "waiting for epoch $EPOCH of $RUN since $(date '+%a %H:%M')"
# If training stops (crash, or stopped to be resumed), keep waiting: it continues from its checkpoint with --resume
warned=0
until grep -q "^epoch $((EPOCH + 1)) iter" "$RUN/log.txt" || [ -f "$RUN/metrics.json" ]; do
  if pgrep -f "python train.py" >/dev/null; then warned=0
  elif [ $warned = 0 ]; then echo "WARNING: training not running at $(date '+%a %H:%M'); still waiting for it to resume"; warned=1
  fi
  sleep 30
done
grep -E "^epoch $EPOCH val" "$RUN/log.txt" | cut -c1-150
grep "best so far" "$RUN/log.txt" | tail -1
cp "$RUN/best.pt" "$RUN/best_ep$EPOCH.pt"
echo "copied best.pt -> best_ep$EPOCH.pt at $(date '+%a %H:%M'); predicting val on the CPU"
CUDA_VISIBLE_DEVICES= $PY predict.py "$RUN" --weights "best_ep$EPOCH.pt" --images cs701-sar-course-data/val/images.json \
  --score-thresh 0 --nms-thresh 0.8 --device cpu --threads 6 --batch-size 4 --workers 2 --out "$ZIP" 2>&1 \
  | grep -v -i "warning\|_C._set"
$PY check_submission.py cs701-sar-course-data/val/images.json "$ZIP" 2>&1 | tail -4
echo "done at $(date '+%a %H:%M')"
