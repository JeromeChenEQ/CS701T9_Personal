#!/usr/bin/env bash
# Make a val submission from a run, at a given epoch, with the thresholds tuned on the holdout (score 0, NMS 0.8).
# Usage: nohup setsid ./predict_at_epoch.sh runs/<out>/<run>/ 64 submission_val_<name>_ep64.zip [WEIGHTS] \
#            > predict_<name>_ep64.out 2>&1 &
# Without WEIGHTS: wait until that epoch is validated (training has moved on to the next one, so best.pt is fully
# written) and freeze best.pt as best_ep<EPOCH>.pt. With WEIGHTS (a file the run writes, e.g. model_ema_ep48.pt
# from --save-epochs, or model_ema.pt at the end): wait for that file. Predicts on the CPU while training holds the
# GPU, on the GPU once training has stopped.
set -u
RUN=${1%/}; EPOCH=$2; ZIP=$3; WEIGHTS=${4:-}
PY=${PY:-python}
cd "$(dirname "$0")"
echo "waiting for epoch $EPOCH of $RUN${WEIGHTS:+ ($WEIGHTS)} since $(date '+%a %H:%M')"
# If training stops (crash, or stopped to be resumed), keep waiting: it continues from its checkpoint with --resume
ready() {
  if [ -n "$WEIGHTS" ]; then [ -f "$RUN/$WEIGHTS" ]
  else grep -q "^epoch $((EPOCH + 1)) iter" "$RUN/log.txt" || [ -f "$RUN/metrics.json" ]; fi
}
warned=0
until ready; do
  if pgrep -f "python train.py" >/dev/null; then warned=0
  elif [ $warned = 0 ]; then echo "WARNING: training not running at $(date '+%a %H:%M'); still waiting for it to resume"; warned=1
  fi
  sleep 30
done
if [ -n "$WEIGHTS" ]; then
  # model_ema.pt is written in place at the end of training: wait until its size has stopped changing
  size=-1; until [ "$size" = "$(stat -c %s "$RUN/$WEIGHTS")" ]; do size=$(stat -c %s "$RUN/$WEIGHTS"); sleep 20; done
  grep -E "^epoch $EPOCH" "$RUN/log.txt" | grep -E "val|saved" | cut -c1-150
else
  grep -E "^epoch $EPOCH val" "$RUN/log.txt" | cut -c1-150
  grep "best so far" "$RUN/log.txt" | tail -1
  cp "$RUN/best.pt" "$RUN/best_ep$EPOCH.pt"
  WEIGHTS=best_ep$EPOCH.pt
  echo "copied best.pt -> $WEIGHTS"
fi
if pgrep -f "python train.py" >/dev/null; then
  echo "predicting val from $WEIGHTS on the CPU at $(date '+%a %H:%M') (training holds the GPU)"
  DEVICE="--device cpu --threads 6 --batch-size 4 --workers 2"; export CUDA_VISIBLE_DEVICES=
else
  echo "predicting val from $WEIGHTS on the GPU at $(date '+%a %H:%M')"
  DEVICE=""
fi
$PY predict.py "$RUN" --weights "$WEIGHTS" --images cs701-sar-course-data/val/images.json \
  --score-thresh 0 --nms-thresh 0.8 $DEVICE --out "$ZIP" 2>&1 | grep -v -i "warning\|_C._set"
$PY check_submission.py cs701-sar-course-data/val/images.json "$ZIP" 2>&1 | tail -4
echo "done at $(date '+%a %H:%M')"
