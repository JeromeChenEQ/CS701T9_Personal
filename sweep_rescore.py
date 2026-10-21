"""Tune box rescoring by the class head on our holdout: every box of an image has the image's class, so a box's
score is multiplied by the class head's probability of the box's class to the power alpha (predict.py
--rescore-alpha). Predicts the holdout once, with the most candidate boxes per image, then evaluates every (alpha,
candidates) pair on that output; the best 100 boxes per image are kept, as COCO AP counts 100. Inference only.
Example: python sweep_rescore.py runs/dinov3/<run> --weights best.pt"""
import argparse
import json
from pathlib import Path

from predict import ImagesToPredict, load_run, predict_raw, to_submission
from sarbench.metrics import classification_metrics, detection_metrics
from train import delta_m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='a run folder of train.py (trained on the 85 %% split, so the holdout is unseen)')
    p.add_argument('--weights', default='best.pt')
    p.add_argument('--gt', type=Path, default=Path('cs701-sar-course-data/detection/instances_val.json'))
    p.add_argument('--score-thresh', type=float, default=0.0)
    p.add_argument('--nms-thresh', type=float, default=0.8)
    p.add_argument('--alpha', type=float, nargs='+', default=[0.0, 0.25, 0.5, 1.0, 2.0])
    p.add_argument('--candidates', type=int, nargs='+', default=[100, 300])
    p.add_argument('--mode', nargs='+', default=['absolute', 'relative'], choices=['absolute', 'relative'],
                   help="absolute: the class probability itself; relative: divided by the image's highest one")
    p.add_argument('--out', type=Path, help="also write every setting's metrics to this JSON file")
    args = p.parse_args()

    model = load_run(args.run, args.weights)
    model.roi_heads.score_thresh, model.roi_heads.nms_thresh = args.score_thresh, args.nms_thresh
    model.roi_heads.detections_per_img = max(args.candidates)
    dataset = ImagesToPredict(args.gt)
    raw = predict_raw(model, dataset)  # each image's boxes come sorted by the detector's score, best first
    ids = [im['id'] for im in dataset.images]
    truth = {a['image_id']: a['category_id'] - 1 for a in json.loads(args.gt.read_text())['annotations']}
    results = []
    print('mode      candidates  alpha  macro_F1  mAP     AP50    AP75    delta_m')
    settings = [(mode, c, a) for mode in args.mode for c in args.candidates for a in args.alpha
                if not (mode == 'relative' and a == 0)]  # alpha 0 is the same in both modes
    for mode, candidates, alpha in settings:
        raw_c = [(i, size, probs, {k: v[:candidates] for k, v in out.items()}) for i, size, probs, out in raw]
        labels, detections = to_submission(raw_c, alpha, relative=mode == 'relative')
        m = {**classification_metrics([truth[i] for i in ids], [labels[i] for i in ids]),
             **detection_metrics(args.gt, detections, ids)}
        results.append({'mode': mode, 'candidates': candidates, 'alpha': alpha, 'delta_m': delta_m(m),
                        **{k: m[k] for k in ('macro_f1', 'mAP', 'AP50', 'AP75')}})
        print(f"{mode:<9s} {candidates:<11d} {alpha:<6g} {m['macro_f1'] * 100:.2f}     {m['mAP'] * 100:.2f}   "
              f"{m['AP50'] * 100:.2f}   {m['AP75'] * 100:.2f}   {delta_m(m):+.2f}", flush=True)
    best = max(results, key=lambda r: r['delta_m'])
    print(f"best: {best['mode']}, candidates {best['candidates']}, alpha {best['alpha']:g}: delta_m {best['delta_m']:+.2f}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
