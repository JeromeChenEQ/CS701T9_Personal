"""Tune the detection score and NMS thresholds of a trained run on our holdout, without retraining: predict the
holdout once per setting and print macro-F1, mAP, AP50 and Δm for each (the classification head does not depend
on them). Inference-only settings: still one model and one forward pass per image.
Example: python sweep_thresholds.py runs/dinov3/<run> --weights best.pt"""
import argparse
import json
from pathlib import Path

from predict import ImagesToPredict, load_run, predict
from sarbench.metrics import classification_metrics, detection_metrics
from train import delta_m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='a run folder of train.py')
    p.add_argument('--weights', default='best.pt')
    p.add_argument('--gt', type=Path, default=Path('cs701-sar-course-data/detection/instances_val.json'),
                   help='COCO ground truth of the holdout (its images are predicted)')
    p.add_argument('--score-thresh', type=float, nargs='+', default=[0.001, 0.0001, 0.0])
    p.add_argument('--nms-thresh', type=float, nargs='+', default=[0.4, 0.5, 0.6])
    p.add_argument('--out', type=Path, help='also write every setting\'s metrics to this JSON file')
    args = p.parse_args()

    model = load_run(args.run, args.weights)
    dataset = ImagesToPredict(args.gt)
    ids = [im['id'] for im in dataset.images]
    truth = {a['image_id']: a['category_id'] - 1 for a in json.loads(args.gt.read_text())['annotations']}
    results = []
    print('score   nms   macro_F1  mAP     AP50    delta_m  boxes  images without a box')
    for score in args.score_thresh:
        for nms in args.nms_thresh:
            model.roi_heads.score_thresh, model.roi_heads.nms_thresh = score, nms
            labels, detections = predict(model, dataset)
            m = {**classification_metrics([truth[i] for i in ids], [labels[i] for i in ids]),
                 **detection_metrics(args.gt, detections, ids)}
            empty = len(ids) - len({d['image_id'] for d in detections})
            results.append({'score_thresh': score, 'nms_thresh': nms, 'delta_m': delta_m(m), 'boxes': len(detections),
                            'images_without_box': empty, **{k: m[k] for k in ('macro_f1', 'mAP', 'AP50', 'AP75')}})
            print(f"{score:<7g} {nms:<5g} {m['macro_f1'] * 100:.2f}     {m['mAP'] * 100:.2f}   {m['AP50'] * 100:.2f}   "
                  f"{delta_m(m):+.2f}   {len(detections):<6d} {empty}", flush=True)
    best = max(results, key=lambda r: r['delta_m'])
    print(f"best: score {best['score_thresh']:g}, NMS {best['nms_thresh']:g}: delta_m {best['delta_m']:+.2f}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
