"""Predict with a trained run on the images of an images.json (the released val or test split) and write the
Codabench submission.zip: classification.csv (image_id,label) and detection.json (COCO results, original pixels).
Example: python predict.py runs/ep36/terramind_pretrained_moelora_kendall \
             --images ~/cs701-sar-course-data/val/images.json --out submission_val.zip
With --gt (a COCO file of the same images, e.g. our holdout's instances_val.json) it also prints the metrics."""
import argparse
import csv
import io
import json
import zipfile
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sarbench.data import MEAN, STD, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from train import build_model, headline


class ImagesToPredict(Dataset):
    """The images listed in images.json ({'id', 'file_name', 'width', 'height'} each), preprocessed exactly as
    for validation: gray, resized to size x size, normalised with the SAR mean and std."""

    def __init__(self, images_json, size=512):
        images_json = Path(images_json)
        data = json.loads(images_json.read_text())
        self.images = data['images'] if isinstance(data, dict) else data
        self.folder = images_json.parent
        self.transform = v2.Compose([v2.ToImage(), v2.Resize((size, size)), v2.ToDtype(torch.float32, scale=True),
                                     v2.Normalize([MEAN], [STD])])

    def path(self, file_name):
        """file_name is relative to the folder of images.json (val/images/...), or to the data root."""
        for folder in (self.folder, self.folder.parent):
            if (folder / file_name).exists():
                return folder / file_name
        raise FileNotFoundError(f'{file_name} next to {self.folder} or {self.folder.parent}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        image = self.transform(Image.open(self.path(im['file_name'])).convert('L'))
        return image, im['id'], (im['height'], im['width'])


def load_run(run, weights, device='cuda', threads=None):
    """The model of a run folder with the given weights file, in eval mode on device. The run's options come from
    metrics.json once training has finished, else the first line of its log.txt (written at the start), so a run
    still training can be predicted from its best.pt."""
    metrics = run / 'metrics.json'
    run_args = Namespace(**(json.loads(metrics.read_text())['args'] if metrics.exists() else
                            json.loads((run / 'log.txt').read_text().splitlines()[0])))
    model = build_model(run_args, pretrained=False)  # the weights come from the weights file
    model.load_state_dict(torch.load(run / weights, map_location='cpu'))
    if device == 'cpu' and threads:
        torch.set_num_threads(threads)
    return model.to(device).eval()


@torch.no_grad()
def predict_raw(model, dataset, device='cuda', batch_size=8, workers=4):
    """-> per image: (image id, original size, class probabilities [9], detections {'boxes', 'scores', 'labels'})
    on the CPU, boxes still on the 512 x 512 input."""
    loader = DataLoader(dataset, batch_size, num_workers=workers, collate_fn=lambda batch: tuple(zip(*batch)))
    raw = []
    for images, ids, sizes in loader:
        # bf16 on the GPU, as in training; fp32 on the CPU
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits, outputs, _ = model(torch.stack(images).to(device))
        probs = logits.float().softmax(dim=1).cpu()
        raw += [(image_id, size, p, {k: v.cpu() for k, v in out.items()})
                for image_id, size, p, out in zip(ids, sizes, probs, outputs)]
    return raw


def to_submission(raw, rescore_alpha=0.0, keep=100, relative=False):
    """predict_raw's output -> ({image id: label 0..8}, COCO detections in original pixels, category 1..9).
    rescore_alpha > 0: every box of an image has the image's class, so each box's score is multiplied by the class
    head's probability for the box's class, to the power alpha, before keeping the image's `keep` best boxes; boxes of
    classes the class head rules out sink in the per-class rankings of COCO AP. 0: the detector's own scores.
    relative: use the probability divided by the image's highest one, so boxes of the predicted class keep their
    scores and only the other classes' boxes are scaled down."""
    labels, detections = {}, []
    for image_id, size, probs, out in raw:
        labels[image_id] = int(probs.argmax())
        weights = probs / probs.max() if relative else probs
        scores = out['scores'] * weights[out['labels'] - 1] ** rescore_alpha if rescore_alpha else out['scores']
        order = scores.argsort(descending=True)[:keep]
        boxes = to_original_xywh(out['boxes'][order], size).tolist()
        detections += [{'image_id': image_id, 'category_id': c, 'bbox': b, 'score': s}
                       for b, c, s in zip(boxes, out['labels'][order].tolist(), scores[order].tolist())]
    return labels, detections


def predict(model, dataset, device='cuda', batch_size=8, workers=4, rescore_alpha=0.0, keep=100, relative=False):
    """-> ({image id: label 0..8}, COCO detections in original pixels with category 1..9)."""
    return to_submission(predict_raw(model, dataset, device, batch_size, workers), rescore_alpha, keep, relative)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='a run folder of train.py (metrics.json and model.pt)')
    p.add_argument('--images', type=Path, required=True, help='images.json of the split to predict')
    p.add_argument('--out', type=Path, default=Path('submission.zip'))
    p.add_argument('--weights', default='model.pt',
                   help="file in the run folder: model.pt (last epoch), model_ema.pt (its averaged weights, "
                        "with --ema-decay) or best.pt (best validated Δm)")
    p.add_argument('--score-thresh', type=float, help="override the run's detection score threshold")
    p.add_argument('--nms-thresh', type=float,
                   help="override the box head's per-class NMS IoU threshold (training's default: 0.5)")
    p.add_argument('--rescore-alpha', type=float, default=0.0,
                   help="multiply each box's score by the class head's probability of the box's class to this power "
                        "(0: off); tuned on the holdout with sweep_rescore.py")
    p.add_argument('--rescore-mode', default='relative', choices=['relative', 'absolute'],
                   help="relative: the box's class probability divided by the image's highest one, so boxes of the "
                        "predicted class keep their scores (holdout: +0.86 delta_m at alpha 4); absolute: the "
                        "probability itself (holdout: worse at every alpha)")
    p.add_argument('--candidates', type=int, default=100,
                   help='boxes per image taken from the detector before rescoring; the best 100 are kept')
    p.add_argument('--gt', type=Path, help='COCO ground truth of the same images: also print the metrics')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                   help='cpu: predict without the GPU, in fp32 (e.g. while a training run holds the GPU memory)')
    p.add_argument('--threads', type=int, help='CPU threads for --device cpu (default: torch default)')
    args = p.parse_args()

    model = load_run(args.run, args.weights, args.device, args.threads)
    if args.score_thresh is not None:
        model.roi_heads.score_thresh = args.score_thresh
    if args.nms_thresh is not None:
        model.roi_heads.nms_thresh = args.nms_thresh
    model.roi_heads.detections_per_img = args.candidates
    dataset = ImagesToPredict(args.images)
    labels, detections = predict(model, dataset, args.device, args.batch_size, args.workers, args.rescore_alpha,
                                 relative=args.rescore_mode == 'relative')

    table = io.StringIO()
    writer = csv.writer(table, lineterminator='\n')
    writer.writerow(['image_id', 'label'])
    writer.writerows(sorted(labels.items()))
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('classification.csv', table.getvalue())
        z.writestr('detection.json', json.dumps(detections))
    print(f'{args.out}: {len(labels)} images, {len(detections)} detections')

    if args.gt:
        gt = json.loads(args.gt.read_text())
        truth = {a['image_id']: a['category_id'] - 1 for a in gt['annotations']}  # every box has the image's class
        ids = [im['id'] for im in dataset.images]
        metrics = {**classification_metrics([truth[i] for i in ids], [labels[i] for i in ids]),
                   **detection_metrics(args.gt, detections, ids)}
        print(headline(metrics))


if __name__ == '__main__':
    main()
