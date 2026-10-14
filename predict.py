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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='a run folder of train.py (metrics.json and model.pt)')
    p.add_argument('--images', type=Path, required=True, help='images.json of the split to predict')
    p.add_argument('--out', type=Path, default=Path('submission.zip'))
    p.add_argument('--weights', default='model.pt',
                   help="file in the run folder: model.pt (last epoch), model_ema.pt (its averaged weights, "
                        "with --ema-decay) or best.pt (best validated Δm)")
    p.add_argument('--score-thresh', type=float, help="override the run's detection score threshold")
    p.add_argument('--gt', type=Path, help='COCO ground truth of the same images: also print the metrics')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    run_args = Namespace(**json.loads((args.run / 'metrics.json').read_text())['args'])
    model = build_model(run_args, pretrained=False)  # the weights come from --weights
    model.load_state_dict(torch.load(args.run / args.weights, map_location='cpu'))
    if args.score_thresh is not None:
        model.roi_heads.score_thresh = args.score_thresh
    model.cuda().eval()

    dataset = ImagesToPredict(args.images)
    loader = DataLoader(dataset, args.batch_size, num_workers=args.workers,
                        collate_fn=lambda batch: tuple(zip(*batch)))
    labels, detections = {}, []
    with torch.no_grad():
        for images, ids, sizes in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, outputs, _ = model(torch.stack(images).cuda())
            for image_id, size, label, out in zip(ids, sizes, logits.argmax(dim=1).tolist(), outputs):
                labels[image_id] = label  # 0..8
                boxes = to_original_xywh(out['boxes'], size).tolist()
                detections += [{'image_id': image_id, 'category_id': c, 'bbox': b, 'score': s}  # category 1..9
                               for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]

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
