"""Check a submission.zip against an images.json before uploading it to Codabench: format errors first, then a
sanity summary of the predictions (no labels needed). Several zips: also how often their class labels agree.
Example: python check_submission.py ~/cs701-sar-course-data/val/images.json submission_val.zip [other.zip ...]"""
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

NAMES = ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine']


def check(images, path):
    """Format errors of one zip (empty list = valid) and its class labels per image."""
    errors = []
    z = zipfile.ZipFile(path)
    if sorted(z.namelist()) != ['classification.csv', 'detection.json']:
        errors.append(f'zip must hold exactly classification.csv and detection.json, has {z.namelist()}')
        return errors, {}, []
    rows = list(csv.reader(io.StringIO(z.read('classification.csv').decode())))
    if rows[0] != ['image_id', 'label']:
        errors.append(f'csv header must be image_id,label, is {rows[0]}')
    labels = {}
    for row in rows[1:]:
        image_id, label = int(row[0]), int(row[1])
        if image_id in labels:
            errors.append(f'image {image_id} listed twice in classification.csv')
        if not 0 <= label <= 8:
            errors.append(f'image {image_id}: label {label} outside 0..8')
        labels[image_id] = label
    if set(labels) != set(images):
        errors.append(f'classification.csv: {len(set(images) - set(labels))} images missing, '
                      f'{len(set(labels) - set(images))} unknown ids')
    detections = json.loads(z.read('detection.json'))
    for d in detections:
        if set(d) != {'image_id', 'category_id', 'bbox', 'score'}:
            errors.append(f'detection with keys {sorted(d)}')
            break
        im = images.get(d['image_id'])
        x, y, w, h = d['bbox']
        if im is None:
            errors.append(f"detection for unknown image {d['image_id']}")
        elif not (1 <= d['category_id'] <= 9 and 0 <= d['score'] <= 1 and w > 0 and h > 0
                  and x >= -1 and y >= -1 and x + w <= im['width'] + 1 and y + h <= im['height'] + 1):
            errors.append(f'bad detection {d}')
        if len(errors) > 20:
            break
    return errors, labels, detections


def main():
    images_json, zips = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    data = json.loads(images_json.read_text())
    images = {im['id']: im for im in (data['images'] if isinstance(data, dict) else data)}
    all_labels = {}
    for path in zips:
        errors, labels, detections = check(images, path)
        print(f'== {path} ({path.stat().st_size / 2**20:.1f} MB)')
        if errors:
            print('INVALID:\n  ' + '\n  '.join(errors[:20]))
            continue
        print(f'valid: {len(labels)} images, {len(detections)} detections')
        all_labels[path.name] = labels
        counts = Counter(labels.values())
        print('predicted classes: ' + ', '.join(f'{NAMES[c]} {counts[c]}' for c in range(9)))
        per_image = Counter(d['image_id'] for d in detections)
        confident = Counter(d['image_id'] for d in detections if d['score'] >= 0.5)
        print(f'detections per image: min {min(per_image.values(), default=0)}, max {max(per_image.values())}; '
              f'images without any: {len(images) - len(per_image)}; '
              f'without one scoring >= 0.5: {len(images) - len(confident)}')
        best = {}
        for d in detections:  # every box of an image has the image's class: does the top box agree?
            if d['score'] > best.get(d['image_id'], (0, None))[0]:
                best[d['image_id']] = (d['score'], d['category_id'] - 1)
        agree = sum(best[i][1] == labels[i] for i in best) / max(len(best), 1)
        print(f'top-scoring box has the predicted image class: {agree * 100:.1f} % of images')
    names = list(all_labels)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = all_labels[names[i]], all_labels[names[j]]
            print(f'{names[i]} vs {names[j]}: same class label on {sum(a[k] == b[k] for k in a) / len(a) * 100:.1f} %')


if __name__ == '__main__':
    main()
