import csv, json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'cs701-sar-course-data'
PREFIX = 'train/'        # e.g. 'train/images/' if images sit in a subfolder
HOLDOUT = 0.15

rows = list(csv.DictReader(open(ROOT / 'train' / 'labels.csv')))
coco = json.load(open(ROOT / 'train' / 'instances.json'))
by_id = {im['id']: im for im in coco['images']}
anns = defaultdict(list)
for a in coco['annotations']:
    anns[a['image_id']].append(a)

random.seed(0)                      # fixed split: keep it for the whole project
by_label = defaultdict(list)
for r in rows:
    by_label[r['label']].append(r)
split = {'train': [], 'val': []}
for rs in by_label.values():        # stratified by class
    random.shuffle(rs)
    k = round(len(rs) * HOLDOUT)
    split['val'] += rs[:k]
    split['train'] += rs[k:]

(ROOT / 'classification').mkdir(exist_ok=True)
(ROOT / 'detection').mkdir(exist_ok=True)
for name, rs in split.items():
    with open(ROOT / 'classification' / f'{name}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'label'])
        for r in rs:
            w.writerow([PREFIX + r['file_name'], r['label']])
    images, annotations = [], []
    for r in rs:
        im = dict(by_id[int(r['image_id'])])
        im['file_name'] = PREFIX + r['file_name']
        images.append(im)
        annotations += anns[im['id']]
    json.dump({'images': images, 'annotations': annotations,
               'categories': coco.get('categories', [])},
              open(ROOT / 'detection' / f'instances_{name}.json', 'w'))

# No labelled test split: train.py skips the test pass when classification/test.csv is missing.
# Remove the placeholder copies of the holdout that earlier versions of this script wrote.
(ROOT / 'classification' / 'test.csv').unlink(missing_ok=True)
(ROOT / 'detection' / 'instances_test.json').unlink(missing_ok=True)
print({k: len(v) for k, v in split.items()})
