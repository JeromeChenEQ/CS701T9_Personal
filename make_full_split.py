"""The 'full' training split: our train split plus the holdout (all 9,392 train images), for the final retrain
(train.py --train-split full). Merges splits/classification/{train,val}.csv and splits/detection/
instances_{train,val}.json into the 'full' files, in splits/ and in the data folder."""
import csv
import json
import shutil
from pathlib import Path

SPLITS = Path(__file__).resolve().parent / 'splits'
DATA = Path(__file__).resolve().parent / 'cs701-sar-course-data'

rows = []
for name in ('train', 'val'):
    with open(SPLITS / 'classification' / f'{name}.csv', newline='') as f:
        rows += list(csv.DictReader(f))
with open(SPLITS / 'classification' / 'full.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image', 'label'])
    writer.writeheader()
    writer.writerows(rows)

coco = [json.loads((SPLITS / 'detection' / f'instances_{name}.json').read_text()) for name in ('train', 'val')]
full = {'images': coco[0]['images'] + coco[1]['images'], 'annotations': coco[0]['annotations'] + coco[1]['annotations'],
        'categories': coco[0]['categories']}
assert len({im['id'] for im in full['images']}) == len(full['images']) == len(rows), 'duplicate or missing images'
assert len({a['id'] for a in full['annotations']}) == len(full['annotations']), 'duplicate annotation ids'
(SPLITS / 'detection' / 'instances_full.json').write_text(json.dumps(full))

for kind, file in (('classification', 'full.csv'), ('detection', 'instances_full.json')):
    shutil.copy(SPLITS / kind / file, DATA / kind / file)
print(f"full split: {len(rows)} images, {len(full['annotations'])} boxes -> splits/ and {DATA.name}/")
