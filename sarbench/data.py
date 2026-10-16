"""The course data in TA Zichen's folder layout (see README): every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

# Mean and std of all pixels of the 9,392 train images, read as gray, resized to
# 512 x 512 and scaled to [0, 1] (computed once).
MEAN, STD = 0.2526, 0.2133


class SARMultiTask(Dataset):
    """The images of classification/{split}.csv with their boxes from detection/instances_{split}.json.
    An item is (image [1, size, size], label 0..8, target), target = {'boxes': XYXY on the resized
    image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}."""

    def __init__(self, root, split, train=False, size=512, limit=None, rot90=False, intensity_jitter=False,
                 rotate_max=0.0, scale_jitter=0.0):
        self.root = Path(root)
        self.size = size
        self.rot90 = train and rot90  # random multiple of 90 degrees, training only
        with open(self.root / 'classification' / f'{split}.csv') as f:
            self.rows = list(csv.DictReader(f))[:limit]
        coco = json.loads((self.root / 'detection' / f'instances_{split}.json').read_text())
        self.coco_images = {img['file_name']: img for img in coco['images']}  # key = CSV 'image'
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []
        jitter = [v2.Lambda(jitter_intensity, tv_tensors.Image)] if train and intensity_jitter else []
        # Training only, both keep the size x size square; boxes are clamped to the image afterwards, and boxes
        # left with no area are dropped with their labels.
        # Rotation: random angle in [-rotate_max, rotate_max] degrees, corners cut and filled with 0; a box becomes
        # the axis-aligned box around its rotated corners.
        # Scale jitter: zoom about the centre by a random factor in [1 - scale_jitter, 1 + scale_jitter]; zooming
        # in crops the border, zooming out pads it with 0. Boxes scale exactly with the pixels.
        geometric = []
        if train and rotate_max > 0:
            geometric.append(v2.RandomRotation(rotate_max, interpolation=v2.InterpolationMode.BILINEAR))
        if train and scale_jitter > 0:
            geometric.append(v2.RandomAffine(degrees=0, scale=(1 - scale_jitter, 1 + scale_jitter),
                                             interpolation=v2.InterpolationMode.BILINEAR))
        if geometric:
            geometric += [v2.ClampBoundingBoxes(), v2.SanitizeBoundingBoxes()]
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            v2.Resize((size, size)),  # stretches non-square images; boxes are scaled with the pixels
            *flips,  # overhead imagery has no canonical "up"
            *geometric,
            v2.ToDtype(torch.float32, scale=True),
            *jitter,  # on [0, 1] intensities, before normalising
            v2.Normalize([MEAN], [STD]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        info = self.coco_images[row['image']]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image = Image.open(self.root / row['image']).convert('L')  # PIL: torchvision.io cannot read .bmp
        labels = torch.tensor([a['category_id'] for a in anns])
        # a dict, so SanitizeBoundingBoxes (rotation, scale jitter) finds the labels of the boxes it drops
        sample = self.transform({'image': image, 'boxes': boxes, 'labels': labels})
        image, boxes, labels = sample['image'], sample['boxes'], sample['labels']
        if self.rot90:  # with the two flips: all 8 orientations of the square
            image, boxes = rotate90(image, boxes, int(torch.randint(4, ())), self.size)
        target = {'boxes': boxes, 'labels': labels, 'image_id': info['id'], 'orig_size': orig_size}
        return image, int(row['label']), target


def jitter_intensity(image, gamma=1.25, contrast=0.1, brightness=0.05):
    """Mild random intensity change of a [0, 1] image: the four source datasets differ in brightness and
    calibration. Gamma in [1/gamma, gamma] (log-uniform), contrast about the mean by a factor in
    [1 - contrast, 1 + contrast], then a brightness shift in [-brightness, brightness]."""
    u = torch.rand(3) * 2 - 1  # three independent draws in [-1, 1]
    out = image.clamp(0, 1) ** (gamma ** u[0])
    mean = out.mean()
    out = (out - mean) * (1 + contrast * u[1]) + mean + brightness * u[2]
    return tv_tensors.wrap(out.clamp(0, 1), like=image)


def rotate90(image, boxes, k, size):
    """Rotate a [C, size, size] image and its XYXY boxes by k x 90 degrees. Exact: pixels move, none are
    interpolated. torch.rot90 moves pixel (x, y) to (y, size - x), so a box becomes (y1, size - x2, y2, size - x1)."""
    for _ in range(k):
        image = torch.rot90(image, 1, dims=(-2, -1))
        boxes = torch.stack([boxes[:, 1], size - boxes[:, 2], boxes[:, 3], size - boxes[:, 0]], dim=1)
    return image, boxes


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
