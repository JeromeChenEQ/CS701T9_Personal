"""The multi-task model: one shared backbone (a ViT, or a multi-scale one such as SAM2's Hiera) feeds a
classification head and a Faster R-CNN detector."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler, BoxCoder, Matcher
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import (FeaturePyramidNetwork, MultiScaleRoIAlign, batched_nms, box_iou, clip_boxes_to_image,
                             generalized_box_iou_loss, remove_small_boxes)
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .adapters import set_task


class GIoURoIHeads(RoIHeads):
    """torchvision's RoIHeads with a GIoU loss (Rezatofighi et al., CVPR 2019) on the decoded boxes in place of
    the smooth L1 loss on the encoded box offsets: COCO mAP averages IoU 0.50-0.95, and GIoU trains the IoU
    itself. As mmdetection's GIoU Faster R-CNN: loss weight 10, normalised like torchvision's box loss (sum over
    the positive RoIs / all sampled RoIs). Inference is unchanged."""

    def __init__(self, *args, giou_weight=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.giou_weight = giou_weight

    def forward(self, features, proposals, image_shapes, targets=None):
        if not self.training:
            return super().forward(features, proposals, image_shapes, targets)
        proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        labels, regression_targets, proposals = torch.cat(labels), torch.cat(regression_targets), torch.cat(proposals)
        loss_classifier = F.cross_entropy(class_logits, labels)
        positive = torch.where(labels > 0)[0]
        # the offsets predicted for each positive RoI's ground-truth class; both they and the regression targets
        # are relative to the RoI, so decoding both with the RoI gives the predicted and the ground-truth box
        offsets = box_regression.reshape(len(labels), -1, 4)[positive, labels[positive]]
        predicted = self.box_coder.decode_single(offsets, proposals[positive])
        truth = self.box_coder.decode_single(regression_targets[positive], proposals[positive])
        loss_box = self.giou_weight * generalized_box_iou_loss(predicted, truth, reduction='sum') / labels.numel()
        return [], {'loss_classifier': loss_classifier, 'loss_box_reg': loss_box}


class CascadeRoIHeads(nn.Module):
    """Cascade R-CNN box heads (Cai & Vasconcelos, CVPR 2018), in place of torchvision's single RoIHeads: three
    stages trained at rising IoU thresholds (0.5, 0.6, 0.7), each refining the boxes of the stage before, for
    tighter boxes at the high IoU thresholds of COCO mAP. As Detectron2 / mmdetection: class-agnostic box
    regression, box coder weights (10, 10, 5, 5), (20, 20, 10, 10), (30, 30, 15, 15), stage loss weights 1, 0.5,
    0.25; in training each stage samples 512 RoIs (25 % positive) from the previous stage's refined boxes plus the
    ground truth. Inference: class scores averaged over the three stages, boxes from the last stage, then
    torchvision's per-class NMS. width: of each stage's two-layer box head (torchvision's single head: 1024)."""

    def __init__(self, box_roi_pool, num_classes, width=512, box_loss='smooth_l1', giou_weight=10.0,
                 score_thresh=0.05, nms_thresh=0.5, detections_per_img=100, iou_thresholds=(0.5, 0.6, 0.7),
                 stage_weights=(1.0, 0.5, 0.25), reg_weights=((10., 10., 5., 5.), (20., 20., 10., 10.), (30., 30., 15., 15.)),
                 batch_size_per_image=512, positive_fraction=0.25):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        size = box_roi_pool.output_size[0] * box_roi_pool.output_size[1]
        self.box_heads = nn.ModuleList(TwoMLPHead(256 * size, width) for _ in iou_thresholds)
        self.cls_scores = nn.ModuleList(nn.Linear(width, num_classes + 1) for _ in iou_thresholds)  # + 1: background
        self.bbox_preds = nn.ModuleList(nn.Linear(width, 4) for _ in iou_thresholds)  # class-agnostic
        self.matchers = [Matcher(t, t, allow_low_quality_matches=False) for t in iou_thresholds]
        self.box_coders = [BoxCoder(w) for w in reg_weights]
        self.sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.stage_weights = stage_weights
        self.box_loss, self.giou_weight = box_loss, giou_weight
        self.score_thresh, self.nms_thresh, self.detections_per_img = score_thresh, nms_thresh, detections_per_img

    def sample(self, stage, proposals, targets):
        """Per image: the proposals plus the ground-truth boxes, matched at this stage's IoU threshold and sampled
        -> (RoIs, labels 0..num_classes with 0 = background, the matched ground-truth box of each RoI)."""
        rois, labels, truth = [], [], []
        for boxes, target in zip(proposals, targets):
            gt = target['boxes']
            boxes = torch.cat([boxes, gt])
            if len(gt) == 0:  # an image whose boxes all left the image under augmentation: only background
                rois.append(boxes), labels.append(torch.zeros(len(boxes), dtype=torch.int64, device=boxes.device))
                truth.append(torch.zeros_like(boxes))
                continue
            match = self.matchers[stage](box_iou(gt, boxes))
            label = target['labels'][match.clamp(min=0)].clone()
            label[match == Matcher.BELOW_LOW_THRESHOLD] = 0
            rois.append(boxes), labels.append(label), truth.append(gt[match.clamp(min=0)])
        positive, negative = self.sampler(labels)
        keep = [torch.where(p | n)[0] for p, n in zip(positive, negative)]
        return ([r[k] for r, k in zip(rois, keep)], [lab[k] for lab, k in zip(labels, keep)],
                [t[k] for t, k in zip(truth, keep)])

    def run_stage(self, stage, features, rois, image_shapes):
        """-> class logits [N, num_classes + 1], refined boxes [N, 4] (clipped to the image), box offsets [N, 4]."""
        x = self.box_heads[stage](self.box_roi_pool(features, rois, image_shapes))
        offsets = self.bbox_preds[stage](x)
        boxes = self.box_coders[stage].decode_single(offsets, torch.cat(rois))
        boxes = torch.cat([clip_boxes_to_image(b, size) for b, size in
                           zip(boxes.split([len(r) for r in rois]), image_shapes)])
        return self.cls_scores[stage](x), boxes, offsets

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            losses = {'loss_classifier': 0.0, 'loss_box_reg': 0.0}
            for stage, weight in enumerate(self.stage_weights):
                rois, labels, truth = self.sample(stage, proposals, targets)
                logits, boxes, offsets = self.run_stage(stage, features, rois, image_shapes)
                rois_all, labels, truth = torch.cat(rois), torch.cat(labels), torch.cat(truth)
                positive = labels > 0
                if self.box_loss == 'giou':
                    predicted = self.box_coders[stage].decode_single(offsets[positive], rois_all[positive])
                    box = self.giou_weight * generalized_box_iou_loss(predicted, truth[positive], reduction='sum')
                else:  # torchvision's smooth L1 on the encoded offsets
                    target = self.box_coders[stage].encode_single(truth[positive], rois_all[positive])
                    box = F.smooth_l1_loss(offsets[positive], target, beta=1 / 9, reduction='sum')
                losses['loss_classifier'] = losses['loss_classifier'] + weight * F.cross_entropy(logits, labels)
                losses['loss_box_reg'] = losses['loss_box_reg'] + weight * box / labels.numel()
                proposals = [b.detach() for b in boxes.split([len(r) for r in rois])]  # the next stage's RoIs
            return [], losses
        rois, scores = proposals, 0.0
        for stage in range(len(self.box_heads)):
            logits, boxes, _ = self.run_stage(stage, features, rois, image_shapes)
            scores = scores + logits.softmax(dim=-1) / len(self.box_heads)
            rois = list(boxes.split([len(r) for r in rois]))
        return self.postprocess(rois, scores.split([len(r) for r in rois])), {}

    def postprocess(self, boxes, scores):
        """As torchvision's postprocess_detections, with one (class-agnostic) box per RoI for every class."""
        result = []
        for b, s in zip(boxes, scores):
            num_classes = s.shape[1] - 1
            b = b[:, None].expand(-1, num_classes, 4).reshape(-1, 4)
            labels = torch.arange(1, num_classes + 1, device=s.device).repeat(len(s))
            s = s[:, 1:].reshape(-1)  # drop background
            keep = torch.where(s > self.score_thresh)[0]
            b, s, labels = b[keep], s[keep], labels[keep]
            keep = remove_small_boxes(b, min_size=1e-2)
            b, s, labels = b[keep], s[keep], labels[keep]
            keep = batched_nms(b, s, labels, self.nms_thresh)[:self.detections_per_img]
            result.append({'boxes': b[keep], 'scores': s[keep], 'labels': labels[keep]})
        return result


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the single stride-16 ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64."""

    def __init__(self, dim=768, channels=256):
        super().__init__()
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GroupNorm(32, dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),  # stride 4
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 8
            nn.Identity(),  # stride 16
            nn.MaxPool2d(2),  # stride 32
        ])
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.GroupNorm(32, channels),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels))
            for c in (dim // 4, dim // 2, dim, dim))

    def forward(self, x):
        levels = [project(rescale(x)) for rescale, project in zip(self.rescale, self.project)]
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


def conv_gn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(16, cout),
                         nn.GELU())


class DetailBranch(nn.Module):
    """Light conv branch on the raw image (after the spatial prior module of ViT-Adapter, Chen et al., 2023):
    pixel detail the 16 px ViT tokens lose, added to the stride-4 and stride-8 pyramid levels. The output
    projections start at 0, so training starts from the model without the branch."""

    def __init__(self, channels=256):
        super().__init__()
        self.stem = nn.Sequential(conv_gn(1, 32, stride=2), conv_gn(32, 32), conv_gn(32, 64), nn.MaxPool2d(2),
                                  conv_gn(64, 128))  # stride 4
        self.down = conv_gn(128, 256, stride=2)  # stride 8
        self.out4, self.out8 = nn.Conv2d(128, channels, 1), nn.Conv2d(256, channels, 1)
        for conv in (self.out4, self.out8):
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, images):
        c4 = self.stem(images)
        return self.out4(c4), self.out8(self.down(c4))


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear, optionally plus the
    detector's per-class evidence) and Faster R-CNN detections."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, loss_weighting='sum', det_fusion=False,
                 detail_branch=False, score_thresh=0.05, box_loss='smooth_l1', detector='faster_rcnn',
                 cascade_width=512):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        # Kendall et al. (CVPR 2018): one learned log-variance s_t per task (0 = classification, 1 = detection);
        # s = 0 at the start, so training starts from the plain sum of the losses
        self.log_vars = nn.Parameter(torch.zeros(2)) if loss_weighting == 'kendall' else None
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        # Detection -> classification fusion: every box of an image has the image's class, so the detector's
        # best score for class c is evidence for label c. logits_c += w_c * log p_c, with p_c the highest class-c
        # probability of the box head over the top proposals; w = 0 at the start, i.e. the plain class head.
        self.fusion_weights = nn.Parameter(torch.zeros(num_classes)) if det_fusion else None
        self.fusion_proposals = 100  # top RPN proposals per image scored for the evidence
        self.multiscale = getattr(backbone, 'multiscale', False)
        # Multi-scale backbone (strides 4-32): a standard FPN, plus a max-pooled stride-64 level as the ViT neck has;
        # same levels '0'-'3', 'pool' either way, so anchors, RPN and RoI heads are unchanged
        self.neck = (FeaturePyramidNetwork(backbone.channels, 256, extra_blocks=LastLevelMaxPool()) if self.multiscale
                     else SimpleFeaturePyramid(backbone.embed_dim))
        self.detail = DetailBranch() if detail_branch else None
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location.
        anchors = AnchorGenerator(((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), ((0.5, 1.0, 2.0),) * 5)
        # Everything else is torchvision's Faster R-CNN default.
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(256, 6), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000), nms_thresh=0.7)
        # box_loss: 'smooth_l1' (torchvision's default) or 'giou'. detector: 'faster_rcnn' (one box head) or
        # 'cascade' (three, each cascade_width wide)
        roi_pool = MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        if detector == 'cascade':
            self.roi_heads = CascadeRoIHeads(roi_pool, num_classes, width=cascade_width, box_loss=box_loss,
                                             score_thresh=score_thresh)
        else:
            self.roi_heads = {'smooth_l1': RoIHeads, 'giou': GIoURoIHeads}[box_loss](
                roi_pool, TwoMLPHead(256 * 7 * 7, 1024), FastRCNNPredictor(1024, num_classes + 1),  # + 1: background
                fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
                bbox_reg_weights=None, score_thresh=score_thresh, nms_thresh=0.5, detections_per_img=100)

    def forward(self, images, targets=None):
        """images [B, 1, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the four Faster R-CNN losses (training mode))."""
        if self.multiscale:  # 4 maps at strides 4-32: classify the mean of the last, detect on all through the FPN
            maps = self.backbone(images)
            logits = self.cls_head(maps[-1].mean(dim=(2, 3)))
            features = self.neck(OrderedDict((str(i), m) for i, m in enumerate(maps)))
        else:
            set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
            tokens = self.backbone(images)
            logits = self.cls_head(tokens.mean(dim=1))
            if self.task_routing:
                set_task(self.backbone, 1)
                tokens = self.backbone(images)
            h, w = images.shape[-2] // 16, images.shape[-1] // 16  # one token per 16 x 16 patch
            features = self.neck(tokens.transpose(1, 2).unflatten(2, (h, w)))  # tokens as a [B, 768, h, w] map
        if self.detail is not None:
            detail4, detail8 = self.detail(images)
            features['0'] = features['0'] + detail4
            features['1'] = features['1'] + detail8
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            if self.fusion_weights is not None:
                logits = logits.float() + self.fusion_weights * self.detection_evidence(features, proposals,
                                                                                       image_list.image_sizes)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}

    def detection_evidence(self, features, proposals, image_sizes):
        """Per image and class c: log of the highest class-c probability the box head gives to any of the top
        RPN proposals -> [B, num_classes]. Detached: the classification loss trains only the fusion weights here,
        never the detector."""
        with torch.no_grad():
            # RPN proposals come sorted by objectness, best first
            top = [p[:self.fusion_proposals] for p in proposals]
            roi = self.roi_heads.box_head(self.roi_heads.box_roi_pool(features, top, image_sizes))
            probs = self.roi_heads.box_predictor(roi)[0].softmax(dim=-1)[:, 1:]  # drop the background column
            best = [p.max(dim=0).values if len(p) else probs.new_zeros(probs.shape[1])
                    for p in probs.split([len(t) for t in top])]
            return torch.stack(best).clamp_min(1e-4).log()  # log p in [-9.2, 0]

    def total_loss(self, losses):
        """losses: 'classification' plus the four Faster R-CNN losses -> the training loss. Plain sum, or with
        Kendall weighting sum_t exp(-s_t) L_t + s_t, where L_1 is the sum of the four detection losses."""
        cls = losses['classification']
        det = sum(value for name, value in losses.items() if name != 'classification')
        if self.log_vars is None:
            return cls + det
        s = self.log_vars.float()
        return torch.exp(-s[0]) * cls + torch.exp(-s[1]) * det + s.sum()

    def task_weights(self):
        """The current weights exp(-s_t) of the classification and detection losses (1, 1 without Kendall)."""
        if self.log_vars is None:
            return 1.0, 1.0
        return tuple(torch.exp(-self.log_vars.detach()).tolist())
