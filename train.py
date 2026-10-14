"""Train one model jointly for classification and detection, then score it on val and test.
Example: python train.py --backbone vit --init pretrained --adapt lora"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.adapters import add_adapters, add_lora_everywhere
from sarbench.backbones import build_backbone
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MultiTaskModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--backbone', required=True, choices=['vit', 'terramind', 'sam2'],
                   help='sam2: SAM2 Hiera-B+ (multi-scale; needs --adapt full, or lora with --lora-targets all)')
    p.add_argument('--init', required=True, choices=['pretrained', 'scratch'])
    p.add_argument('--adapt', required=True, choices=['full', 'lora', 'moelora'])
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16, help='images per optimizer step')
    p.add_argument('--accum-steps', type=int, default=1,
                   help='split every batch into this many forward/backward passes (less GPU memory, same update)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--loss-weighting', default='sum', choices=['sum', 'kendall'],
                   help='sum: plain sum of the task losses; kendall: learned uncertainty weights (Kendall et al., 2018)')
    p.add_argument('--loss-weight-lr', type=float, default=1e-3,
                   help='for the two Kendall log-variances (no weight decay)')
    p.add_argument('--label-smoothing', type=float, default=0.0, help='for the classification cross-entropy')
    p.add_argument('--det-fusion', action='store_true',
                   help="add the detector's per-class scores to the classification logits (learned weight per class)")
    p.add_argument('--rot90', action='store_true', help='augment training images with random 90-degree rotations')
    p.add_argument('--intensity-jitter', action='store_true',
                   help='augment training images with mild random gamma, contrast and brightness changes')
    p.add_argument('--adapt-mlp', action='store_true',
                   help='LoRA / MoE-LoRA also on the MLP layers of every block (default: attention only)')
    p.add_argument('--detail-branch', action='store_true',
                   help='light conv branch on the raw image, added to the stride-4 and stride-8 detection features')
    p.add_argument('--score-thresh', type=float, default=0.05,
                   help='drop detections scoring below this (COCO mAP usually gains from a lower value, e.g. 0.001)')
    p.add_argument('--lora-rank', type=int, default=16, help='rank of the plain LoRA (alpha = 2 x rank)')
    p.add_argument('--lora-targets', default='attn', choices=['attn', 'all'],
                   help='attn: attn.qkv and attn.proj of every ViT block; all: every Linear layer of the backbone')
    p.add_argument('--unfreeze-last', type=int, default=0,
                   help='train the last N backbone blocks in full (no adapters on them); the others keep adapters')
    p.add_argument('--unfrozen-lr', type=float, default=2e-5,
                   help='for the unfrozen blocks (pretrained weights: gentler than the adapters, as in full fine-tuning)')
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N images of every split (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--resume', action='store_true',
                   help="continue from the run's checkpoint.pt (same options as the interrupted run), if there is one")
    args = p.parse_args()
    if args.init == 'scratch' and args.adapt != 'full':
        p.error('--init scratch is only valid with --adapt full')
    if (args.adapt_mlp or args.unfreeze_last) and args.adapt == 'full':
        p.error('--adapt-mlp and --unfreeze-last need --adapt lora or moelora')
    if args.lora_targets == 'all' and args.adapt != 'lora':
        p.error('--lora-targets all needs --adapt lora')
    if args.lora_targets == 'all' and (args.adapt_mlp or args.unfreeze_last):
        p.error('--lora-targets all already covers every layer: no --adapt-mlp or --unfreeze-last')
    if args.backbone == 'sam2' and (args.adapt == 'moelora' or (args.adapt == 'lora' and args.lora_targets != 'all')
                                    or args.detail_branch or args.det_fusion):
        p.error('--backbone sam2 needs --adapt full or --adapt lora --lora-targets all, without --detail-branch '
                'or --det-fusion')
    if args.batch_size % args.accum_steps:
        p.error('--batch-size must be a multiple of --accum-steps')
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    return args


def build_model(args, pretrained):
    """The model a run's options describe, on the CPU. Adapters are created on the CPU, so they are added before
    .cuda(). Also used by predict.py on the saved args of a run; options newer than that run get their defaults."""
    option = lambda name, default: getattr(args, name, default)  # noqa: E731
    backbone = build_backbone(args.backbone, pretrained=pretrained)
    if args.adapt != 'full':
        if option('lora_targets', 'attn') == 'all':
            add_lora_everywhere(backbone, option('lora_rank', 16))  # also freezes the backbone
        else:
            add_adapters(backbone, args.adapt, mlp=option('adapt_mlp', False),  # also freezes the backbone
                         unfreeze_last=option('unfreeze_last', 0), rank=option('lora_rank', 16))
    return MultiTaskModel(backbone, task_routing=args.adapt == 'moelora',
                          loss_weighting=option('loss_weighting', 'sum'), det_fusion=option('det_fusion', False),
                          detail_branch=option('detail_branch', False), score_thresh=option('score_thresh', 0.05))


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, split):
    """Micro-batches (--batch-size / --accum-steps images) of `split`; only the training set is shuffled and
    augmented."""
    train = split == 'train'
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit, rot90=args.rot90,
                           intensity_jitter=args.intensity_jitter)
    return DataLoader(dataset, args.batch_size // args.accum_steps, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0)  # training workers live across epochs


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file, accum_steps, label_smoothing):
    """One pass over the training set; the loss is cross-entropy plus the four Faster R-CNN losses. Gradients of
    `accum_steps` micro-batches are averaged before each optimizer step, which equals one step on their union."""
    model.train()
    start, seen = time.time(), 0
    optimizer.zero_grad()
    for step, (images, labels, targets) in enumerate(loader, 1):
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):  # bf16 has fp32's range: no loss scaling needed
            logits, _, det_losses = model(images.cuda(), targets)
            losses = {'classification': F.cross_entropy(logits, labels.cuda(), label_smoothing=label_smoothing),
                      **det_losses}
        loss = model.total_loss(losses)
        group_start = (step - 1) // accum_steps * accum_steps  # micro-batches already used in earlier groups
        (loss / min(accum_steps, len(loader) - group_start)).backward()  # the last group may be shorter
        if step % accum_steps == 0 or step == len(loader):
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            if model.log_vars is not None:  # the loss includes + s_t, so it can go below 0
                parts += ' weights cls {:.3f} det {:.3f}'.format(*model.task_weights())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} ({parts}) '
                f'lr {optimizer.param_groups[1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0


@torch.no_grad()
def evaluate(model, loader, gt_json):
    """Classification and COCO detection metrics on one split, and the detections in COCO result format."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, out in zip(targets, outputs):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size']).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50'))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = args.out / f'{args.backbone}_{args.init}_{args.adapt}'
    # Each option away from the reference adds a suffix, so variants sit next to, not over, the plain run
    suffixes = [args.loss_weighting] if args.loss_weighting != 'sum' else []
    suffixes += [f'ls{args.label_smoothing:g}'] if args.label_smoothing else []
    suffixes += ['detfusion'] if args.det_fusion else []
    suffixes += ['rot90'] if args.rot90 else []
    suffixes += ['jitter'] if args.intensity_jitter else []
    suffixes += ['mlp'] if args.adapt_mlp else []
    suffixes += ['detail'] if args.detail_branch else []
    suffixes += ['all'] if args.lora_targets == 'all' else []
    suffixes += [f'r{args.lora_rank}'] if args.adapt == 'lora' and args.lora_rank != 16 else []
    suffixes += [f'unfreeze{args.unfreeze_last}'] if args.unfreeze_last else []
    suffixes += [f'st{args.score_thresh:g}'] if args.score_thresh != 0.05 else []
    run_dir = run_dir.with_name('_'.join([run_dir.name, *suffixes]))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / 'checkpoint.pt'
    resuming = args.resume and checkpoint_path.exists()
    log_file = open(run_dir / 'log.txt', 'a' if resuming else 'w')  # a resumed run continues the same log
    log(json.dumps(vars(args), default=str), log_file)

    # Data
    train_loader, val_loader = (make_loader(args, split) for split in ('train', 'val'))
    has_test = (args.data / 'classification' / 'test.csv').exists()  # course data: no labelled test split
    test_loader = make_loader(args, 'test') if has_test else None

    model = build_model(args, pretrained=args.init == 'pretrained').cuda()
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr, all new modules at --lr, unfrozen
    # pretrained blocks at --unfrozen-lr
    unfrozen = set()
    if args.unfreeze_last:
        blocks = list(model.backbone.blocks)
        unfrozen = {id(p) for blk in blocks[len(blocks) - args.unfreeze_last:] for p in blk.parameters()}
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad and id(p) not in unfrozen]
    no_decay = ('log_vars', 'fusion_weights')  # learned scalars: weight decay would pull them back to the plain model
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.') and name not in no_decay]
    groups = [{'params': backbone_params, 'lr': args.backbone_lr}, {'params': head_params}]
    if unfrozen:
        groups.append({'params': [p for p in model.backbone.parameters() if id(p) in unfrozen],
                       'lr': args.unfrozen_lr})
    if model.log_vars is not None:
        groups.append({'params': [model.log_vars], 'lr': args.loss_weight_lr, 'weight_decay': 0.0})
    if model.fusion_weights is not None:
        groups.append({'params': [model.fusion_weights], 'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)  # optimizer steps, not micro-batches
    warmup, total = steps_per_epoch, args.epochs * steps_per_epoch  # one epoch of warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))

    # Train, validating every --eval-every epochs; the reported model is the last-epoch model. After every
    # validation the whole training state goes to checkpoint.pt, which --resume continues from.
    history, first_epoch, earlier_seconds = [], 1, 0.0
    if resuming:
        # On the CPU: the RNG states must stay CPU tensors; the weights and optimizer state are copied to the GPU
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['schedule'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        history, first_epoch, earlier_seconds = state['history'], state['epoch'] + 1, state['train_seconds']
        log(f"resumed from {checkpoint_path} after epoch {state['epoch']}", log_file)
    start = time.time()
    for epoch in range(first_epoch, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file, args.accum_steps,
                        args.label_smoothing)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = evaluate(model, val_loader, args.data / 'detection' / 'instances_val.json')
            history.append({'epoch': epoch, 'task_weights': model.task_weights(), **val})
            if model.fusion_weights is not None:
                history[-1]['fusion_weights'] = model.fusion_weights.detach().tolist()
                log('fusion weights: ' + ' '.join(f'{w:.3f}' for w in history[-1]['fusion_weights']), log_file)
            log(f'epoch {epoch} val: {headline(val)}', log_file)
            state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'schedule': schedule.state_dict(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state(), 'history': history,
                     'train_seconds': earlier_seconds + time.time() - start}
            torch.save(state, run_dir / 'checkpoint.tmp')
            (run_dir / 'checkpoint.tmp').replace(checkpoint_path)  # rename: a crash mid-save keeps the old one
    train_hours = (earlier_seconds + time.time() - start) / 3600
    if first_epoch > args.epochs:  # resumed after the last epoch: its validation detections are not saved
        val_detections = evaluate(model, val_loader, args.data / 'detection' / 'instances_val.json')[1]

    # Evaluate on test, once, if a labelled test split exists
    test, test_detections = None, None
    if has_test:
        test, test_detections = evaluate(model, test_loader, args.data / 'detection' / 'instances_test.json')
        log(f'test: {headline(test)}', log_file)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)

    # Save
    summary = {'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
               'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'val_history': history, 'val': history[-1], 'test': test}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    if has_test:
        (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
    torch.save(model.state_dict(), run_dir / 'model.pt')
    checkpoint_path.unlink(missing_ok=True)  # finished: model.pt and metrics.json replace it


if __name__ == '__main__':
    main()
