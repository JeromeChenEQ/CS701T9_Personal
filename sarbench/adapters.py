"""LoRA and MoE-LoRA adapters for the attention layers of a frozen backbone."""
from torch import nn


class LoRA(nn.Module):
    """y = W x + (alpha / r) * B A x, where W is the wrapped layer (frozen by add_adapters)."""

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)  # A
        self.up = nn.Linear(rank, base.out_features, bias=False)  # B
        nn.init.zeros_(self.up.weight)  # B = 0: training starts from exactly the pretrained layer
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


class MoELoRA(nn.Module):
    """y = W x + (alpha / r) * sum_e g_e(x) B_e A_e x, with gates g(x) = softmax(router[task](x)) per token.

    The experts' A_e are stacked in `down` and their B_e side by side in `up`, so gating each
    expert's r hidden units and applying `up` sums the experts.
    """

    def __init__(self, base: nn.Linear, experts=4, rank=4, alpha=8, tasks=2):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, experts * rank, bias=False)  # A_1 ... A_E
        self.up = nn.Linear(experts * rank, base.out_features, bias=False)  # B_1 ... B_E
        nn.init.zeros_(self.up.weight)
        self.routers = nn.ModuleList(nn.Linear(base.in_features, experts, bias=False) for _ in range(tasks))
        self.rank = rank
        self.scale = alpha / rank
        self.task = 0  # which router to use; see set_task

    def forward(self, x):
        gates = self.routers[self.task](x).softmax(dim=-1)
        gates = gates.repeat_interleave(self.rank, dim=-1)  # expert e's gate on each of its r hidden units
        return self.base(x) + self.scale * self.up(self.down(x) * gates)


def add_lora_everywhere(backbone, rank):
    """Freeze the backbone, then wrap every nn.Linear in it with LoRA of `rank`, alpha = 2 * rank (the same
    scale 2 as the default rank-16 LoRA). For backbones without ViT blocks, e.g. the hierarchical SAM2 Hiera."""
    backbone.requires_grad_(False)
    targets = [(parent, name, child) for parent in backbone.modules()
               for name, child in parent.named_children() if isinstance(child, nn.Linear)]
    for parent, name, child in targets:  # collected first, so no LoRA wraps another LoRA's layers
        setattr(parent, name, LoRA(child, rank=rank, alpha=2 * rank))


def depthwise_and_norm_params(backbone):
    """The backbone's small per-channel parameters that LoRA on the Linear layers leaves frozen: depthwise
    convolutions (weights and biases; ConvNeXt's 7 x 7 token mixers), LayerNorms (weights and biases) and layer
    scales (`gamma`). DINOv3 ConvNeXt-B: 958,592 values, against 83.8 M in its Linear layers."""
    params = []
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d) and module.groups > 1 and module.groups == module.in_channels:
            params += list(module.parameters(recurse=False))
        elif isinstance(module, nn.LayerNorm):  # timm's LayerNorm2d is a LayerNorm too
            params += list(module.parameters(recurse=False))
        else:
            params += [p for name, p in module.named_parameters(recurse=False) if name == 'gamma']
    return params


def add_adapters(backbone, kind, mlp=False, unfreeze_last=0, rank=16):
    """Freeze the backbone, then wrap attn.qkv and attn.proj of every block (kind: 'lora' or 'moelora'); with
    mlp=True also every Linear of the block's MLP (ViT: fc1, fc2; TerraMind's gated MLP: fc1, fc2, fc3).
    unfreeze_last=N: the last N blocks get no adapters and are trained in full instead (partial fine-tuning).
    rank: of the plain LoRA, with alpha = 2 * rank (default 16, alpha 32); MoE-LoRA keeps its 4 experts x rank 4."""
    adapter = {'lora': lambda base: LoRA(base, rank=rank, alpha=2 * rank), 'moelora': MoELoRA}[kind]
    backbone.requires_grad_(False)
    blocks = list(backbone.blocks)
    for blk in blocks[len(blocks) - unfreeze_last:]:
        blk.requires_grad_(True)
    for blk in blocks[:len(blocks) - unfreeze_last]:
        blk.attn.qkv = adapter(blk.attn.qkv)
        blk.attn.proj = adapter(blk.attn.proj)
        if mlp:
            for name, layer in list(blk.mlp.named_children()):
                if isinstance(layer, nn.Linear):
                    setattr(blk.mlp, name, adapter(layer))


def set_task(module, task):
    """Route every MoELoRA inside `module` through the router of `task` (0 = classification, 1 = detection)."""
    for layer in module.modules():
        if isinstance(layer, MoELoRA):
            layer.task = task
