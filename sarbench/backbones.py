"""The two ViT-B/16 backbones, built for 512 px input: images [B, 1, 512, 512] -> patch tokens [B, 1024, 768];
and multi-scale (hierarchical) backbones such as SAM2's Hiera: images -> feature maps at strides 4 to 32."""
import timm
from terratorch.registry import BACKBONE_REGISTRY
from torch import nn


class ViT(nn.Module):
    """timm ViT-B/16 pretrained on ImageNet-21k."""

    def __init__(self, pretrained: bool):
        super().__init__()
        # in_chans=1: timm sums the RGB patch-embedding weights; img_size=512: it resamples the position embedding
        self.vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrained,
                                     img_size=512, in_chans=1, num_classes=0)
        self.embed_dim = 768

    @property
    def blocks(self):
        return self.vit.blocks

    def forward(self, x):
        tokens = self.vit.forward_features(x)  # [B, 1 + N, 768], final norm applied
        return tokens[:, self.vit.num_prefix_tokens:]  # drop the CLS token


class TerraMind(nn.Module):
    """TerraMind-1.0-base (IBM/ESA) through its Sentinel-1 GRD input."""

    def __init__(self, pretrained: bool):
        super().__init__()
        self.terramind = BACKBONE_REGISTRY.build('terratorch_terramind_v1_base', pretrained=pretrained,
                                                 modalities=['S1GRD'])
        self.embed_dim = 768

    @property
    def blocks(self):
        return self.terramind.encoder

    def forward(self, x):
        per_block = self.terramind({'S1GRD': x.repeat(1, 2, 1, 1)})  # gray image as both VV and VH
        return per_block[-1]  # last block's tokens [B, N, 768]; terratorch already applied encoder_norm


class Hierarchical(nn.Module):
    """A timm backbone with feature maps at strides 4, 8, 16 and 32 (features_only), e.g. SAM2's Hiera:
    images [B, 1, 512, 512] -> 4 maps [B, C_i, 512 / s_i, 512 / s_i]. No ViT blocks: adapters go on every
    nn.Linear (sarbench.adapters.add_lora_everywhere), and the detector gets a standard FPN."""
    multiscale = True

    def __init__(self, model_name, pretrained: bool, drop_path=0.0, **kwargs):
        super().__init__()
        # in_chans=1: timm sums the RGB weights of the first conv. drop_path: stochastic depth, rising linearly
        # to this rate in the last block (training only; a regulariser even with the weights frozen). kwargs: extra
        # timm options, e.g. img_size for Swin
        self.net = timm.create_model(model_name, pretrained=pretrained, features_only=True, in_chans=1,
                                     drop_path_rate=drop_path, **kwargs)
        # Swin's maps come channels-last [B, H, W, C]; the FPN takes [B, C, H, W]
        self.channels_last = str(getattr(self.net, 'output_fmt', 'NCHW')).endswith('NHWC')
        self.channels = self.net.feature_info.channels()
        self.embed_dim = self.channels[-1]
        # ConvNeXt (DINOv3): under bf16 autocast, cuDNN rejects its 7 x 7 depthwise convs (CUDNN_STATUS_BAD_PARAM,
        # torch 2.9 / cuDNN 9.10, RTX 3060) on the channels-last-strided maps the blocks pass on; a contiguous copy
        # of the input avoids it (and is faster than running those convs in fp32). No-op for SAM2.
        for name, module in self.net.named_modules():
            if name.endswith('conv_dw'):
                module.register_forward_pre_hook(lambda module, args: (args[0].contiguous(), *args[1:]))

    def forward(self, x):
        maps = self.net(x)
        return [m.permute(0, 3, 1, 2) for m in maps] if self.channels_last else maps


# The multi-scale backbones (strides 4-32), by --backbone name: timm model name and extra timm options
HIERARCHICAL = {
    'sam2': ('sam2_hiera_base_plus.fb_r896', {}),  # SAM2 Hiera-B+ (Meta), SA-1B / SA-V natural images and video
    'dinov3_convnext_b': ('convnext_base.dinov3_lvd1689m', {}),  # DINOv3 ConvNeXt-B (Meta), LVD-1689M web images
    # Swin-B (Microsoft), ImageNet-22k, pretrained at 384 px with 12 x 12 windows; img_size: built for 512 px input
    'swin_b': ('swin_base_patch4_window12_384.ms_in22k', {'img_size': 512}),
}


def build_backbone(name, pretrained, drop_path=0.0):
    """name: 'vit', 'terramind', or a multi-scale backbone of HIERARCHICAL. pretrained=False builds the same
    architecture with random weights. drop_path: stochastic depth rate, multi-scale backbones only."""
    if name in HIERARCHICAL:
        model_name, kwargs = HIERARCHICAL[name]
        return Hierarchical(model_name, pretrained, drop_path, **kwargs)
    return {'vit': ViT, 'terramind': TerraMind}[name](pretrained)
