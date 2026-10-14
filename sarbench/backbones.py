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

    def __init__(self, model_name, pretrained: bool):
        super().__init__()
        # in_chans=1: timm sums the RGB weights of the first conv
        self.net = timm.create_model(model_name, pretrained=pretrained, features_only=True, in_chans=1)
        self.channels = self.net.feature_info.channels()
        self.embed_dim = self.channels[-1]

    def forward(self, x):
        return self.net(x)


def build_backbone(name, pretrained):
    """name: 'vit', 'terramind' or 'sam2' (SAM2 Hiera-B+, Meta). pretrained=False builds the same architecture
    with random weights."""
    if name == 'sam2':
        return Hierarchical('sam2_hiera_base_plus.fb_r896', pretrained)
    return {'vit': ViT, 'terramind': TerraMind}[name](pretrained)
