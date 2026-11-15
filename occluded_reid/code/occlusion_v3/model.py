"""OSNet with small, selectable spatial-context adapters."""
import torch
from torch import nn
from torch.nn import functional as F

from reid.model import OcclusionOSNet


VARIANTS = ('baseline', 'transformer', 'semantic_detail')


class SpatialTransformer(nn.Module):
    """One encoder block over OSNet's 16x8 feature tokens."""

    def __init__(self, channels=512, heads=8, mlp_ratio=2):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, 16 * 8, channels))
        nn.init.trunc_normal_(self.position, std=.02)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * mlp_ratio), nn.GELU(),
                                 nn.Linear(channels * mlp_ratio, channels))
        self.scale_attention = nn.Parameter(torch.full((channels,), .1))
        self.scale_mlp = nn.Parameter(torch.full((channels,), .1))

    def forward(self, feature_map):
        batch, channels, height, width = feature_map.shape
        if (height, width) != (16, 8):
            raise ValueError(f'Expected OSNet map 16x8, got {height}x{width}')
        tokens = feature_map.flatten(2).transpose(1, 2)
        positioned = self.norm1(tokens + self.position.to(tokens.dtype))
        context, _ = self.attention(positioned, positioned, positioned, need_weights=False)
        tokens = tokens + self.scale_attention * context
        tokens = tokens + self.scale_mlp * self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class SemanticDetailFusion(nn.Module):
    """Use a global semantic query to select and fuse high-frequency locations.

    Inspired by the high-frequency token selection in Li et al., ECCV 2024;
    this adapts its CLS-to-patch ranking to an OSNet feature map and a learned
    residual gate. It does not reproduce the paper's ViT/FFT augmentation.
    """

    def __init__(self, channels=512, hidden=128, keep_ratio=.5):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.query = nn.Linear(channels, hidden, bias=False)
        self.key = nn.Conv2d(channels, hidden, 1, bias=False)
        self.value = nn.Conv2d(channels, hidden, 1, bias=False)
        self.output = nn.Linear(hidden, channels, bias=False)
        self.gate = nn.Conv2d(channels * 2, 1, 1)
        self.residual_scale = nn.Parameter(torch.tensor(.1))

    def forward(self, feature_map):
        height, width = feature_map.shape[-2:]
        if (height, width) != (16, 8):
            raise ValueError(f'Expected OSNet map 16x8, got {height}x{width}')
        low_frequency = F.avg_pool2d(feature_map, 3, stride=1, padding=1, count_include_pad=False)
        high_frequency = feature_map - low_frequency
        query = F.normalize(self.query(low_frequency.mean((2, 3))).float(), dim=-1)
        keys = F.normalize(self.key(low_frequency).flatten(2).transpose(1, 2).float(), dim=-1)
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / .1
        keep = max(1, round(scores.shape[-1] * self.keep_ratio))
        selected = scores.topk(keep, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, selected, True)
        weights = F.softmax(scores.masked_fill(~mask, torch.finfo(scores.dtype).min), dim=-1)
        values = self.value(high_frequency).flatten(2).transpose(1, 2).float()
        detail = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        detail = self.output(detail).to(feature_map.dtype)[:, :, None, None]
        gate = torch.sigmoid(self.gate(torch.cat((feature_map, detail.expand_as(feature_map)), dim=1)))
        return feature_map + self.residual_scale * gate * detail


class StructuralOSNet(nn.Module):
    """Keep the official OSNet stem/blocks/FC and insert a map-level adapter."""

    def __init__(self, num_classes, variant='baseline'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        self.variant = variant
        self.backbone = OcclusionOSNet(num_classes, 'aug')
        if variant == 'transformer':
            self.context = SpatialTransformer()
        elif variant == 'semantic_detail':
            self.context = SemanticDetailFusion()
        else:
            self.context = nn.Identity()

    def load_imagenet(self, path):
        return self.backbone.load_imagenet(path)

    def forward(self, images, details=False):
        maps = self.context(self.backbone.featuremaps(images))
        raw = maps.mean((2, 3))
        features = self.backbone.fc(raw)
        logits = self.backbone.classifier(features)
        if details:
            return {'maps': maps, 'raw': raw, 'features': features, 'logits': logits}
        return F.normalize(features, dim=1)
