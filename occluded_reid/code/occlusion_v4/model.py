"""OSNet with one isolated channel-affinity adapter after conv5."""
import torch
from torch import nn
from torch.nn import functional as F

from occlusion_v4.mdta import MDTAAdapter
from reid.model import OcclusionOSNet


class MDTAOSNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = OcclusionOSNet(num_classes, 'aug')
        self.context = MDTAAdapter(channels=512, heads=4, residual_scale=0.05)

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
