"""Executable AGW with unused duplicate ResNets removed.

Architecture matches mangye16/Cross-Modal-Re-ID-baseline e817a267.
Non-local equations below are adapted from its MIT-licensed model.py.
The upstream inter_channels=1 is intentional and retained for equivalence.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from vendor.agw.resnet import resnet50


class NonLocal(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.g = nn.Conv2d(channels, 1, 1)
        self.theta = nn.Conv2d(channels, 1, 1)
        self.phi = nn.Conv2d(channels, 1, 1)
        self.out = nn.Sequential(nn.Conv2d(1, channels, 1), nn.BatchNorm2d(channels))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x):
        b, _, h, w = x.shape
        affinity = self.theta(x).flatten(2).transpose(1, 2) @ self.phi(x).flatten(2)
        attended = affinity / (h * w) @ self.g(x).flatten(2).transpose(1, 2)
        return x + self.out(attended.transpose(1, 2).reshape(b, 1, h, w))


class AGW(nn.Module):
    def __init__(self, classes, weights=None):
        super().__init__()
        body = resnet50(pretrained=False, last_conv_stride=1)
        self.initialization = {'kind': 'random', 'sha256': None}
        if weights:
            path = Path(weights)
            state = torch.load(path, map_location='cpu', weights_only=True)
            state = {k: v for k, v in state.items() if not k.startswith('fc.')}
            body.load_state_dict(state, strict=True)
            self.initialization = {'kind': 'ImageNet', 'path': str(path.resolve()),
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.visible = nn.Sequential(body.conv1, body.bn1, body.relu, body.maxpool)
        self.thermal = deepcopy(self.visible)
        self.stages = nn.ModuleList([body.layer1, body.layer2, body.layer3, body.layer4])
        self.nonlocal_blocks = nn.ModuleDict({
            's2b3': NonLocal(512), 's2b4': NonLocal(512),
            's3b4': NonLocal(1024), 's3b5': NonLocal(1024), 's3b6': NonLocal(1024)})
        self.neck = nn.BatchNorm1d(2048)
        nn.init.normal_(self.neck.weight, 1, .01)
        self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=.001)

    def forward(self, visible=None, thermal=None):
        parts = []
        if visible is not None:
            parts.append(self.visible(visible))
        if thermal is not None:
            parts.append(self.thermal(thermal))
        if not parts:
            raise ValueError('Provide visible and/or thermal images.')
        x = torch.cat(parts) if len(parts) == 2 else parts[0]
        for s, stage in enumerate(self.stages, 1):
            for b, block in enumerate(stage, 1):
                x = block(x)
                key = f's{s}b{b}'
                if key in self.nonlocal_blocks:
                    x = self.nonlocal_blocks[key](x)
        # Float32 GeM also avoids overflowing x**3 under AMP.
        pooled = x.float().clamp_min(0).pow(3).mean((2, 3)).add(1e-12).pow(1 / 3)
        neck = self.neck(pooled)
        return {'pooled': pooled, 'embedding': F.normalize(neck, dim=1),
                'logits': self.classifier(neck)}


def load_upstream(model, upstream):
    """Map active layers only; used by the numerical equivalence check."""
    model.visible.load_state_dict(nn.Sequential(
        upstream.visible_module.visible.conv1, upstream.visible_module.visible.bn1,
        upstream.visible_module.visible.relu, upstream.visible_module.visible.maxpool).state_dict())
    model.thermal.load_state_dict(nn.Sequential(
        upstream.thermal_module.thermal.conv1, upstream.thermal_module.thermal.bn1,
        upstream.thermal_module.thermal.relu, upstream.thermal_module.thermal.maxpool).state_dict())
    for ours, name in zip(model.stages, ['layer1', 'layer2', 'layer3', 'layer4']):
        ours.load_state_dict(getattr(upstream.base_resnet.base, name).state_dict())
    for stage, names in [(2, ['s2b3', 's2b4']), (3, ['s3b4', 's3b5', 's3b6'])]:
        for name, original in zip(names, getattr(upstream, f'NL_{stage}')):
            current = model.nonlocal_blocks[name]
            for field in ['g', 'theta', 'phi']:
                getattr(current, field).load_state_dict(getattr(original, field)[0].state_dict()
                    if isinstance(getattr(original, field), nn.Sequential) else getattr(original, field).state_dict())
            current.out.load_state_dict(original.W.state_dict())
    model.neck.load_state_dict(upstream.bottleneck.state_dict())
    model.classifier.load_state_dict(upstream.classifier.state_dict())
