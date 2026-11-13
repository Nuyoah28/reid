"""OSNet extension. The backbone is loaded directly from the pinned upstream file.

This avoids importing optional, unused torchreid data/engine dependencies on Windows.
The OSNet architecture and its MIT license remain in code/deep-person-reid.
"""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SOURCE=Path(__file__).resolve().parents[1]/'deep-person-reid'/'torchreid'/'models'/'osnet.py'
spec=importlib.util.spec_from_file_location('upstream_osnet',SOURCE)
upstream=importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


class OcclusionOSNet(upstream.OSNet):
    """Official OSNet x1.0 with optional counterfactual leakage calibrated pooling.

    A single image and no synthetic mask are required at inference. Reliability
    predicts stability under changed occluder appearance, not human segmentation.
    """
    def __init__(self,num_classes,variant='clcp',floor=0.05):
        super().__init__(num_classes,[upstream.OSBlock]*3,[2,2,2],[64,256,384,512],loss='triplet')
        self.variant=variant
        self.floor=floor
        self.reliability=None
        if variant in ('visibility','clcp','full'):
            self.reliability=nn.Sequential(
                nn.Conv2d(512,32,1,bias=False),nn.GroupNorm(4,32),nn.SiLU(),
                nn.Conv2d(32,32,3,padding=1,groups=32),nn.SiLU(),nn.Conv2d(32,1,1))
            nn.init.zeros_(self.reliability[-1].weight)
            nn.init.constant_(self.reliability[-1].bias,2.)

    def pool(self,maps,weights=None):
        if weights is None:return maps.mean((2,3))
        return (maps*weights).sum((2,3))/weights.sum((2,3)).clamp_min(1e-6)

    def forward(self,images,details=False):
        maps=self.featuremaps(images)
        # Detaching the head input isolates calibration learning from the backbone.
        logits_r=self.reliability(maps.detach()) if self.reliability is not None else None
        weights=self.floor+(1-self.floor)*logits_r.sigmoid() if logits_r is not None else None
        raw=self.pool(maps,weights)
        features=self.fc(raw)
        if details:return {'maps':maps,'raw':raw,'features':features,'logits':self.classifier(features),'reliability_logits':logits_r,'weights':weights}
        return F.normalize(features,dim=1)

    def load_imagenet(self,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        state=state.get('state_dict',state)
        own=self.state_dict()
        matched={k.removeprefix('module.'):v for k,v in state.items() if k.removeprefix('module.') in own and v.shape==own[k.removeprefix('module.')].shape and not k.removeprefix('module.').startswith('classifier.')}
        result=self.load_state_dict(matched,strict=False)
        if len(matched)<200:raise RuntimeError(f'Unexpected ImageNet checkpoint: only {len(matched)} layers match')
        return {'matched_tensors':len(matched),'missing':result.missing_keys}
