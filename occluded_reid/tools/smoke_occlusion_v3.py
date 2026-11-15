"""Check all v3 adapters on one realistic CUDA batch before training."""
from pathlib import Path
import json
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT/'code'))
from occlusion_v3.model import VARIANTS, StructuralOSNet
from reid.losses import batch_hard


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda')
    torch.manual_seed(17)
    images = torch.randn(24, 3, 256, 128, device=device)
    labels = torch.arange(6, device=device).repeat_interleave(4)
    output = {'gpu': torch.cuda.get_device_name(0), 'variants': []}
    for variant in VARIANTS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = StructuralOSNet(6, variant)
        model.load_imagenet(ROOT/'weights/osnet_x1_0_imagenet.pth')
        model.to(device).train()
        start = time.perf_counter()
        with torch.autocast('cuda'):
            result = model(images, details=True)
            loss = F.cross_entropy(result['logits'], labels, label_smoothing=.1)
            loss = loss + .5*batch_hard(F.normalize(result['features'].float(), dim=1), labels)
        loss.backward()
        torch.cuda.synchronize()
        row = {'variant': variant, 'loss_finite': bool(torch.isfinite(loss)),
               'feature_shape': list(result['features'].shape),
               'trainable_parameters': sum(p.numel() for p in model.parameters()),
               'peak_cuda_mb': torch.cuda.max_memory_allocated()/2**20,
               'forward_backward_seconds': time.perf_counter()-start}
        if not row['loss_finite'] or row['feature_shape'] != [24, 512]:
            raise RuntimeError(f'Smoke test failed: {row}')
        output['variants'].append(row)
        print(json.dumps(row), flush=True)
        del model, result, loss
    (ROOT/'research/occlusion_v3_smoke.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', **output}), flush=True)


if __name__ == '__main__':
    main()
