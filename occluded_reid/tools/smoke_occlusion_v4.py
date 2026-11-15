"""Run an RTX 4060-sized MDTA training batch before the registered run."""
from pathlib import Path
import json
import time
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT/'code'))
from occlusion_v4.model import MDTAOSNet
from reid.losses import batch_hard


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda')
    torch.manual_seed(17)
    images = torch.randn(24, 3, 256, 128, device=device)
    labels = torch.arange(6, device=device).repeat_interleave(4)
    model = MDTAOSNet(6)
    model.load_imagenet(ROOT/'weights/osnet_x1_0_imagenet.pth')
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autocast('cuda'):
        out = model(images, details=True)
        loss = F.cross_entropy(out['logits'], labels, label_smoothing=.1)
        loss = loss + .5*batch_hard(F.normalize(out['features'].float(), dim=1), labels)
    loss.backward()
    torch.cuda.synchronize()
    result = {'gpu': torch.cuda.get_device_name(0), 'variant': 'mdta',
              'loss_finite': bool(torch.isfinite(loss)),
              'feature_shape': list(out['features'].shape),
              'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'adapter_parameters': sum(p.numel() for p in model.context.parameters()),
              'peak_cuda_mb': torch.cuda.max_memory_allocated()/2**20,
              'forward_backward_seconds': time.perf_counter()-start}
    if not result['loss_finite'] or result['feature_shape'] != [24, 512]:
        raise RuntimeError(f'Smoke test failed: {result}')
    (ROOT/'research/occlusion_v4_smoke.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', **result}), flush=True)


if __name__ == '__main__':
    main()
