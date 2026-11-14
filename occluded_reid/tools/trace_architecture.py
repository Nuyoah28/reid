"""Trace the actual full-variant modules on CPU for the architecture atlas.

No dataset, checkpoint, package installation, or training source edit is needed.
Functional operations (detach, sums, cosine, L2 normalization) are documented in
the atlas separately because nn.Module hooks do not enumerate them.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from reid.model import OcclusionOSNet


def shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shape(item) for item in value]
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items() if item is not None}
    return None


def main():
    torch.set_num_threads(4)
    model = OcclusionOSNet(751, 'full').eval()
    calls, stages, handles = [], [], []
    watched = {'conv1', 'maxpool', 'conv2', 'conv3', 'conv4', 'conv5',
               'reliability', 'fc', 'classifier'}

    def hook(name, stage=False):
        def record(module, inputs, output):
            entry = {'module': name, 'type': type(module).__name__,
                     'spec': module.extra_repr(), 'input': shape(inputs),
                     'output': shape(output),
                     'parameters': sum(p.numel() for p in module.parameters())}
            (stages if stage else calls).append(entry)
        return record

    for name, module in model.named_modules():
        if name in watched:
            handles.append(module.register_forward_hook(hook(name, True)))
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        result = model(torch.zeros(1, 3, 256, 128), details=True)
    for handle in handles:
        handle.remove()
    sources = [ROOT/'code/reid/model.py', ROOT/'code/reid/occlusion.py',
               ROOT/'code/train.py', ROOT/'code/reid/data.py',
               ROOT/'code/reid/losses.py',
               ROOT/'code/deep-person-reid/torchreid/models/osnet.py']
    all_parameters = sum(p.numel() for p in model.parameters())
    classifier = sum(p.numel() for p in model.classifier.parameters())
    report = {
        'variant': 'full', 'input': [1, 3, 256, 128], 'classes': 751,
        'trace_mode': 'CPU eval + inference_mode, details=True, no weights loaded',
        'note': 'Classifier is called by details=True for training details, but '
                'not by the deployed normalized-embedding forward. Leaf module '
                'hooks omit functional ops. ChannelGate is shared across four '
                'branches within each OSBlock; calls repeat without new weights.',
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(
            path.read_bytes()).hexdigest() for path in sources},
        'parameters': {'including_classifier': all_parameters,
                       'classifier': classifier,
                       'inference': all_parameters-classifier,
                       'reliability_head': sum(p.numel() for p in model.reliability.parameters())},
        'stored_module_types': dict(Counter(type(m).__name__ for m in model.modules())),
        'executed_leaf_types': dict(Counter(x['type'] for x in calls)),
        'stages': stages, 'outputs': shape(result), 'leaf_calls': calls,
    }
    destination = ROOT/'research/architecture'
    destination.mkdir(exist_ok=True)
    (destination/'architecture_trace.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'parameters': report['parameters'], 'stages': stages,
                      'stored_conv2d': report['stored_module_types']['Conv2d'],
                      'executed_conv2d': report['executed_leaf_types']['Conv2d'],
                      'leaf_calls': len(calls)}, indent=2))


if __name__ == '__main__':
    main()
