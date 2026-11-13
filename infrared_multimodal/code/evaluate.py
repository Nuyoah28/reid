"""Evaluate only a fixed completed checkpoint; no reranking or flip ensemble."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import Images, read_records, sysu_gallery, train_partition
from metrics import retrieval
from model import AGW


@torch.inference_mode()
def extract(model, rows, config, device, workers=0, batch=32):
    features = []
    loader = DataLoader(Images(rows, config['height'], config['width']), batch_size=batch, num_workers=workers)
    for images, _, modalities, _ in loader:
        # Evaluation loaders must each contain a single modality.
        if modalities.unique().numel() != 1:
            raise ValueError('Mixed evaluation modalities.')
        m = int(modalities[0])
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            out = model(visible=images.to(device)) if m == 0 else model(thermal=images.to(device))
        features.append(out['embedding'].float().cpu())
    if not features:
        raise ValueError('No evaluation images.')
    return torch.cat(features).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--batch', type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']
    if ckpt['epoch'] != config['epochs']:
        raise ValueError('Only fixed final-epoch checkpoints may be evaluated by this runner.')
    model = AGW(config['classes']).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    rows = read_records(config['dataset'], args.data, 'train' if config['protocol'] == 'dev' else 'test', config['trial'])
    if config['protocol'] == 'dev':
        _, rows, _ = train_partition(rows, 'dev')
    results = {}
    if config['dataset'] == 'regdb' or config['protocol'] == 'dev':
        # Development data is a held-out-identity full-gallery probe, not the SYSU official protocol.
        visible, thermal = [[r for r in rows if r.modality == m] for m in [0, 1]]
        fv, ft = [extract(model, part, config, device, args.workers, args.batch) for part in [visible, thermal]]
        distance = -fv @ ft.T
        results['visible_to_infrared'] = retrieval(distance, visible, thermal)
        results['infrared_to_visible'] = retrieval(distance.T, thermal, visible)
    else:
        query = [r for r in rows if r.modality == 1]
        fq = extract(model, query, config, device, args.workers, args.batch)
        for mode in ['all', 'indoor']:
            trials = []
            for trial in range(10):
                gallery = sysu_gallery(rows, mode, trial)
                fg = extract(model, gallery, config, device, args.workers, args.batch)
                trials.append(retrieval(-fq @ fg.T, query, gallery, sysu=True))
            results[mode+'_single_shot'] = {'mean': {k: float(np.mean([r[k] for r in trials])) for k in trials[0]},
                                                  'gallery_trials': trials}
    payload = {'checkpoint': str(args.checkpoint.resolve()), 'epoch': ckpt['epoch'],
               'dataset': config['dataset'], 'protocol': config['protocol'], 'trial': config['trial'],
               'feature': 'L2 normalized BNNeck, cosine', 'reranking': False, 'test_flip': False,
               'units': 'fractions (multiply by 100 for percentage)', 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
