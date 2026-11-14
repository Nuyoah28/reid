"""Frozen-checkpoint exploratory transfer; never a training/selection signal."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train import extract
from reid.data import Images, scan_occluded
from reid.metrics import evaluate_rank
from occlusion_v2.method import build_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if digest not in {entry['checkpoint_sha256'] for entry in manifest['runs']}:
        raise ValueError('Checkpoint not in the pre-evaluation frozen manifest')
    if args.output.exists():
        raise FileExistsError('Do not overwrite a frozen external result')
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    model = build_model(len(config['split']['train_ids']), config['variant'])
    model.load_state_dict(saved['model'], strict=True)
    device = torch.device('cuda'); model.to(device).eval()
    query, gallery = scan_occluded(ROOT/'data')
    def embed(records):
        return extract(model, DataLoader(Images(records, 256, 128), batch_size=96,
                                         num_workers=0, pin_memory=True), device)
    q, g = embed(query), embed(gallery)
    result = evaluate_rank(q[0], g[0], q[1], g[1], q[2], g[2])
    report = {'dataset': 'Occluded-REID', 'variant': config['variant'], 'seed': config['seed'],
              'epochs': saved['epoch'], 'train_ids': len(config['split']['train_ids']),
              'checkpoint_sha256': digest, 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'evaluated_at_utc': datetime.now(timezone.utc).isoformat(), 'evaluation': result,
              'protocol': 'All 1000 occluded query / 1000 whole gallery, 200 target IDs. No reranking, flip, target training or selection. All pre-registered candidates evaluated.',
              'limitation': 'Exploratory transfer: v1 target-domain scores were already inspected. 651-ID/640-batch training, not comparable to v1 751-ID/2000-batch scores.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
