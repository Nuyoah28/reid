"""Frozen-checkpoint exploratory transfer evaluation on Occluded-REID."""
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
from occlusion_v3.model import StructuralOSNet

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    entries = {row['checkpoint_sha256'] for row in manifest['runs']}
    if checkpoint_sha not in entries:
        raise ValueError('Checkpoint is absent from the pre-evaluation freeze manifest')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite frozen target result: {args.output}')
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    model = StructuralOSNet(len(config['split']['train_ids']), config['variant'])
    model.load_state_dict(saved['model'], strict=True)
    device = torch.device('cuda')
    model.to(device).eval()
    data_root = ROOT/'data'
    query, gallery = scan_occluded(data_root)
    record_rows = ['%s\t%d\t%d' % (Path(path).relative_to(data_root).as_posix(), pid, camera)
                   for path, pid, camera in query + gallery]
    target_record_sha = hashlib.sha256('\n'.join(record_rows).encode()).hexdigest()

    def embed(records):
        loader = DataLoader(Images(records, 256, 128), batch_size=96,
                            num_workers=0, pin_memory=True)
        return extract(model, loader, device)

    q, g = embed(query), embed(gallery)
    metrics = evaluate_rank(q[0], g[0], q[1], g[1], q[2], g[2])
    report = {'dataset': 'Occluded-REID', 'variant': config['variant'], 'seed': config['seed'],
              'epochs': saved['epoch'], 'train_ids': len(config['split']['train_ids']),
              'checkpoint_sha256': checkpoint_sha,
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'target_data_record_sha256': target_record_sha,
              'evaluated_at_utc': datetime.now(timezone.utc).isoformat(), 'evaluation': metrics,
              'protocol': 'All 1000 occluded queries against all 1000 whole gallery images; no reranking, flip, target training, or selection.',
              'limitation': 'Exploratory transfer only: this dataset was inspected in v1/v2.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
