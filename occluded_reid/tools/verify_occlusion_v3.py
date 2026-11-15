"""Audit the registered v3 matrix, raw logs, checkpoints and frozen evaluations."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(require_checkpoints=False):
    protocol_path = ROOT/'research/occlusion_v3_protocol.json'
    protocol = read(protocol_path)
    directory = ROOT/'results/occlusion_v3_summary'
    manifest_path = directory/'external_freeze_manifest.json'
    manifest = read(manifest_path)
    expected = [(v, s) for s in protocol['seeds'] for v in protocol['variants_in_order']]
    assert [(r['variant'], r['seed']) for r in manifest['runs']] == expected, 'Incomplete or reordered matrix'
    assert manifest['protocol_sha256'] == digest(protocol_path), 'Protocol changed after training'
    for rel, checksum in manifest['source_sha256'].items():
        assert digest(ROOT/rel) == checksum, f'Frozen source changed: {rel}'
    frozen_at = datetime.fromisoformat(manifest['frozen_at_utc'])
    records = []
    reference = None
    for row in manifest['runs']:
        name = row['run']
        folder = ROOT/'results'/name
        config, history, result, external = [read(folder/f) for f in
            ('config.json', 'history.json', 'results.json', 'occluded_reid.json')]
        assert name == f"occlusion_v3_{row['variant']}_s{row['seed']}"
        assert config == result['config'] and history == result['history']
        assert config['variant'] == result['variant'] == external['variant'] == row['variant']
        assert config['seed'] == result['seed'] == external['seed'] == row['seed']
        assert config['protocol_sha256'] == digest(protocol_path)
        assert config['source_sha256'] == manifest['source_sha256']
        assert row['development_result_sha256'] == digest(folder/'results.json')
        assert result['checkpoint_sha256'] == external['checkpoint_sha256'] == row['checkpoint_sha256']
        assert external['manifest_sha256'] == digest(manifest_path)
        assert datetime.fromisoformat(config['created_at_utc']) <= datetime.fromisoformat(result['completed_at_utc']) <= frozen_at
        assert datetime.fromisoformat(external['evaluated_at_utc']) >= frozen_at
        shared = {k: config[k] for k in ('epochs', 'steps', 'batch_size', 'instances', 'height', 'width',
                 'protocol_sha256', 'source_sha256', 'pretrained_sha256', 'data_record_sha256', 'split', 'train_images',
                 'query_images', 'gallery_images', 'torch', 'torchvision', 'cuda', 'gpu', 'optimizer', 'loss')}
        if reference is None:
            reference = shared
        assert shared == reference, f'Unequal recipe: {name}'
        assert config['epochs'] == protocol['epochs'] == result['trained_epochs'] == external['epochs']
        assert config['steps'] == protocol['pk_steps_per_epoch']
        assert len(config['split']['train_ids']) == 651 and len(config['split']['val_ids']) == 100
        assert not set(config['split']['train_ids']) & set(config['split']['val_ids'])
        checkpoint = ROOT/row['checkpoint']
        if require_checkpoints or checkpoint.exists():
            assert checkpoint.is_file() and digest(checkpoint) == row['checkpoint_sha256']
        logs = [json.loads(line) for line in (folder/'train.log').read_text(encoding='utf-8').splitlines()
                if line.startswith('{')]
        logged_epochs = {x['epoch']: {k: v for k, v in x.items() if k != 'event'} for x in logs
                         if x.get('event') == 'epoch'}
        assert len(history) == protocol['epochs']
        for epoch, item in enumerate(history, 1):
            assert item['epoch'] == epoch and logged_epochs[epoch] == item
            assert item['pk_batches'] == protocol['pk_steps_per_epoch']
            assert item['optimizer_updates'] + item['amp_skipped_updates'] == item['pk_batches']
            assert all(math.isfinite(v) for v in item.values() if isinstance(v, (int, float)))
        for condition in ('clean', 'synthetic_noise_0.3', 'synthetic_noise_0.5'):
            metrics = result['evaluation'][condition]
            assert metrics['valid_queries'] == 100 and metrics['gallery_size'] == 1476
            for key in ('mAP', 'rank1', 'rank5', 'mINP'):
                assert math.isfinite(metrics[key]) and 0 <= metrics[key] <= 100
        metrics = external['evaluation']
        assert metrics['valid_queries'] == 1000 and metrics['gallery_size'] == 1000
        assert external['target_data_record_sha256']
        for key in ('mAP', 'rank1', 'rank5', 'mINP'):
            assert math.isfinite(metrics[key]) and 0 <= metrics[key] <= 100
        records.append({'run': name, 'updates': sum(x['optimizer_updates'] for x in history),
                        'amp_skips': sum(x['amp_skipped_updates'] for x in history),
                        'checkpoint_verified_locally': checkpoint.exists()})
    return {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'runs': records, 'checks': ['complete registered matrix', 'fixed source/protocol/data/init and matched training recipe',
            'disjoint train/development identities', 'history/log/update agreement',
            'all development completed before checkpoint freeze', 'all external evaluations use frozen checkpoints',
            'query/gallery counts and metrics present'],
            'limitations': ['Two seeds are not a significance test.', 'Occluded-REID was viewed in earlier stages; exploratory transfer only.',
            'Audit checks recorded consistency, not novelty, causality, convergence, or SOTA.']}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-checkpoints', action='store_true')
    args = parser.parse_args()
    report = verify(args.require_checkpoints)
    (ROOT/'results/occlusion_v3_summary/verification.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'runs': len(report['runs'])}))


if __name__ == '__main__':
    main()
