"""Audit the registered v5 matrix, source/data hashes and frozen evaluation."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(require_checkpoints=False):
    from train_occlusion_v5 import source_hashes
    protocol_path = ROOT/'research/occlusion_v5_protocol.json'
    protocol = read(protocol_path)
    summary = ROOT/'results/occlusion_v5_summary'
    manifest_path = summary/'external_freeze_manifest.json'
    manifest = read(manifest_path)
    expected = [(variant, seed) for seed in protocol['seeds']
                for variant in protocol['variants_in_order']]
    assert [(row['variant'], row['seed']) for row in manifest['runs']] == expected
    assert manifest['protocol_sha256'] == digest(protocol_path)
    assert manifest['source_sha256'] == source_hashes()
    assert manifest['target_data_record_sha256']
    for rel, checksum in manifest['source_sha256'].items():
        assert digest(ROOT/rel) == checksum, f'Frozen source changed: {rel}'
    frozen_at = datetime.fromisoformat(manifest['frozen_at_utc'])
    reference = None
    target_hashes = set()
    records = []
    for row in manifest['runs']:
        name = row['run']
        folder = ROOT/'results'/name
        config, history, result, external = [read(folder/filename) for filename in
            ('config.json', 'history.json', 'results.json', 'occluded_reid.json')]
        assert name == f"{protocol['run_tag']}_{row['variant']}_s{row['seed']}"
        assert config == result['config'] and history == result['history']
        assert config['variant'] == result['variant'] == external['variant'] == row['variant']
        assert config['run_tag'] == protocol['run_tag']
        assert config['seed'] == result['seed'] == external['seed'] == row['seed']
        assert config['protocol_sha256'] == digest(protocol_path)
        assert config['source_sha256'] == manifest['source_sha256']
        assert row['development_result_sha256'] == digest(folder/'results.json')
        assert result['checkpoint_sha256'] == external['checkpoint_sha256'] == row['checkpoint_sha256']
        assert external['manifest_sha256'] == digest(manifest_path)
        assert external['target_data_record_sha256'] == manifest['target_data_record_sha256']
        assert datetime.fromisoformat(config['created_at_utc']) <= \
               datetime.fromisoformat(result['completed_at_utc']) <= frozen_at
        assert datetime.fromisoformat(external['evaluated_at_utc']) >= frozen_at
        shared_keys = ('epochs', 'steps', 'batch_size', 'instances', 'height', 'width',
                       'protocol_sha256', 'source_sha256', 'pretrained_sha256',
                       'data_record_sha256', 'split', 'query_camera_by_identity',
                       'train_images', 'query_images', 'gallery_images', 'torch',
                       'torchvision', 'cuda', 'gpu', 'optimizer', 'loss')
        shared = {key: config[key] for key in shared_keys}
        if reference is None:
            reference = shared
        assert shared == reference, f'Unequal training recipe: {name}'
        assert config['epochs'] == protocol['epochs'] == result['trained_epochs'] == external['epochs']
        assert config['steps'] == protocol['steps_per_epoch']
        assert len(config['split']['train_ids']) == 651 and len(config['split']['val_ids']) == 100
        assert not set(config['split']['train_ids']) & set(config['split']['val_ids'])
        checkpoint = ROOT/row['checkpoint']
        if require_checkpoints or checkpoint.exists():
            assert checkpoint.is_file() and digest(checkpoint) == row['checkpoint_sha256']
        log_path = folder/'train.log'
        logs = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()
                if line.startswith('{')]
        logged_epochs = {item['epoch']: {key: value for key, value in item.items() if key != 'event'}
                         for item in logs if item.get('event') == 'epoch'}
        assert len(history) == protocol['epochs']
        for epoch, item in enumerate(history, 1):
            assert item['epoch'] == epoch and item['pk_batches'] == protocol['steps_per_epoch']
            assert logged_epochs[epoch] == {'variant': row['variant'], 'seed': row['seed'],
                                              **item}
            assert item['optimizer_updates'] + item['amp_skipped_updates'] == item['pk_batches']
            assert all(math.isfinite(value) for value in item.values()
                       if isinstance(value, (int, float)))
        for condition in ('clean', 'synthetic_donor_0.3', 'synthetic_donor_0.5'):
            metrics = result['evaluation'][condition]
            assert metrics['valid_queries'] == config['query_images']
            assert metrics['gallery_size'] == config['gallery_images']
            for key in ('mAP', 'rank1', 'rank5', 'mINP'):
                assert math.isfinite(metrics[key]) and 0 <= metrics[key] <= 100
        metrics = external['evaluation']
        assert metrics['valid_queries'] == 1000 and metrics['gallery_size'] == 1000
        target_hashes.add(external['target_data_record_sha256'])
        for key in ('mAP', 'rank1', 'rank5', 'mINP'):
            assert math.isfinite(metrics[key]) and 0 <= metrics[key] <= 100
        records.append({'run': name, 'updates': sum(item['optimizer_updates'] for item in history),
                        'amp_skips': sum(item['amp_skipped_updates'] for item in history),
                        'checkpoint_verified_locally': checkpoint.exists()})
    assert len(target_hashes) == 1, 'External runs did not use the same target data records'
    report = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'runs': records,
              'checks': ['complete registered matrix', 'fixed source, protocol, data, initialization and matched recipe',
                         'disjoint train/development identities', 'history, raw log and optimizer update accounting',
                         'all development completed before checkpoint freeze',
                         'all external evaluations use frozen checkpoints and the same target data hash'],
              'limitations': ['Five seeds are screening evidence, not a significance test.',
                              'Occluded-REID was viewed earlier; exploratory transfer only.',
                              'The audit verifies records and implementation, not novelty, causality, convergence or SOTA.']}
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-checkpoints', action='store_true')
    args = parser.parse_args()
    report = verify(args.require_checkpoints)
    (ROOT/'results/occlusion_v5_summary/verification.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'runs': len(report['runs'])}))


if __name__ == '__main__':
    main()
