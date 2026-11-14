"""Audit all registered runs without using target scores for selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def instant(value):
    return datetime.fromisoformat(value)


def verify(tag='peer_v2', require_checkpoints=False):
    protocol_path = ROOT/'research/occlusion_v2_protocol.json'
    protocol = read(protocol_path)
    directory = ROOT/'results'/f'{tag}_summary'
    manifest_path = directory/'external_freeze_manifest.json'
    manifest = read(manifest_path)
    expected = [(v, s) for s in protocol['seeds'] for v in protocol['variants_in_order']]
    assert [(r['variant'], r['seed']) for r in manifest['runs']] == expected, 'Incomplete / reordered run matrix'
    assert manifest['registered_protocol_sha256'] == digest(protocol_path), 'Protocol changed'
    frozen_at = instant(manifest['frozen_at_utc'])
    evaluator_freeze = read(ROOT/'research/occlusion_v2_evaluation_source_freeze.json')
    assert instant(evaluator_freeze['recorded_at_utc']) < frozen_at
    for path, checksum in evaluator_freeze['sources'].items():
        assert digest(ROOT/path) == checksum, 'Evaluator source changed: ' + path
    reference = None
    audits = []
    common_keys = ('epochs', 'steps', 'workers', 'height', 'width', 'batch_size', 'instances', 'views',
                   'lr', 'head_lr_multiplier', 'fixbase_epochs', 'tau', 'identity_temperature',
                   'calibration_weight', 'evidence_weight', 'pool_floor', 'split', 'train_images',
                   'query_images', 'gallery_images', 'identities_under_k', 'sampling', 'source_sha256',
                   'pretrained_sha256', 'data_record_sha256', 'torch', 'torchvision', 'cuda', 'python',
                   'gpu', 'protocol', 'recipe_version', 'registered_protocol_sha256')
    for entry in manifest['runs']:
        name = entry['run']
        assert name == f"{tag}_{entry['variant']}_s{entry['seed']}", 'Run name mismatch'
        root = ROOT/'results'/name
        config, history, result, external = [read(root/f) for f in
                                             ('config.json', 'history.json', 'results.json', 'occluded_reid.json')]
        assert result['config'] == config and result['history'] == history, name + ': archive mismatch'
        assert (config['variant'], config['seed']) == (entry['variant'], entry['seed'])
        assert (result['variant'], result['seed']) == (entry['variant'], entry['seed'])
        assert (external['variant'], external['seed']) == (entry['variant'], entry['seed'])
        assert config['epochs'] == protocol['epochs'] == result['trained_epochs'] == external['epochs']
        assert config['steps'] == protocol['pk_steps_per_epoch']
        assert config['registered_protocol_sha256'] == digest(protocol_path)
        assert result['checkpoint_sha256'] == entry['checkpoint_sha256'] == external['checkpoint_sha256']
        assert entry['development_result_sha256'] == digest(root/'results.json')
        assert external['manifest_sha256'] == digest(manifest_path)
        assert instant(config['created_at_utc']) <= instant(result['completed_at_utc']) <= frozen_at
        assert frozen_at <= instant(external['evaluated_at_utc']), 'External results predate freeze'
        common = {key: config[key] for key in common_keys}
        if reference is None:
            reference = common
        assert common == reference, name + ': unfair recipe / source change'
        train_ids, val_ids = set(config['split']['train_ids']), set(config['split']['val_ids'])
        assert len(train_ids) == 651 and len(val_ids) == 100 and not train_ids & val_ids
        assert (config['train_images'], config['query_images'], config['gallery_images']) == (11360, 100, 1476)
        assert external['train_ids'] == 651
        for path, checksum in config['source_sha256'].items():
            assert digest(ROOT/path) == checksum, 'Source changed: ' + path
        checkpoint = ROOT/entry['checkpoint']
        if require_checkpoints or checkpoint.exists():
            assert checkpoint.is_file() and digest(checkpoint) == entry['checkpoint_sha256'], name + ': checkpoint mismatch'
        logs = [json.loads(line) for line in (root/'train.log').read_text(encoding='utf-8').splitlines()
                if line.startswith('{')]
        logged_epochs = {r['epoch']: {k: v for k, v in r.items() if k != 'event'}
                         for r in logs if r.get('event') == 'epoch'}
        assert len(history) == protocol['epochs']
        for epoch, row in enumerate(history, 1):
            assert row['epoch'] == epoch and row == logged_epochs[epoch], name + ': history/log mismatch'
            assert row['pk_batches'] == protocol['pk_steps_per_epoch']
            assert row['optimizer_updates'] + row['amp_skipped_updates'] == row['pk_batches']
            skips = sum(r.get('event') == 'amp_skipped_update' and r.get('epoch') == epoch for r in logs)
            assert row['amp_skipped_updates'] == skips, name + ': AMP count mismatch'
            assert row['backbone_frozen'] == (epoch <= config['fixbase_epochs'])
            assert all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float)))
        for condition in ('clean', 'synthetic_noise_0.3', 'synthetic_noise_0.5'):
            metrics = result['evaluation'][condition]
            assert (metrics['valid_queries'], metrics['invalid_queries'], metrics['gallery_size']) == (100, 0, 1476)
            for key in ('mAP', 'rank1', 'rank5', 'mINP'):
                assert math.isfinite(metrics[key]) and 0 <= metrics[key] <= 100
        metrics = external['evaluation']
        assert (metrics['valid_queries'], metrics['invalid_queries'], metrics['gallery_size']) == (1000, 0, 1000)
        for key in ('mAP', 'rank1', 'rank5', 'mINP'):
            assert math.isfinite(metrics[key]) and 0 <= metrics[key] <= 100
        audits.append({'run': name, 'pk_batches': sum(h['pk_batches'] for h in history),
                       'optimizer_updates': sum(h['optimizer_updates'] for h in history),
                       'amp_skipped_updates': sum(h['amp_skipped_updates'] for h in history),
                       'checkpoint_verified_locally': checkpoint.exists(),
                       'archived_sha256': {f: digest(root/f) for f in
                                          ('config.json', 'history.json', 'results.json', 'train.log', 'occluded_reid.json')}})
    return {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'protocol_sha256': digest(protocol_path), 'manifest_sha256': digest(manifest_path),
            'matrix': expected, 'runs': audits,
            'checks': ['complete registered matrix', 'fixed recipe/source/data-record hashes',
                       'disjoint train/development identities', 'raw log/history/AMP update consistency',
                       'all development completed before freeze; all external results after freeze',
                       'matching checkpoint and evaluation manifest hashes', 'query/gallery sizes and metric ranges'],
            'limitations': ['Audit proves recorded consistency, not method effectiveness or causality.',
                           'Data record hashes cover paths/identity/camera, not every image byte.',
                           'cuDNN benchmark allows nondeterministic kernels; two seeds do not establish significance.',
                           'External benchmark already inspected during v1; exploratory transfer only.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='peer_v2')
    parser.add_argument('--require-checkpoints', action='store_true')
    args = parser.parse_args()
    report = verify(args.tag, args.require_checkpoints)
    destination = ROOT/'results'/f'{args.tag}_summary'/'verification.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'runs': len(report['runs']), 'output': str(destination)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
