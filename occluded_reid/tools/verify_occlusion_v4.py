"""Audit the v4 adapter runs, baseline references, logs and target records."""
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
    protocol_path = ROOT/'research/occlusion_v4_protocol.json'
    protocol = read(protocol_path)
    directory = ROOT/'results/occlusion_v4_summary'
    manifest_path = directory/'external_freeze_manifest.json'
    manifest = read(manifest_path)
    assert manifest['protocol_sha256'] == digest(protocol_path), 'Protocol changed after training'
    assert manifest['target_data_record_sha256'] == protocol['target_data_record_sha256']
    for rel, checksum in manifest['source_sha256'].items():
        assert digest(ROOT/rel) == checksum, f'Frozen source changed: {rel}'
    assert [row['seed'] for row in manifest['runs']] == protocol['seeds']

    for ref in protocol['baseline_references']:
        checkpoint = ROOT/'runs'/ref['run']/'last.pt'
        assert checkpoint.is_file() and digest(checkpoint) == ref['checkpoint_sha256']
        baseline = read(ROOT/'results'/ref['run']/'occluded_reid.json')
        baseline_dev = read(ROOT/'results'/ref['run']/'results.json')['evaluation']
        baseline_dev_map = (baseline_dev['synthetic_noise_0.3']['mAP']+
                            baseline_dev['synthetic_noise_0.5']['mAP'])/2
        assert abs(baseline_dev_map-ref['development_map']) < 1e-9
        assert baseline['checkpoint_sha256'] == ref['checkpoint_sha256']
        assert baseline['target_data_record_sha256'] == protocol['target_data_record_sha256']
        assert abs(baseline['evaluation']['mAP']-ref['external_map']) < 1e-9
        assert abs(baseline['evaluation']['rank1']-ref['external_rank1']) < 1e-9

    records, target_hashes = [], set()
    for row in manifest['runs']:
        name = row['run']
        folder = ROOT/'results'/name
        config, history, result, external = [read(folder/f) for f in
            ('config.json', 'history.json', 'results.json', 'occluded_reid.json')]
        assert name == f"occlusion_v4_mdta_s{row['seed']}" and row['variant'] == 'mdta'
        assert config['variant'] == result['variant'] == external['variant'] == 'mdta'
        assert config['seed'] == result['seed'] == external['seed'] == row['seed']
        assert config['protocol_sha256'] == digest(protocol_path)
        assert config['source_sha256'] == manifest['source_sha256']
        assert row['development_result_sha256'] == digest(folder/'results.json')
        assert result['checkpoint_sha256'] == external['checkpoint_sha256'] == row['checkpoint_sha256']
        assert external['manifest_sha256'] == digest(manifest_path)
        assert external['target_data_record_sha256'] == manifest['target_data_record_sha256']
        target_hashes.add(external['target_data_record_sha256'])
        assert result['trained_epochs'] == protocol['epochs'] == len(history)
        assert len(config['split']['train_ids']) == 651 and len(config['split']['val_ids']) == 100
        assert not set(config['split']['train_ids']) & set(config['split']['val_ids'])
        checkpoint = ROOT/row['checkpoint']
        if require_checkpoints or checkpoint.exists():
            assert checkpoint.is_file() and digest(checkpoint) == row['checkpoint_sha256']
        logs = [json.loads(line) for line in (folder/'train.log').read_text(encoding='utf-8').splitlines()
                if line.startswith('{')]
        logged = {item['epoch']: {k: v for k, v in item.items() if k != 'event'}
                  for item in logs if item.get('event') == 'epoch'}
        for epoch, item in enumerate(history, 1):
            assert item['epoch'] == epoch and logged[epoch] == item
            assert item['pk_batches'] == protocol['pk_steps_per_epoch']
            assert item['optimizer_updates'] + item['amp_skipped_updates'] == item['pk_batches']
            assert all(math.isfinite(v) for v in item.values() if isinstance(v, (int, float)))
        metrics = external['evaluation']
        assert metrics['valid_queries'] == 1000 and metrics['gallery_size'] == 1000
        for key in ('mAP', 'rank1', 'rank5', 'mINP'):
            assert math.isfinite(metrics[key]) and 0 <= metrics[key] <= 100
        records.append({'run': name, 'updates': sum(item['optimizer_updates'] for item in history),
                        'amp_skips': sum(item['amp_skipped_updates'] for item in history),
                        'checkpoint_verified_locally': checkpoint.exists()})
    assert len(target_hashes) == 1, 'Target query/gallery record hashes differ across adapter seeds'
    return {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'runs': records, 'target_data_record_sha256': next(iter(target_hashes)),
            'checks': ['baseline checkpoint and metric references match v3',
                       'protocol, source and target record hashes frozen',
                       'split, recipe, epoch logs and optimizer updates match registration',
                       'external evaluations use frozen MDTA checkpoints and identical target records'],
            'limitations': ['Two seeds are not a significance test.',
                            'Occluded-REID was previously inspected; exploratory transfer only.',
                            'Audit checks recorded consistency, not novelty or causality.']}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-checkpoints', action='store_true')
    args = parser.parse_args()
    report = verify(args.require_checkpoints)
    (ROOT/'results/occlusion_v4_summary/verification.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'runs': len(report['runs'])}))


if __name__ == '__main__':
    main()
