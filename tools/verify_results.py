"""Audit the frozen short-budget experiment records before reporting them.

This checks provenance and protocol consistency, not scientific superiority.
It does not require the large local datasets or checkpoints.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_metrics(metrics, queries, gallery):
    for key in ('mAP', 'rank1', 'rank5', 'mINP'):
        require(math.isfinite(metrics[key]) and 0 <= metrics[key] <= 100,
                f'Invalid {key}: {metrics[key]}')
    require(metrics['valid_queries'] == queries, 'Unexpected query count')
    require(metrics['invalid_queries'] == 0, 'Invalid queries need investigation')
    require(metrics['gallery_size'] == gallery, 'Unexpected gallery count')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='frozen')
    parser.add_argument('--output', type=Path, default=ROOT/'research'/'verification.json')
    args = parser.parse_args()
    protocol = read(ROOT/'research'/'frozen_protocol.json')
    asset = read(ROOT/'research'/'asset_imagenet.json')
    checks = []
    line_ending_differences = set()
    reference = None
    shared_keys = ('seed', 'epochs', 'steps', 'batch_size', 'instances', 'height',
                   'width', 'workers', 'lr', 'head_lr_multiplier', 'fixbase_epochs',
                   'tau', 'calibration_weight', 'evidence_weight', 'split',
                   'source_sha256', 'pretrained_sha256', 'recipe_version')
    for variant in protocol['formal_variants']:
        name = f'{args.tag}_official_{variant}_s{protocol["seed"]}'
        folder = ROOT/'results'/name
        result = read(folder/'results.json')
        config = result['config']
        require(result['variant'] == config['variant'] == variant, f'{name}: variant')
        require(result['protocol'] == config['protocol'] == 'official', f'{name}: protocol')
        require(result['trained_epochs'] == config['epochs'] == protocol['epochs'], f'{name}: epochs')
        require(config['steps'] == protocol['pk_steps_per_epoch'], f'{name}: steps')
        require(config['seed'] == protocol['seed'], f'{name}: seed')
        require(config['batch_size'] == protocol['batch_size'], f'{name}: batch size')
        require(config['eval_every'] == 0, f'{name}: test-time selection')
        require(config['pretrained_sha256'] == asset['sha256'], f'{name}: initialization')
        require(not set(config['split']['train_ids']) & set(config['split']['test_ids']),
                f'{name}: train/test identity leakage')
        require(config['train_images'] == 12936 and len(config['split']['train_ids']) == 751,
                f'{name}: training data')
        for source, expected in config['source_sha256'].items():
            content = (ROOT/source).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                # Git on Windows may convert text between LF and CRLF. Accept
                # only that reversible conversion, not arbitrary source edits.
                normalized = content.replace(b'\r\n', b'\n')
                equivalents = (normalized, normalized.replace(b'\n', b'\r\n'))
                require(any(hashlib.sha256(value).hexdigest() == expected for value in equivalents),
                        f'{name}: source changed after training: {source}')
                line_ending_differences.add(source)
        if reference is not None:
            for key in shared_keys:
                require(config[key] == reference[key], f'{name}: unequal recipe: {key}')
        reference = config
        history = result['history']
        require([epoch['epoch'] for epoch in history] == list(range(1, protocol['epochs'] + 1)),
                f'{name}: incomplete training history')
        require(all('validation' not in epoch for epoch in history), f'{name}: interim test evaluation')
        require(all(math.isfinite(epoch['loss']) for epoch in history), f'{name}: nonfinite loss')
        archived_history = read(folder/'history.json')
        require(archived_history == history, f'{name}: history archive mismatch')
        require(read(folder/'config.json') == config, f'{name}: config archive mismatch')
        log = [json.loads(line) for line in (folder/'train.log').read_text(encoding='utf-8').splitlines()]
        steps = [(r['epoch'], r['step']) for r in log if r.get('event') == 'step']
        require(steps == [(epoch, step) for epoch in range(1, protocol['epochs'] + 1)
                          for step in (25, 50, 75, 100)], f'{name}: unexpected update log')
        require(set(result['evaluation']) == {'clean', 'synthetic_noise_0.3', 'synthetic_noise_0.5'},
                f'{name}: evaluation conditions')
        for metrics in result['evaluation'].values():
            check_metrics(metrics, 3368, 15913)
        external = read(folder/'occluded_reid.json')
        require(external['variant'] == variant and external['seed'] == protocol['seed']
                and external['epochs'] == protocol['epochs'] and external['train_protocol'] == 'official',
                f'{name}: external checkpoint mismatch')
        check_metrics(external['evaluation'], 1000, 1000)
        checks.append({'run': name, 'status': 'passed', 'updates': config['epochs']*config['steps'],
                       'inference_parameters': config['inference_parameters'],
                       'train_seconds': sum(epoch['train_seconds'] for epoch in history),
                       'peak_cuda_mb': max(epoch['peak_cuda_mb'] for epoch in history)})
    report = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'checks': checks, 'source_sha256': reference['source_sha256'],
              'source_line_ending_only_differences': sorted(line_ending_differences),
              'pretrained_sha256': asset['sha256'],
              'scope': 'Archive consistency, actual update logs, unchanged source, fixed recipe, '
                       'identity split, and metric counts/ranges. Does not establish novelty, '
                       'convergence, significance, or state-of-the-art performance.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'runs': checks}, indent=2))


if __name__ == '__main__':
    main()
