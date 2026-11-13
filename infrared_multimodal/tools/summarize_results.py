"""Audit completed four-way official pilots and generate an unselected summary."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ['agw', 'center', 'unbounded', 'bcrt']
METRICS = ['rank1', 'rank5', 'rank10', 'mAP', 'mINP']
COMPARISON_KEYS = ['dataset', 'data', 'protocol', 'trial', 'seed', 'epochs', 'steps',
                   'identities', 'instances', 'height', 'width', 'lr', 'weight', 'alpha',
                   'retain', 'warmup', 'workers', 'device', 'weights']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def audit(directory, dataset, variant):
    config, history, evaluation = [read(directory / f'{name}.json')
                                  for name in ['config', 'history', 'evaluation']]
    assert config['dataset'] == evaluation['dataset'] == dataset
    assert config['variant'] == variant
    assert config['protocol'] == evaluation['protocol'] == 'official'
    assert config['trial'] == evaluation['trial'] == 1 and config['seed'] == 1
    assert config['epochs'] == evaluation['epoch'] == len(history)
    assert [row['epoch'] for row in history] == list(range(1, len(history) + 1))
    assert not (set(config['training_ids']) & set(config['test_ids']))
    assert config['classes'] == len(config['training_ids'])
    assert all(row['steps'] == config['steps'] and row['skipped_updates'] == 0 for row in history)
    assert all(math.isfinite(value) for row in history for value in row.values()
               if isinstance(value, (int, float)))
    for relative, expected in config['source_hashes'].items():
        assert digest(ROOT / relative.replace('\\', '/')) == expected, relative
    assert not evaluation['reranking'] and not evaluation['test_flip']
    metrics = {}
    for protocol, result in evaluation['results'].items():
        if dataset == 'sysu':
            assert len(result['gallery_trials']) == 10
            for key, value in result['mean'].items():
                mean = math.fsum(t[key] for t in result['gallery_trials']) / 10
                assert math.isclose(value, mean, rel_tol=1e-10, abs_tol=1e-12)
            result = result['mean']
        assert all(0 <= result[key] <= 1 for key in METRICS)
        metrics[protocol] = {key: result[key] * 100 for key in METRICS}
    if dataset == 'regdb':
        metrics['bidirectional_mean'] = {
            key: math.fsum(row[key] for row in metrics.values()) / len(metrics)
            for key in METRICS}
    checkpoint = ROOT / 'runs' / directory.name / 'last.pt'
    return {
        'archive': str(directory.relative_to(ROOT)), 'dataset': dataset, 'variant': variant,
        'epochs': len(history), 'steps_per_epoch': config['steps'],
        'optimizer_updates': sum(row['steps'] for row in history),
        'training_ids': len(config['training_ids']), 'training_images': config['training_images'],
        'metrics_percent': metrics, 'last_epoch': history[-1],
        'artifact_sha256': {path.name: digest(path) for path in directory.glob('*.json')},
        'local_checkpoint_sha256': digest(checkpoint) if checkpoint.is_file() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='official_pilot_v1')
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in {'.', '..'}:
        raise ValueError('Use a simple directory-name tag.')
    runs = [audit(ROOT / 'results' / f'{args.tag}_{dataset}_official_{variant}_t1_s1', dataset, variant)
            for dataset in ['regdb', 'sysu'] for variant in VARIANTS]
    assert len({(r['epochs'], r['steps_per_epoch']) for r in runs}) == 1
    for dataset in ['regdb', 'sysu']:
        configs = [read(ROOT / 'results' / f'{args.tag}_{dataset}_official_{variant}_t1_s1' / 'config.json')
                   for variant in VARIANTS]
        baseline = {key: configs[0][key] for key in COMPARISON_KEYS}
        assert all({key: config[key] for key in COMPARISON_KEYS} == baseline for config in configs[1:])
        assert len({json.dumps(config['training_ids']) for config in configs}) == 1
        assert len({json.dumps(config['test_ids']) for config in configs}) == 1
    epochs, steps = runs[0]['epochs'], runs[0]['steps_per_epoch']
    lines = [f'# Official short-budget pilot: {args.tag}', '',
             f'Fixed final epoch; {epochs} epochs x {steps} steps, trial 1 / seed 1 only. '
             'No reranking, flip ensemble, test-selected checkpoint or full-convergence claim.', '',
             'Metrics below are percentages. SYSU means use 10 single-shot gallery samples; '
             'these are not 10 independent training seeds.', '',
             '| Variant | RegDB V-to-I R1 / mAP | RegDB I-to-V R1 / mAP | '
             'RegDB mean R1 / mAP | SYSU all R1 / mAP | SYSU indoor R1 / mAP |',
             '|---|---:|---:|---:|---:|---:|']
    def pair(row):
        return f"{row['rank1']:.2f} / {row['mAP']:.2f}"
    for variant in VARIANTS:
        regdb, sysu = [next(r['metrics_percent'] for r in runs
                           if r['dataset'] == dataset and r['variant'] == variant)
                       for dataset in ['regdb', 'sysu']]
        cells = [variant, pair(regdb['visible_to_infrared']), pair(regdb['infrared_to_visible']),
                 pair(regdb['bidirectional_mean']), pair(sysu['all_single_shot']),
                 pair(sysu['indoor_single_shot'])]
        lines.append('| ' + ' | '.join(cells) + ' |')
    notes = ['Matched finite training histories; zero skipped updates; disjoint train/test identities; '
             'current source hashes match all archived configurations.',
             'Single-seed short pilots do not establish significance, novelty or a complete AGW reproduction.',
             'Boundary diagnostics certify fixed batch Euclidean prototypes only, not retrieval generalization.']
    lines += ['', *notes, '']
    output = ROOT / 'results' / f'{args.tag}_summary'
    output.with_suffix('.json').write_text(json.dumps({
        'kind': 'SHORT_BUDGET_OFFICIAL_PROTOCOL_PILOT_NOT_FULL_REPRODUCTION',
        'units': 'percent', 'audit_passed': True, 'notes': notes, 'runs': runs}, indent=2), encoding='utf-8')
    output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
