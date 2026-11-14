"""Execute the pre-registered matrix sequentially, then freeze/evaluate ALL runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from train_occlusion_v2 import source_hashes


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='peer_v2')
    args = p.parse_args()
    if not args.tag.replace('_', '').replace('-', '').isalnum():
        p.error('Use a simple alphanumeric tag')
    protocol_path = ROOT/'research/occlusion_v2_protocol.json'
    protocol = read(protocol_path)
    current_sources = source_hashes()
    runs = []
    for seed in protocol['seeds']:
        for variant in protocol['variants_in_order']:
            name = f'{args.tag}_{variant}_s{seed}'
            output, archive = ROOT/'runs'/name, ROOT/'results'/name
            if archive.exists() and any(archive.iterdir()) and not (output/'config.json').exists():
                raise FileExistsError(f'Archive without local checkpoint: {name}; use another tag')
            archive.mkdir(parents=True, exist_ok=True)
            if (output/'config.json').exists():
                old = read(output/'config.json')
                for key, expected in {'variant': variant, 'seed': seed, 'epochs': protocol['epochs'],
                                      'steps': protocol['pk_steps_per_epoch'],
                                      'registered_protocol_sha256': digest(protocol_path),
                                      'source_sha256': current_sources}.items():
                    if old[key] != expected:
                        raise ValueError(f'Cannot resume {name}: changed {key}')
            command = [sys.executable, '-u', str(ROOT/'code/train_occlusion_v2.py'), '--variant', variant,
                       '--seed', str(seed), '--epochs', str(protocol['epochs']),
                       '--steps', str(protocol['pk_steps_per_epoch']), '--workers', '0', '--output', str(output)]
            if (output/'last.pt').exists():
                command += ['--resume', str(output/'last.pt')]
            if not (output/'results.json').exists():
                print(json.dumps({'event': 'training', 'run': name}), flush=True)
                with (archive/'train.log').open('a', encoding='utf-8') as log:
                    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f'{name} failed; inspect {archive}/train.log')
            result = read(output/'results.json')
            if result['trained_epochs'] != protocol['epochs'] or digest(output/'last.pt') != result['checkpoint_sha256']:
                raise ValueError(f'Incomplete or mismatched checkpoint: {name}')
            for filename in ('config.json', 'history.json', 'results.json'):
                shutil.copy2(output/filename, archive/filename)
            runs.append({'run': name, 'variant': variant, 'seed': seed,
                         'checkpoint': (output/'last.pt').relative_to(ROOT).as_posix(),
                         'checkpoint_sha256': digest(output/'last.pt'),
                         'development_result_sha256': digest(archive/'results.json')})
            print(json.dumps({'event': 'development_complete', 'run': name,
                              'evaluation': result['evaluation']}), flush=True)
    manifest_dir = ROOT/'results'/f'{args.tag}_summary'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir/'external_freeze_manifest.json'
    manifest = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                'registered_protocol_sha256': digest(protocol_path), 'runs': runs,
                'rule': 'All registered development runs finished before any v2 external evaluation. Evaluate every frozen run; do not tune against target results.'}
    if manifest_path.exists():
        old = read(manifest_path)
        if old['runs'] != runs or old['registered_protocol_sha256'] != manifest['registered_protocol_sha256']:
            raise ValueError('Existing external freeze manifest differs; never overwrite it')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for run in runs:
        output = ROOT/'results'/run['run']/'occluded_reid.json'
        if output.exists():
            old = read(output)
            if old['checkpoint_sha256'] != run['checkpoint_sha256'] or old['manifest_sha256'] != digest(manifest_path):
                raise ValueError('Existing external evaluation does not match its frozen checkpoint/manifest')
            continue
        print(json.dumps({'event': 'external_evaluation', 'run': run['run']}), flush=True)
        subprocess.run([sys.executable, '-u', str(ROOT/'code/evaluate_occlusion_v2.py'),
                        '--checkpoint', str(ROOT/run['checkpoint']), '--manifest', str(manifest_path),
                        '--output', str(output)], cwd=ROOT, check=True)
    print(json.dumps({'event': 'suite_complete', 'runs': len(runs)}), flush=True)


if __name__ == '__main__':
    main()
