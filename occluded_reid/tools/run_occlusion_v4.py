"""Train the registered MDTA seeds, freeze all runs, then evaluate them once."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from reid.data import scan_occluded
from train_occlusion_v4 import record_manifest, source_hashes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    protocol_path = ROOT/'research/occlusion_v4_protocol.json'
    protocol = read(protocol_path)
    protocol_sha = digest(protocol_path)
    query, gallery = scan_occluded(ROOT/'data')
    target_sha = record_manifest(query+gallery, ROOT/'data')
    if target_sha != protocol['target_data_record_sha256']:
        raise ValueError('Target records differ from the pre-registered dataset hash')

    for ref in protocol['baseline_references']:
        checkpoint = ROOT/'runs'/ref['run']/'last.pt'
        if not checkpoint.is_file() or digest(checkpoint) != ref['checkpoint_sha256']:
            raise ValueError(f"Frozen baseline reference mismatch: {ref['run']}")
        baseline = read(ROOT/'runs'/ref['run']/'results.json')
        external = read(ROOT/'results'/ref['run']/'occluded_reid.json')
        baseline_dev = baseline['evaluation']
        baseline_dev_map = (baseline_dev['synthetic_noise_0.3']['mAP']+
                            baseline_dev['synthetic_noise_0.5']['mAP'])/2
        if abs(baseline_dev_map-ref['development_map']) > 1e-9:
            raise ValueError(f"Baseline development metric mismatch: {ref['run']}")
        if external['target_data_record_sha256'] != target_sha:
            raise ValueError(f"Baseline target record mismatch: {ref['run']}")

    runs = []
    for seed in protocol['seeds']:
        name = f'occlusion_v4_mdta_s{seed}'
        output, archive = ROOT/'runs'/name, ROOT/'results'/name
        archive.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(ROOT/'code/train_occlusion_v4.py'),
                   '--seed', str(seed), '--output', str(output)]
        if (output/'last.pt').exists() and not (output/'results.json').exists():
            command += ['--resume', str(output/'last.pt')]
        if not (output/'results.json').exists():
            print(json.dumps({'event': 'training', 'run': name}), flush=True)
            with (archive/'train.log').open('a', encoding='utf-8') as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f'{name} failed; preserve its log/checkpoint before recovery')
        result = read(output/'results.json')
        if result['trained_epochs'] != protocol['epochs']:
            raise ValueError(f'Incomplete run: {name}')
        if result['checkpoint_sha256'] != digest(output/'last.pt'):
            raise ValueError(f'Checkpoint hash mismatch: {name}')
        for filename in ('config.json', 'history.json', 'results.json'):
            shutil.copy2(output/filename, archive/filename)
        runs.append({'run': name, 'variant': 'mdta', 'seed': seed,
                     'checkpoint': (output/'last.pt').relative_to(ROOT).as_posix(),
                     'checkpoint_sha256': digest(output/'last.pt'),
                     'development_result_sha256': digest(archive/'results.json')})
        print(json.dumps({'event': 'development_complete', 'run': name,
                          'evaluation': result['evaluation']}), flush=True)

    summary_dir = ROOT/'results/occlusion_v4_summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = summary_dir/'external_freeze_manifest.json'
    manifest = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': protocol_sha, 'source_sha256': source_hashes(),
                'target_data_record_sha256': target_sha, 'runs': runs,
                'rule': 'All registered development runs completed before freeze. Evaluate all frozen checkpoints; never tune against Occluded-REID.'}
    if manifest_path.exists():
        previous = read(manifest_path)
        if (previous['runs'] != runs or previous['protocol_sha256'] != protocol_sha or
                previous['target_data_record_sha256'] != target_sha or
                previous['source_sha256'] != manifest['source_sha256']):
            raise ValueError('Existing freeze manifest differs; preserve it and use a new protocol/tag')
        manifest = previous
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for row in runs:
        output = ROOT/'results'/row['run']/'occluded_reid.json'
        if output.exists():
            old = read(output)
            if old['checkpoint_sha256'] != row['checkpoint_sha256'] or old['manifest_sha256'] != digest(manifest_path):
                raise ValueError('Existing external result does not match frozen checkpoint and manifest')
            continue
        print(json.dumps({'event': 'external_evaluation', 'run': row['run']}), flush=True)
        subprocess.run([sys.executable, '-u', str(ROOT/'code/evaluate_occlusion_v4.py'),
                        '--checkpoint', str(ROOT/row['checkpoint']), '--manifest', str(manifest_path),
                        '--output', str(output)], cwd=ROOT, check=True)
    print(json.dumps({'event': 'suite_complete', 'runs': len(runs)}), flush=True)


if __name__ == '__main__':
    main()
