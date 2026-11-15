"""Run the registered v5 matrix, freeze all checkpoints, then evaluate target data."""
from datetime import datetime, timezone
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from train_occlusion_v5 import VARIANTS, source_hashes
from reid.data import scan_occluded


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-plan', action='store_true',
                        help='list the registered runs without starting training')
    args = parser.parse_args()
    protocol_path = ROOT/'research/occlusion_v5_protocol.json'
    protocol = read(protocol_path)
    if args.print_plan:
        names = [f"{protocol['run_tag']}_{variant}_s{seed}"
                 for seed in protocol['seeds'] for variant in protocol['variants_in_order']]
        print(json.dumps({'runs': names, 'total_runs': len(names),
                          'optimizer_updates_per_run': protocol['epochs']*protocol['steps_per_epoch']},
                         ensure_ascii=False, indent=2))
        return
    protocol_sha = digest(protocol_path)
    runs = []
    for seed in protocol['seeds']:
        for variant in protocol['variants_in_order']:
            name = f"{protocol['run_tag']}_{variant}_s{seed}"
            output, archive = ROOT/'runs'/name, ROOT/'results'/name
            archive.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(ROOT/'code/train_occlusion_v5.py'),
                       '--variant', variant, '--seed', str(seed), '--output', str(output)]
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
            runs.append({'run': name, 'variant': variant, 'seed': seed,
                         'checkpoint': output.joinpath('last.pt').relative_to(ROOT).as_posix(),
                         'checkpoint_sha256': digest(output/'last.pt'),
                         'development_result_sha256': digest(archive/'results.json')})
            print(json.dumps({'event': 'development_complete', 'run': name,
                              'evaluation': result['evaluation']}), flush=True)

    frozen_sources = source_hashes()
    for row in runs:
        config = read(ROOT/'results'/row['run']/'config.json')
        if config['source_sha256'] != frozen_sources or config['protocol_sha256'] != protocol_sha:
            raise ValueError(f"Source or protocol changed during the registered matrix: {row['run']}")
    target_query, target_gallery = scan_occluded(ROOT/'data')
    target_rows = ['%s\t%d\t%d\t%s' % (Path(path).relative_to(ROOT/'data').as_posix(), pid, camera,
                                         digest(Path(path)))
                   for path, pid, camera in target_query+target_gallery]
    target_hash = hashlib.sha256('\n'.join(target_rows).encode()).hexdigest()

    summary_dir = ROOT/'results/occlusion_v5_summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = summary_dir/'external_freeze_manifest.json'
    manifest = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': protocol_sha, 'source_sha256': frozen_sources,
                'target_data_record_sha256': target_hash,
                'runs': runs,
                'rule': 'All registered development runs completed before freeze. Evaluate every frozen checkpoint; never tune against Occluded-REID.'}
    if manifest_path.exists():
        previous = read(manifest_path)
        if previous['runs'] != runs or previous['protocol_sha256'] != protocol_sha or \
                previous['source_sha256'] != frozen_sources or \
                previous['target_data_record_sha256'] != target_hash:
            raise ValueError('Existing freeze manifest differs; preserve it and use a new protocol/tag')
        manifest = previous
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for row in runs:
        output = ROOT/'results'/row['run']/'occluded_reid.json'
        if output.exists():
            old = read(output)
            if old['checkpoint_sha256'] != row['checkpoint_sha256'] or \
                    old['manifest_sha256'] != digest(manifest_path):
                raise ValueError('Existing external result does not match frozen checkpoint and manifest')
            continue
        print(json.dumps({'event': 'external_evaluation', 'run': row['run']}), flush=True)
        subprocess.run([sys.executable, '-u', str(ROOT/'code/evaluate_occlusion_v5.py'),
                        '--checkpoint', str(ROOT/row['checkpoint']), '--manifest', str(manifest_path),
                        '--output', str(output)], cwd=ROOT, check=True)
    print(json.dumps({'event': 'suite_complete', 'runs': len(runs)}), flush=True)


if __name__ == '__main__':
    main()
