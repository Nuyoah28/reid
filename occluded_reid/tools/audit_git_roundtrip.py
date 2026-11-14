"""Verify staged experiment files reproduce recorded bytes after Git checkout."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifest = ROOT/'results/peer_v2_summary/external_freeze_manifest.json'
    frozen = read(manifest)
    paths = {manifest, ROOT/'research/occlusion_v2_protocol.json',
             ROOT/'research/occlusion_v2_evaluation_source_freeze.json'}
    evaluator = read(ROOT/'research/occlusion_v2_evaluation_source_freeze.json')
    paths.update(ROOT/p for p in evaluator['sources'])
    for row in frozen['runs']:
        folder = ROOT/'results'/row['run']
        paths.update(folder/name for name in ('config.json', 'history.json', 'results.json',
                                             'train.log', 'occluded_reid.json'))
        paths.update(ROOT/p for p in read(folder/'config.json')['source_sha256'])
    rows = []
    for path in sorted(paths):
        relative = path.relative_to(REPO).as_posix()
        # --filters applies the index path's checkout (including eol) rules.
        checkout = subprocess.check_output(['git', 'cat-file', '--filters', ':'+relative], cwd=REPO)
        original = path.read_bytes()
        rows.append({'path': relative, 'passed': checkout == original,
                     'working_sha256': hashlib.sha256(original).hexdigest(),
                     'checkout_sha256': hashlib.sha256(checkout).hexdigest()})
    passed = all(row['passed'] for row in rows)
    report = {'status': 'passed' if passed else 'failed',
              'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'files': rows,
              'scope': 'Staged v2 frozen source, protocol, manifest and raw result bytes; Git checkout filters applied.',
              'limitation': 'Checks staged index content and checkout filters, not a rerun or a guarantee of deterministic CUDA training.'}
    output = ROOT/'research/git_checkout_byte_audit.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'files': len(rows),
                      'failures': [r['path'] for r in rows if not r['passed']]}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
