"""Compare moved v1 files against the original Git tree, allowing checkout EOLs."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = '3acc4de06c3c336478e32f088ff52ef814bc5b39'


def main():
    listing = subprocess.check_output(['git', 'ls-tree', '-rz', BASE, '--', 'code/', 'results/'], cwd=ROOT)
    entries = []
    for line in listing.split(b'\0'):
        if line:
            metadata, path = line.split(b'\t', 1)
            entries.append((metadata.split()[2].decode(), path.decode('utf-8')))
    response = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT,
                              input=('\n'.join(oid for oid, _ in entries)+'\n').encode(),
                              stdout=subprocess.PIPE, check=True).stdout
    offset, rows = 0, []
    for oid, original in entries:
        end = response.index(b'\n', offset)
        _, kind, size = response[offset:end].split()
        assert kind == b'blob'
        size = int(size)
        blob = response[end+1:end+1+size]
        offset = end + size + 2
        destination = ROOT/'occluded_reid'/original
        current = destination.read_bytes() if destination.is_file() else None
        status = 'missing' if current is None else 'changed'
        if current == blob:
            status = 'identical_bytes'
        elif current is not None and b'\0' not in current and b'\0' not in blob:
            try:
                current.decode('utf-8'); blob.decode('utf-8')
                if current.replace(b'\r\n', b'\n') == blob.replace(b'\r\n', b'\n'):
                    status = 'checkout_line_endings_only'
            except UnicodeDecodeError:
                pass
        rows.append({'original': original, 'destination': destination.relative_to(ROOT).as_posix(),
                     'git_blob': oid, 'comparison': status,
                     'git_sha256': hashlib.sha256(blob).hexdigest(),
                     'working_file_sha256': hashlib.sha256(current).hexdigest() if current is not None else None})
    counts = dict(Counter(row['comparison'] for row in rows))
    passed = all(row['comparison'] in ('identical_bytes', 'checkout_line_endings_only') for row in rows)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'base_commit': BASE,
              'status': 'passed' if passed else 'failed', 'counts': counts, 'files': rows,
              'scope': 'All previously tracked v1 code and result files moved to occluded_reid. New v2 files and intentionally edited documentation/tools are outside this comparison.',
              'interpretation': 'Git blobs are LF-normalized; this Windows checkout uses core.autocrlf=true. Initial strict blob-to-working-file byte comparison reported these checkout differences. This audit separately reports exact bytes and UTF-8 text differing ONLY by CRLF/LF; it does not rewrite files. Raw source hashes recorded by v1/v2 experiments are checked by their experiment verifiers.',
              'limitation': 'The Git tree is not a pre-migration working-directory byte snapshot. Content equivalence to Git is verified; historical working-file bytes are established only where experiment hashes exist.'}
    output = ROOT/'occluded_reid/research/workspace_migration_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'files': len(rows), 'counts': counts}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
