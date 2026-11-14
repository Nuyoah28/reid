"""Inspect and safely extract user-provided ZIP files into the isolated project."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    for pattern, name in [('RegDB*.zip', 'regdb_archive'), ('SYSU-MM01.zip', 'sysu_archive')]:
        paths = list((ROOT/'data/archives').glob(pattern))
        if not paths:
            paths = list(ROOT.parent.glob(pattern))  # Compatibility with pre-reorganization checkouts.
        if len(paths) != 1:
            raise ValueError(f'Expected one {pattern}, got {paths}')
        source = paths[0]
        destination = ROOT/'data'/name
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            files = archive.infolist()
            print(name, 'files=', len(files), 'first=', [f.filename for f in files[:8]], flush=True)
            for member in files:
                normalized = member.filename.replace('\\', '/')
                relative = PurePosixPath(normalized)
                if relative.is_absolute() or '..' in relative.parts or ':' in normalized:
                    raise ValueError(f'Unsafe archive member: {member.filename}')
                target = (destination/normalized).resolve()
                target.relative_to(destination.resolve())
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists() or target.stat().st_size != member.file_size:
                        with archive.open(member) as input_file, target.open('wb') as output:
                            while chunk := input_file.read(2**20):
                                output.write(chunk)
            bad = archive.testzip()
            if bad:
                raise ValueError(f'Bad ZIP CRC: {bad}')
        records.append({'archive': source.name, 'archive_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'files': len(files), 'destination': str(destination), 'zip_crc': 'passed',
                        'provenance': 'User-provided local archive; not redistributed.'})
    (ROOT/'research/dataset_archives.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
