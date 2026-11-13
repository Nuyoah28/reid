"""Check identity leakage, image counts and original official split files."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from data import read_records, train_partition


def main():
    report = {}
    roots = {'regdb': ROOT/'data/regdb_archive/RegDB', 'sysu': ROOT/'data/sysu_archive'}
    for dataset, root in roots.items():
        trials = []
        for trial in range(1, 11) if dataset == 'regdb' else [1]:
            train, test = [read_records(dataset, root, s, trial) for s in ['train', 'test']]
            train_ids, test_ids = [{r.pid for r in rows} for rows in [train, test]]
            if train_ids & test_ids or {r.path for r in train} & {r.path for r in test}:
                raise AssertionError('Train/test overlap.')
            dev, held, _ = train_partition(train, 'dev')
            for rows in [dev, held, test]:
                for pid in {r.pid for r in rows}:
                    if {r.modality for r in rows if r.pid == pid} != {0, 1}:
                        raise AssertionError(f'{dataset}: missing modality for person {pid}.')
            trials.append({'trial': trial, 'train_ids': len(train_ids), 'test_ids': len(test_ids),
                           'train_visible': sum(r.modality == 0 for r in train),
                           'train_infrared': sum(r.modality == 1 for r in train),
                           'test_visible': sum(r.modality == 0 for r in test),
                           'test_infrared': sum(r.modality == 1 for r in test),
                           'dev_train_ids': len({r.pid for r in dev}), 'dev_heldout_ids': len({r.pid for r in held}),
                           'identity_overlap': 0})
        bad, total = [], 0
        # Decode every supplied benchmark image once, including identities not used in a split.
        for path in root.rglob('*'):
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                total += 1
                try:
                    with Image.open(path) as image:
                        image.load()
                except Exception as exc:
                    bad.append({'file': str(path), 'error': str(exc)})
        if bad:
            raise ValueError(bad)
        splits = list((root/('idx' if dataset == 'regdb' else 'exp')).glob('*.txt'))
        ignored = ('RegDB *_sym_*.npy label arrays are not used; identity is parsed from original image folders.'
                   if dataset == 'regdb' else 'Only official exp/train_id.txt, val_id.txt and test_id.txt splits are used.')
        report[dataset] = {'root': str(root), 'trials': trials, 'decoded_images': total, 'corrupt_images': bad,
                           'split_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(splits)},
                           'ignored_extras': ignored}
    (ROOT/'research/data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
