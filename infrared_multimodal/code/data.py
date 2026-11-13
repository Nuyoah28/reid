"""Raw-image RegDB / SYSU-MM01 loaders and paired identity sampling.

Datasets must be obtained under their owners' terms; this module downloads none.
"""
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T


@dataclass(frozen=True)
class Record:
    path: str
    pid: int
    modality: int  # 0 visible, 1 infrared (converted to 3 channels)
    camera: int


def regdb(root, split, trial):
    if not 1 <= trial <= 10:
        raise ValueError('RegDB trial must be in 1..10.')
    root = Path(root).resolve()
    records = []
    for modality, name in enumerate(['visible', 'thermal']):
        index = root / 'idx' / f'{split}_{name}_{trial}.txt'
        if not index.is_file():
            raise FileNotFoundError(f'Missing authorized RegDB data: {index}')
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            filename, label = line.rsplit(maxsplit=1)
            path = root / filename.replace('\\', '/').lstrip('/')
            if not path.is_file():
                raise FileNotFoundError(path)
            # Official index labels restart at 0 independently in train/test.
            # Original person folders provide a common identity namespace.
            identity = int(Path(filename.replace('\\', '/')).parent.name)
            records.append(Record(str(path), identity, modality, modality + 1))
    return records


def sysu(root, split):
    root = Path(root).resolve()
    ids = set()
    for name in (['train', 'val'] if split == 'train' else ['test']):
        file = root / 'exp' / f'{name}_id.txt'
        if not file.is_file():
            raise FileNotFoundError(f'Missing authorized SYSU-MM01 data: {file}')
        ids.update(int(x) for x in file.read_text().strip().replace('\n', ',').split(',') if x.strip())
    records = []
    for pid in sorted(ids):
        for camera in range(1, 7):
            folder = root / f'cam{camera}' / f'{pid:04d}'
            for path in sorted(folder.glob('*')):
                if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    records.append(Record(str(path), pid, int(camera in [3, 6]), camera))
    if not records:
        raise FileNotFoundError(f'No SYSU images found in {root}')
    return records


def read_records(dataset, root, split, trial=1):
    return regdb(root, split, trial) if dataset == 'regdb' else sysu(root, split)


def train_partition(records, protocol, seed=17, fraction=.2):
    ids = sorted({r.pid for r in records})
    if len(ids) < 4:
        raise ValueError('At least four training identities are needed.')
    held = set()
    if protocol == 'dev':
        held = set(random.Random(seed).sample(ids, max(2, int(len(ids) * fraction))))
    training = [r for r in records if r.pid not in held]
    validation = [r for r in records if r.pid in held]
    mapping = {pid: i for i, pid in enumerate(sorted({r.pid for r in training}))}
    return training, validation, mapping


def sysu_gallery(records, mode='all', trial=0):
    cameras = {1, 2, 4, 5} if mode == 'all' else {1, 2}
    groups = {}
    for record in records:
        if record.camera in cameras:
            groups.setdefault((record.pid, record.camera), []).append(record)
    rng = random.Random(trial)
    return [rng.choice(sorted(group, key=lambda r: r.path)) for _, group in sorted(groups.items())]


class Images(Dataset):
    def __init__(self, records, height=288, width=144, training=False, mapping=None):
        self.records, self.mapping = records, mapping
        operations = [T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR)]
        if training:
            operations += [T.Pad(10), T.RandomCrop((height, width)), T.RandomHorizontalFlip()]
        operations += [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])]
        self.transform = T.Compose(operations)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row.path) as image:
            x = self.transform(image.convert('RGB'))
        label = row.pid if self.mapping is None else self.mapping[row.pid]
        return x, label, row.modality, row.camera


class PairedPK(Sampler):
    """P identities, K VIS + K IR images each; batch order VIS then IR."""
    def __init__(self, records, identities=4, instances=2, steps=0, seed=1):
        self.groups = {}
        for index, row in enumerate(records):
            self.groups.setdefault((row.pid, row.modality), []).append(index)
        self.ids = sorted({pid for pid, _ in self.groups})
        if identities < 2 or instances < 1 or len(self.ids) < identities:
            raise ValueError('Need P>=2, K>=1 and at least P paired identities.')
        if any((pid, modality) not in self.groups for pid in self.ids for modality in [0, 1]):
            raise ValueError('Each training identity needs both modalities.')
        self.p, self.k, self.seed, self.epoch = identities, instances, seed, 0
        self.steps = steps or max(1, len(records) // (2 * identities * instances))

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            chosen = rng.choice(self.ids, self.p, replace=False)
            yield [int(i) for modality in [0, 1] for pid in chosen
                   for i in rng.choice(self.groups[(int(pid), modality)], self.k,
                                       replace=len(self.groups[(int(pid), modality)]) < self.k)]
