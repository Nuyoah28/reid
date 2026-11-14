import copy
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,Sampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF

PATTERN=re.compile(r'(-?\d+)_c(\d+)')

def scan_market(folder):
    records=[]
    for path in sorted(Path(folder).glob('*.jpg')):
        match=PATTERN.search(path.name)
        if match is None:raise ValueError(f'Invalid Market filename {path}')
        pid,cam=map(int,match.groups())
        if pid!=-1:records.append((str(path),pid,cam-1))
    if not records:raise FileNotFoundError(f'No images under {folder}')
    return records


def scan_occluded(root):
    """Authors' Occluded-REID: 200 held-out IDs, 1,000 queries/1,000 gallery."""
    root=Path(root)
    groups=[]
    for subfolder,cam in [('occluded_body_images',0),('whole_body_images',1)]:
        records=[]
        for path in sorted((root/subfolder).glob('*/*.tif')):
            pid=int(path.stem.split('_')[0])
            records.append((str(path),pid,cam))
        if not records:raise FileNotFoundError(f'No Occluded-REID images under {root/subfolder}')
        groups.append(records)
    query,gallery=groups
    if len(query)!=1000 or len(gallery)!=1000:raise ValueError('Expected 1000 occluded/1000 whole images')
    qids={x[1] for x in query};gids={x[1] for x in gallery}
    if qids!=gids or len(qids)!=200:raise ValueError('Expected identical 200 query/gallery identities')
    return query,gallery


def training_split(root,holdout_ids=100,seed=2026):
    """Reserve train identities for tuning; official test identities never enter training."""
    records=scan_market(Path(root)/'bounding_box_train')
    ids=sorted({r[1] for r in records});random.Random(seed).shuffle(ids)
    if not 0<holdout_ids<len(ids):raise ValueError('Invalid validation ID count')
    val_ids=set(ids[:holdout_ids]);train_ids=set(ids[holdout_ids:])
    train=[r for r in records if r[1] in train_ids]
    validation=[r for r in records if r[1] in val_ids]
    query=[];gallery=[]
    groups=defaultdict(list)
    for record in validation:groups[record[1]].append(record)
    for pid,group in groups.items():
        cameras={r[2] for r in group}
        if len(cameras)<2:continue
        q=group[0];query.append(q);gallery.extend(r for r in group if r!=q)
    return train,query,gallery,{'train_ids':sorted(train_ids),'val_ids':sorted(val_ids),'split_seed':seed}


class Images(Dataset):
    def __init__(self,records,height=256,width=128,train=False,occlusion=0.,seed=0,fill='noise'):
        self.records=records;self.train=train;self.occlusion=occlusion;self.seed=seed;self.fill=fill
        self.pid_map={pid:i for i,pid in enumerate(sorted({r[1] for r in records}))}
        operations=[T.Resize((height,width),interpolation=T.InterpolationMode.BICUBIC)]
        if train:operations += [T.RandomHorizontalFlip(),T.Pad(10),T.RandomCrop((height,width))]
        operations += [T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.transform=T.Compose(operations)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        path,pid,cam=self.records[index]
        with Image.open(path) as image:value=self.transform(image.convert('RGB'))
        if self.occlusion:
            # Stable per filename; independent of OS/Python hash randomization and global RNG.
            digest=int.from_bytes(hashlib.sha256((Path(path).name+str(self.seed)).encode()).digest()[:8],'little')
            rng=random.Random(digest);_,h,w=value.shape
            rh=max(1,round(h*self.occlusion));y=rng.randrange(h-rh+1)
            if self.fill=='noise':
                generator=torch.Generator().manual_seed(digest%(2**63-1))
                value[:,y:y+rh,:]=torch.randn((3,rh,w),generator=generator)
            else:value[:,y:y+rh,:]=0
        return value,self.pid_map[pid] if self.train else pid,cam


class PKSampler(Sampler):
    """P identities and K instances; enough positives for batch-hard triplet loss."""
    def __init__(self,records,batch_size=32,instances=4,seed=1,steps=None):
        if batch_size%instances or batch_size//instances<2:raise ValueError('Require P>=2 and batch_size=P*K')
        self.groups=defaultdict(list)
        for index,(_,pid,_) in enumerate(records):self.groups[pid].append(index)
        self.ids=sorted(self.groups);self.p=batch_size//instances;self.k=instances
        if len(self.ids)<self.p:raise ValueError('Insufficient training identities')
        self.seed=seed;self.epoch=0;self.steps=steps or len(records)//batch_size
    def __len__(self):return self.steps*self.p*self.k
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch*1009)
        for _ in range(self.steps):
            for pid in rng.sample(self.ids,self.p):
                pool=self.groups[pid]
                yield from rng.sample(pool,self.k) if len(pool)>=self.k else rng.choices(pool,k=self.k)
