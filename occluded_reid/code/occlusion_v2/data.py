"""Source-image-aware PK sampling for unbiased leave-image-out supervision."""
import random
from reid.data import Images, PKSampler


class IndexedImages(Images):
    def __getitem__(self, index):
        return (*super().__getitem__(index), index)


class PeerPKSampler(PKSampler):
    """Keep all train IDs; guarantee another source image for small-ID pools.

    For pools smaller than K, first visit every image once, then fill the rest.
    The same sampler is used by ALL v2 baselines and candidates.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.k < 2 or any(len(pool) < 2 for pool in self.groups.values()):
            raise ValueError('Peer supervision requires K>=2 and >=2 source images per identity')

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch*1009)
        for _ in range(self.steps):
            for pid in rng.sample(self.ids, self.p):
                pool = self.groups[pid]
                if len(pool) >= self.k:
                    yield from rng.sample(pool, self.k)
                else:
                    draw = rng.sample(pool, len(pool)) + rng.choices(pool, k=self.k-len(pool))
                    rng.shuffle(draw)
                    yield from draw
