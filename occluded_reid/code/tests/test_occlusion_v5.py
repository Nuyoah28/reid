import sys
import tempfile
import unittest
from pathlib import Path
from collections import Counter

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reid.factorial_occlusion import factorial_occlusion, interaction_penalty
from reid.data import Images
from train_occlusion_v5 import DonorOcclusionQueries


class FactorialOcclusionTests(unittest.TestCase):
    def test_twelve_groups_balance_six_identity_weights(self):
        images = torch.randn(24, 3, 16, 8)
        labels = torch.arange(6).repeat_interleave(4)
        _, _, metadata = factorial_occlusion(
            images, labels, generator=torch.Generator().manual_seed(19), groups=12)
        counts = Counter(identity for group in metadata for identity in group['identities'])
        self.assertEqual(counts, Counter({identity: 4 for identity in range(6)}))

    def test_shared_donor_and_geometry_across_target_identities(self):
        images = torch.arange(8*3*16*8, dtype=torch.float32).reshape(8, 3, 16, 8)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        views, targets, metadata = factorial_occlusion(
            images, labels, generator=torch.Generator().manual_seed(7), groups=3)
        self.assertEqual(tuple(views.shape), (3, 4, 3, 16, 8))
        self.assertEqual(tuple(targets.shape), (3, 4))
        for group, info in enumerate(metadata):
            top, left, bottom, right = info['box']
            self.assertTrue(torch.equal(views[group, 0, :, top:bottom, left:right],
                                        views[group, 2, :, top:bottom, left:right]))
            self.assertTrue(torch.equal(views[group, 1, :, top:bottom, left:right],
                                        views[group, 3, :, top:bottom, left:right]))
            first_id, second_id = info['identities']
            self.assertEqual(targets[group].tolist(), [first_id, first_id, second_id, second_id])

    def test_interaction_penalty_distinguishes_identity_conditioned_effect(self):
        offset = 0.3
        orthogonal = (1-offset**2)**0.5
        shared_change = torch.tensor([[[offset, orthogonal, 0.], [-offset, orthogonal, 0.],
                                       [offset, 0., orthogonal], [-offset, 0., orthogonal]]])
        value = interaction_penalty(shared_change)
        self.assertLess(value.item(), 1e-6)
        identity_specific = torch.tensor([[[1., 0.], [0., 1.], [1., 1.], [1., 1.]]])
        self.assertGreater(interaction_penalty(identity_specific).item(), 0.1)

    def test_development_donor_intervention_is_deterministic_and_uses_train_patch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            query_path, donor_path = root/'query.jpg', root/'donor.jpg'
            Image.new('RGB', (128, 256), (255, 0, 0)).save(query_path)
            Image.new('RGB', (128, 256), (0, 0, 255)).save(donor_path)
            query = [(str(query_path), 10, 0)]
            donors = [(str(donor_path), 1, 1)]
            clean = Images(query, 256, 128)[0][0]
            dataset = DonorOcclusionQueries(query, donors, 0.3, seed=44)
            first = dataset[0][0]
            second = dataset[0][0]
            self.assertTrue(torch.equal(first, second))
            changed = (first-clean).abs().sum(dim=0) > 0
            donor = Images(donors, 256, 128)[0][0]
            self.assertGreater(torch.count_nonzero(changed).item(), 0)
            self.assertLess(torch.count_nonzero(changed).item(), changed.numel())
            self.assertTrue(torch.equal(first[:, changed], donor[:, changed]))


if __name__ == '__main__':
    unittest.main()
