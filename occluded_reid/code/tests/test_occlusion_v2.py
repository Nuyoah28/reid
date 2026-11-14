import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occlusion_v2.data import PeerPKSampler
from occlusion_v2.method import VARIANTS, build_model, peer_identity_target, auxiliary_losses


class PeerEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.labels = torch.tensor([0, 0, 1, 1])
        self.clean = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])[:, :, None, None].expand(-1, -1, 2, 2).clone()
        self.visible = torch.ones(4, 1, 8, 8)

    def test_support_distinguishes_identity_from_shared_stability(self):
        a = self.clean.clone()
        a[0, :, 0, 0] = torch.tensor([0., 1.])  # stable evidence of another identity
        target, info = peer_identity_target(self.clean, a, a, self.labels, self.visible)
        self.assertLess(target[0, 0, 0, 0].item(), .001)
        self.assertGreater(target[0, 0, 1, 1].item(), .999)
        self.assertTrue(torch.isfinite(info['identity_margin']).all())

    def test_anchor_image_excluded_including_its_repeated_draws(self):
        clean = torch.tensor([[0., 1.], [0., 1.], [1., 0.], [0., 1.], [0., 1.]])[:, :, None, None]
        labels = torch.tensor([0, 0, 0, 1, 1]); indices = torch.tensor([3, 3, 4, 5, 6])
        a = torch.tensor([[1., 0.]]).repeat(5, 1)[:, :, None, None]
        target, _ = peer_identity_target(clean, a, a, labels, torch.ones(5, 1, 1, 1), image_indices=indices)
        self.assertGreater(target[0].item(), .999)
        self.assertGreater(target[1].item(), .999)
        clean[0] = torch.tensor([[-1.], [0.]])[:, :, None]
        clean[1] = clean[0]
        changed, _ = peer_identity_target(clean, a, a, labels, torch.ones(5, 1, 1, 1), image_indices=indices)
        self.assertTrue(torch.equal(target[:2], changed[:2]))

    def test_self_exclusion_and_stop_gradient(self):
        clean = self.clean.clone().requires_grad_(); a = self.clean.clone().requires_grad_()
        target, _ = peer_identity_target(clean, a, a, self.labels, self.visible)
        self.assertFalse(target.requires_grad)
        altered = clean.detach().clone(); altered[0] = torch.rand_like(altered[0])*100
        changed, _ = peer_identity_target(altered, a, a, self.labels, self.visible)
        self.assertTrue(torch.allclose(target[0], changed[0]))

    def test_pessimistic_two_intervention_support(self):
        bad = self.clean.clone(); bad[0] = self.clean[2]
        target, _ = peer_identity_target(self.clean, self.clean, bad, self.labels, self.visible)
        self.assertLess(target[0].mean().item(), .001)

    def test_geometry_and_invalid_batch(self):
        target, _ = peer_identity_target(self.clean, self.clean, self.clean, self.labels, self.visible*0)
        self.assertEqual(target.sum().item(), 0.)
        with self.assertRaises(ValueError):
            peer_identity_target(self.clean, self.clean, self.clean, torch.arange(4), self.visible)
        with self.assertRaises(ValueError):
            peer_identity_target(self.clean, self.clean, self.clean, self.labels, self.visible, 0)

    def test_small_pool_sampling_never_loses_other_image(self):
        records = [(str(i), i//2, 0) for i in range(10)]
        sampler = PeerPKSampler(records, 8, 4, seed=9, steps=20)
        draws = list(sampler)
        for i in range(0, len(draws), 4):
            self.assertEqual(len(set(draws[i:i+4])), 2)
        self.assertEqual(draws, list(sampler))
        sampler.epoch = 1
        self.assertNotEqual(draws, list(sampler))

    def test_inverse_multiplicity_gives_each_source_one_vote(self):
        clean = torch.tensor([[1., 1.], [1., 0.], [1., 0.], [0., 1.], [-1., 0.], [-1., 0.]])[:, :, None, None]
        labels = torch.tensor([0, 0, 0, 0, 1, 1]); indices = torch.tensor([0, 1, 1, 2, 3, 4])
        target, _ = peer_identity_target(clean, clean, clean, labels, torch.ones(6, 1, 1, 1), image_indices=indices)
        keep = torch.tensor([0, 1, 3, 4, 5])
        unique, _ = peer_identity_target(clean[keep], clean[keep], clean[keep], labels[keep], torch.ones(5, 1, 1, 1), image_indices=indices[keep])
        self.assertTrue(torch.allclose(target[0], unique[0]))

    def test_same_reliability_architecture_for_all_target_ablations(self):
        states = []
        for variant in ('stability', 'peer', 'peer_stability', 'peer_evidence'):
            torch.manual_seed(4)
            states.append(build_model(3, variant).state_dict())
        for state in states[1:]:
            self.assertEqual(states[0].keys(), state.keys())
            self.assertTrue(all(torch.equal(states[0][k], state[k]) for k in state))

    def test_auxiliary_backprop_and_single_image_inference(self):
        labels = self.labels
        for variant in VARIANTS:
            model = build_model(2, variant).train()
            out = model(torch.rand(12, 3, 64, 32), details=True)
            cal, evidence, diagnostic = auxiliary_losses(out, labels, self.visible, variant)
            loss = F.cross_entropy(out['logits'], labels.repeat(3)) + .5*cal + .5*evidence
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0.)
            self.assertTrue(torch.isfinite(diagnostic['target_mean']))
            model.eval()
            with torch.no_grad():
                embedding = model(torch.rand(1, 3, 64, 32))
            self.assertEqual(tuple(embedding.shape), (1, 512))
            self.assertAlmostEqual(embedding.norm().item(), 1., places=5)


if __name__ == '__main__':
    unittest.main()
