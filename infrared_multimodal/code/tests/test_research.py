"""Geometry, sampling, protocol and upstream equivalence checks, without benchmark data."""
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from data import Record, PairedPK, train_partition, sysu_gallery, regdb
from metrics import retrieval
from model import AGW, load_upstream
from objectives import safe_step, transport_loss, weighted_triplet, objective
from vendor.agw.resnet import resnet50


class GeometryTests(unittest.TestCase):
    def test_exact_cap_and_margin_retention(self):
        x = torch.tensor([[.2, 0.]])
        d = torch.tensor([[2., 0.]])
        prototypes = torch.tensor([[0., 0.], [1., 0.]])
        step, eligible = safe_step(x, d, prototypes, torch.tensor([0]), alpha=1., retain=.5)
        self.assertTrue(eligible.item())
        self.assertAlmostEqual(step.item(), .075, places=6)
        moved = x + step[:, None] * d
        gap = ((moved-prototypes[1])**2).sum()-((moved-prototypes[0])**2).sum()
        self.assertAlmostEqual(gap.item(), .3, places=6)

    def test_many_random_nonspherical_constraints(self):
        torch.manual_seed(10)
        prototypes = torch.randn(7, 24) * 3  # no unit-prototype assumption
        own = torch.arange(7).repeat(5)
        x = prototypes[own] + torch.randn(35, 24) * .1
        direction = torch.randn_like(x) * 10
        step, eligible = safe_step(x, direction, prototypes, own, alpha=2., retain=.7)
        self.assertTrue(eligible.all())
        dist0 = (x[:, None]-prototypes[None]).square().sum(2)
        dist1 = (x[:, None]+step[:, None, None]*direction[:, None]-prototypes[None]).square().sum(2)
        gap0 = dist0-dist0.gather(1, own[:, None]); gap1 = dist1-dist1.gather(1, own[:, None])
        self.assertTrue((gap1+1e-3 >= .7*gap0).all())

    def test_ineligible_and_safe_directions(self):
        p = torch.tensor([[0., 0.], [1., 0.]])
        steps, ok = safe_step(torch.tensor([[.8, 0.], [.1, 0.]]),
                              torch.tensor([[-1., 0.], [-1., 0.]]), p, torch.tensor([0, 0]))
        self.assertEqual(ok.tolist(), [False, True])
        self.assertEqual(steps.tolist(), [0., .5])

    def test_zero_residual_and_gradients(self):
        z = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
        y, m = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1])
        loss, logs = transport_loss(z, y, m)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(logs['virtual_pairs'], 4)
        self.assertEqual(logs['boundary_violations'], 0.)

    def test_no_eligible_samples_stays_finite(self):
        z = torch.ones(4, 3, requires_grad=True)
        loss, logs = transport_loss(z, torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1]))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(logs['eligible_fraction'], 0.)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_paired_ids_required(self):
        with self.assertRaises(ValueError):
            transport_loss(torch.randn(3, 4), torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1]))


class ProtocolTests(unittest.TestCase):
    def test_regdb_independent_split_labels_use_original_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'idx').mkdir()
            for split, pid in [('train', 285), ('test', 88)]:
                for folder, name in [('Visible', 'visible'), ('Thermal', 'thermal')]:
                    image = root/folder/str(pid)/'example.bmp'
                    image.parent.mkdir(parents=True)
                    image.touch()
                    (root/'idx'/f'{split}_{name}_1.txt').write_text(f'{folder}/{pid}/example.bmp 0\n')
            training, test = regdb(root, 'train', 1), regdb(root, 'test', 1)
            self.assertEqual({r.pid for r in training}, {285})
            self.assertEqual({r.pid for r in test}, {88})

    def setUp(self):
        self.rows = [Record(f'{pid}_{m}_{k}', pid, m, m+1) for pid in range(10) for m in [0, 1] for k in range(3)]

    def test_pk_order_and_determinism(self):
        sampler = PairedPK(self.rows, identities=4, instances=2, steps=3)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        for batch in first:
            rows = [self.rows[i] for i in batch]
            self.assertEqual([r.modality for r in rows], [0]*8+[1]*8)
            self.assertEqual([r.pid for r in rows[:8]], [r.pid for r in rows[8:]])
        sampler.epoch = 1
        self.assertNotEqual(first, list(sampler))

    def test_identity_holdout(self):
        train, val, mapping = train_partition(self.rows, 'dev')
        self.assertFalse({r.pid for r in train} & {r.pid for r in val})
        self.assertEqual(set(mapping.values()), set(range(8)))

    def test_known_metrics_and_sysu_exclusion(self):
        query = [Record('q', 1, 1, 3)]
        gallery = [Record('excluded', 1, 0, 2), Record('negative', 2, 0, 1), Record('positive', 1, 0, 1)]
        out = retrieval(np.array([[0., 1., 2.]]), query, gallery, sysu=True)
        self.assertEqual(out['rank1'], 0.)
        self.assertEqual(out['mAP'], .5)
        self.assertEqual(out['mINP'], .5)
        gallery = [Record('a', 1, 0, 1), Record('b', 1, 0, 1), Record('c', 2, 0, 1)]
        self.assertEqual(retrieval(np.array([[0., 1., 2.]]), query, gallery)['mAP'], 1.)

    def test_metrics_equal_upstream(self):
        from vendor.agw.eval_metrics import eval_regdb, eval_sysu
        rng = np.random.default_rng(12)
        gallery = [Record(str(i), i//3, 0, 1) for i in range(90)]
        query = [Record(str(i), i, 1, 6) for i in range(30)]
        dist = rng.random((30, 90))
        for is_sysu, function in [(False, eval_regdb), (True, eval_sysu)]:
            args = (dist, np.arange(30), np.repeat(np.arange(30), 3))
            if is_sysu:
                args += (np.full(30, 6), np.ones(90))
            cmc, ap, inp = function(*args)
            ours = retrieval(dist, query, gallery, sysu=is_sysu)
            np.testing.assert_allclose([ours['rank1'], ours['mAP'], ours['mINP']], [cmc[0], ap, inp], atol=1e-7)


class BaselineTests(unittest.TestCase):
    def test_forward_matches_active_upstream_layers(self):
        torch.manual_seed(1); torch.set_num_threads(4)
        sys.path.insert(0, str(CODE/'vendor/agw'))
        spec = importlib.util.spec_from_file_location('upstream_agw_model', CODE/'vendor/agw/model.py')
        upstream_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream_module)
        # Avoid any implicit pretrained download in this offline equivalence test.
        with patch.object(upstream_module, 'resnet50', side_effect=lambda **kwargs: resnet50(pretrained=False, last_conv_stride=1)):
            original = upstream_module.embed_net(4).eval()
        # Nonzero final BN makes the attention computation observable.
        for stage in [original.NL_2, original.NL_3]:
            for block in stage:
                block.W[1].weight.data.fill_(.1)
        ours = AGW(4).eval(); load_upstream(ours, original)
        visible, thermal = torch.randn(2, 3, 64, 32), torch.randn(2, 3, 64, 32)
        with torch.no_grad():
            expected_pool, expected_neck = original(visible, thermal)
            actual = ours(visible, thermal)
        torch.testing.assert_close(actual['embedding'], expected_neck, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(torch.nn.functional.normalize(actual['pooled'], dim=1), expected_pool, rtol=2e-4, atol=2e-5)

    def test_triplet_matches_upstream(self):
        from vendor.agw.loss import TripletLoss_WRT
        torch.manual_seed(3)
        x, labels = torch.randn(12, 16), torch.arange(3).repeat(4)
        old, _ = TripletLoss_WRT()(x, labels)
        torch.testing.assert_close(weighted_triplet(x, labels), old, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
