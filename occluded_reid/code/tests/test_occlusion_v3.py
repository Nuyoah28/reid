import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from occlusion_v3.model import SemanticDetailFusion, SpatialTransformer, StructuralOSNet


class StructuralAdapterTests(unittest.TestCase):
    def test_transformer_shape_and_gradients(self):
        block = SpatialTransformer()
        x = torch.randn(2, 512, 16, 8, requires_grad=True)
        y = block(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(block.attention.in_proj_weight.grad.abs().sum().item(), 0.)

    def test_semantic_detail_selects_tokens_and_backpropagates(self):
        block = SemanticDetailFusion(keep_ratio=.5)
        x = torch.randn(2, 512, 16, 8, requires_grad=True)
        y = block(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(block.value.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(block.query.weight.grad.abs().sum().item(), 0.)

    def test_single_image_embedding_for_all_variants(self):
        for variant in ('baseline', 'transformer', 'semantic_detail'):
            model = StructuralOSNet(7, variant).eval()
            with torch.inference_mode():
                feature = model(torch.randn(1, 3, 256, 128))
            self.assertEqual(tuple(feature.shape), (1, 512))
            self.assertAlmostEqual(feature.norm().item(), 1., places=5)


if __name__ == '__main__':
    unittest.main()
