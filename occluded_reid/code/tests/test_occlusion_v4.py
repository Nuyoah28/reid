import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from occlusion_v4.mdta import ChannelLayerNorm2d, MDTAAdapter
from occlusion_v4.model import MDTAOSNet


class MDTATests(unittest.TestCase):
    def test_channel_norm_is_spatially_local(self):
        norm = ChannelLayerNorm2d(8)
        x = torch.randn(2, 8, 5, 3)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.allclose(y.mean(1), torch.zeros_like(y[:, 0]), atol=2e-6))

    def test_channel_attention_shape_and_gradients(self):
        block = MDTAAdapter(channels=32, heads=4)
        x = torch.randn(2, 32, 16, 8, requires_grad=True)
        y = block(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(block.qkv.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(block.temperature.grad.abs().sum().item(), 0.)

    def test_single_image_embedding_shape(self):
        model = MDTAOSNet(7).eval()
        with torch.inference_mode():
            feature = model(torch.randn(1, 3, 256, 128))
        self.assertEqual(tuple(feature.shape), (1, 512))
        self.assertAlmostEqual(feature.norm().item(), 1., places=5)


if __name__ == '__main__':
    unittest.main()
