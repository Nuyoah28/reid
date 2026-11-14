import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reid.metrics import evaluate_rank
from reid.occlusion import influence_target,paired_occlusion,surviving_evidence_loss
from reid.losses import batch_hard
from reid.data import PKSampler
from reid.model import OcclusionOSNet

class ResearchTests(unittest.TestCase):
    def test_cross_camera_filter_and_junk(self):
        q=torch.tensor([[1.,0.]])
        g=torch.tensor([[1.,0.],[1.,0.],[.8,.2],[0.,1.]])
        result=evaluate_rank(q,g,[1],[-1,1,1,2],[0],[2,0,1,1])
        self.assertEqual(result['rank1'],100.)
        self.assertEqual(result['mAP'],100.)
    def test_average_precision_and_invalid_query(self):
        q=torch.eye(2);g=torch.tensor([[1.,0.],[.9,.1],[0.,1.]])
        result=evaluate_rank(q,g,[1,7],[2,1,1],[0,0],[1,1,1])
        self.assertAlmostEqual(result['mAP'],(1/2+2/3)/2*100)
        self.assertEqual(result['invalid_queries'],1)
    def test_occlusion_target_detects_pollution_outside_mask(self):
        a=torch.zeros(2,3,4,2);a[:,0]=1
        b=a.clone();b[0,:,0,0]=torch.tensor([0.,1.,0.])
        visible=torch.ones(2,1,16,8);visible[:,:,8:]=0
        target=influence_target(a,b,visible,tau=.1)
        self.assertLess(target[0,0,0,0].item(),.001)
        self.assertEqual(target[1,0,0,0].item(),1.)
        self.assertEqual(target[:,:,2:].sum().item(),0.)
        self.assertFalse(target.requires_grad)
    def test_identical_intervention_reduces_to_geometry(self):
        maps=torch.rand(2,8,4,2);mask=torch.rand(2,1,16,8)
        self.assertTrue(torch.allclose(influence_target(maps,maps,mask),influence_target(maps,maps,mask,geometric_only=True),atol=2e-6))
    def test_paired_intervention_preserves_visible_pixels(self):
        x=torch.arange(8.).view(8,1,1,1).expand(8,3,32,16).clone()
        labels=torch.arange(8)//2
        a,b,visible=paired_occlusion(x,labels,generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a*visible,x*visible))
        self.assertTrue(torch.equal(b*visible,x*visible))
        self.assertGreater((a-b).abs().sum().item(),0)
        self.assertTrue(torch.all((visible.mean((1,2,3))>0)&(visible.mean((1,2,3))<1)))
    def test_evidence_teacher_detached(self):
        c=torch.rand(2,8,4,2,requires_grad=True);a=torch.rand_like(c,requires_grad=True);b=torch.rand_like(c,requires_grad=True)
        loss=surviving_evidence_loss(c,a,b,torch.ones(2,1,4,2));loss.backward()
        self.assertIsNone(c.grad);self.assertGreater(a.grad.abs().sum().item(),0)
    def test_evidence_ignores_regions_without_support(self):
        generator=torch.Generator().manual_seed(31)
        clean=torch.rand(2,8,4,2,generator=generator)
        a=torch.rand(2,8,4,2,generator=generator,requires_grad=True)
        b=torch.rand(2,8,4,2,generator=generator,requires_grad=True)
        target=torch.zeros(2,1,4,2);target[:,:,:2]=1
        altered_clean=clean.clone();altered_clean[:,:,2:]=1000
        loss=surviving_evidence_loss(clean,a,b,target)
        self.assertTrue(torch.allclose(loss,surviving_evidence_loss(altered_clean,a,b,target)))
        loss.backward()
        self.assertEqual(a.grad[:,:,2:].abs().sum().item(),0.)
        self.assertGreater(a.grad[:,:,:2].abs().sum().item(),0.)
    def test_calibration_isolates_backbone_but_retrieval_can_train_it(self):
        torch.set_num_threads(2)
        model=OcclusionOSNet(3,'full').train()
        images=torch.rand(4,3,64,32)
        output=model(images,details=True)
        calibration=torch.nn.functional.binary_cross_entropy_with_logits(
            output['reliability_logits'],torch.full_like(output['reliability_logits'],.2))
        calibration.backward()
        self.assertTrue(all(p.grad is None for name,p in model.named_parameters()
                            if not name.startswith('reliability.')))
        self.assertGreater(model.reliability[-1].bias.grad.abs().sum().item(),0.)
        model.zero_grad(set_to_none=True)
        output=model(images,details=True);output['maps'].retain_grad()
        output['features'].square().mean().backward()
        self.assertGreater(output['maps'].grad.abs().sum().item(),0.)
    def test_zero_evidence_is_finite(self):
        maps=torch.randn(2,8,4,2,requires_grad=True)
        loss=surviving_evidence_loss(maps,maps,maps,torch.zeros(2,1,4,2))
        self.assertTrue(torch.isfinite(loss));self.assertEqual(float(loss),0.)
    def test_pk_positive_negative_composition(self):
        records=[(str(i),i//5,0) for i in range(30)]
        sampler=PKSampler(records,12,4,seed=1,steps=3)
        indices=list(sampler)
        for offset in range(0,len(indices),12):
            _,counts=np.unique([records[i][1] for i in indices[offset:offset+12]],return_counts=True)
            self.assertEqual(sorted(counts.tolist()),[4,4,4])
    def test_triplet_separates_ids(self):
        labels=torch.tensor([0,0,1,1]);good=torch.tensor([[1.,0.],[1.,.01],[-1.,0.],[-1.,.01]])
        bad=torch.tensor([[1.,0.],[-1.,0.],[1.,.01],[-1.,.01]])
        self.assertLess(batch_hard(good,labels),batch_hard(bad,labels))
    def test_pooling_matches_official_when_disabled(self):
        torch.set_num_threads(2)
        model=OcclusionOSNet(3,'plain').eval();x=torch.rand(2,3,64,32)
        with torch.no_grad():
            native=super(OcclusionOSNet,model).forward(x)
            extended=model(x)
        self.assertTrue(torch.allclose(torch.nn.functional.normalize(native,dim=1),extended,atol=1e-6))

if __name__=='__main__':unittest.main()
