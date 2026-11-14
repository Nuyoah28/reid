"""Paired occluder interventions and detached influence targets (project design)."""
import math
import torch
from torch.nn import functional as F


def paired_occlusion(images,labels,area=(0.15,0.45),generator=None):
    """Hold identity and rectangle fixed; intervene only on occluder appearance.

    Donors always have another training identity. No test images enter augmentation.
    Two independently sampled donor crops avoid identical donor interventions.
    """
    batch,_,height,width=images.shape
    first,second=images.clone(),images.clone()
    visible=torch.ones(batch,1,height,width,device=images.device)
    for i in range(batch):
        candidates=torch.where(labels!=labels[i])[0]
        if len(candidates)==0:raise ValueError('At least two identities are required')
        fraction=float(torch.empty(1).uniform_(*area,generator=generator))
        aspect=math.exp(float(torch.empty(1).uniform_(math.log(0.4),math.log(2.5),generator=generator)))
        rh=min(height-1,max(1,round(math.sqrt(height*width*fraction/aspect))))
        rw=min(width-1,max(1,round(math.sqrt(height*width*fraction*aspect))))
        y=int(torch.randint(height-rh+1,(1,),generator=generator));x=int(torch.randint(width-rw+1,(1,),generator=generator))
        choices=torch.randperm(len(candidates),generator=generator)
        d1=int(candidates[int(choices[0])]);d2=int(candidates[int(choices[1%len(choices)])])
        crop1=images[d1,:,y:y+rh,x:x+rw]
        crop2=images[d2,:,y:y+rh,x:x+rw].flip(-1)
        first[i,:,y:y+rh,x:x+rw]=crop1
        second[i,:,y:y+rh,x:x+rw]=crop2
        visible[i,:,y:y+rh,x:x+rw]=0
    return first,second,visible


@torch.no_grad()
def influence_target(maps_a,maps_b,visible,tau=0.15,geometric_only=False):
    if tau<=0:raise ValueError('tau must be positive')
    geometric=F.adaptive_avg_pool2d(visible,maps_a.shape[-2:])
    if geometric_only:return geometric
    # A visible cell can still change when only the distant occluder is replaced.
    discrepancy=(1-F.cosine_similarity(maps_a.float(),maps_b.float(),dim=1)).clamp_min(0).unsqueeze(1)
    return geometric*torch.exp(-discrepancy/tau)


def surviving_evidence_loss(clean,a,b,target):
    """Compare only evidence present in the intervened views, without completing it.

    The clean teacher is re-pooled using the SAME detached influence target.
    Full clean embeddings would incorrectly supervise genuinely missing evidence.
    """
    target=target.detach()
    def pool(value):return (value.float()*target).sum((2,3))/target.sum((2,3)).clamp_min(1e-6)
    teacher=pool(clean.detach())
    cost=0.5*((1-F.cosine_similarity(pool(a),teacher))+(1-F.cosine_similarity(pool(b),teacher)))
    budget=target.mean((1,2,3)).detach()
    return (cost*budget).sum()/budget.sum().clamp_min(1e-6)
