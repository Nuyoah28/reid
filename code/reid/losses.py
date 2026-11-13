import torch
from torch.nn import functional as F


def batch_hard(features,labels,margin=.3):
    distances=torch.cdist(features.float(),features.float(),p=2)
    same=labels[:,None].eq(labels[None,:])
    positive=same&~torch.eye(len(labels),device=labels.device,dtype=torch.bool)
    valid=positive.any(1)&(~same).any(1)
    if not valid.any():raise ValueError('Batch hard loss needs >=2 IDs with repeated instances')
    hardest_positive=distances.masked_fill(~positive,float('-inf')).max(1).values
    hardest_negative=distances.masked_fill(same,float('inf')).min(1).values
    return F.relu(hardest_positive[valid]-hardest_negative[valid]+margin).mean()
