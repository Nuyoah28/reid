"""AGW controls and proposed boundary-constrained residual transport (BCRT).

BCRT is a research candidate implemented here, not an established new SOTA.
The transport direction is estimated from ANOTHER identity in the minibatch.
An analytic step bound prevents crossing fixed centroid decision boundaries.
No synthesized images, pixel alignment, memory bank or inference head is used.
"""
import torch
from torch.nn import functional as F


def weighted_triplet(x, labels):
    """AGW weighted regularized triplet, FP32; softmax over valid entries."""
    if labels.unique().numel() < 2:
        raise ValueError('Weighted triplet needs at least two identities.')
    squared = x.square().sum(1)[:, None] + x.square().sum(1)[None, :]
    d = squared.addmm(x, x.T, beta=1, alpha=-2).clamp_min(1e-12).sqrt()
    positive = labels[:, None].eq(labels[None, :])
    # Retain the upstream 1e-6 normalizer (its effect is observable for negatives).
    def weights(values, mask):
        maximum = (values * mask).max(1, keepdim=True).values
        exponent = (values - maximum).exp() * mask
        return exponent / (exponent.sum(1, keepdim=True) + 1e-6)
    pos = (weights(d * positive, positive) * d).sum(1)
    neg = (weights(-d * ~positive, ~positive) * d).sum(1)
    return F.softplus(pos - neg).mean()


def centroids(z, labels, modalities):
    ids, inv = labels.unique(sorted=True, return_inverse=True)
    centers = []
    for m in [0, 1]:
        rows = []
        for identity in ids:
            mask = (labels == identity) & (modalities == m)
            if not mask.any():
                raise ValueError('Every sampled identity must occur in both modalities.')
            rows.append(z[mask].mean(0))
        centers.append(torch.stack(rows))
    return centers[0], centers[1], inv


@torch.no_grad()
def safe_step(x, direction, prototypes, own, alpha=.5, retain=.5):
    """Keep >=retain of each initially positive squared-distance margin.

    For every impostor k, g_k(t)=g_k(0)+2t*d dot (p_y-p_k).
    If any original margin is non-positive the anchor is ineligible.
    Prototypes need not have unit norm; there is no spherical-bound shortcut.
    """
    if alpha < 0 or not 0 <= retain <= 1 or prototypes.shape[0] < 2:
        raise ValueError('Need alpha>=0, retain in [0,1] and >=2 prototypes.')
    dist = (x[:, None, :] - prototypes[None, :, :]).square().sum(2)
    gap = dist - dist.gather(1, own[:, None])
    self_mask = F.one_hot(own, prototypes.shape[0]).bool()
    slope = 2 * (direction[:, None, :] * (prototypes[own, None, :] - prototypes[None, :, :])).sum(2)
    eligible = gap.masked_fill(self_mask, torch.inf).amin(1) > 1e-6
    bound = torch.where((slope < -1e-8) & ~self_mask,
                        (1 - retain) * gap / (-slope).clamp_min(1e-8), torch.inf)
    step = bound.amin(1).clamp(0, alpha)
    return torch.where(eligible, step, 0), eligible


def transport_loss(z, labels, modalities, constrained=True, alpha=.5, retain=.5, temperature=.1):
    """Full donors excluding the anchor identity, with detached geometry.

    All variants use exactly the same forward images and features. BCRT and
    the unconstrained control differ ONLY in step selection. No gradient can
    flow into a donor residual or the prototype target.
    """
    cv, ct, own = centroids(z, labels, modalities)
    p = cv.shape[0]
    if p < 2:
        raise ValueError('Residual transport requires at least 2 identities.')
    delta = (ct - cv).detach()
    prototype = ((cv + ct) / 2).detach()
    donors = torch.arange(p, device=z.device).expand(z.shape[0], -1)
    allowed = donors != own[:, None]
    anchors = z[:, None, :].expand(-1, p, -1)[allowed]
    directions = delta[donors[allowed]] * (1 - 2 * modalities[:, None].expand(-1, p)[allowed])[:, None]
    targets = own[:, None].expand(-1, p)[allowed]
    # Eligibility is shared with the unbounded control; only step caps differ.
    bounded, eligible = safe_step(anchors.detach(), directions, prototype, targets, alpha, retain)
    steps = bounded if constrained else torch.full_like(bounded, alpha) * eligible
    virtual = F.normalize(anchors + steps[:, None] * directions, dim=1)
    logits = virtual @ F.normalize(prototype, dim=1).T / temperature
    per_example = F.cross_entropy(logits, targets, reduction='none')
    active = eligible.float()
    loss = (per_example * active).sum() / active.sum().clamp_min(1)
    # The certificate concerns unnormalized virtual points and fixed prototypes.
    with torch.no_grad():
        dist0 = (anchors[:, None, :] - prototype[None, :, :]).square().sum(2)
        moved = anchors + steps[:, None] * directions
        dist1 = (moved[:, None, :] - prototype[None, :, :]).square().sum(2)
        mask = ~F.one_hot(targets, p).bool()
        gap0 = dist0 - dist0.gather(1, targets[:, None])
        gap1 = dist1 - dist1.gather(1, targets[:, None])
        violations = ((gap1 + 1e-6 < retain * gap0) & mask & eligible[:, None]).float().sum()
    return loss, {'eligible_fraction': active.mean().item(),
                  'mean_step': steps[eligible].mean().item() if eligible.any() else 0.,
                  'capped_fraction': ((bounded < alpha) & eligible).float().sum().item() / max(eligible.sum().item(), 1),
                  'boundary_violations': violations.item(), 'virtual_pairs': len(steps)}


def objective(output, labels, modalities, variant, weight=.2, alpha=.5, retain=.5):
    pooled = output['pooled'].float()
    ce = F.cross_entropy(output['logits'].float(), labels)
    triplet = weighted_triplet(pooled, labels)
    z = F.normalize(pooled, dim=1)
    extra = z.sum() * 0
    diagnostic = {}
    if variant == 'center':
        cv, ct, _ = centroids(z, labels, modalities)
        extra = (cv - ct).square().sum(1).mean()
    elif variant in ['unbounded', 'bcrt']:
        extra, diagnostic = transport_loss(z, labels, modalities, variant == 'bcrt', alpha, retain)
    elif variant != 'agw':
        raise ValueError(variant)
    total = ce + triplet + weight * extra
    return total, {'ce': ce.item(), 'wrt': triplet.item(), 'extra': extra.item(),
                   'total': total.item(), **diagnostic}
