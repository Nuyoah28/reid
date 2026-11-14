"""Peer-identity evidence calibration, a project hypothesis rather than proven novelty.

OSNet and the reliability head are reused from the frozen v1 implementation.
No CLIP, body-part memory bank, or external segmentation module is imported.
The new intervention target uses other images of the identity, never its own
global descriptor, and is a detached training signal only.
"""
import torch
from torch.nn import functional as F
from reid.model import OcclusionOSNet
from reid.occlusion import influence_target, surviving_evidence_loss

VARIANTS = ('repeat3', 'paste3', 'stability', 'peer', 'peer_stability', 'peer_evidence')


def build_model(num_classes, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    return OcclusionOSNet(num_classes, 'aug' if variant in ('repeat3', 'paste3') else 'clcp')


@torch.no_grad()
def peer_identity_target(clean, a, b, labels, visible, temperature=0.1, image_indices=None):
    """Conservative local identity margin against leave-one-image-out peers.

    p_i = normalize(mean(clean GAP_j : y_j=y_i, j!=i)).
    Negative prototypes are normalized clean GAP class means in this batch.
    margin_v(s) = cos(F_v(s), p_i) - max_{c!=y_i} cos(F_v(s), n_c).
    q(s) = geometry(s) * sigmoid(min(margin_a, margin_b) / T).

    Clean/occluded maps and prototypes are detached. The minimum asks that
    identity support survives BOTH donor interventions, rather than averaging
    away one failed intervention. This is not a foreground segmentation label.
    Repeated draws of the same source image are excluded from its positive set.
    Duplicate peer draws get inverse-multiplicity weights, one vote per image.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if clean.shape != a.shape or a.shape != b.shape or len(labels) != len(clean):
        raise ValueError('Clean and paired features must have equal shapes')
    labels = labels.to(clean.device)
    same = labels[:, None].eq(labels[None, :])
    if image_indices is None:
        image_indices = torch.arange(len(labels), device=labels.device)
    image_indices = image_indices.to(labels.device)
    same_image = image_indices[:, None].eq(image_indices[None, :])
    if bool((same_image & ~same).any()):
        raise ValueError('The same source image cannot have different identity labels')
    peers = same & ~same_image
    if not bool(peers.any(1).all()) or labels.unique().numel() < 2:
        raise ValueError('Require another image of each identity and at least two identities')
    global_clean = clean.float().mean((2, 3))
    unique_weight = same_image.sum(0).float().reciprocal()
    peer_weights = peers.float() * unique_weight[None, :]
    positives = F.normalize(peer_weights @ global_clean / peer_weights.sum(1, keepdim=True), dim=1)
    classes, inverse = labels.unique(sorted=True, return_inverse=True)
    membership = F.one_hot(inverse, len(classes)).float()
    unique_membership = membership * unique_weight[:, None]
    negatives = F.normalize(unique_membership.T @ global_clean / unique_membership.sum(0)[:, None], dim=1)
    positive_scores, negative_scores, margins = [], [], []
    own = inverse[:, None, None, None].eq(torch.arange(len(classes), device=labels.device)[None, :, None, None])
    for maps in (a, b):
        normalized = F.normalize(maps.float(), dim=1)
        positive = torch.einsum('bchw,bc->bhw', normalized, positives).unsqueeze(1)
        all_scores = torch.einsum('bchw,pc->bphw', normalized, negatives)
        negative = all_scores.masked_fill(own, -torch.inf).amax(1, keepdim=True)
        positive_scores.append(positive)
        negative_scores.append(negative)
        margins.append(positive - negative)
    margin = torch.minimum(margins[0], margins[1])
    identity = torch.sigmoid(margin / temperature)
    geometry = F.adaptive_avg_pool2d(visible.float(), a.shape[-2:])
    target = geometry * identity
    return target, {
        'identity_support': identity,
        'identity_margin': margin,
        'positive_similarity': torch.stack(positive_scores).mean(0),
        'negative_similarity': torch.stack(negative_scores).mean(0),
    }


def auxiliary_losses(out, labels, visible, variant, tau=0.15, temperature=0.1, image_indices=None):
    zero = out['features'].new_zeros((), dtype=torch.float32)
    diagnostics = {k: zero for k in ('target_mean', 'identity_mean', 'margin_mean', 'calibration_mae')}
    if variant in ('repeat3', 'paste3'):
        return zero, zero, diagnostics
    clean, a, b = out['maps'].chunk(3)
    stable = influence_target(a, b, visible, tau)
    if variant == 'stability':
        target = stable
    else:
        target, info = peer_identity_target(clean, a, b, labels, visible, temperature, image_indices)
        diagnostics['identity_mean'] = info['identity_support'].mean()
        diagnostics['margin_mean'] = info['identity_margin'].mean()
        if variant in ('peer_stability', 'peer_evidence'):
            # stable already contains geometry; apply its appearance attenuation once.
            geometry = F.adaptive_avg_pool2d(visible.float(), a.shape[-2:])
            attenuation = torch.where(geometry > 0, stable / geometry.clamp_min(1e-6), 0.)
            target = target * attenuation
    _, ra, rb = out['reliability_logits'].chunk(3)
    calibration = 0.5 * (F.binary_cross_entropy_with_logits(ra.float(), target)
                         + F.binary_cross_entropy_with_logits(rb.float(), target))
    # Directly reused v1 consistency loss: explicitly not a new module claim.
    evidence = surviving_evidence_loss(clean, a, b, target) if variant == 'peer_evidence' else zero
    diagnostics['target_mean'] = target.mean()
    diagnostics['calibration_mae'] = 0.5 * ((ra.detach().float().sigmoid()-target).abs().mean()
                                          + (rb.detach().float().sigmoid()-target).abs().mean())
    return calibration, evidence, diagnostics
