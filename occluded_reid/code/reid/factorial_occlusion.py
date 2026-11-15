"""Matched 2x2 interventions over target identity and donor appearance."""
from itertools import combinations
import math

import torch


def factorial_occlusion(images, labels, generator=None, groups=12, area=(0.15, 0.45)):
    """Build [target identity, donor appearance] groups with shared geometry.

    Each group contains identity-A/donor-1, identity-A/donor-2,
    identity-B/donor-1, identity-B/donor-2 in that order.
    """
    batch, _, height, width = images.shape
    label_values = labels.detach().cpu().tolist()
    members = {}
    for index, label in enumerate(label_values):
        members.setdefault(int(label), []).append(index)
    identities = sorted(members)
    identity_pairs = list(combinations(identities, 2))
    if len(identity_pairs) < groups or len(identities) < 4:
        raise ValueError('Need at least four identities and enough unique identity pairs')
    identity_order = torch.randperm(len(identities), generator=generator).tolist()
    rotating = [identities[index] for index in identity_order]
    schedule = []
    for _ in range(len(rotating)-1):
        schedule.extend((rotating[index], rotating[-1-index])
                        for index in range(len(rotating)//2))
        rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
    selected_pairs = schedule[:groups]
    view_groups, label_groups, metadata = [], [], []
    for first_id, second_id in selected_pairs:
        other_ids = [identity for identity in members if identity not in (first_id, second_id)]
        donor_order = torch.randperm(len(other_ids), generator=generator).tolist()
        donor_ids = [other_ids[index] for index in donor_order[:2]]
        target_indices = []
        donor_indices = []
        for identity in (first_id, second_id):
            choice = int(torch.randint(len(members[identity]), (1,), generator=generator))
            target_indices.append(members[identity][choice])
        for identity in donor_ids:
            choice = int(torch.randint(len(members[identity]), (1,), generator=generator))
            donor_indices.append(members[identity][choice])

        fraction = float(torch.empty(1).uniform_(*area, generator=generator))
        aspect = math.exp(float(torch.empty(1).uniform_(math.log(0.4), math.log(2.5), generator=generator)))
        region_h = min(height - 1, max(1, round(math.sqrt(height * width * fraction / aspect))))
        region_w = min(width - 1, max(1, round(math.sqrt(height * width * fraction * aspect))))
        top = int(torch.randint(height - region_h + 1, (1,), generator=generator))
        left = int(torch.randint(width - region_w + 1, (1,), generator=generator))
        bottom, right = top + region_h, left + region_w

        donor_patches = [images[index, :, top:bottom, left:right] for index in donor_indices]
        views = []
        for target_index in target_indices:
            for patch in donor_patches:
                view = images[target_index].clone()
                view[:, top:bottom, left:right] = patch
                views.append(view)
        view_groups.append(torch.stack(views))
        label_groups.append(torch.tensor([first_id, first_id, second_id, second_id],
                                         device=labels.device, dtype=labels.dtype))
        metadata.append({'identities': (first_id, second_id), 'donors': tuple(donor_ids),
                         'box': (top, left, bottom, right),
                         'target_indices': tuple(target_indices),
                         'donor_indices': tuple(donor_indices)})
    return torch.stack(view_groups), torch.stack(label_groups), metadata


def interaction_penalty(embeddings):
    """Penalize identity-dependent changes between the two donor appearances."""
    if embeddings.ndim != 3 or embeddings.shape[1] != 4:
        raise ValueError('Expected embeddings shaped [groups, 4, dimensions]')
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    identity_a_change = normalized[:, 0] - normalized[:, 1]
    identity_b_change = normalized[:, 2] - normalized[:, 3]
    return (identity_a_change - identity_b_change).square().sum(dim=-1).mean()
