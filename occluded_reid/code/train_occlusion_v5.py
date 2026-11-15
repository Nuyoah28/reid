"""Matched OSNet training for cross-factor counterfactual occlusion studies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from train import seed_all, seed_worker, extract
from reid.data import Images, PKSampler, scan_market, training_split
from reid.factorial_occlusion import factorial_occlusion, interaction_penalty
from reid.losses import batch_hard
from reid.metrics import evaluate_rank
from reid.occlusion import paired_occlusion
from occlusion_v5.model import build_model

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('plain', 'occlusion_aug', 'pairwise_consistency', 'factorial_interaction')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = sorted((ROOT/'code/reid').glob('*.py'))
    paths += sorted((ROOT/'code/occlusion_v3').glob('*.py'))
    paths += sorted((ROOT/'code/occlusion_v5').glob('*.py'))
    paths += [Path(__file__), ROOT/'code/train.py', ROOT/'code/evaluate_occlusion_v5.py',
              ROOT/'code/deep-person-reid/torchreid/models/osnet.py',
              ROOT/'tools/run_occlusion_v5.py', ROOT/'tools/verify_occlusion_v5.py',
              ROOT/'tools/summarize_occlusion_v5.py']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def validation_split(data_root, split):
    records = scan_market(Path(data_root)/'bounding_box_train')
    heldout = set(split['val_ids'])
    by_identity = {}
    for record in records:
        if record[1] in heldout:
            by_identity.setdefault(record[1], []).append(record)
    query, gallery = [], []
    query_camera = {}
    for identity in sorted(by_identity):
        group = by_identity[identity]
        cameras = sorted({record[2] for record in group})
        if len(cameras) < 2:
            continue
        camera = cameras[(identity + split['split_seed']) % len(cameras)]
        query_camera[str(identity)] = camera
        query.extend(record for record in group if record[2] == camera)
        gallery.extend(record for record in group if record[2] != camera)
    if not query or not gallery:
        raise ValueError('The fixed identity holdout did not produce a cross-camera split')
    if {Path(row[0]).name for row in query} & {Path(row[0]).name for row in gallery}:
        raise RuntimeError('Development query/gallery image leakage')
    return query, gallery, query_camera


def record_manifest(records, data_root):
    rows = ['%s\t%d\t%d\t%s' % (Path(path).relative_to(data_root).as_posix(), pid, camera,
                                  digest(path)) for path, pid, camera in records]
    return hashlib.sha256('\n'.join(rows).encode()).hexdigest()


class DonorOcclusionQueries(Dataset):
    """Apply one fixed, deterministic Market-train donor rectangle per query."""
    def __init__(self, query, donors, amount, seed=1729, height=256, width=128):
        self.query_records = query
        self.query = Images(query, height, width)
        self.donors = Images(donors, height, width)
        self.amount, self.seed = amount, seed
        if not donors:
            raise ValueError('At least one training donor is required')

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index):
        image, pid, camera = self.query[index]
        source_path = Path(self.query_records[index][0])
        token = f'{source_path.name}\t{self.amount:.3f}\t{self.seed}'
        value = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little')
        rng = random.Random(value)
        h, w = image.shape[-2:]
        aspect = math.exp(rng.uniform(math.log(0.4), math.log(2.5)))
        region_h = min(h-1, max(1, round(math.sqrt(h*w*self.amount/aspect))))
        region_w = min(w-1, max(1, round(math.sqrt(h*w*self.amount*aspect))))
        top, left = rng.randrange(h-region_h+1), rng.randrange(w-region_w+1)
        donor_index = rng.randrange(len(self.donors))
        donor = self.donors[donor_index][0]
        image[:, top:top+region_h, left:left+region_w] = \
            donor[:, top:top+region_h, left:left+region_w]
        return image, pid, camera


def run_v5_evaluation(model, query, gallery, donors, eval_args, device):
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        loader_args = {'batch_size': eval_args.eval_batch_size, 'num_workers': 0,
                       'pin_memory': device.type == 'cuda'}
        gallery_features = extract(model, DataLoader(
            Images(gallery, eval_args.height, eval_args.width), **loader_args), device)
        results = {}
        conditions = [(0., 'clean')]
        conditions.extend((amount, f'synthetic_donor_{amount:g}')
                          for amount in eval_args.eval_occlusion if amount > 0)
        for amount, name in conditions:
            dataset = Images(query, eval_args.height, eval_args.width) if amount == 0 else \
                DonorOcclusionQueries(query, donors, amount, height=eval_args.height,
                                      width=eval_args.width)
            query_features = extract(model, DataLoader(dataset, **loader_args), device)
            results[name] = evaluate_rank(query_features[0], gallery_features[0],
                query_features[1], gallery_features[1], query_features[2], gallery_features[2])
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    device = torch.device('cuda')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    seed_all(args.seed)
    data = ROOT/'data/Market-1501-v15.09.15'
    pretrained = ROOT/'weights/osnet_x1_0_imagenet.pth'
    train, _, _, split = training_split(data)
    query, gallery, query_camera = validation_split(data, split)
    if set(split['train_ids']) & set(split['val_ids']):
        raise RuntimeError('Train/development identity leakage')
    model = build_model(len(split['train_ids']))
    initialization = model.load_imagenet(pretrained)
    model.to(device)
    seed_all(args.seed)
    protocol_path = ROOT/'research/occlusion_v5_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    config = {
        'run_tag': protocol['run_tag'], 'variant': args.variant, 'seed': args.seed,
        'epochs': protocol['epochs'], 'steps': protocol['steps_per_epoch'],
        'batch_size': protocol['batch_size'], 'instances': protocol['instances'],
        'height': protocol['image_size'][0], 'width': protocol['image_size'][1],
        'protocol_sha256': digest(protocol_path), 'source_sha256': source_hashes(),
        'pretrained_sha256': digest(pretrained), 'split': split,
        'query_camera_by_identity': query_camera,
        'data_record_sha256': {name: record_manifest(records, data) for name, records in
                               [('train', train), ('query', query), ('gallery', gallery)]},
        'train_images': len(train), 'query_images': len(query), 'gallery_images': len(gallery),
        'optimizer': protocol['optimizer'], 'loss': protocol['loss'],
        'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__,
        'cuda': torch.version.cuda, 'python': platform.python_version(),
        'gpu': torch.cuda.get_device_name(0), 'initialization': initialization,
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'inference_parameters': sum(p.numel() for n, p in model.named_parameters()
                                    if not n.startswith('backbone.classifier.')),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_before_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    groups = {'backbone': [], 'heads': []}
    for name, parameter in model.named_parameters():
        key = 'heads' if name.startswith(('backbone.fc.', 'backbone.classifier.')) else 'backbone'
        groups[key].append(parameter)
    optimizer = torch.optim.Adam([
        {'params': groups['backbone'], 'lr': .0003, 'multiplier': 1.},
        {'params': groups['heads'], 'lr': .0015, 'multiplier': 5.}],
        weight_decay=.0005, amsgrad=True)
    scaler = torch.amp.GradScaler('cuda')
    sampler = PKSampler(train, protocol['batch_size'], protocol['instances'], args.seed,
                         protocol['steps_per_epoch'])
    loader_generator = torch.Generator().manual_seed(args.seed)
    occlusion_generator = torch.Generator().manual_seed(args.seed + 5000)
    loader = DataLoader(Images(train, *protocol['image_size'], train=True),
                        batch_size=protocol['batch_size'], sampler=sampler, num_workers=0,
                        pin_memory=True, worker_init_fn=seed_worker, generator=loader_generator,
                        persistent_workers=False)
    history, start_epoch = [], 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        old = saved['config']
        for key in config:
            if key not in ('created_at_utc', 'git_head_before_run') and old[key] != config[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        config = old
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        history, start_epoch = saved['history'], saved['epoch']
        random.setstate(saved['rng']['python'])
        np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'])
        torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        loader_generator.set_state(saved['rng']['loader'])
        occlusion_generator.set_state(saved['rng']['occlusion'])
    elif (args.output/'config.json').exists():
        if (args.output/'history.json').exists() or (args.output/'last.tmp').exists():
            raise FileExistsError(f'Incomplete checkpoint state requires inspection: {args.output}')
        old = json.loads((args.output/'config.json').read_text(encoding='utf-8'))
        for key in config:
            if key not in ('created_at_utc', 'git_head_before_run') and old[key] != config[key]:
                raise ValueError(f'Existing startup config mismatch: {key}')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'start', 'variant': args.variant, 'seed': args.seed,
                      'epochs': protocol['epochs'], 'steps': protocol['steps_per_epoch'],
                      'gpu': config['gpu']}, ensure_ascii=True), flush=True)
    for epoch in range(start_epoch, protocol['epochs']):
        sampler.epoch = epoch
        model.train()
        if epoch < protocol['fixbase_epochs']:
            model.eval()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(('backbone.fc.', 'backbone.classifier.')))
            model.backbone.fc.train()
            model.backbone.classifier.train()
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        warm = min(5, max(1, protocol['epochs']//5))
        factor = (epoch+1)/warm if epoch < warm else .5*(1+math.cos(
            math.pi*(epoch-warm)/max(1, protocol['epochs']-warm)))
        for group in optimizer.param_groups:
            group['lr'] = .0003*factor*group['multiplier']
        totals = {'loss': 0., 'ce': 0., 'triplet': 0., 'pairwise': 0., 'interaction': 0.}
        skipped, t0 = 0, time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for step, (images, labels, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            metadata = None
            if args.variant == 'plain':
                model_images, model_labels = images, labels
            elif args.variant in ('occlusion_aug', 'pairwise_consistency'):
                first, second, _ = paired_occlusion(images, labels, generator=occlusion_generator)
                model_images = torch.cat((first, second))
                model_labels = labels.repeat(2)
            else:
                views, targets, metadata = factorial_occlusion(images, labels,
                    generator=occlusion_generator, groups=protocol['factorial_groups'])
                model_images = views.flatten(0, 1)
                model_labels = targets.flatten()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                out = model(model_images, details=True)
                ce = F.cross_entropy(out['logits'], model_labels, label_smoothing=.1)
                triplet = batch_hard(F.normalize(out['features'].float(), dim=1), model_labels)
                pairwise, interaction = ce.new_zeros(()), ce.new_zeros(())
                if args.variant == 'pairwise_consistency':
                    paired = F.normalize(out['features'].float(), dim=1).reshape(2, len(labels), -1)
                    pairwise = (1-F.cosine_similarity(paired[0], paired[1], dim=-1)).mean()
                elif args.variant == 'factorial_interaction':
                    grouped = out['features'].float().reshape(protocol['factorial_groups'], 4, -1)
                    interaction = interaction_penalty(grouped)
                loss = ce + .5*triplet + protocol['pairwise_weight']*pairwise + \
                    protocol['interaction_weight']*interaction
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped += 1
            for key, value in [('loss', loss), ('ce', ce), ('triplet', triplet),
                               ('pairwise', pairwise), ('interaction', interaction)]:
                totals[key] += float(value.detach())
        row = {'epoch': epoch+1, 'lr': .0003*factor, 'pk_batches': step,
               'optimizer_updates': step-skipped, 'amp_skipped_updates': skipped,
               'train_seconds': time.perf_counter()-t0,
               'peak_cuda_mb': torch.cuda.max_memory_allocated()/2**20,
               **{key: value/step for key, value in totals.items()}}
        history.append(row)
        (args.output/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
               'loader': loader_generator.get_state(), 'occlusion': occlusion_generator.get_state()}
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch+1,
                    'history': history, 'config': config, 'rng': rng}, args.output/'last.tmp')
        (args.output/'last.tmp').replace(args.output/'last.pt')
        print(json.dumps({'event': 'epoch', 'variant': args.variant, 'seed': args.seed,
                          **row}, ensure_ascii=True), flush=True)
    eval_args = argparse.Namespace(eval_batch_size=96, workers=0,
        height=protocol['image_size'][0], width=protocol['image_size'][1],
        eval_occlusion=protocol['development_occlusion'])
    evaluation = run_v5_evaluation(model, query, gallery, train, eval_args, device)
    result = {'variant': args.variant, 'seed': args.seed, 'trained_epochs': history[-1]['epoch'],
              'evaluation': evaluation, 'history': history, 'config': config,
              'checkpoint_sha256': digest(args.output/'last.pt'),
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'note': 'Synthetic horizontal bands are a development stress test, not a real-occlusion benchmark.'}
    (args.output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'variant': args.variant, 'seed': args.seed,
                      'evaluation': evaluation}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
