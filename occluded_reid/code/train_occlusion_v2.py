"""Matched three-view, short-budget occlusion experiments; legacy sources frozen."""
import argparse
from collections import Counter
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
from torch.utils.data import DataLoader
from train import seed_all, seed_worker, run_evaluation
from reid.data import training_split
from reid.losses import batch_hard
from reid.occlusion import paired_occlusion
from occlusion_v2.data import IndexedImages, PeerPKSampler
from occlusion_v2.method import VARIANTS, build_model, auxiliary_losses

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = sorted((ROOT/'code/reid').glob('*.py'))
    paths += sorted((ROOT/'code/occlusion_v2').glob('*.py'))
    paths += [Path(__file__), ROOT/'code/train.py', ROOT/'code/deep-person-reid/torchreid/models/osnet.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def record_manifest(records, data_root):
    lines = ['%s\t%d\t%d' % (Path(p).relative_to(data_root).as_posix(), pid, cam) for p, pid, cam in records]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if args.epochs < 3 or args.steps < 1:
        p.error('Require epochs>=3 and steps>=1')
    if (args.output/'config.json').exists() and not args.resume:
        p.error('Output exists; use a fresh name or explicit --resume')
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    seed_all(args.seed)
    data = ROOT/'data/Market-1501-v15.09.15'
    pretrained = ROOT/'weights/osnet_x1_0_imagenet.pth'
    train, query, gallery, split = training_split(data)
    if set(split['train_ids']) & set(split['val_ids']):
        raise RuntimeError('Identity leakage')
    model = build_model(len(split['train_ids']), args.variant)
    initialization = model.load_imagenet(pretrained)
    model.to(device)
    seed_all(args.seed)
    backbone, heads = [], []
    for name, param in model.named_parameters():
        (heads if name.startswith(('classifier.', 'reliability.')) else backbone).append(param)
    optimizer = torch.optim.Adam([
        {'params': backbone, 'lr': .0003, 'multiplier': 1.},
        {'params': heads, 'lr': .0015, 'multiplier': 5.}], weight_decay=5e-4, amsgrad=True)
    scaler = torch.amp.GradScaler('cuda')
    sampler = PeerPKSampler(train, 24, 4, args.seed, args.steps)
    loader_gen = torch.Generator().manual_seed(args.seed)
    occlusion_gen = torch.Generator().manual_seed(args.seed+5000)
    loader = DataLoader(IndexedImages(train, 256, 128, train=True), batch_size=24,
                        sampler=sampler, num_workers=args.workers, pin_memory=True,
                        worker_init_fn=seed_worker, generator=loader_gen, persistent_workers=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    config.update({
        'protocol': 'v2_market_train_holdout', 'recipe_version': 'occlusion_v2.1',
        'height': 256, 'width': 128, 'batch_size': 24, 'instances': 4, 'views': 3,
        'lr': .0003, 'head_lr_multiplier': 5., 'fixbase_epochs': 2,
        'tau': .15, 'identity_temperature': .1, 'calibration_weight': .5, 'evidence_weight': .5,
        'pool_floor': .05, 'split': split, 'train_images': len(train),
        'query_images': len(query), 'gallery_images': len(gallery),
        'identities_under_k': sum(n < 4 for n in Counter(r[1] for r in train).values()),
        'sampling': 'PeerPKSampler; all IDs retained; exclude duplicate source images in peer target',
        'source_sha256': source_hashes(), 'pretrained_sha256': sha256(pretrained),
        'registered_protocol_sha256': sha256(ROOT/'research/occlusion_v2_protocol.json'),
        'data_record_sha256': {name: record_manifest(records, data) for name, records in
                               [('train', train), ('query', query), ('gallery', gallery)]},
        'initialization': initialization, 'torch': torch.__version__,
        'torchvision': __import__('torchvision').__version__, 'cuda': torch.version.cuda,
        'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0),
        'git_head_before_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'determinism': 'Seeded with cuDNN benchmark; not guaranteed bitwise deterministic',
        'inference_parameters': sum(v.numel() for k, v in model.named_parameters() if not k.startswith('classifier.')),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation': 'Fixed last epoch only. No intermediate validation or external target-domain training.',
    })
    history, start = [], 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        old = saved['config']
        for key in config:
            if key not in ('output', 'created_at_utc', 'git_head_before_run') and old[key] != config[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        config = old
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        history, start = saved['history'], saved['epoch']
        rng = saved['rng']
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])
        loader_gen.set_state(rng['loader']); occlusion_gen.set_state(rng['occlusion'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'start', 'variant': args.variant, 'seed': args.seed, 'epochs': args.epochs,
                      'steps': args.steps, 'resume_epoch': start, 'views': 3, 'gpu': config['gpu']}), flush=True)
    for epoch in range(start, args.epochs):
        model.train(); sampler.epoch = epoch
        if epoch < 2:
            model.eval()
            for name, param in model.named_parameters():
                param.requires_grad_(name.startswith(('classifier.', 'reliability.')))
            model.classifier.train()
            if model.reliability is not None:
                model.reliability.train()
        else:
            for param in model.parameters():
                param.requires_grad_(True)
        warm = min(5, max(1, args.epochs//5))
        factor = (epoch+1)/warm if epoch < warm else .5*(1+math.cos(math.pi*(epoch-warm)/max(1, args.epochs-warm)))
        for group in optimizer.param_groups:
            group['lr'] = .0003*factor*group['multiplier']
        keys = ('loss', 'ce', 'triplet', 'calibration', 'evidence', 'target_mean', 'identity_mean', 'margin_mean', 'calibration_mae')
        sums = dict.fromkeys(keys, 0.)
        skipped_updates = 0
        t0 = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for step, (images, labels, _, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            # Consume the same donor/geometry RNG in repeat3 as in all other variants.
            a, b, visible = paired_occlusion(images, labels, generator=occlusion_gen)
            inputs = torch.cat([images, images, images] if args.variant == 'repeat3' else [images, a, b])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                out = model(inputs, details=True)
                ce = F.cross_entropy(out['logits'], labels.repeat(3), label_smoothing=.1)
                tri = torch.stack([batch_hard(F.normalize(f.float(), dim=1), labels)
                                   for f in out['features'].chunk(3)]).mean()
                cal, evidence, diagnostic = auxiliary_losses(out, labels, visible, args.variant, image_indices=indices)
                loss = ce + .5*tri + .5*cal + .5*evidence
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; preserve log and investigate')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < old_scale:
                skipped_updates += 1
                # Count this explicitly; a skipped AMP update is not an optimizer update.
                print(json.dumps({'event': 'amp_skipped_update', 'epoch': epoch+1, 'step': step}), flush=True)
            values = dict(loss=loss, ce=ce, triplet=tri, calibration=cal, evidence=evidence, **diagnostic)
            for key in keys:
                sums[key] += float(values[key].detach())
            if step % 20 == 0:
                print(json.dumps({'event': 'step', 'epoch': epoch+1, 'step': step,
                                  'loss': sums['loss']/step, 'elapsed_seconds': time.perf_counter()-t0}), flush=True)
        row = {'epoch': epoch+1, 'lr': .0003*factor, 'backbone_frozen': epoch < 2,
               'train_seconds': time.perf_counter()-t0, 'pk_batches': step,
               'amp_skipped_updates': skipped_updates, 'optimizer_updates': step-skipped_updates,
               'peak_cuda_mb': torch.cuda.max_memory_allocated()/2**20,
               **{key: value/step for key, value in sums.items()}}
        history.append(row)
        (args.output/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
               'cuda': torch.cuda.get_rng_state_all(), 'loader': loader_gen.get_state(), 'occlusion': occlusion_gen.get_state()}
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch+1, 'history': history, 'config': config, 'rng': rng}, args.output/'last.tmp')
        (args.output/'last.tmp').replace(args.output/'last.pt')
        print(json.dumps({'event': 'epoch', **row}), flush=True)
    eval_args = argparse.Namespace(eval_batch_size=96, workers=args.workers, height=256, width=128,
                                   eval_occlusion=[0., .3, .5])
    evaluation = run_evaluation(model, query, gallery, eval_args, device)
    result = {'variant': args.variant, 'seed': args.seed, 'trained_epochs': history[-1]['epoch'],
              'evaluation': evaluation, 'history': history, 'config': config,
              'checkpoint_sha256': sha256(args.output/'last.pt'),
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'note': '651-ID short-budget development run; synthetic noise is NOT a real occluded benchmark.'}
    (args.output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'evaluation': evaluation}), flush=True)


if __name__ == '__main__':
    main()
