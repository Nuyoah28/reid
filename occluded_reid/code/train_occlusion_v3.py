"""Matched single-view structural experiments on the fixed Market split."""
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
from torch.utils.data import DataLoader

from train import seed_all, seed_worker, run_evaluation
from reid.data import Images, PKSampler, training_split
from reid.losses import batch_hard
from occlusion_v3.model import VARIANTS, StructuralOSNet

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = sorted((ROOT/'code/reid').glob('*.py'))
    paths += sorted((ROOT/'code/occlusion_v3').glob('*.py'))
    paths += [Path(__file__), ROOT/'code/train.py', ROOT/'code/evaluate_occlusion_v3.py',
              ROOT/'code/deep-person-reid/torchreid/models/osnet.py',
              ROOT/'tools/run_occlusion_v3.py', ROOT/'tools/verify_occlusion_v3.py']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def record_manifest(records, data_root):
    rows = ['%s\t%d\t%d' % (Path(path).relative_to(data_root).as_posix(), pid, camera)
            for path, pid, camera in records]
    return hashlib.sha256('\n'.join(rows).encode()).hexdigest()


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
    train, query, gallery, split = training_split(data)
    if set(split['train_ids']) & set(split['val_ids']):
        raise RuntimeError('Train/development identity leakage')
    model = StructuralOSNet(len(split['train_ids']), args.variant)
    initialization = model.load_imagenet(pretrained)
    model.to(device)
    seed_all(args.seed)
    protocol_path = ROOT/'research/occlusion_v3_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    config = {
        'variant': args.variant, 'seed': args.seed,
        'epochs': protocol['epochs'], 'steps': protocol['pk_steps_per_epoch'],
        'batch_size': protocol['batch_size'], 'instances': protocol['instances'],
        'height': protocol['image_size'][0], 'width': protocol['image_size'][1],
        'protocol_sha256': digest(protocol_path), 'source_sha256': source_hashes(),
        'pretrained_sha256': digest(pretrained), 'split': split,
        'data_record_sha256': {name: record_manifest(records, data) for name, records in
                               [('train', train), ('query', query), ('gallery', gallery)]},
        'train_images': len(train), 'query_images': len(query), 'gallery_images': len(gallery),
        'optimizer': protocol['optimizer'], 'loss': protocol['loss'],
        'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__,
        'cuda': torch.version.cuda, 'python': platform.python_version(),
        'gpu': torch.cuda.get_device_name(0),
        'initialization': initialization,
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'inference_parameters': sum(p.numel() for n, p in model.named_parameters()
                                    if not n.startswith('backbone.classifier.')),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_before_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    groups = {'backbone': [], 'heads': []}
    for name, parameter in model.named_parameters():
        key = 'heads' if name.startswith(('backbone.fc.', 'backbone.classifier.', 'context.')) else 'backbone'
        groups[key].append(parameter)
    optimizer = torch.optim.Adam([
        {'params': groups['backbone'], 'lr': .0003, 'multiplier': 1.},
        {'params': groups['heads'], 'lr': .0015, 'multiplier': 5.}],
        weight_decay=.0005, amsgrad=True)
    scaler = torch.amp.GradScaler('cuda')
    sampler = PKSampler(train, 24, 4, args.seed, 80)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(Images(train, 256, 128, train=True), batch_size=24, sampler=sampler,
                        num_workers=0, pin_memory=True, worker_init_fn=seed_worker,
                        generator=loader_generator, persistent_workers=False)
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
    elif (args.output/'config.json').exists():
        raise FileExistsError(f'Existing run requires --resume: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'start', 'variant': args.variant, 'seed': args.seed,
                      'epochs': 8, 'steps': 80, 'gpu': config['gpu']}, ensure_ascii=True), flush=True)
    for epoch in range(start_epoch, 8):
        sampler.epoch = epoch
        model.train()
        if epoch < 2:
            model.eval()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(('backbone.fc.', 'backbone.classifier.', 'context.')))
            model.backbone.fc.train()
            model.backbone.classifier.train()
            model.context.train()
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        warm = min(5, max(1, 8//5))
        factor = (epoch+1)/warm if epoch < warm else .5*(1+math.cos(math.pi*(epoch-warm)/max(1, 8-warm)))
        for group in optimizer.param_groups:
            group['lr'] = .0003*factor*group['multiplier']
        totals = {'loss': 0., 'ce': 0., 'triplet': 0.}
        skipped, t0 = 0, time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for step, (images, labels, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                out = model(images, details=True)
                ce = F.cross_entropy(out['logits'], labels, label_smoothing=.1)
                triplet = batch_hard(F.normalize(out['features'].float(), dim=1), labels)
                loss = ce + .5*triplet
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
                print(json.dumps({'event': 'amp_skipped_update', 'epoch': epoch+1, 'step': step}), flush=True)
            totals['loss'] += float(loss.detach())
            totals['ce'] += float(ce.detach())
            totals['triplet'] += float(triplet.detach())
        row = {'epoch': epoch+1, 'lr': .0003*factor, 'pk_batches': step,
               'optimizer_updates': step-skipped, 'amp_skipped_updates': skipped,
               'train_seconds': time.perf_counter()-t0,
               'peak_cuda_mb': torch.cuda.max_memory_allocated()/2**20,
               **{k: v/step for k, v in totals.items()}}
        history.append(row)
        (args.output/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
               'loader': loader_generator.get_state()}
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch+1,
                    'history': history, 'config': config, 'rng': rng}, args.output/'last.tmp')
        (args.output/'last.tmp').replace(args.output/'last.pt')
        print(json.dumps({'event': 'epoch', **row}, ensure_ascii=True), flush=True)
    eval_args = argparse.Namespace(eval_batch_size=96, workers=0, height=256, width=128,
                                   eval_occlusion=[0., .3, .5])
    evaluation = run_evaluation(model, query, gallery, eval_args, device)
    result = {'variant': args.variant, 'seed': args.seed, 'trained_epochs': history[-1]['epoch'],
              'evaluation': evaluation, 'history': history, 'config': config,
              'checkpoint_sha256': digest(args.output/'last.pt'),
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'note': 'Fixed final epoch; synthetic bands are a stress test, not a real occlusion benchmark.'}
    (args.output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'evaluation': evaluation}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
