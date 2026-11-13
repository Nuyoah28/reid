"""Reproducible small-GPU AGW controls; never evaluates the test set during training."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import Images, PairedPK, read_records, train_partition
from model import AGW
from objectives import objective

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'code').rglob('*.py')) if 'upstream' not in p.parts}


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['regdb', 'sysu'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--variant', choices=['agw', 'center', 'unbounded', 'bcrt'], default='agw')
    p.add_argument('--protocol', choices=['dev', 'official'], default='dev')
    p.add_argument('--trial', type=int, default=1)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--identities', type=int, default=4)
    p.add_argument('--instances', type=int, default=2)
    p.add_argument('--height', type=int, default=288)
    p.add_argument('--width', type=int, default=144)
    p.add_argument('--lr', type=float, default=.1)
    p.add_argument('--weight', type=float, default=.2)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--retain', type=float, default=.5)
    p.add_argument('--warmup', type=int, default=5, help='Delay auxiliary loss until this epoch count.')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--weights', type=Path, default=ROOT/'weights/resnet50-19c8e357.pth')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    return p


def main():
    args = parser().parse_args()
    if args.epochs < 1 or args.steps < 0 or not 0 <= args.retain <= 1 or args.alpha < 0:
        raise ValueError('Invalid epochs, steps or transport parameters.')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    rows = read_records(args.dataset, args.data, 'train', args.trial)
    train, held, mapping = train_partition(rows, args.protocol)
    # Detect accidentally mixed training/testing identities before optimization.
    test = read_records(args.dataset, args.data, 'test', args.trial)
    if {r.pid for r in rows} & {r.pid for r in test}:
        raise ValueError('Train/test identity overlap: check the original dataset indexes.')
    if not args.weights.is_file() and not args.resume:
        raise FileNotFoundError(f'ImageNet weights required: run tools/prepare.py first ({args.weights}).')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    config.update(source_hashes=source_hashes(), training_ids=sorted(mapping),
                  validation_ids=sorted({r.pid for r in held}), test_ids=sorted({r.pid for r in test}),
                  training_images=len(train), classes=len(mapping),
                  protocol_note='Fixed final epoch; no test-driven checkpoint selection. RegDB trial is one split.')
    history_file = output/'history.json'
    if (output/'config.json').exists():
        previous = json.loads((output/'config.json').read_text())
        if previous != config or not args.resume:
            raise ValueError('Existing run: resume with identical configuration/source, or choose a new output.')
    elif args.resume:
        raise ValueError('Resume only inside the original run directory with config.json.')
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    model = AGW(len(mapping), args.weights if not args.resume else None).to(device)
    heads = list(model.neck.parameters()) + list(model.classifier.parameters())
    head_ids = {id(p) for p in heads}
    optimizer = torch.optim.SGD([{'params': [p for p in model.parameters() if id(p) not in head_ids], 'scale': .1},
                                 {'params': heads, 'scale': 1.}], lr=args.lr, momentum=.9, weight_decay=5e-4, nesterov=True)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=128.)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = PairedPK(train, args.identities, args.instances, args.steps, args.seed)
    loader = DataLoader(Images(train, args.height, args.width, True, mapping), batch_sampler=sampler,
                        num_workers=args.workers, pin_memory=amp, worker_init_fn=seed_worker, generator=generator)
    start, history = 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Checkpoint configuration mismatch.')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler']); start = checkpoint['epoch']
        history = json.loads(history_file.read_text())[:start]
        random.setstate(checkpoint['random']); np.random.set_state(checkpoint['numpy'])
        torch.set_rng_state(checkpoint['torch'].cpu()); generator.set_state(checkpoint['loader'].cpu())
        if amp:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda']])
        model.initialization = checkpoint['initialization']
    for epoch in range(start, args.epochs):
        sampler.epoch = epoch
        model.train()
        # AGW-style warmup and step schedule; auxiliary warmup is shared by controls.
        lr = args.lr * ((epoch+1)/10 if epoch < 10 else 1 if epoch < 20 else .1 if epoch < 50 else .01)
        for group in optimizer.param_groups:
            group['lr'] = lr * group['scale']
        totals, begin, skipped = {}, time.perf_counter(), 0
        if amp:
            torch.cuda.reset_peak_memory_stats()
        for images, labels, modalities, _ in loader:
            images, labels, modalities = images.to(device), labels.to(device), modalities.to(device)
            # Explicit reordering keeps labels paired with the model's VIS-then-IR concatenation.
            order = torch.cat([(modalities == m).nonzero().flatten() for m in [0, 1]])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                result = model(images[modalities == 0], images[modalities == 1])
            loss, logs = objective(result, labels[order], modalities[order], args.variant,
                                   args.weight if epoch >= args.warmup else 0., args.alpha, args.retain)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}: {logs}')
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            skipped += int(scaler.get_scale() < previous_scale)
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.) + value
        row = {'epoch': epoch+1, 'steps': len(sampler), 'lr_head': lr,
               'skipped_updates': skipped, 'amp_scale': scaler.get_scale(),
               'seconds': time.perf_counter()-begin,
               'peak_cuda_mib': torch.cuda.max_memory_allocated()/2**20 if amp else 0,
               **{key: value/len(sampler) for key, value in totals.items()}}
        history.append(row)
        history_file.write_text(json.dumps(history, indent=2), encoding='utf-8')
        snapshot = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch+1, 'config': config, 'initialization': model.initialization,
                    'random': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                    'loader': generator.get_state(), 'cuda': torch.cuda.get_rng_state_all() if amp else []}
        temporary = output/'last.tmp'
        torch.save(snapshot, temporary); temporary.replace(output/'last.pt')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
