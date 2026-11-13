"""GPU pipeline checks and exact layer trace. Synthetic tensors are NOT ReID results."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from model import AGW
from objectives import objective, safe_step


def shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return [shape(x) for x in value] if isinstance(value, (list, tuple)) else None


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    payload = {'kind': 'SYNTHETIC_PIPELINE_CHECK_NOT_BENCHMARK', 'gpu': torch.cuda.get_device_name(),
               'input': [16, 3, 288, 144], 'P': 4, 'K_per_modality': 2, 'variants': {}}
    for variant in ['agw', 'center', 'unbounded', 'bcrt']:
        torch.manual_seed(19)
        model = AGW(206).to(device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        scaler = torch.amp.GradScaler('cuda', init_scale=128.)
        visible, thermal = [torch.randn(8, 3, 288, 144, device=device) for _ in [0, 1]]
        labels = torch.arange(4, device=device).repeat_interleave(2).repeat(2)
        modalities = torch.arange(2, device=device).repeat_interleave(8)
        torch.cuda.reset_peak_memory_stats()
        before = model.classifier.weight.detach().clone()
        start = time.perf_counter()
        logs = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                output = model(visible, thermal)
            loss, diagnostic = objective(output, labels, modalities, variant)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
            if not torch.isfinite(loss) or not finite:
                invalid = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                raise FloatingPointError(f'{variant}: nonfinite loss or gradient: {diagnostic}, {invalid[:6]}')
            scaler.step(optimizer); scaler.update()
            logs.append(diagnostic)
        torch.cuda.synchronize()
        payload['variants'][variant] = {'steps': logs, 'finite_gradients': finite,
            'parameters_changed': not torch.equal(before, model.classifier.weight),
            'seconds_3_steps': time.perf_counter()-start, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'parameters_with_206_classes': sum(p.numel() for p in model.parameters())}
        if not payload['variants'][variant]['parameters_changed']:
            raise AssertionError('Optimizer failed to update parameters.')
        del model, optimizer, scaler, output, loss
        gc.collect(); torch.cuda.empty_cache()
    model = AGW(206).to(device).eval()
    calls, handles = [], []
    def hook(name):
        def record(module, inputs, output):
            calls.append({'module': name, 'type': type(module).__name__, 'spec': module.extra_repr(),
                          'input': shape(inputs), 'output': shape(output)})
        return record
    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        model(torch.randn(1, 3, 288, 144, device=device), torch.randn(1, 3, 288, 144, device=device))
    for handle in handles:
        handle.remove()
    inference = torch.randn(1, 3, 288, 144, device=device)
    with torch.inference_mode(), torch.autocast('cuda'):
        for _ in range(20):
            model(visible=inference)
        torch.cuda.synchronize(); start = time.perf_counter()
        for _ in range(80):
            model(visible=inference)
        torch.cuda.synchronize()
    payload['inference'] = {'batch': 1, 'amp_fp16': True, 'iterations': 80,
                            'mean_ms': (time.perf_counter()-start)*1000/80,
                            'excludes': 'image I/O and retrieval; random initialization; no accuracy meaning'}
    # Mechanism stress case: fixed step crosses a boundary, capped step retains half the initial margin.
    x, d, p = torch.tensor([[.2, 0.]]), torch.tensor([[2., 0.]]), torch.tensor([[0., 0.], [1., 0.]])
    step, _ = safe_step(x, d, p, torch.tensor([0]), alpha=.5, retain=.5)
    payload['analytic_case'] = {'initial_margin': .6, 'unbounded_step': .5, 'unbounded_margin': -1.4,
                              'bounded_step': step.item(), 'bounded_margin': .3,
                              'scope': 'Fixed unnormalized Euclidean prototypes only; no classifier/generalization guarantee.'}
    payload['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT/'code').glob('*.py'))}
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/pipeline_check.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (ROOT/'research/architecture/layer_trace.json').write_text(json.dumps(calls, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
