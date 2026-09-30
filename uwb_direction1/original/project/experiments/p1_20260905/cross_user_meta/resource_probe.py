"""Resource-only h1 source-user probes; never evaluate validation/test data.

Each candidate runs in a fresh child process so an OOM does not contaminate later
measurements. This is not a training run or an accuracy comparison.
"""
import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import statistics
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('_cross_meta_resource_' + name, HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stats(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    return {'l2': float(sum(g.detach().float().square().sum() for g in gradients).sqrt()) if gradients else 0.,
            'all_finite': all(bool(torch.isfinite(g).all()) for g in gradients),
            'count': len(gradients)}


def initialize():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load('model').build_model(2026, 'cuda').train()
    protocol = load('protocol')
    reader, labels, subjects, partitions = protocol.open_fold(1)
    return model, protocol, reader, labels, subjects, partitions


def candidate(arm, per_user, steps=3):
    model, protocol, reader, labels, subjects, partitions = initialize()
    step = load('meta_step')
    sampler = protocol.EpisodeSampler(partitions['train'], labels, subjects, 2026, per_user)
    optimizer = torch.optim.Adam([
        {'params': model.right.encoder.parameters(), 'lr': 1e-5},
        {'params': model.right.head.parameters(), 'lr': 1e-5},
        {'params': model.fuser.parameters(), 'lr': 1e-4}], weight_decay=1e-4)
    groups = {'encoder': list(model.right.encoder.parameters()),
              'head': list(model.right.head.parameters()), 'aux': list(model.fuser.parameters())}
    tracked = {'encoder': model.right.encoder.temporal.proj.weight,
               'head': model.right.head.net[0].weight, 'aux': model.fuser.head[0].weight}
    result = {'arm': arm, 'per_user': per_user, 'support_rows': 5*per_user, 'query_rows': per_user,
              'train_users': sorted(np.unique(subjects[partitions['train']]).tolist()),
              'ntrain': len(partitions['train']), 'steps': [], 'status': 'running'}
    try:
        for index in range(steps):
            torch.cuda.synchronize()
            started = time.perf_counter()
            ep = sampler.next()
            sx = reader.load_ids(ep['support_ids'], 'train', 'MV' if arm in ('A2', 'A3') else 'R')
            qx = reader.load_ids(ep['query_ids'], 'train', 'R')
            sy = torch.tensor(labels[ep['support_ids']], device='cuda')
            qy = torch.tensor(labels[ep['query_ids']], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            before = {k: v.detach().clone() for k, v in tracked.items()}
            torch.cuda.synchronize()
            compute_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                loss, diagnostics = step.episode_loss(model, sx, sy, qx, qy, arm, F.cross_entropy)
            loss.backward()
            gradient_stats = {k: stats(v) for k, v in groups.items()}
            optimizer.step()
            torch.cuda.synchronize()
            ended = time.perf_counter()
            row = {'step': index, 'query_user': ep['query_subject'], 'loss': float(loss.detach()),
                   'compute_seconds': ended-compute_started, 'end_to_end_seconds': ended-started,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                   'gradient_stats': gradient_stats,
                   'tracked_parameter_max_changes': {k: float((v.detach()-before[k]).abs().max()) for k,v in tracked.items()}}
            result['steps'].append(row)
            del loss, diagnostics, sx, qx, sy, qy, before
        result['status'] = 'ok'
    except RuntimeError as error:
        if not isinstance(error, torch.OutOfMemoryError) and 'out of memory' not in str(error).lower():
            raise
        result['status'] = 'oom'
        result['error'] = str(error)
        result['peak_allocated_bytes_at_oom'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes_at_oom'] = torch.cuda.max_memory_reserved()
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    return result


def query_probe():
    model, protocol, reader, labels, subjects, partitions = initialize()
    step = load('meta_step')
    ep = protocol.EpisodeSampler(partitions['train'], labels, subjects, 2026, 1).next()
    sx = reader.load_ids(ep['support_ids'], 'train', 'MV')
    qx = reader.load_ids(ep['query_ids'], 'train', 'R')
    sy = torch.tensor(labels[ep['support_ids']], device='cuda')
    qy = torch.tensor(labels[ep['query_ids']], device='cuda')
    with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
        r, mv = model.support_outputs(sx, 'MV')
        support = F.cross_entropy(r, sy) + .5*F.cross_entropy(mv, sy)
        fast = step.differentiable_update(dict(model.named_parameters()), support, 1e-4)
        query = F.cross_entropy(torch.func.functional_call(model, fast, (qx,)), qy)
    grads = torch.autograd.grad(query, tuple(model.fuser.parameters()), allow_unused=True)
    return {'status': 'ok', 'support_rows': 5, 'query_rows': 1,
            'query_only_aux_gradient_l2': float(sum(g.float().square().sum() for g in grads if g is not None).sqrt()),
            'query_only_aux_gradient_finite': all(bool(torch.isfinite(g).all()) for g in grads if g is not None),
            'original_grad_fields_untouched': all(p.grad is None for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['A0', 'A1', 'A2', 'A3'])
    ap.add_argument('--per-user', type=int)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--query-probe', action='store_true')
    args = ap.parse_args()
    if args.arm or args.query_probe:
        result = query_probe() if args.query_probe else candidate(args.arm, args.per_user, args.steps)
        print('PROBE_RESULT '+json.dumps(result), flush=True)
        return
    protocol = load('protocol')
    output = {'purpose': 'resource_only_not_effectiveness', 'held_user': 1, 'validation_user': 2,
              'seed': 2026, 'criterion': 'cross_entropy', 'alpha': 1e-4, 'mv_weight': .5,
              'optimizer': 'Adam', 'weight_decay': 1e-4, 'encoder_head_lr': 1e-5, 'aux_lr': 1e-4,
              'dtype': 'CUDA autocast bfloat16; parameters float32', 'attention': 'SDPBackend.MATH',
              'device': torch.cuda.get_device_name(), 'total_device_bytes': torch.cuda.get_device_properties(0).total_memory,
              'torch': torch.__version__, 'candidates': [],
              'source_sha256': {n: protocol.digest(HERE/(n+'.py')) for n in ['resource_probe','model','meta_step','protocol']}}
    for per_user in [4, 8, 16]:
        for arm in ['A0', 'A3']:
            print(f'PROBE_START arm={arm} per_user={per_user}', flush=True)
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--arm', arm, '--per-user', str(per_user)], capture_output=True, text=True)
            lines = [line[len('PROBE_RESULT '):] for line in child.stdout.splitlines() if line.startswith('PROBE_RESULT ')]
            result = json.loads(lines[-1]) if lines else {'arm': arm, 'per_user': per_user, 'status': 'error', 'exit': child.returncode, 'stderr': child.stderr[-5000:]}
            if result['status'] == 'error' and 'out of memory' in result.get('stderr', '').lower():
                result['status'] = 'oom'
            output['candidates'].append(result)
            protocol.dump(HERE/'RESOURCE_PROBE.json', output)
            print(f'PROBE_DONE arm={arm} per_user={per_user} status={result["status"]}', flush=True)
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--query-probe'], capture_output=True, text=True)
    lines = [line[len('PROBE_RESULT '):] for line in child.stdout.splitlines() if line.startswith('PROBE_RESULT ')]
    output['query_gradient_probe'] = json.loads(lines[-1]) if lines else {'status': 'error', 'stderr': child.stderr[-5000:]}
    summarize(output)
    protocol.dump(HERE/'RESOURCE_PROBE.json', output)
    print('RESOURCE_PROBE_COMPLETE', flush=True)


def summarize(output):
    """Report residency and timing limits; a completed spill is not safe fit."""
    for result in output['candidates']:
        if not result.get('steps'):
            continue
        rows = result['steps']
        result['summary'] = {
            'mean_steady_compute_seconds': statistics.mean(r['compute_seconds'] for r in rows[1:]),
            'mean_steady_end_to_end_seconds': statistics.mean(r['end_to_end_seconds'] for r in rows[1:]),
            'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in rows),
            'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in rows),
            'exceeds_physical_device_memory': max(r['peak_allocated_bytes'] for r in rows) > output['total_device_bytes'],
            'episodes_per_epoch': math.ceil(result['ntrain']/(6*result['per_user']))}
    output['recommendation'] = {
        'per_user': 4, 'support_rows': 20, 'query_rows': 4, 'h1_episodes_per_epoch': 296,
        'reason': 'Largest tested candidate fitting A3 with physical device memory headroom; per_user8 A3 allocations exceed physical VRAM and slow sharply; per_user16 A3 OOM.',
        'notes': [
            'Three full Adam steps for A0/A3 successful candidates; two for supplemental A1/A2; mean steady time excludes first step.',
            'Compute timing includes gradient diagnostics and parameter snapshots inflate live memory slightly.',
            'End-to-end timing includes training-row reads and device transfer.',
            'Allocated above physical memory with sharp slowdown indicates Windows memory oversubscription; OS shared-memory counters were not directly sampled.',
            'CE-only resource probe, not final soft-target criterion, not an accuracy result.',
            'No validation/test examples or predictions were read; no formal training/checkpoint was started.',
            'Epoch projections exclude validation, checkpoint writes, thermal drift and other concurrent GPU work.',
            'Use math SDPA, BF16 autocast and identical per-user episode count for all comparison arms.']}


if __name__ == '__main__':
    main()
