"""Method B: per-view motion evidence from the RF frames and the 15-segment targets.

Energies are computed from the unnormalised analytic signal, never from the
per-sample standardised maps the classifier eats, so a genuinely weak clip stays
weak instead of being rescaled into a strong one.
"""
import argparse
import json

import numpy as np
import torch

import protocol as P
from protocol import CACHE, ROOT, digest, dump

ENERGY_CACHE = CACHE / 'motion_energy.npz'
EPS = 1e-8


def frame_evidence(rf):
    """rf: (V,T,M) float. Returns phase energy, amplitude-change energy, concentration.

    Phase energy uses the absolute phase increment so approach and recession add
    up instead of cancelling; signed direction stays the job of the classifier
    input, not of this process target.
    """
    from uwb_osr.data import _analytic
    z = _analytic(np.asarray(rf, dtype=np.float64))
    z = z - z.mean(axis=1, keepdims=True)
    d = z[:, 1:] * np.conj(z[:, :-1])
    w = np.sqrt(np.abs(d))
    phase = np.angle(d)
    e_phase = (w * np.abs(phase)).sum(-1)
    e_amp = np.abs(np.abs(z[:, 1:]) - np.abs(z[:, :-1])).sum(-1)
    concentration = np.abs((w * np.exp(1j * phase)).sum(-1)) / (w.sum(-1) + EPS)

    def pad(a):
        return np.concatenate([a[:, :1], a], axis=1).astype('float32')

    return pad(e_phase), pad(e_amp), pad(concentration)


def build_energy_cache(chunk=32):
    if ENERGY_CACHE.exists():
        meta = json.loads(ENERGY_CACHE.with_suffix('.json').read_text(encoding='utf-8'))
        assert meta['cache_sha256'] == digest(ENERGY_CACHE), 'motion energy cache changed on disk'
        return
    source = ROOT / 'cache/X_clutterremoved.npy'
    x = np.load(source, mmap_mode='r')
    n = len(x)
    e_phase = np.zeros((n, 3, P.FRAMES), 'float32')
    e_amp = np.zeros_like(e_phase)
    concentration = np.zeros_like(e_phase)
    for start in range(0, n, chunk):
        block = np.asarray(x[start:start + chunk], dtype=np.float64)
        for j, row in enumerate(block):
            e_phase[start + j], e_amp[start + j], concentration[start + j] = frame_evidence(row)
        if start % 1024 == 0:
            print('motion_energy', start, n, flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ENERGY_CACHE, e_phase=e_phase, e_amp=e_amp, concentration=concentration)
    dump(ENERGY_CACHE.with_suffix('.json'),
         {'definition': 'per frame sum over range of sqrt(abs(d))*abs(angle(d)), '
                        'amplitude change, and circular concentration. d is the analytic '
                        'frame-to-frame product after full-clip static removal.',
          'source': str(source), 'source_sha256': digest(source),
          'code_sha256': digest(__file__), 'shape': [n, 3, P.FRAMES],
          'causal': False, 'calibrated_velocity': False,
          'cache_sha256': digest(ENERGY_CACHE)})


def load_energy(kind='phase'):
    build_energy_cache()
    with np.load(ENERGY_CACHE) as a:
        energy = a['e_phase' if kind == 'phase' else 'e_amp'].astype('float32')
        concentration = a['concentration'].astype('float32')
    return energy, concentration


def fold_scale(energy, train_ids, valid=None):
    """Per-view 95th percentile of training frames only; clip at five for safety."""
    frames = energy[np.asarray(train_ids)]
    if valid is not None:
        frames = np.where(valid[np.asarray(train_ids)], frames, np.nan)
        scale = np.nanpercentile(frames, 95, axis=(0, 2))
    else:
        scale = np.percentile(frames, 95, axis=(0, 2))
    return np.maximum(scale, EPS).astype('float32')


def fuse_process_targets(energy, quality, frame_valid, eps=EPS):
    """energy, quality, frame_valid: B,V,90 tensors. Returns activity, progress, masks."""
    b, v, t = energy.shape
    assert t == P.FRAMES and quality.shape == frame_valid.shape == energy.shape
    q = quality.float() * frame_valid.float()
    denominator = q.sum(1)
    frame_ok = denominator > eps
    fused = (energy.float() * q).sum(1) / denominator.clamp_min(eps)

    cnt = frame_ok.reshape(b, P.SEGMENTS, P.FRAMES_PER_SEGMENT).sum(-1)
    summed = (fused * frame_ok).reshape(b, P.SEGMENTS, P.FRAMES_PER_SEGMENT).sum(-1)
    segment_energy = summed / cnt.clamp_min(1)
    seg_ok = cnt > 0
    total = segment_energy.sum(-1)
    valid = (total > eps) & seg_ok.any(-1)

    smoothed = (segment_energy + eps) * seg_ok
    activity = smoothed / smoothed.sum(-1, keepdim=True).clamp_min(eps)
    activity = torch.where(valid[:, None], activity, torch.zeros_like(activity))
    progress = activity.cumsum(-1)
    return activity.detach(), progress.detach(), seg_ok, valid


def build_fold_targets(fold, train_ids, mode='multi', kind='phase', quality_mode='equal',
                       shift=None, device='cuda'):
    """Frozen 15-segment targets for one fold.

    mode: multi (weighted mean over the three radars), self (deployed radar
    only), multi_max (per frame maximum over radars). Scales, shifts and every
    fitted statistic come from train_ids only.
    """
    energy, concentration = load_energy(kind)
    views = [0, 1, 2] if mode != 'self' else [P.DEPLOY_VIEW]
    energy = energy[:, views]
    concentration = concentration[:, views]
    valid = np.ones(energy.shape, dtype=bool)
    if shift is not None:
        shifted = np.zeros_like(energy)
        shifted_valid = np.zeros_like(valid)
        for j, v in enumerate(views):
            k = int(shift[v])
            src = slice(max(0, k), P.FRAMES + min(0, k))
            dst = slice(max(0, -k), P.FRAMES - max(0, k))
            shifted[:, j, dst] = energy[:, j, src]
            shifted_valid[:, j, dst] = True
        energy, valid = shifted, shifted_valid
    scale = fold_scale(energy, train_ids, valid)
    scaled = np.clip(energy / scale[None, :, None], 0, 5)
    quality = 0.25 + 0.75 * concentration if quality_mode == 'quality' else np.ones_like(scaled)
    e = torch.tensor(scaled, device=device)
    q = torch.tensor(quality.astype('float32'), device=device)
    m = torch.tensor(valid, device=device)
    if mode == 'multi_max':
        e = torch.where(m, e, torch.full_like(e, -1.0)).max(1, keepdim=True).values.clamp_min(0)
        m = m.any(1, keepdim=True)
        q = torch.ones_like(e)
    activity, progress, seg_ok, sample_ok = fuse_process_targets(e, q, m)
    meta = {'mode': mode, 'kind': kind, 'quality_mode': quality_mode,
            'views': [P.VIEW_NAMES[v] for v in views], 'clip': 5.0,
            'scale_percentile95': scale.tolist(),
            'frame_shift': None if shift is None else [int(s) for s in shift],
            'target_transform_fit_ids_hash': P.hash_ids(train_ids)}
    return activity, progress, seg_ok, sample_ok, meta


def shuffle_time(activity, generator):
    """B-TIME control: permute the segment order, then rebuild the CDF from it."""
    perm = torch.randperm(P.SEGMENTS, generator=generator, device=activity.device)
    shuffled = activity[:, perm]
    return shuffled, shuffled.cumsum(-1)


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    build_energy_cache()
    energy, concentration = load_energy()
    print('energy', energy.shape, 'finite', bool(np.isfinite(energy).all()),
          'zero rows', int((energy.sum(-1) <= EPS).sum()))
    print('MOTION_ENERGY_OK')
