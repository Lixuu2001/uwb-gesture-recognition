"""Matched-width ablation discarding motion sign BEFORE normalization."""
import argparse
import gc
from pathlib import Path
import numpy as np
from pipeline import BASE, ROOT, digest, dump, norm, doppler_map

DEST = BASE / 'prepared/phase_abs_R.npy'
META = BASE / 'followup_results/phase_abs_R_manifest.json'


def unsigned_from_motion(motion):
    return norm(np.abs(motion)).astype('float32')


def unsigned_phase(raw):
    return unsigned_from_motion(doppler_map(raw))


def prepare():
    src = ROOT / 'cache/X_clutterremoved.npy'
    if DEST.exists() or META.exists():
        if not (DEST.exists() and META.exists()): raise RuntimeError('Incomplete unsigned feature cache')
        import json
        m = json.loads(META.read_text())
        assert m['cache_sha'] == digest(DEST) and m['source_sha'] == digest(src)
        assert m['code_sha'] == digest(__file__)
        print('PHASE_ABS_CACHE_VERIFIED'); return
    x = np.load(src, mmap_mode='r')
    tmp = DEST.with_name('phase_abs_R.pending.npy')
    out = np.lib.format.open_memmap(tmp, mode='w+', dtype='float16', shape=(len(x), 1, 90, 189))
    signed = np.load(BASE / 'prepared/only.npy', mmap_mode='r')
    for i in range(len(x)):
        raw = np.asarray(x[i, 2:3], dtype=np.float32)
        motion = doppler_map(raw)
        if i < 3:
            np.testing.assert_array_equal(norm(motion).astype('float16'), signed[i, 2:3])
        out[i] = unsigned_from_motion(motion)
        if i % 2000 == 0: print('PREPARE_UNSIGNED', i, len(x), flush=True)
    out.flush(); del out
    tmp.replace(DEST)
    dump(META, {'definition': 'norm(abs(amplitude_weighted_wrapped_phase_increment)); Right only',
                'source': str(src), 'source_sha': digest(src), 'cache_sha': digest(DEST),
                'code_sha': digest(__file__), 'shape': [len(x), 1, 90, 189],
                'no_fitted_parameters': True, 'signed_cache_equivalence_checked': 3})
    print('PHASE_ABS_READY', str(DEST), flush=True)


def run(folds):
    import torch
    import train
    prepare()
    args = argparse.Namespace(tag='sv_R_phase_abs', mode='concat', views=[2], readout='mean',
         folds=folds, seed=2026, epochs=40, batch=128, microbatch=32, teacher='', save_teacher=False,
         kd_weight=.5, aux='none', input_variant='phase_absolute_before_normalization',
         input_cache_sha=digest(DEST), wrapper_sha=digest(__file__))
    train.setup(args.seed)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    data = train.load_gpu('concat', [2])
    unsigned = np.load(DEST, mmap_mode='r')
    for i in range(0, len(unsigned), 128):
        part = torch.from_numpy(np.array(unsigned[i:i+128], copy=True))
        data[i:i+len(part), :, :, 189:].copy_(part)
    for fold in folds:
        train.run(args, data, fold); gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--folds', nargs='+', type=int, default=list(range(1, 9)))
    args = ap.parse_args()
    if args.prepare: prepare()
    else: run(args.folds)


if __name__ == '__main__': main()
