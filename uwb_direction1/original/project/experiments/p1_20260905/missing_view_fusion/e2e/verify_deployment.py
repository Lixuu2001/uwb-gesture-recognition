"""Replay completed formal predictions from standalone Right-only exports."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_deployment_name = '_e2e_standalone_deployment_replay'
if _deployment_name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_deployment_name, HERE / 'deploy.py')
    _deployment = importlib.util.module_from_spec(_spec)
    sys.modules[_deployment_name] = _deployment
    _spec.loader.exec_module(_deployment)
_deployment = sys.modules[_deployment_name]
BASE = _deployment.BASE
load_deployment = _deployment.load_deployment
predict_logits = _deployment.predict_logits


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def summarize_replays(records):
    """Formal acceptance requires exactly one passing run per seed-2026 pair."""
    expected = {(mode, fold, 2026) for mode in ('R', 'MV') for fold in range(1, 9)}
    identities = [(row.get('mode'), row.get('fold'), row.get('seed')) for row in records]
    complete = len(identities) == len(expected) and set(identities) == expected
    return {'required_seed': 2026, 'expected_runs': len(expected),
            'complete_eight_fold_pairing': complete,
            'passed': complete and all(row['passed'] for row in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=HERE / 'runs')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Exact replay requires the CUDA evaluation environment')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    amplitude = np.load(BASE / 'prepared/off.npy', mmap_mode='r')
    phase = np.load(BASE / 'prepared/only.npy', mmap_mode='r')
    records = []
    for directory in sorted(args.runs.iterdir()):
        status_path = directory / 'status.json'
        if not directory.is_dir() or not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if status.get('status') != 'complete':
            continue
        checkpoint = directory / 'deployment.pt'
        archive = directory / 'predictions.npz'
        if any(digest(directory / name) != expected for name, expected in status['artifacts'].items()):
            raise ValueError(f'Changed saved artifact in {directory}')
        with np.load(archive, allow_pickle=False) as saved:
            ids, expected = saved['ids'], saved['logits']
        # Match ViewReader: concatenate prepared channels, quantize to float16.
        # Only the Right slice is fetched; labels never enter inference.
        right = torch.from_numpy(np.concatenate((amplitude[ids, 2:3], phase[ids, 2:3]), -1)).half()
        model = load_deployment(checkpoint, device='cuda')
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        if any(status[key] != config[key] for key in ('fold', 'mode', 'seed')):
            raise ValueError(f'Status/config replay identity mismatch in {directory}')
        actual = predict_logits(model, right, batch=config['microbatch'], precision='bf16').numpy()
        error = np.abs(actual - expected)
        matches = int((actual.argmax(1) == expected.argmax(1)).sum())
        passed = bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-5) and matches == len(ids))
        records.append({'run': directory.name, 'fold': status['fold'], 'mode': status['mode'],
                        'seed': status['seed'],
                        'n_test': len(ids), 'prediction_matches': matches,
                        'max_absolute_logit_error': float(error.max()),
                        'mean_absolute_logit_error': float(error.mean()),
                        'deployment_sha256': digest(checkpoint), 'passed': passed})
        print(f'DEPLOYMENT_REPLAY {directory.name} matches={matches}/{len(ids)} max_error={error.max():.8g} passed={passed}', flush=True)
        del model, right
        gc.collect()
        torch.cuda.empty_cache()
    report = {'scope': 'formal seed-2026 sixteen-run acceptance; all test rows; standalone deployment graph',
              'device': torch.cuda.get_device_name(), 'precision': 'CUDA BF16 autocast',
              'input': 'prepared Right amplitude+phase, quantized to float16 before inference',
              'code_sha256': {name: digest(HERE / name) for name in ('deploy.py', 'verify_deployment.py')},
              **summarize_replays(records), 'runs': records}
    path = args.runs / 'DEPLOYMENT_CHECK.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)
    if not report['passed']:
        raise AssertionError(f'Formal sixteen-run coverage or deployment prediction parity failed; see {path}')
    print(f'DEPLOYMENT_CHECK_OK runs={len(records)} report={path}', flush=True)


if __name__ == '__main__':
    main()
