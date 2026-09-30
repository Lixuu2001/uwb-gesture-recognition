"""Fixed low-probability same-class cross-user statistics training experiment."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = BASE/'cross_user_meta/runs_s2026'
CONTROL_ROOT = BASE/'mild_time_augmentation/runs'
spec = importlib.util.spec_from_file_location('_mild_style_original_trainer', BASE/'cross_user_meta/trainer.py')
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
original_epoch = T.warmup_epoch


def assert_control_match(source, config, manifest, result):
    reference = manifest['source']
    for key in ('data', 'runtime'):
        if source[key] != reference[key]: raise ValueError(f'Control {key} differs')
    for path, digest in source['code'].items():
        if reference['code'].get(path) != digest: raise ValueError('Original model/training code differs')
    for key, value in config.items():
        if manifest['config'].get(key) != value: raise ValueError(f'Control config differs: {key}')
    identity = result['identity']
    expected = dict(manifest['config'], augmentation_fold=identity['fold'], active_mode='CONTROL')
    if identity['source'] != reference or identity['config'] != expected:
        raise ValueError('Reference is not a manifest-bound disabled-augmentation CONTROL')


def verify_controls(source, config):
    manifest = json.loads((CONTROL_ROOT/'manifest.json').read_text(encoding='utf-8'))
    environment = json.loads((BASE/'mild_time_augmentation/ENVIRONMENT.json').read_text(encoding='utf-8'))
    for key, value in [('python', sys.version), ('numpy', np.__version__), ('torch', torch.__version__), ('cuda', torch.version.cuda)]:
        if environment[key] != value: raise ValueError(f'Control environment differs: {key}; rerun matching controls')
    for i, (path, digest) in enumerate(manifest['source']['code'].items()):
        if T.protocol.digest(path) != digest: raise ValueError('Reference source file changed')
        if T.protocol.digest(CONTROL_ROOT/'source'/f'{i:02d}_{Path(path).name}') != digest:
            raise ValueError('Reference source snapshot changed')
    ledger = []
    for fold in range(1, 9):
        path = CONTROL_ROOT/f'CONTROL/fold{fold}/warmup/result.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        assert_control_match(source, config, manifest, result)
        for name, digest in result['files'].items():
            if T.protocol.digest(path.parent/name) != digest: raise ValueError('Reference model/predictions changed')
        ledger.append({'fold': fold, 'path': str(path), 'result_sha256': T.protocol.digest(path),
                       'initial_state_sha256': result['identity']['initial_state_sha256'],
                       'selected_state_sha256': result['selected_state_sha256'], 'files': result['files']})
    return {'controls_reused': True, 'numpy_version': np.__version__,
            'control_manifest_sha256': T.protocol.digest(CONTROL_ROOT/'manifest.json'),
            'environment_sha256': T.protocol.digest(BASE/'mild_time_augmentation/ENVIRONMENT.json'),
            'controls': ledger}


def style_epoch(model, optimizer, sampler, reader, labels, criterion, config):
    from style import sampler_seed
    seed = sampler_seed(copy.deepcopy(sampler.state_dict()), config['style_fold'])
    reader.configure(seed)
    with reader.epoch_context(model):
        row = original_epoch(model, optimizer, sampler, reader, labels, criterion, config)
    row.update(style_seed=seed, style_trace=reader.trace.hexdigest(), style_counts=dict(reader.counts))
    return row


def main():
    from style import StyleReader
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--mode', choices=['STYLE', 'CONTROL'], default='STYLE')
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(1, 9)))
    parser.add_argument('--out', type=Path, default=HERE/'runs')
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or any(f not in range(1, 9) for f in args.folds):
        raise ValueError('Require unique folds 1..8')
    if args.mode == 'CONTROL' and not args.smoke:
        raise ValueError('CONTROL mode is the identity smoke; formal reference controls are separately verified')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    config = copy.deepcopy(original['config'])
    source = T.source_identity()
    ledger = verify_controls(source, config)
    source['style'] = {'numpy': np.__version__, 'control_manifest_sha256': ledger['control_manifest_sha256']}
    for path in (Path(__file__).resolve(), HERE/'style.py'):
        source['code'][str(path)] = T.protocol.digest(path)
    if not args.smoke:
        precheck = json.loads((HERE/'PRETRAIN_CHECK.json').read_text(encoding='utf-8'))
        if precheck.get('status') != 'verified' or precheck.get('source_code') != source['code']:
            raise ValueError('Verified smoke/recovery check for these exact training sources is required')
        ledger['pretrain_check_sha256'] = T.protocol.digest(HERE/'PRETRAIN_CHECK.json')
    config['style'] = {'mode': args.mode, 'probability': .25, 'share': .1, 'seed': 20260909,
                       'donor_policy': 'same_class_cross_user_uniform_user_then_instance',
                       'statistics': 'float32_detached_population_std_clamp1e-6'}
    if args.smoke:
        config.update(warmup_epochs=1, warmup_min=1, epochs=1, min_epochs=1, smoke_steps=3, smoke=True)
    T.freeze_manifest(args.out, source, config)
    T.protocol.dump(args.out/'CONTROL_REUSE.json', ledger)
    T.protocol.dump(args.out/'ENVIRONMENT.json', {'python': sys.version, 'executable': sys.executable,
                    'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__})
    T.warmup_epoch = style_epoch
    try:
        for fold in args.folds:
            reader, labels, subjects, parts = T.protocol.open_fold(fold)
            matrix, metadata = T.smoothing.prepare_matrix(fold, OLD/'matrices', source['data'])
            control_path = Path(ledger['controls'][fold-1]['path'])
            control = json.loads(control_path.read_text(encoding='utf-8'))
            if control['identity']['partitions'] != {k: T.protocol.array_hash(v) for k, v in parts.items()}:
                raise ValueError('Current fold differs from reference CONTROL')
            if control['identity']['matrix_sha256'] != metadata['matrix_sha256']:
                raise ValueError('Current smoothing matrix differs from reference CONTROL')
            reader = StyleReader(reader, labels, subjects, enabled=args.mode == 'STYLE')
            root = args.out/args.mode
            T.protocol.dump(root/f'fold{fold}/split.json', {'ids': {k: v.tolist() for k, v in parts.items()},
                        'training_subjects': np.unique(subjects[parts['train']]).tolist(), 'val': fold%8+1, 'test': fold})
            T.protocol.dump(args.out/'queue_status.json', {'state': 'running', 'fold': fold,
                            'mode': args.mode, 'folds': args.folds})
            foldconfig = dict(config, style_fold=fold, active_mode=args.mode)
            result = T.run_arm(root, fold, 'warmup', None, reader, labels, subjects, parts,
                               matrix, metadata, source, foldconfig)
            if result['identity']['initial_state_sha256'] != control['identity']['initial_state_sha256']:
                raise ValueError('Initialization differs from reference CONTROL')
    except BaseException as error:
        T.protocol.dump(args.out/'queue_status.json', {'state': 'failed', 'error': str(error)})
        raise
    T.protocol.dump(args.out/'queue_status.json', {'state': 'complete', 'folds': args.folds,
                    'mode': args.mode, 'smoke': args.smoke})
    print('MILD_STYLE_QUEUE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
