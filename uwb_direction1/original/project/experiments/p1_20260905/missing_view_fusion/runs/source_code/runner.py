"""Sequential controlled frozen-feature screens; validation selects Right only."""
import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score

from data import BASE, HERE, digest, dump, extract, fit_scale, prototype_banks, setup
from models import MASKS, MissingFusion, SharedSpecific, PrototypeCompletion, sigmoid_contrast

STAGES = {1: ('R_ONLY', 'RANDOM_ZERO', 'JOINT_ZERO', 'JOINT_TOKEN', 'JOINT_CONTRAST'),
          2: ('SS_R_ONLY', 'SS_JOINT', 'SS_ALIGN'),
          3: ('PROTO_R_ONLY', 'PROTO_MULTI', 'PROTO_MLP')}
DEFAULTS = dict(batch=256, epochs=40, min_epochs=12, patience=8, lr=.001,
                weight_decay=.0001, contrast_weight=.05, align_weight=.05,
                view_weight=.05, temperature=.2, split_seed=2026)


def subset_schedule(arm, generator):
    if arm in ('R_ONLY', 'SS_R_ONLY'):
        return (MASKS[0],)*4
    if arm == 'RANDOM_ZERO':
        return tuple(MASKS[k] for k in torch.randint(4, (4,), generator=generator).tolist())
    return MASKS


def derangement(labels, subjects, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    permutation = indices.copy()
    for user, label in sorted(set(zip(subjects.tolist(), labels.tolist()))):
        ids = indices[(subjects == user) & (labels == label)]
        if len(ids) < 2:
            raise ValueError('global pairing control requires at least two rows per cell')
        rng.shuffle(ids)
        permutation[ids] = np.roll(ids, 1)
    assert np.all(permutation != indices)
    return permutation


def classification_loss(logits, labels, matrix):
    onehot = F.one_hot(labels, 12).float()
    target = .5*(onehot+onehot@matrix.detach())
    return F.cross_entropy(logits, labels)+2*F.kl_div(
        F.log_softmax(logits/2, -1), target, reduction='batchmean')


def select_best(history):
    row = min(history, key=lambda item: item['val_ce'])
    return row['epoch'], row['val_ce']


def verify_completed(directory, status):
    for name, key in [('predictions.npz', 'predictions_sha256'), ('deployment.pt', 'deployment_sha256')]:
        path = Path(directory)/name
        if not path.exists() or digest(path) != status[key]:
            raise ValueError(f'completed artifact missing or corrupt: {path}')


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_model(arm, training_features, labels, seed):
    setup(seed)
    dim = training_features.shape[-1]
    if arm.startswith('SS_'):
        model = SharedSpecific(dim)
    elif arm.startswith('PROTO_'):
        banks = prototype_banks(training_features, labels).cpu()
        if arm != 'PROTO_MULTI':
            banks = banks[2:3].repeat(3, 1, 1)
        model = PrototypeCompletion(dim, banks, trainable_bank=arm == 'PROTO_MLP')
    else:
        model = MissingFusion(dim, learned_tokens=arm in ('JOINT_TOKEN', 'JOINT_CONTRAST'))
    return model.to(training_features.device)


@torch.no_grad()
def predict(model, right, base, batch=256):
    if right.ndim != 2:
        raise ValueError('validation/test accepts only Right rank-2 features')
    model.eval()
    result = []
    for begin in range(0, len(right), batch):
        j = slice(begin, begin+batch)
        result.append((base[j]+model.deploy(right[j])).cpu())
    return torch.cat(result).numpy()


def train_arm(arm, fold, seed, arrays, meta, out, overrides=None, shuffle=False):
    settings = DEFAULTS | (overrides or {})
    name = f'{arm}{"_SHUF" if shuffle else ""}_s{seed}_h{fold}'
    dest = Path(out)/name
    dest.mkdir(parents=True, exist_ok=True)
    sources = {p.name: digest(p) for p in (HERE/'data.py', HERE/'models.py', HERE/'runner.py')}
    matrix_path = BASE/f'label_smoothing_results/global_matrix_h{fold}.npz'
    config = settings | {'arm': arm, 'fold': fold, 'seed': seed, 'shuffle': shuffle,
                         'sampler_seed': 7777+fold, 'mask_seed': 31337+fold,
                         'cache_sha256': meta['cache_sha256'], 'code_sha256': sources,
                         'matrix_sha256': digest(matrix_path), 'validation': 'Right hard CE including epoch 0'}
    status_path = dest/'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding='utf-8'))
        old_config = json.loads((dest/'config.json').read_text(encoding='utf-8'))
        if old_config != config:
            raise ValueError(f'run identity changed; use separate output directory: {dest}')
        if status['status'] == 'complete':
            verify_completed(dest, status)
            print('SKIP', name, flush=True)
            return status
    dump(dest/'config.json', config)
    dump(status_path, {'status': 'running', 'name': name, 'started': time.time()})
    started = time.time()
    device = 'cuda'
    x = torch.tensor(arrays['train_x'], device=device, dtype=torch.float32)
    mu, sd = fit_scale(x)
    x = (x-mu)/sd
    yr = torch.tensor(arrays['train_y'], device=device)
    base = torch.tensor(arrays['train_base'], device=device)
    val_x = (torch.tensor(arrays['val_x'], device=device).float()-mu[:, 2])/sd[:, 2]
    val_base = torch.tensor(arrays['val_base'], device=device)
    test_x = (torch.tensor(arrays['test_x'], device=device).float()-mu[:, 2])/sd[:, 2]
    test_base = torch.tensor(arrays['test_base'], device=device)
    with np.load(matrix_path) as archive:
        for k in ('train_ids', 'val_ids', 'test_ids'):
            np.testing.assert_array_equal(archive[k], arrays[k])
        matrix = torch.tensor(archive['global_matrix'], device=device)
    if shuffle:
        perm = derangement(arrays['train_y'], arrays['train_subjects'], 31337+fold)
        x[:, :2] = x[perm, :2].clone()
        np.save(dest/'shuffle.npy', perm)
    model = build_model(arm, x, yr, seed)
    initial = state_hash(model)
    opt = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    sampler = torch.Generator().manual_seed(7777+fold)
    mask_rng = torch.Generator().manual_seed(31337+fold)
    val_y = torch.tensor(arrays['val_y'])
    initial_logits = predict(model, val_x, val_base)
    np.testing.assert_array_equal(initial_logits, arrays['val_base'])
    history = [{'epoch': 0, 'val_ce': float(F.cross_entropy(torch.tensor(initial_logits), val_y)),
                'val_accuracy': float((initial_logits.argmax(1) == arrays['val_y']).mean())}]
    best_epoch, best = select_best(history)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sampling_hashes = []
    with (dest/'metrics.jsonl').open('w', encoding='utf-8') as metrics:
        metrics.write(json.dumps(history[0])+'\n')
        metrics.flush()
        for epoch in range(1, settings['epochs']+1):
            tick = time.time()
            model.train()
            ids = torch.randperm(len(x), generator=sampler).numpy()
            sampling_hashes.append(hashlib.sha256(ids.tobytes()).hexdigest())
            total = 0.
            for begin in range(0, len(ids), settings['batch']):
                j = ids[begin:begin+settings['batch']]
                opt.zero_grad(set_to_none=True)
                if arm.startswith('PROTO_'):
                    correction, _ = model(x[j, 2])
                    loss = classification_loss(base[j]+correction, yr[j], matrix)
                else:
                    codes = []
                    loss = x.new_zeros(())
                    for mask in subset_schedule(arm, mask_rng):
                        correction, code = model(x[j], mask)
                        loss = loss+classification_loss(base[j]+correction, yr[j], matrix)/4
                        codes.append(code)
                    if arm == 'JOINT_CONTRAST':
                        loss = loss+settings['contrast_weight']*sigmoid_contrast(
                            torch.cat(codes), yr[j].repeat(4), settings['temperature'])
                    if arm == 'SS_ALIGN':
                        auxiliary = model.auxiliary(x[j])
                        loss = loss+settings['align_weight']*auxiliary['align']+settings['view_weight']*auxiliary['view']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite loss in {name} epoch {epoch}')
                loss.backward()
                opt.step()
                total += float(loss.detach())*len(j)
            vl = predict(model, val_x, val_base)
            ce = float(F.cross_entropy(torch.tensor(vl), val_y))
            row = {'epoch': epoch, 'train_loss': total/len(x), 'val_ce': ce,
                   'val_accuracy': float((vl.argmax(1) == arrays['val_y']).mean()),
                   'seconds': time.time()-tick}
            history.append(row)
            metrics.write(json.dumps(row)+'\n')
            metrics.flush()
            if ce < best:
                best, best_epoch = ce, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0:
                print(f'TRAIN {name} epoch={epoch} best={best_epoch} val_ce={ce:.4f}', flush=True)
            if epoch >= settings['min_epochs'] and epoch-best_epoch >= settings['patience']:
                break
    assert select_best(history) == (best_epoch, best)
    model.load_state_dict(best_state)
    test_logits = predict(model, test_x, test_base)
    val_logits = predict(model, val_x, val_base)
    test_pred = test_logits.argmax(1)
    # Export includes sufficient lineage/scaling for the raw-Right wrapper.
    torch.save({'state': best_state, 'right_mu': mu[:, 2].cpu(), 'right_sd': sd[:, 2].cpu(),
                'config': config, 'feature_dim': x.shape[-1], 'encoder_identity': meta['identity']}, dest/'deployment.pt')
    np.savez_compressed(dest/'predictions.npz', ids=arrays['test_ids'], y=arrays['test_y'],
                        logits=test_logits, pred=test_pred, base_logits=arrays['test_base'],
                        train_ids=arrays['train_ids'], val_ids=arrays['val_ids'],
                        val_y=arrays['val_y'], val_logits=val_logits, val_base_logits=arrays['val_base'])
    dump(dest/'history.json', history)
    result = {'status': 'complete', 'name': name, 'arm': arm, 'fold': fold, 'seed': seed,
              'shuffle': shuffle, 'accuracy': float((test_pred == arrays['test_y']).mean()),
              'base_accuracy': meta['base_accuracy'],
              'macro_f1': float(f1_score(arrays['test_y'], test_pred, labels=np.unique(arrays['test_y']),
                                         average='macro', zero_division=0)),
              'best_epoch': best_epoch, 'best_val_ce': best, 'epochs': len(history)-1,
              'initial_state_sha256': initial, 'sampling_hashes': sampling_hashes,
              'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'seconds': time.time()-started, 'n_test': len(test_pred),
              'predictions_sha256': digest(dest/'predictions.npz'),
              'deployment_sha256': digest(dest/'deployment.pt')}
    dump(status_path, result)
    print(f'COMPLETE {name} accuracy={100*result["accuracy"]:.3f} base={100*meta["base_accuracy"]:.3f} '
          f'best_epoch={best_epoch} seconds={result["seconds"]:.1f}', flush=True)
    del model, opt, x, val_x, test_x, base, val_base, test_base
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    ap.add_argument('--seeds', type=int, nargs='+', default=[2026])
    ap.add_argument('--arms', nargs='+')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--out', type=Path, default=HERE/'runs')
    ap.add_argument('--shuffle', action='store_true')
    args = ap.parse_args()
    setup(2026)
    args.out.mkdir(parents=True, exist_ok=True)
    if __import__('os').name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    stages = args.stages if not args.arms else [0]
    try:
        for stage in stages:
            arms = args.arms if stage == 0 else STAGES[stage]
            dump(args.out/'queue_status.json', {'status': 'running', 'stage': stage, 'arms': arms})
            for fold in args.folds:
                arrays, meta = extract(fold)
                for seed in args.seeds:
                    for arm in arms:
                        if arm not in sum(STAGES.values(), ()):
                            raise ValueError(arm)
                        train_arm(arm, fold, seed, arrays, meta, args.out,
                                  {'epochs': args.epochs}, args.shuffle)
                del arrays
            print(f'STAGE_COMPLETE {stage}', flush=True)
        dump(args.out/'queue_status.json', {'status': 'complete', 'stages': stages,
                                          'folds': args.folds, 'seeds': args.seeds})
    except Exception as exc:
        dump(args.out/'queue_status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
