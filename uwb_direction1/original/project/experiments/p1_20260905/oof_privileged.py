"""Cross-user out-of-fold privileged soft-target generation and training."""
import argparse
import gc
import json
import time
import warnings
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from pipeline import BASE, RESULTS, split, digest, dump
from train import Classifier, setup, load_gpu, predict, digest_state

OUT = BASE/'oof_privileged_results'
TAGS = {'single': 'sv_R_oofshape_single', 'multi': 'sv_R_oofshape_multi'}


def _pool_one_view(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (90, 189):
        raise ValueError(f'expected (N,90,189), got {values.shape}')
    return values.reshape(len(values), 10, 9, 21, 9).mean((2, 4)).reshape(len(values), 210)


def pooled_features_from_arrays(amplitude, phase, views):
    amplitude = np.asarray(amplitude)
    phase = np.asarray(phase)
    if amplitude.shape != phase.shape or amplitude.ndim != 4:
        raise ValueError('amplitude and phase must have matching (N,V,90,189) shapes')
    parts = []
    for view in views:
        parts.extend((_pool_one_view(amplitude[:, view]), _pool_one_view(phase[:, view])))
    return np.concatenate(parts, axis=1)


def cross_user_oof(features, y, subjects, train_ids, reserved_ids, seed=2026):
    features = np.asarray(features); y = np.asarray(y); subjects = np.asarray(subjects)
    train_ids = np.asarray(train_ids); reserved_ids = np.asarray(reserved_ids)
    if np.intersect1d(train_ids, reserved_ids).size:
        raise ValueError('training and reserved indices overlap')
    nclass = int(y.max()) + 1
    probabilities = np.full((len(y), nclass), np.nan, dtype=np.float64)
    history = []
    for held_subject in np.unique(subjects[train_ids]):
        predict_ids = train_ids[subjects[train_ids] == held_subject]
        fit_ids = train_ids[subjects[train_ids] != held_subject]
        if np.intersect1d(np.r_[fit_ids, predict_ids], reserved_ids).size:
            raise ValueError('reserved index entered an inner fold')
        scaler = StandardScaler().fit(features[fit_ids])
        classifier = LogisticRegression(C=1, solver='lbfgs', max_iter=1000,
                                        tol=1e-4, random_state=seed)
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
            warnings.simplefilter('always')
            classifier.fit(scaler.transform(features[fit_ids]), y[fit_ids])
            if not np.array_equal(classifier.classes_, np.arange(nclass)):
                raise ValueError('an inner fit is missing output classes')
            probabilities[predict_ids] = classifier.predict_proba(
                scaler.transform(features[predict_ids]))
        history.append({'held_subject': int(held_subject), 'fit_ids': fit_ids.tolist(),
                        'predict_ids': predict_ids.tolist(),
                        'iterations': classifier.n_iter_.tolist(),
                        'warnings': [str(item.message) for item in captured]})
    if not np.isfinite(probabilities[train_ids]).all():
        raise ValueError('some training rows lack OOF probabilities')
    if not np.isnan(probabilities[reserved_ids]).all():
        raise ValueError('reserved rows received OOF probabilities')
    np.testing.assert_allclose(probabilities[train_ids].sum(1), 1, atol=1e-7)
    return probabilities, history


def shape_matched_target(labels, probabilities, true_mass):
    probabilities = probabilities.detach().float()
    onehot = F.one_hot(labels, num_classes=probabilities.shape[-1]).float()
    non_target = probabilities*(1-onehot)
    denominator = non_target.sum(-1, keepdim=True)
    uniform = (1-onehot)/(probabilities.shape[-1]-1)
    shape = torch.where(denominator > 1e-8, non_target/denominator.clamp_min(1e-8), uniform)
    mass = torch.as_tensor(true_mass, dtype=probabilities.dtype, device=probabilities.device)
    return onehot*mass + shape*(1-mass)


def shape_matched_loss(student, labels, probabilities, true_mass, temp=2):
    target = shape_matched_target(labels, probabilities, true_mass)
    return F.kl_div(F.log_softmax(student.float()/temp, dim=-1), target,
                    reduction='batchmean') * temp**2


def multiview_features():
    path = OUT/'proxy_LTR.npy'; meta_path = OUT/'proxy_LTR.json'
    if path.exists():
        metadata = json.loads(meta_path.read_text())
        assert metadata['sha'] == digest(path)
        return np.load(path, mmap_mode='r'), metadata
    OUT.mkdir(exist_ok=True)
    amplitude = np.load(BASE/'prepared/off.npy', mmap_mode='r')
    phase = np.load(BASE/'prepared/only.npy', mmap_mode='r')
    result = np.empty((len(amplitude), 1260), dtype=np.float32)
    for begin in range(0, len(result), 128):
        a = np.asarray(amplitude[begin:begin+128], dtype=np.float32)
        p = np.asarray(phase[begin:begin+128], dtype=np.float32)
        result[begin:begin+len(a)] = pooled_features_from_arrays(a, p, [0, 1, 2])
    np.save(path, result)
    metadata = {'sha': digest(path), 'shape': list(result.shape),
                'definition': 'view-major L/T/R; each amplitude then signed phase; 10x21 local means',
                'source_amplitude_sha': digest(BASE/'prepared/off.npy'),
                'source_phase_sha': digest(BASE/'prepared/only.npy')}
    dump(meta_path, metadata)
    return np.load(path, mmap_mode='r'), metadata


def prepare_proxy(fold, features, feature_meta, force=False):
    OUT.mkdir(exist_ok=True)
    path = OUT/f'proxy_h{fold}.npz'; meta_path = path.with_suffix('.json')
    y = np.load(BASE/'prepared/y.npy'); subjects = np.load(BASE/'prepared/subj.npy')
    tr, va, te = split(y, subjects, fold, 2026); reserved = np.r_[va, te]
    if path.exists() and not force:
        metadata = json.loads(meta_path.read_text())
        assert metadata['code_sha'] == digest(__file__)
        assert metadata['feature_sha'] == feature_meta['sha']
        assert metadata['archive_sha'] == digest(path)
        with np.load(path) as archive:
            for key, ids in [('train_ids', tr), ('val_ids', va), ('test_ids', te)]:
                np.testing.assert_array_equal(archive[key], ids)
        return metadata
    multi_full, history = cross_user_oof(features, y, subjects, tr, reserved, seed=2026)
    source = BASE/f'confusion_kd_results/matrix_h{fold}.npz'
    source_meta = json.loads(source.with_suffix('.json').read_text())
    assert source_meta['matrix_sha'] == digest(source)
    with np.load(source) as archive:
        np.testing.assert_array_equal(archive['train_ids'], tr)
        np.testing.assert_array_equal(archive['val_ids'], va)
        np.testing.assert_array_equal(archive['test_ids'], te)
        single = archive['oof_train_probabilities'].copy()
    multi = multi_full[tr]
    warnings_found = sum(len(row['warnings']) for row in history)
    if warnings_found:
        raise RuntimeError(f'multiview proxy emitted {warnings_found} warnings')
    np.savez_compressed(path, single=single, multi=multi, train_ids=tr,
                        val_ids=va, test_ids=te, y_train=y[tr])
    metadata = {'fold': fold, 'code_sha': digest(__file__), 'feature_sha': feature_meta['sha'],
                'archive_sha': digest(path), 'single_source_sha': digest(source),
                'single_oof_accuracy': float((single.argmax(1) == y[tr]).mean()),
                'multi_oof_accuracy': float((multi.argmax(1) == y[tr]).mean()),
                'warnings': warnings_found, 'inner_folds': history,
                'validation_or_test_used': False}
    dump(meta_path, metadata)
    print('OOF_PROXY_READY', fold, metadata['single_oof_accuracy'],
          metadata['multi_oof_accuracy'], flush=True)
    return metadata


def load_fold_probabilities(fold, arm, y, tr, va, te, device='cuda'):
    path = OUT/f'proxy_h{fold}.npz'; metadata = json.loads(path.with_suffix('.json').read_text())
    assert metadata['archive_sha'] == digest(path)
    assert metadata['warnings'] == 0 and not metadata['validation_or_test_used']
    with np.load(path) as archive:
        for key, ids in [('train_ids', tr), ('val_ids', va), ('test_ids', te)]:
            np.testing.assert_array_equal(archive[key], ids)
        np.testing.assert_array_equal(archive['y_train'], y[tr])
        values = archive[arm].copy()
    np.testing.assert_allclose(values.sum(1), 1, atol=1e-7)
    full = torch.full((len(y), 12), float('nan'), dtype=torch.float32, device=device)
    full[torch.as_tensor(tr, device=device)] = torch.as_tensor(values, dtype=torch.float32,
                                                               device=device)
    global_path = BASE/f'label_smoothing_results/global_matrix_h{fold}.npz'
    with np.load(global_path) as archive:
        global_matrix = archive['global_matrix']
        np.testing.assert_array_equal(archive['train_ids'], tr)
        np.testing.assert_array_equal(archive['val_ids'], va)
        np.testing.assert_array_equal(archive['test_ids'], te)
    diagonal = np.diag(global_matrix)
    np.testing.assert_allclose(diagonal, np.repeat(diagonal[0], len(diagonal)), atol=1e-7)
    true_mass = float(.5*(1+diagonal[0]))
    return full, path, metadata, global_path, true_mass


def run_student(fold, arm, data):
    tag = TAGS[arm]; stem = RESULTS/f'{tag}_s2026_h{fold}'
    if stem.with_suffix('.json').exists():
        print('SKIP', tag, fold, flush=True); return
    y = np.load(BASE/'prepared/y.npy'); subjects = np.load(BASE/'prepared/subj.npy')
    tr, va, te = split(y, subjects, fold, 2026)
    probabilities, proxy_path, proxy_meta, global_path, true_mass = load_fold_probabilities(
        fold, arm, y, tr, va, te)
    setup(2026); model = Classifier([2], 'concat', 'mean').cuda(); init = digest_state(model.head)
    config = {'tag': tag, 'seed': 2026, 'folds': [fold], 'mode': 'concat', 'views': [2],
              'readout': 'mean', 'aux': 'none', 'teacher': '', 'epochs': 40, 'batch': 128,
              'microbatch': 32, 'kd_weight': .5, 'soft_target': f'cross_user_oof_shape_{arm}',
              'temperature': 2, 'true_class_mass': true_mass,
              'global_projection_matrix_sha': digest(global_path), 'proxy_sha': digest(proxy_path),
              'proxy_feature_sha': proxy_meta['feature_sha'] if arm == 'multi'
                                   else proxy_meta['single_source_sha'],
              'wrapper_sha': digest(__file__)}
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    yy = torch.as_tensor(y, device='cuda'); tr_gpu = torch.tensor(tr, device='cuda')
    order = torch.Generator(device='cpu').manual_seed(2026+fold*1000)
    torch.manual_seed(2026+100000); torch.cuda.manual_seed_all(2026+100000)
    best = float('inf'); best_ep = 0; history = []; start = time.time()
    for epoch in range(1, 41):
        model.train(); total = 0.; correct = 0; count = 0; tic = time.time()
        ids = tr_gpu[torch.randperm(len(tr), generator=order).to('cuda')]
        for begin in range(0, len(ids), 128):
            batch = ids[begin:begin+128]; opt.zero_grad(set_to_none=True)
            for mb in batch.split(32):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z, _ = model(data[mb])
                    loss = F.cross_entropy(z, yy[mb]) + .5*shape_matched_loss(
                        z, yy[mb], probabilities[mb], true_mass)
                (loss*len(mb)/len(batch)).backward()
                total += float(loss.detach())*len(mb)
                correct += int((z.argmax(-1) == yy[mb]).sum()); count += len(mb)
            opt.step()
        vl = predict(model, data, va, 32)
        ce = float(F.cross_entropy(torch.from_numpy(vl), torch.from_numpy(y[va])))
        row = {'epoch': epoch, 'train_loss': total/count, 'train_acc': correct/count,
               'val_ce': ce, 'val_acc': float((vl.argmax(-1) == y[va]).mean()),
               'seconds': time.time()-tic}
        history.append(row)
        if ce < best:
            best = ce; best_ep = epoch
            torch.save({'state': model.state_dict(), 'args': config, 'fold': fold,
                        'epoch': epoch}, stem.with_suffix('.pt'))
        print(stem.name, json.dumps(row), flush=True)
        if epoch >= 12 and epoch-best_ep >= 8:
            break
    checkpoint = torch.load(stem.with_suffix('.pt'), map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['state']); del checkpoint
    logits = predict(model, data, te, 32); pred = logits.argmax(-1)
    val_logits = predict(model, data, va, 32)
    np.savez_compressed(str(stem)+'_predictions.npz', ids=te, y=y[te], pred=pred,
                        logits=logits, train_ids=tr, val_ids=va, val_y=y[va],
                        val_logits=val_logits)
    result = {'tag': tag, 'seed': 2026, 'held_subject': fold, 'mode': 'concat',
              'views': [2], 'readout': 'mean', 'aux': 'none', 'teacher': None,
              'config': config, 'parameters': sum(p.numel() for p in model.parameters()),
              'initial_head_hash': init, 'n_train': len(tr), 'n_val': len(va),
              'n_test': len(te), 'best_epoch': best_ep, 'best_val_ce': best,
              'accuracy': float((pred == y[te]).mean()),
              'macro_f1': float(f1_score(y[te], pred, labels=np.unique(y[te]),
                                         average='macro', zero_division=0)),
              'evaluated_classes': np.unique(y[te]).tolist(),
              'confusion': confusion_matrix(y[te], pred, labels=np.arange(12)).tolist(),
              'history': history, 'elapsed_seconds': time.time()-start,
              'train_py_sha256': digest(__file__),
              'original_loop_source_sha': digest(BASE/'train.py'),
              'pipeline_sha256': digest(BASE/'pipeline.py')}
    dump(stem.with_suffix('.json'), result)
    print('COMPLETE', stem.name, result['accuracy'], flush=True)
    del model, opt, probabilities; gc.collect(); torch.cuda.empty_cache()


def smoke():
    setup(2026); model = Classifier([2], 'concat').cuda()
    z, _ = model(torch.randn(4, 1, 90, 378, device='cuda'))
    labels = torch.tensor([0, 3, 5, 11], device='cuda')
    probabilities = F.one_hot(labels, num_classes=12).float()*.7 + .3/12
    loss = F.cross_entropy(z, labels) + .5*shape_matched_loss(
        z, labels, probabilities, .82)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.temporal.proj.weight.grad.abs().sum() > 0
    print('OOF_PRIVILEGED_SMOKE_OK', float(loss.detach()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--force-prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    parser.add_argument('--arms', choices=['single', 'multi'], nargs='+', default=['single', 'multi'])
    args = parser.parse_args(); OUT.mkdir(exist_ok=True)
    if args.smoke:
        smoke(); return
    if args.prepare:
        features, metadata = multiview_features()
        for fold in args.folds:
            prepare_proxy(fold, features, metadata, force=args.force_prepare)
        return
    gpu_data = load_gpu('concat', [2])
    for fold in args.folds:
        for arm in args.arms:
            run_student(fold, arm, gpu_data)


if __name__ == '__main__':
    main()
