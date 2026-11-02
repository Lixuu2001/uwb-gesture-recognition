"""Single trainer for every arm: base, method A relation, method B process.

One optimisation loop, one early-stopping rule, one sampler registry. Arms
differ only in which auxiliary target they read, so a difference in the result
table cannot come from a quietly different training budget.
"""
import argparse
import gc
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch.nn import functional as F

import protocol as P
import relation as REL
import motion_targets as MT
import sampling
from model_adapter import build_student, classification_features, student_aux_features
from objectives import classification_loss
from process import process_loss
from protocol import RUNS, dump, digest, hash_ids

# method_tag -> what that arm is allowed to read and optimise.
ARMS = {
    'A-GS':    dict(sampler='relation', contrast=False, relation=None),
    'A-SC':    dict(sampler='relation', contrast=True, relation=None),
    'A-TS':    dict(sampler='relation', contrast=True, relation='target_single'),
    'A-US':    dict(sampler='relation', contrast=True, relation='teacher_single'),
    'A-MV':    dict(sampler='relation', contrast=True, relation='multi'),
    'A-SHUF':  dict(sampler='relation', contrast=True, relation='multi', shuffle_targets=True),
    # The two runs already on disk were produced before shuffle_scope existed and
    # therefore used the microbatch-local variant; these tags reproduce them.
    'A-SHUF-LOCAL': dict(sampler='relation', contrast=True, relation='multi',
                         shuffle_targets=True, shuffle_scope='microbatch'),
    'B-SHUF-LOCAL': dict(sampler='permutation', process=dict(mode='multi'),
                         shuffle_targets=True, shuffle_scope='microbatch'),
    'A-RKD':   dict(sampler='relation', contrast=False, relation='multi', relation_form='rkd'),
    'B-GS':    dict(sampler='permutation'),
    'B-SELF':  dict(sampler='permutation', process=dict(mode='self')),
    'B-MV':    dict(sampler='permutation', process=dict(mode='multi')),
    'B-Q':     dict(sampler='permutation', process=dict(mode='multi', quality_mode='quality')),
    'B-AMP':   dict(sampler='permutation', process=dict(mode='multi', kind='amp')),
    'B-MAX':   dict(sampler='permutation', process=dict(mode='multi_max')),
    'B-SHUF':  dict(sampler='permutation', process=dict(mode='multi'), shuffle_targets=True),
    'B-TIME':  dict(sampler='permutation', process=dict(mode='multi'), shuffle_time=True),
    'B-CDF':   dict(sampler='permutation', process=dict(mode='multi'), lambda_cdf=0.0),
    'AB-MV':   dict(sampler='relation', contrast=True, relation='multi', process=dict(mode='multi')),
    'AB-USMV': dict(sampler='relation', contrast=True, relation='teacher_single', process=dict(mode='multi')),
    'AB-MVSELF': dict(sampler='relation', contrast=True, relation='multi', process=dict(mode='self')),
}

DEFAULTS = dict(lambda_contrast=0.1, lambda_relation=0.1, lambda_activity=0.1, lambda_cdf=0.05,
                tau_student=0.2, tau_teacher=0.2, ramp_epochs=5, max_epochs=40, min_stop_epoch=12,
                patience=8, batch=128, microbatch=32, lr=1e-4, weight_decay=1e-4,
                relation_dim=128, sampler='permutation', contrast=False, relation=None,
                relation_form='kl', process=None, shuffle_targets=False, shuffle_time=False,
                # 'global' deranges every (class,user) cell over the whole training
                # pool. 'microbatch' is the superseded local variant that only
                # perturbed 31.6% of exposures; kept solely to reproduce the runs
                # already on disk, never for a new control.
                shuffle_scope='global')


def spec_for(arm):
    if arm not in ARMS:
        raise KeyError(f'unknown method_tag {arm}')
    spec = dict(DEFAULTS)
    spec.update(ARMS[arm])
    spec['method_tag'] = arm
    return spec


def run_name(arm, model_seed, fold):
    return f'{arm}_split{P.SPLIT_SEED}_model{model_seed}_h{fold}'


@torch.no_grad()
def predict(model, data, ids, microbatch=32):
    model.eval()
    out = []
    for i in range(0, len(ids), microbatch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _, _, _ = classification_features(model, data[ids[i:i + microbatch]])
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def load_student_data():
    """Right amplitude and phase-motion, concatenated on the feature axis."""
    from train import load_gpu
    return load_gpu('concat', [P.DEPLOY_VIEW])


def run(arm, fold, model_seed, data, force=False):
    spec = spec_for(arm)
    name = run_name(arm, model_seed, fold)
    dest = RUNS / name
    status_path = dest / 'status.json'
    if status_path.exists() and not force:
        state = json.loads(status_path.read_text(encoding='utf-8'))
        if state.get('status') == 'complete':
            print('SKIP', name, flush=True)
            return state
    dest.mkdir(parents=True, exist_ok=True)
    dump(status_path, {'status': 'running', 'started': time.time(), 'name': name})
    try:
        state = _train(spec, fold, model_seed, data, dest, name)
    except Exception as error:                       # a failed arm must stay visible
        dump(status_path, {'status': 'failed', 'name': name, 'error': repr(error)})
        raise
    return state


def _train(spec, fold, model_seed, data, dest, name):
    y, subj = P.labels()
    tr, va, te = P.fold_split(fold, y, subj)
    matrix_np, mass, matrix_path = P.global_matrix(fold)
    matrix = torch.tensor(matrix_np, device='cuda')
    yy = torch.as_tensor(y.astype('int64'), device='cuda')
    ss = torch.as_tensor(subj.astype('int64'), device='cuda')

    wants_relation = spec['relation'] is not None
    wants_contrast = bool(spec['contrast'])
    wants_process = spec['process'] is not None
    model, heads, fingerprint = build_student(model_seed, relation=wants_relation or wants_contrast,
                                              process=wants_process, relation_dim=spec['relation_dim'])

    ledger = {'teacher_training_views': [], 'teacher_target_views': [],
              'teacher_checkpoint_sha256': None, 'teacher_training_ids_hash': None,
              'target_transform_fit_ids_hash': None}
    targets = None
    if wants_relation:
        import teacher_targets
        targets, ledger = teacher_targets.relation_targets(fold, spec['relation'], tr)
    process_meta = None
    activity = progress = seg_ok = sample_ok = None
    if wants_process:
        options = dict(spec['process'])
        activity, progress, seg_ok, sample_ok, process_meta = MT.build_fold_targets(
            fold, tr, **options)
        if spec['shuffle_time']:
            generator = torch.Generator(device='cuda').manual_seed(P.AUXILIARY_SEED + fold)
            activity, progress = MT.shuffle_time(activity, generator)
            process_meta['time_permuted'] = True
        if not ledger['target_transform_fit_ids_hash']:
            ledger['target_transform_fit_ids_hash'] = process_meta['target_transform_fit_ids_hash']
            ledger['teacher_target_views'] = process_meta['views']

    sampler_seed = P.SAMPLER_SEED + fold
    if spec['sampler'] == 'relation':
        groups = sampling.relation_epochs(y, subj, tr, spec['max_epochs'], sampler_seed)
    else:
        groups = sampling.permutation_epochs(tr, spec['max_epochs'], spec['batch'], sampler_seed)
        groups = groups.reshape(spec['max_epochs'], groups.shape[1], -1, spec['microbatch'])
    manifest_hash = sampling.manifest_hash(groups)

    parameters = list(model.parameters()) + list(heads.parameters())
    optimiser = torch.optim.Adam(parameters, lr=spec['lr'], weight_decay=spec['weight_decay'])
    torch.manual_seed(model_seed + 100000)
    torch.cuda.manual_seed_all(model_seed + 100000)
    shuffle_generator = torch.Generator(device='cuda').manual_seed(P.AUXILIARY_SEED + 7 * fold)
    shuffle_lookup, shuffle_changed = None, None
    if spec['shuffle_targets'] and spec['shuffle_scope'] == 'global':
        lookup, shuffle_changed = REL.global_within_class_derangement(
            y, subj, tr, P.AUXILIARY_SEED + 7 * fold)
        shuffle_lookup = torch.as_tensor(lookup, device='cuda')

    def mismatch(ids, labels, subjects):
        """Instance-mismatched target ids for the SHUF controls."""
        if shuffle_lookup is not None:
            return shuffle_lookup[ids]
        return REL.shuffle_within_class(ids, labels, subjects, shuffle_generator)

    best, best_epoch, history, start = float('inf'), 0, [], time.time()
    gradient_audit = None
    for epoch in range(1, spec['max_epochs'] + 1):
        ramp = min(epoch / spec['ramp_epochs'], 1.0)
        model.train(); heads.train()
        totals = {'loss': 0., 'base': 0., 'contrast': 0., 'relation': 0.,
                  'activity': 0., 'cdf': 0.}
        correct = count = 0
        tic = time.time()
        for step in groups[epoch - 1]:
            optimiser.zero_grad(set_to_none=True)
            valid_groups = [g[g >= 0] for g in step if bool((g >= 0).any())]
            total_in_step = sum(len(g) for g in valid_groups)
            for group in valid_groups:
                ids = torch.as_tensor(group, device='cuda')
                labels, subjects = yy[ids], ss[ids]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, _, temporal, spatial = classification_features(model, data[ids])
                base = classification_loss(logits, labels, matrix)
                loss = base
                aux_total = None
                parts = {'base': float(base.detach())}
                if wants_relation or wants_contrast:
                    aux_in = student_aux_features(temporal, spatial).float()
                    projected = heads['relation'](aux_in)
                    target_ids = ids
                    if spec['shuffle_targets'] and wants_relation:
                        target_ids = mismatch(ids, labels, subjects)
                    if spec['relation_form'] == 'rkd' and wants_relation:
                        rel = REL.rkd_distance_loss(projected, targets[target_ids], labels, subjects)
                        con = projected.sum() * 0
                    else:
                        rel, con = REL.cross_user_relation_losses(
                            projected, targets[target_ids] if wants_relation else projected,
                            labels, subjects, spec['tau_student'], spec['tau_teacher'],
                            want_relation=wants_relation, want_contrast=wants_contrast)
                    contribution = ramp * (spec['lambda_contrast'] * con
                                           + spec['lambda_relation'] * rel)
                    aux_total = contribution if aux_total is None else aux_total + contribution
                    loss = loss + contribution
                    parts['relation'] = float(rel.detach()); parts['contrast'] = float(con.detach())
                if wants_process:
                    target_ids = ids
                    if spec['shuffle_targets']:
                        target_ids = mismatch(ids, labels, subjects)
                    predicted = heads['process'](temporal.float())
                    kl, cdf = process_loss(predicted, activity[target_ids], progress[target_ids],
                                           seg_ok[target_ids], sample_ok[target_ids])
                    contribution = ramp * (spec['lambda_activity'] * kl + spec['lambda_cdf'] * cdf)
                    aux_total = contribution if aux_total is None else aux_total + contribution
                    loss = loss + contribution
                    parts['activity'] = float(kl.detach()); parts['cdf'] = float(cdf.detach())
                if gradient_audit is None:
                    gradient_audit = _audit_gradients(model, heads, targets, aux_total)
                (loss * len(group) / total_in_step).backward()
                totals['loss'] += float(loss.detach()) * len(group)
                for key, value in parts.items():
                    totals[key] += value * len(group)
                correct += int((logits.argmax(-1) == labels).sum()); count += len(group)
            optimiser.step()
        val_logits = predict(model, data, va, spec['microbatch'])
        val_ce = float(F.cross_entropy(torch.from_numpy(val_logits),
                                       torch.from_numpy(y[va].astype('int64'))))
        row = {'epoch': epoch, 'ramp': ramp, 'train_acc': correct / count,
               'val_ce': val_ce, 'val_acc': float((val_logits.argmax(-1) == y[va]).mean()),
               'seconds': time.time() - tic}
        row.update({k: v / count for k, v in totals.items()})
        history.append(row)
        if val_ce < best:
            best, best_epoch = val_ce, epoch
            torch.save({'model': model.state_dict(), 'heads': heads.state_dict(),
                        'epoch': epoch, 'fold': fold, 'method_tag': spec['method_tag'],
                        'model_seed': model_seed}, dest / 'best.pt')
        print(name, json.dumps(row), flush=True)
        if epoch >= spec['min_stop_epoch'] and epoch - best_epoch >= spec['patience']:
            break

    checkpoint = torch.load(dest / 'best.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model']); heads.load_state_dict(checkpoint['heads'])
    del checkpoint
    logits = predict(model, data, te, spec['microbatch'])
    prediction = logits.argmax(-1)
    val_logits = predict(model, data, va, spec['microbatch'])
    detach_check = _auxiliary_free_logits(model, data, te[:64], logits[:64], spec['microbatch'])

    np.savez_compressed(dest / 'predictions.npz', ids=te, y=y[te], pred=prediction, logits=logits)
    np.savez_compressed(dest / 'validation_predictions.npz', ids=va, y=y[va], logits=val_logits)
    np.savez_compressed(dest / 'split.npz', train_ids=tr, val_ids=va, test_ids=te)
    np.savez_compressed(dest / 'sampling.npz', groups=groups)
    torch.save({'model': model.state_dict(), 'method_tag': spec['method_tag'], 'fold': fold},
               dest / 'deployment_state.pt')
    (dest / 'metrics.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in history) + '\n', encoding='utf-8')

    config = dict(spec)
    config.update({'fold': fold, 'model_seed': model_seed, 'split_seed': P.SPLIT_SEED,
                   'sampler_seed': sampler_seed, 'auxiliary_seed': P.AUXILIARY_SEED,
                   'process_targets': process_meta, 'true_mass': mass,
                   'deployment_view': P.VIEW_NAMES[P.DEPLOY_VIEW]})
    dump(dest / 'config.json', config)
    dump(dest / 'provenance.json', P.provenance(
        student_input_views=[P.VIEW_NAMES[P.DEPLOY_VIEW]],
        teacher_training_views=ledger['teacher_training_views'],
        teacher_target_views=ledger['teacher_target_views'],
        teacher_training_ids_hash=ledger['teacher_training_ids_hash'],
        student_train_ids_hash=hash_ids(tr), validation_ids_hash=hash_ids(va),
        test_ids_hash=hash_ids(te),
        teacher_checkpoint_sha256=ledger['teacher_checkpoint_sha256'],
        target_transform_fit_ids_hash=ledger['target_transform_fit_ids_hash'],
        label_statistics_source=matrix_path, split_seed=P.SPLIT_SEED, model_seed=model_seed,
        sampler_seed=sampler_seed, auxiliary_seed=P.AUXILIARY_SEED,
        initial_classifier_hash=fingerprint, sampler_manifest_hash=manifest_hash,
        code_snapshot_sha256=P.code_snapshot(),
        aux_settings={k: spec[k] for k in ('lambda_contrast', 'lambda_relation', 'lambda_activity',
                                           'lambda_cdf', 'tau_student', 'tau_teacher',
                                           'ramp_epochs', 'relation_dim', 'relation_form',
                                           'shuffle_targets', 'shuffle_time', 'shuffle_scope')}
        | {'shuffle_changed_fraction': shuffle_changed},
        gradient_audit=gradient_audit, auxiliary_free_logits_max_abs_diff=detach_check,
        relation_source=spec['relation'], process_source=process_meta))

    result = {'status': 'complete', 'name': dest.name, 'method_tag': spec['method_tag'],
              'held_subject': fold, 'model_seed': model_seed, 'best_epoch': best_epoch,
              'stopped_epoch': history[-1]['epoch'], 'best_val_ce': best,
              'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
              'accuracy': float((prediction == y[te]).mean()),
              'macro_f1': float(f1_score(y[te], prediction, labels=np.unique(y[te]),
                                         average='macro', zero_division=0)),
              'evaluated_classes': np.unique(y[te]).tolist(),
              'confusion': confusion_matrix(y[te], prediction, labels=np.arange(12)).tolist(),
              'parameters_deployed': int(sum(p.numel() for p in model.parameters())),
              'parameters_auxiliary': int(sum(p.numel() for p in heads.parameters())),
              'elapsed_seconds': time.time() - start,
              'best_checkpoint_sha256': digest(dest / 'best.pt')}
    dump(dest / 'status.json', result)
    print('COMPLETE', dest.name, result['accuracy'], result['macro_f1'], flush=True)
    del model, heads, optimiser, targets, matrix
    gc.collect(); torch.cuda.empty_cache()
    return result


def _audit_gradients(model, heads, targets, aux_total):
    """Does the AUXILIARY loss alone reach the backbone?

    Reading .grad after the combined backward cannot answer this: the
    classification cross-entropy already makes every encoder gradient non-zero,
    so a detached auxiliary term would pass unnoticed. The gradient of the
    auxiliary term alone is taken here without touching the optimiser state.
    """
    watched = {'temporal_proj': model.encoder.temporal.proj.weight,
               'spatial_proj': model.encoder.spatial.proj.weight,
               'head_first': model.head.net[0].weight}
    audit = {'auxiliary_only': None, 'auxiliary_present': aux_total is not None,
             'teacher_targets_require_grad': bool(targets is not None and targets.requires_grad)}
    if aux_total is not None:
        grads = torch.autograd.grad(aux_total, list(watched.values()),
                                    retain_graph=True, allow_unused=True)
        audit['auxiliary_only'] = {
            name: (None if g is None else float(g.abs().sum()))
            for name, g in zip(watched, grads)}
        audit['auxiliary_reaches_encoder'] = bool(
            (audit['auxiliary_only']['temporal_proj'] or 0) > 0
            or (audit['auxiliary_only']['spatial_proj'] or 0) > 0)
    return audit


@torch.no_grad()
def _auxiliary_free_logits(model, data, ids, reference, microbatch):
    """Dropping the auxiliary heads must not move a single deployed logit."""
    model.eval()
    out = []
    for i in range(0, len(ids), microbatch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(data[ids[i:i + microbatch]])
        out.append(logits.float().cpu().numpy())
    return float(np.abs(np.concatenate(out) - reference).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=list(P.FOLDS))
    ap.add_argument('--model-seed', type=int, default=2026)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    for arm in args.arms:
        spec_for(arm)
    import os
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    P.verify_manifest()
    data = load_student_data()
    print('START', args.arms, 'gpu_data_GB', data.numel() * data.element_size() / 1e9, flush=True)
    for arm in args.arms:
        for fold in args.folds:
            run(arm, fold, args.model_seed, data, force=args.force)


if __name__ == '__main__':
    main()
