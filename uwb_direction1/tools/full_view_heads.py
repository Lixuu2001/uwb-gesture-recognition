"""Train matched three-view readout controls on verified frozen radar features."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from audit_restored import digest, dump, metrics, validate_cache

MASKS = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1], [0,1,1], [1,1,1]], dtype=torch.float32)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(771, 256), nn.GELU(),
                                 nn.Dropout(.1), nn.Linear(256, 12))

    def forward(self, features, mask=None):
        if features.ndim != 3 or features.shape[1:] != (3, 256):
            raise ValueError('Expected [N,3,256] frozen features')
        if mask is None:
            mask = features.new_ones(3)
        flags = mask.expand(len(features), 3)
        x = torch.cat(((features * flags[:, :, None]).flatten(1), flags), dim=1)
        hidden = self.net[1](self.net[0](x))
        return self.net[3](self.net[2](hidden)), hidden


def contrast_masks(labels, users, crossuser):
    same_class = labels[:, None] == labels[None, :]
    self_pair = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    if crossuser:
        different_user = users[:, None] != users[None, :]
        positive = same_class & different_user
        eligible = (~same_class) | different_user
    else:
        positive, eligible = same_class & ~self_pair, ~self_pair
    return positive, eligible


def supcon(hidden, labels, users, crossuser=False, temperature=.1):
    positive, eligible = contrast_masks(labels, users, crossuser)
    counts = positive.sum(1)
    valid = counts > 0
    if not valid.any():
        return hidden.sum() * 0
    z = F.normalize(hidden, dim=1)
    logits = z @ z.T / temperature
    log_denom = torch.logsumexp(logits.masked_fill(~eligible, -torch.inf), dim=1)
    selected = torch.where(positive, logits - log_denom[:, None], torch.zeros_like(logits))
    return -(selected.sum(1)[valid] / counts[valid]).mean()


def sample_epoch(labels, users, seed, held, epoch):
    """Indices are local to explicit source arrays, never validation/test arrays."""
    cells = [np.flatnonzero((labels == c) & (users == u))
             for c in range(12) for u in np.unique(users)
             if np.any((labels == c) & (users == u))]
    if any(len(cell) < 2 for cell in cells):
        raise ValueError('Need two distinct source records in every retained cell')
    rng = np.random.default_rng(seed + 10000 * held + 100000 * epoch)
    steps = math.ceil(len(labels) / (2 * len(cells)))
    batches = []
    for _ in range(steps):
        batch = np.concatenate([rng.choice(cell, 2, replace=False) for cell in cells])
        batches.append(rng.permutation(batch))
    return np.stack(batches)


@torch.no_grad()
def evaluate(model, x, batch=256):
    model.eval()
    return torch.cat([model(x[start:start+batch])[0] for start in range(0, len(x), batch)])


def atomic_save(path, obj):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def fit_arm(out, config, identity, arm, held, features, y, users, roles, ids):
    out.mkdir(parents=True, exist_ok=True)
    result_path, progress_path = out/'result.json', out/'progress.pt'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['identity'] != identity:
            raise ValueError('Completed identity mismatch')
        for filename, sha in result['files'].items():
            if digest(out/filename) != sha:
                raise ValueError('Completed artifact mismatch: ' + filename)
        return result
    train, val, test = (roles == i for i in range(3))
    xtr, xva = torch.from_numpy(features[train]), torch.from_numpy(features[val])
    ytr, yva = torch.from_numpy(y[train]).long(), torch.from_numpy(y[val]).long()
    utr = torch.from_numpy(users[train]).long()
    torch.manual_seed(config['model_seed'])
    model = Head()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'],
                                 weight_decay=config['optimizer']['weight_decay'])
    if progress_path.exists():
        saved = torch.load(progress_path, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Resume identity mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        history, best, best_ce, best_epoch = (saved[k] for k in ('history','best','best_ce','best_epoch'))
    else:
        logits = evaluate(model, xva)
        best_ce, best_epoch = F.cross_entropy(logits, yva).item(), 0
        best = {k: v.clone() for k, v in model.state_dict().items()}
        history = [{'epoch': 0, 'validation_ce': best_ce}]
    for epoch in range(history[-1]['epoch'] + 1, config['epochs'] + 1):
        if epoch > config['min_epochs'] and epoch - 1 - best_epoch >= config['patience']:
            break
        start = time.perf_counter()
        batches = sample_epoch(y[train], users[train], config['model_seed'], held, epoch)
        subset_rng = np.random.default_rng(config['model_seed'] + 700000 + held * 10000 + epoch)
        losses, cls_losses, con_losses = [], [], []
        model.train()
        for rows in batches:
            mask = MASKS[subset_rng.integers(7)] if arm == 'subset_ce' else None
            logits, hidden = model(xtr[rows], mask)
            ce = F.cross_entropy(logits, ytr[rows])
            con = hidden.sum() * 0
            if arm in ('full_supcon', 'full_crossuser'):
                con = supcon(hidden, ytr[rows], utr[rows], arm == 'full_crossuser',
                             config['supcon']['temperature'])
            loss = ce + config['supcon']['weight'] * con
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item()); cls_losses.append(ce.item()); con_losses.append(con.item())
        val_logits = evaluate(model, xva)
        val_ce = F.cross_entropy(val_logits, yva).item()
        if val_ce < best_ce:
            best_ce, best_epoch = val_ce, epoch
            best = {k: v.clone() for k, v in model.state_dict().items()}
        row = {'epoch': epoch, 'validation_ce': val_ce, 'train_loss': float(np.mean(losses)),
               'train_ce': float(np.mean(cls_losses)), 'train_contrast': float(np.mean(con_losses)),
               'source_batch_ids_sha256': array_hash(ids[train][batches]),
               'seconds': time.perf_counter()-start}
        history.append(row)
        atomic_save(progress_path, dict(identity=identity, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    history=history, best=best, best_ce=best_ce, best_epoch=best_epoch))
        dump(out/'progress.json', {'state':'training','pid':os.getpid(),'epoch':epoch,
                                   'best_epoch':best_epoch,'best_val_ce':best_ce,'last':row})
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'fold':held,'arm':arm,'epoch':epoch,'val_ce':val_ce,
                              'seconds':row['seconds']}), flush=True)
    model.load_state_dict(best)
    # Test is evaluated only after validation checkpoint selection is finished.
    test_logits = evaluate(model, torch.from_numpy(features[test])).numpy()
    val_logits = evaluate(model, xva).numpy()
    np.testing.assert_allclose(F.cross_entropy(torch.from_numpy(val_logits), yva).item(), best_ce,
                               atol=1e-7, rtol=0)
    atomic_save(out/'best.pt', {'identity':identity,'model':best,'epoch':best_epoch,
                              'validation_ce':best_ce})
    np.savez_compressed(out/'predictions.npz', ids=ids[test], y=y[test], logits=test_logits,
                        validation_ids=ids[val], validation_logits=val_logits)
    result = {'identity':identity,'arm':arm,'held_user':held,'validation_user':held%8+1,
              'selected_epoch':best_epoch,'selected_val_ce':best_ce,'history':history,
              'metrics':metrics(y[test], test_logits.argmax(1)),
              'files':{name:digest(out/name) for name in ('best.pt','predictions.npz','progress.pt')}}
    dump(result_path, result)
    dump(out/'progress.json', {'state':'complete','selected_epoch':best_epoch})
    print(json.dumps({'fold':held,'arm':arm,'complete':True,
                      'macro_accuracy':result['metrics']['macro_accuracy']}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--audit', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parents[1]/'research/FULL_VIEW_HEAD_PROTOCOL.json')
    ap.add_argument('--benchmark', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    config = json.loads(args.protocol.read_text())
    audit = json.loads(args.audit.read_text())
    assert audit['state'] == 'verified'
    identity = {'protocol_sha256':digest(args.protocol),'script_sha256':digest(__file__),
                'audit_sha256':digest(args.audit),'torch':torch.__version__,'numpy':np.__version__}
    args.out.mkdir(parents=True, exist_ok=True)
    # OS-level exclusive lock prevents overlapping hourly tasks from duplicating a queue.
    import fcntl
    lock = (args.out/'queue.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.out/'identity.json').exists():
        if json.loads((args.out/'identity.json').read_text()) != identity:
            raise ValueError('Queue identity mismatch')
    else:
        dump(args.out/'identity.json', identity)
        dump(args.out/'protocol.json', config)
    labels = np.load(args.data/'cache/y_clutterremoved.npy')
    subjects = np.load(args.data/'cache/subj_clutterremoved.npy')
    results = []
    try:
        for held in range(1, 9):
            folder = args.data/f'experiments/p1_20260928/multiview_s2026/fold{held}'
            path = folder/'predictions.npz'
            source_sha = digest(path)
            assert source_sha == audit['artifact_hashes'][str(path.relative_to(args.data))]
            with np.load(path, allow_pickle=False) as cache:
                validate_cache(cache, folder/'split.json', labels, subjects, held)
                features, y, users, roles, ids = (cache[k] for k in ('features','y','subjects','role','ids'))
            if args.benchmark:
                train = roles == 0
                torch.manual_seed(config['model_seed'])
                model = Head()
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                rows = sample_epoch(y[train], users[train], 2026, held, 1)[0]
                x = torch.from_numpy(features[train][rows])
                yy, uu = torch.from_numpy(y[train][rows]).long(), torch.from_numpy(users[train][rows]).long()
                started = time.perf_counter()
                for _ in range(20):
                    logits, hidden = model(x)
                    loss = F.cross_entropy(logits, yy) + .05 * supcon(hidden, yy, uu, True)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                result = {'state':'benchmark_only','steps':20,'batch':len(rows),
                          'seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters()),
                          'source_ids':ids[train][rows].tolist(),'no_validation_or_test_fitting':True}
                dump(args.out/'benchmark.json', result)
                print(json.dumps({k:v for k,v in result.items() if k != 'source_ids'}))
                return
            for arm in config['arms']:
                dump(args.out/'queue_status.json', {'state':'running','pid':os.getpid(),'fold':held,'arm':arm})
                arm_identity = dict(identity, fold=held, arm=arm, frozen_features_sha256=source_sha)
                row = fit_arm(args.out/f'fold{held}'/arm, config, arm_identity, arm, held,
                              features, y, users, roles, ids)
                results.append(row)
        scores = {arm:np.array([r['metrics']['macro_accuracy'] for r in results if r['arm'] == arm])
                  for arm in config['arms']}
        old = np.array([r['subsets']['LTR']['macro_accuracy'] for r in audit['folds']])
        scores['historical_LTR'] = old
        candidate = scores['full_crossuser']
        summary = {arm:{'mean_user_macro_accuracy':float(s.mean()),'worst_user_macro_accuracy':float(s.min()),
                         'per_user':s.tolist()} for arm,s in scores.items()}
        paired = {arm:{'mean_difference_pp':float(100*(candidate-s).mean()),
                       'improved_users':int((candidate>s).sum()),'per_user_pp':(100*(candidate-s)).tolist()}
                  for arm,s in scores.items() if arm != 'full_crossuser'}
        passed = all((candidate-scores[a]).mean() >= .01 for a in ('full_ce','full_supcon','historical_LTR'))
        passed = passed and (candidate>scores['full_ce']).sum() >= 6
        report = {'state':'complete','identity':identity,'summary':summary,'paired':paired,
                  'continue_criterion_met':bool(passed),'novelty_claim':False,'encoder_retrained':False,
                  'runs':[{'arm':r['arm'],'held_user':r['held_user'],'selected_epoch':r['selected_epoch'],
                           'selected_val_ce':r['selected_val_ce'],'files':r['files']} for r in results]}
        dump(args.out/'RESULTS.json', report)
        dump(args.out/'queue_status.json', {'state':'complete','pid':os.getpid(),'runs':len(results)})
        print(json.dumps({'state':'complete','summary':summary,'continue_criterion_met':bool(passed)}), flush=True)
    except BaseException as error:
        dump(args.out/'queue_status.json', {'state':'failed','pid':os.getpid(),
                                           'error_type':type(error).__name__,'error':str(error)})
        raise


if __name__ == '__main__':
    main()
