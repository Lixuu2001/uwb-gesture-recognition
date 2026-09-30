"""Raw-input joint training. Every classification loss reaches the encoder."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score

from data_access import HERE, PARENT, BASE, load_fold, digest, dump
from model import build_model, state_hash
from checkpointing import save_progress, load_progress, atomic_save
from runner import classification_loss
from data import setup


def objective(outputs, labels, matrix):
    return torch.stack([classification_loss(z.float(), labels, matrix) for z, _ in outputs]).mean()


def should_stop(epoch, best_epoch, minimum=12, patience=8):
    return epoch >= minimum and epoch-best_epoch >= patience


@torch.no_grad()
def predict(model, right, batch):
    if right.ndim != 4 or right.shape[1] != 1:
        raise ValueError('Right-only prediction input required')
    model.eval(); results = []
    for start in range(0, len(right), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            results.append(model(right[start:start+batch]).float().cpu().numpy())
    return np.concatenate(results)


def source_identity():
    paths = [HERE/'train_e2e.py', HERE/'model.py', HERE/'data_access.py', HERE/'checkpointing.py',
             PARENT/'models.py', PARENT/'data.py', PARENT/'runner.py']
    return {str(p): digest(p) for p in paths}


def train_one(args, fold, mode, reader, arrays, mu, sd, source):
    dest = args.out/f'E2E_{mode}_s{args.seed}_h{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    config = {'fold': fold, 'mode': mode, 'seed': args.seed, 'split_seed': 2026,
              'sampler_seed': 7777+fold, 'dropout_seed': args.seed+100000,
              'epochs': args.epochs, 'batch': 128, 'microbatch': args.microbatch,
              'encoder_lr': 1e-5, 'fuser_lr': 1e-3, 'weight_decay': 1e-4,
              'minimum_epochs': 12, 'patience': 8, 'source_identity': source['identity'],
              'source_code': source_identity(), 'cache_sha256': source['cache_sha256']}
    cp = dest/'config.json'; status_file = dest/'status.json'
    if cp.exists() and json.loads(cp.read_text(encoding='utf-8')) != config:
        raise ValueError(f'configuration changed; use a new output directory: {dest}')
    if status_file.exists():
        status = json.loads(status_file.read_text(encoding='utf-8'))
        if status['status'] == 'complete':
            for filename in ('predictions.npz', 'deployment.pt'):
                if digest(dest/filename) != status['artifacts'][filename]:
                    raise ValueError(f'completed artifact changed: {dest/filename}')
            print('SKIP', dest.name, flush=True); return
    dump(cp, config)
    dump(status_file, {'status': 'running', 'fold': fold, 'mode': mode, 'started': time.time()})
    started = time.time()
    try:
        train_x = reader.load('train', mode)
        val_x = reader.load('val', 'R')
        test_x = reader.load('test', 'R')
        labels = torch.tensor(arrays['train_y'], device='cuda')
        matrix = torch.tensor(np.load(BASE/f'label_smoothing_results/global_matrix_h{fold}.npz')['global_matrix'], device='cuda')
        model, lineage = build_model(fold, args.seed, mu, sd)
        assert lineage['checkpoint_sha256'] == source['identity'][lineage['checkpoint_path']]
        dump(dest/'initialization.json', lineage)
        optimizer = torch.optim.Adam([{'params': model.encoder.parameters(), 'lr': 1e-5},
                                      {'params': model.fuser.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
        sampler = torch.Generator().manual_seed(7777+fold)
        torch.manual_seed(args.seed+100000); torch.cuda.manual_seed_all(args.seed+100000)
        run_identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        info = {'epoch': 0, 'best_epoch': 0, 'history': [], 'sampling_hashes': [], 'elapsed': 0.,
                'run_identity': run_identity}
        if (dest/'progress.pt').exists():
            info = load_progress(dest/'progress.pt', model, optimizer, sampler, expected_identity=run_identity)
            print(f'RESUME {dest.name} after_epoch={info["epoch"]}', flush=True)
        else:
            val_logits = predict(model, val_x, args.microbatch)
            best = float(F.cross_entropy(torch.tensor(val_logits), torch.tensor(arrays['val_y'])))
            info.update(best=best, best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            info['history'].append({'epoch': 0, 'val_ce': best,
                                    'val_accuracy': float((val_logits.argmax(1) == arrays['val_y']).mean())})
            save_progress(dest/'progress.pt', model, optimizer, sampler, info)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(info['epoch']+1, args.epochs+1):
            if should_stop(info['epoch'], info['best_epoch']):
                break
            tick = time.time(); model.train()
            order = torch.randperm(len(train_x), generator=sampler).numpy()
            info['sampling_hashes'].append(hashlib.sha256(order.tobytes()).hexdigest())
            total = 0.; max_encoder_gradient = 0.
            for begin in range(0, len(order), 128):
                batch = order[begin:begin+128]
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, len(batch), args.microbatch):
                    j = batch[start:start+args.microbatch]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model.training_outputs(train_x[j], mode)
                        loss = objective(outputs, labels[j], matrix)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    (loss*len(j)/len(batch)).backward()
                    total += float(loss.detach())*len(j)
                gradient = model.encoder.temporal.proj.weight.grad
                max_encoder_gradient = max(max_encoder_gradient, float(gradient.detach().float().norm()))
                optimizer.step()
            val_logits = predict(model, val_x, args.microbatch)
            ce = float(F.cross_entropy(torch.tensor(val_logits), torch.tensor(arrays['val_y'])))
            row = {'epoch': epoch, 'train_loss': total/len(train_x), 'val_ce': ce,
                   'val_accuracy': float((val_logits.argmax(1) == arrays['val_y']).mean()),
                   'max_encoder_gradient_norm': max_encoder_gradient,
                   'seconds': time.time()-tick, 'peak_memory_mb': torch.cuda.max_memory_allocated()/2**20}
            info['history'].append(row)
            info['epoch'] = epoch
            info['elapsed'] += row['seconds']
            if ce < info['best']:
                info.update(best=ce, best_epoch=epoch,
                            best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            save_progress(dest/'progress.pt', model, optimizer, sampler, info)
            dump(dest/'history.json', info['history'])
            dump(status_file, {'status': 'running', 'fold': fold, 'mode': mode, 'epoch': epoch,
                               'best_epoch': info['best_epoch'], 'best_val_ce': info['best']})
            print(f'EPOCH {dest.name} {epoch} val_ce={ce:.5f} best={info["best_epoch"]} '
                  f'encoder_grad={max_encoder_gradient:.5g} seconds={row["seconds"]:.1f}', flush=True)
        model.load_state_dict(info['best_state'])
        logits = predict(model, test_x, args.microbatch)
        vl = predict(model, val_x, args.microbatch)
        prediction = logits.argmax(1)
        atomic_save(dest/'deployment.pt', {'state': info['best_state'], 'config': config,
                                          'initialization': lineage, 'best_epoch': info['best_epoch']})
        np.savez_compressed(dest/'predictions.npz', ids=arrays['test_ids'], y=arrays['test_y'],
                            pred=prediction, logits=logits, base_logits=arrays['test_base'],
                            train_ids=arrays['train_ids'], val_ids=arrays['val_ids'],
                            val_y=arrays['val_y'], val_logits=vl)
        final_encoder = state_hash(model.encoder)
        result = {'status': 'complete', 'fold': fold, 'mode': mode, 'seed': args.seed,
                  'accuracy': float((prediction == arrays['test_y']).mean()),
                  'base_accuracy': source['base_accuracy'],
                  'macro_f1': float(f1_score(arrays['test_y'], prediction, labels=np.unique(arrays['test_y']),
                                             average='macro', zero_division=0)),
                  'n_test': len(prediction), 'best_epoch': info['best_epoch'], 'best_val_ce': info['best'],
                  'epochs': info['epoch'], 'sampling_hashes': info['sampling_hashes'],
                  'initial_encoder_sha256': lineage['initial_encoder_sha256'],
                  'final_encoder_sha256': final_encoder,
                  'encoder_changed': final_encoder != lineage['initial_encoder_sha256'],
                  'encoder_gradient_observed': any(r.get('max_encoder_gradient_norm', 0)>0 for r in info['history']),
                  'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'elapsed_training_seconds': info['elapsed'], 'invocation_seconds': time.time()-started,
                  'artifacts': {n: digest(dest/n) for n in ('deployment.pt', 'predictions.npz')}}
        dump(dest/'history.json', info['history']); dump(status_file, result)
        print(f'COMPLETE {dest.name} accuracy={100*result["accuracy"]:.3f} '
              f'encoder_changed={result["encoder_changed"]}', flush=True)
        del model, optimizer, train_x, val_x, test_x, info
        gc.collect(); torch.cuda.empty_cache()
    except BaseException as exc:
        dump(status_file, {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                           'fold': fold, 'mode': mode, 'error': repr(exc),
                           'recoverable_checkpoint': (dest/'progress.pt').exists()})
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=HERE/'runs')
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    ap.add_argument('--modes', nargs='+', choices=['R', 'MV'], default=['R', 'MV'])
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--microbatch', type=int, default=16)
    args = ap.parse_args(); setup(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    snapshots = args.out/'source_code'; snapshots.mkdir(exist_ok=True)
    for path, sha in source_identity().items():
        source = Path(path)
        target = snapshots/(('e2e_' if source.parent == HERE else 'parent_')+source.name)
        if target.exists() and digest(target) != sha:
            raise ValueError('source snapshot changed; use a new output directory')
        if not target.exists():
            target.write_bytes(source.read_bytes())
    if __import__('os').name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for fold in args.folds:
            reader, arrays, mu, sd, source = load_fold(fold)
            for mode in args.modes:
                dump(args.out/'queue_status.json', {'status': 'running', 'fold': fold, 'mode': mode})
                train_one(args, fold, mode, reader, arrays, mu, sd, source)
            del arrays, reader
        dump(args.out/'queue_status.json', {'status': 'complete', 'folds': args.folds, 'modes': args.modes})
    except BaseException as exc:
        dump(args.out/'queue_status.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                          'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
