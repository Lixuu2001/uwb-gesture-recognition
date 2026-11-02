"""LOSO runner for reliability-gated end-to-end feature completion."""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, confusion_matrix

from pipeline import BASE, RESULTS, dump, digest, split
from train import Classifier, setup, load_gpu, digest_state
from reconstruction import fit_scale, reconstruction_loss
from e2e_reconstruction import extract
from gated_reconstruction import GatedCompletion

OUT = BASE / 'gated_reconstruction_results'


def build(fold, view, mode, mu, sd, seed):
    setup(seed)
    source = RESULTS / f'teacher_concat_s2026_h{fold}.pt'
    base = Classifier([0, 1, 2], 'off', 'concat')
    ck = torch.load(source, map_location='cpu', weights_only=False)
    assert ck['fold'] == fold and ck['args']['seed'] == 2026
    base.load_state_dict(ck['state'])
    enc = base.encoder
    assert enc.temporal.xmv == 0 and enc.spatial.xmv == 0
    setup(seed + 7000)
    model = GatedCompletion(enc, view, mode, mu.cpu(), sd.cpu()).cuda()
    # Start close to the zero-filled deployment fallback. Both gated arms use
    # this identical initialization; the joint arm differs only by its loss.
    with torch.no_grad():
        model.gate[2].bias.fill_(-1.0)
    return model


@torch.no_grad()
def evaluate(model, data, target, ids, batch=64):
    model.eval(); logits = []
    num = torch.zeros((), device='cuda'); den = num.clone()
    missing = [v for v in range(3) if v != model.view]
    for i in range(0, len(ids), batch):
        j = ids[i:i + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z, parts, gates = model(data[j])
        logits.append(z.float().cpu().numpy())
        num += (parts[:, missing].float() - target[j][:, missing]).square().sum()
        den += target[j][:, missing].square().sum()
    return np.concatenate(logits), float(num / den), float(num / (len(ids) * 2 * 992))


def run(args, fold, mode, data, features, y, subjects, teacher_sha):
    tag = f'{mode}_L_s{args.seed}_h{fold}'
    dest = OUT / tag
    if dest.with_suffix('.json').exists():
        print('SKIP', tag, flush=True); return
    tr, va, te = split(y, subjects, fold, 2026)
    mu, sd = fit_scale(features, tr)
    target = (features.float() - mu) / sd
    target.requires_grad_(False)
    model = build(fold, args.view, mode, mu, sd, args.seed)
    initial = digest_state(model)
    head_initial = digest_state(model.head)
    enc_initial = digest_state(model.encoder)
    opt = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters()
                    if not n.startswith('encoder.')], 'lr': 1e-3},
    ], weight_decay=1e-4)
    yy = torch.tensor(y, device='cuda')
    order = torch.Generator().manual_seed(args.seed + fold * 1000)
    torch.manual_seed(args.seed + 100000); torch.cuda.manual_seed_all(args.seed + 100000)
    best = float('inf'); best_ep = 0; history = []; tic = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); idx = tr[torch.randperm(len(tr), generator=order).numpy()]
        total = 0.; rec_sum = 0.; gate_sum = 0.
        for i in range(0, len(idx), 128):
            batch_ids = idx[i:i + 128]; opt.zero_grad(set_to_none=True)
            pieces = np.array_split(batch_ids, int(np.ceil(len(batch_ids) / 32)))
            for mb in pieces:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z, parts, gates = model(data[mb])
                    rec = reconstruction_loss(parts, target[mb], args.view)
                    loss = F.cross_entropy(z, yy[mb])
                    if mode == 'gate_joint': loss = loss + 0.1 * rec
                (loss * len(mb) / len(batch_ids)).backward()
                total += float(loss.detach()) * len(mb)
                rec_sum += float(rec.detach()) * len(mb)
                gate_sum += float(gates.detach().mean()) * len(mb)
            opt.step()
        vl, vnm, vmse = evaluate(model, data, target, va)
        ce = F.cross_entropy(torch.tensor(vl), torch.tensor(y[va])).item()
        history.append({'epoch': ep, 'train_loss': total / len(tr),
                        'train_rec': rec_sum / len(tr), 'train_gate': gate_sum / len(tr),
                        'val_ce': ce, 'val_accuracy': float(np.mean(vl.argmax(1) == y[va])),
                        'val_nmse': vnm})
        if ce < best:
            best = ce; best_ep = ep
            torch.save({'state': model.state_dict(), 'fold': fold, 'view': args.view,
                        'mode': mode, 'seed': args.seed, 'epoch': ep,
                        'teacher_sha': teacher_sha}, dest.with_suffix('.pt'))
        print('EPOCH', tag, ep, 'val_ce', round(ce, 4), 'rec', round(vmse, 4),
              'gate', round(gate_sum / len(tr), 3), flush=True)
        if ep >= 12 and ep - best_ep >= 6: break
    ck = torch.load(dest.with_suffix('.pt'), map_location='cuda', weights_only=False)
    model.load_state_dict(ck['state']); del ck
    logits, nmse, mse = evaluate(model, data, target, te); pred = logits.argmax(1)
    val_logits, _, _ = evaluate(model, data, target, va)
    _, _, test_gate = evaluate_gates(model, data, te)
    np.savez_compressed(str(dest) + '_predictions.npz', ids=te, y=y[te], pred=pred,
                        logits=logits, train_ids=tr, val_ids=va, val_logits=val_logits)
    res = {
        'tag': tag, 'fold': fold, 'view': args.view, 'mode': mode, 'seed': args.seed,
        'encoder_seed': 2026, 'accuracy': float(np.mean(pred == y[te])),
        'macro_f1': float(f1_score(y[te], pred, labels=np.unique(y[te]), average='macro', zero_division=0)),
        'confusion': confusion_matrix(y[te], pred, labels=np.arange(12)).tolist(),
        'missing_nmse': nmse, 'missing_mse': mse, 'test_gate_mean': test_gate,
        'initial_hash': initial, 'initial_head_hash': head_initial,
        'initial_encoder_hash': enc_initial, 'teacher_sha': teacher_sha,
        'parameters': sum(p.numel() for p in model.parameters()),
        'best_epoch': best_ep, 'best_val_ce': best, 'n_test': len(te),
        'history': history, 'seconds': time.time() - tic, 'config': vars(args),
        'source_sha': digest(__file__),
    }
    dump(dest.with_suffix('.json'), res)
    print('COMPLETE', tag, 'acc', round(res['accuracy'] * 100, 3),
          'nmse', round(nmse, 3), 'gate', round(test_gate, 3),
          'sec', round(res['seconds'], 1), flush=True)
    del model, opt, target; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_gates(model, data, ids, batch=64):
    model.eval(); total = 0.; count = 0
    for i in range(0, len(ids), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, _, gates = model(data[ids[i:i + batch]])
        total += float(gates.float().mean()) * len(ids[i:i + batch]); count += len(ids[i:i + batch])
    return None, None, total / count


def smoke():
    features, y, subjects, sha = extract(1)
    tr, _, _ = split(y, subjects, 1, 2026); mu, sd = fit_scale(features, tr)
    data = load_gpu('off', [0]); model = build(1, 0, 'gate_joint', mu, sd, 2026)
    j = tr[:8]; z, p, g = model(data[j].float())
    t = (features[j].float() - mu) / sd
    loss = F.cross_entropy(z, torch.tensor(y[j], device='cuda')) + .1 * reconstruction_loss(p, t, 0)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.temporal.proj.weight.grad.abs().sum() > 0
    print('SMOKE_OK loss', float(loss.detach()), 'encoder_gradient',
          float(model.encoder.temporal.proj.weight.grad.abs().sum()),
          'gate_mean', float(g.mean()), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--folds', nargs='+', type=int, default=list(range(1, 9)))
    ap.add_argument('--modes', nargs='+', choices=['gate_capacity', 'gate_joint'],
                    default=['gate_capacity', 'gate_joint'])
    ap.add_argument('--view', type=int, choices=[0, 1, 2], default=0)
    ap.add_argument('--seed', type=int, default=2026); ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    OUT.mkdir(exist_ok=True); setup(args.seed)
    import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if args.smoke: smoke(); return
    data = load_gpu('off', [args.view])
    for fold in args.folds:
        ft, y, s, sha = extract(fold)
        for mode in args.modes: run(args, fold, mode, data, ft, y, s, sha)
        del ft; gc.collect(); torch.cuda.empty_cache()
    print('GATED_COMPLETE', flush=True)


if __name__ == '__main__': main()
