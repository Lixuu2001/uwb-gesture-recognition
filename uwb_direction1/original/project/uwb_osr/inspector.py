"""Deep-inspection OSR head (paper-faithful, Eom IoTJ-2025 Stage-2) with the
multi-view extension from the design doc (per-view inspectors + mean/variance).

Stage-2 (this module) runs on top of a frozen Stage-1 encoder/classifier:
  * one inspector per (known class k, view v):  feature f_v -> latent l
  * prototype phi_k^v = mean latent of class-k real samples (updated per epoch)
  * profile-distance loss: pull class-k close to phi, push others away
  * per-class threshold sigma_k^v = RMS distance of class-k samples to phi
Decision uses the multi-view anomaly score
      s = mean_v(d_v)/mean_v(sigma_v) + beta * var_v(d_v)/mean_v(sigma_v)^2
where high s => unknown. confidence-of-known = -s.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import auroc, oscr, best_threshold_acc_f1, fixed_eta_acc_f1


class Inspector(nn.Module):
    def __init__(self, in_dim, hidden=1024, latent=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent))

    def forward(self, f):
        return self.net(f)


@torch.no_grad()
def extract(model, ds, cfg, device, view_indices=None):
    """Frozen forward pass: return per-view features, logits, labels.
    `view_indices` (E2): only forward the chosen view slots (the model's
    view_emb is sliced to match). When None, all trained views are used.
    F1 (cfg.inspector_feats): "pre" reads the encoder feature codes (default,
    old pipeline); "fused" reads the post-cross-view-attention codes."""
    model.eval()
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=cfg.num_workers)
    fkey = ("f_views_fused"
            if getattr(cfg, "inspector_feats", "pre") == "fused"
            else "f_views")
    F, L, Y = [], [], []
    for x, y in loader:
        x = x.to(device)
        if view_indices is not None:
            x = x[:, view_indices]
        out = model(x, view_indices=view_indices)
        F.append(out[fkey].cpu())
        L.append(out["logits"].cpu())
        Y.append(y)
    return torch.cat(F), torch.cat(L).numpy(), torch.cat(Y).numpy()


def _make_mixup_negatives(features, labels, cfg):
    """Eq.12-style synthetic negatives from different known gesture labels.

    This branch can mix the target class with another known class, producing
    closer near-boundary negatives than mixing only non-target samples. It is
    useful for paper-faithful diagnostics, but is intentionally gated by
    cfg.paper_neg_mixup because the local single-view sweep did not improve.
    """
    if cfg.n_neg_syn <= 0 or len(features) < 2:
        return features.new_empty((0, features.size(1)))
    labels = torch.as_tensor(labels, device=features.device)
    unique = torch.unique(labels)
    if unique.numel() < 2:
        return features.new_empty((0, features.size(1)))

    n = cfg.n_neg_syn
    i = torch.randint(0, len(features), (n,), device=features.device)
    j = torch.empty_like(i)
    for cls in unique:
        mask = labels[i] == cls
        count = int(mask.sum().item())
        if count == 0:
            continue
        candidates = torch.where(labels != cls)[0]
        picks = candidates[torch.randint(0, len(candidates), (count,),
                                        device=features.device)]
        j[mask] = picks

    if cfg.ood_mix_low == cfg.ood_mix_high:
        lam = torch.full((n, 1), float(cfg.ood_mix_low), device=features.device)
    else:
        lam = torch.empty(n, 1, device=features.device).uniform_(
            cfg.ood_mix_low, cfg.ood_mix_high)
    return lam * features[i] + (1.0 - lam) * features[j]


def _augment(pos, neg, cfg, all_real=None, all_labels=None):
    """Build Stage-2 training data for one inspector.

    pos/neg are defined relative to the current one-vs-rest inspector. The
    default negative synthesis mixes only non-target real features; enabling
    --paper-neg-mixup switches to all-known-label Eq.12-style mixup.
    """
    if cfg.n_pos_syn > 0 and len(pos) > 0:
        idx = torch.randint(0, len(pos), (cfg.n_pos_syn,))
        a = 1.0 + cfg.pos_jitter * torch.randn(cfg.n_pos_syn, 1)
        b = cfg.pos_jitter * torch.randn(cfg.n_pos_syn, pos.size(1))
        pos = torch.cat([pos, a * pos[idx] + b], 0)
    if cfg.paper_neg_mixup and all_real is not None and all_labels is not None:
        # Paper-facing branch: mix any two different known labels, including
        # target + non-target pairs for the current inspector.
        neg = torch.cat([neg, _make_mixup_negatives(all_real, all_labels, cfg)], 0)
    elif cfg.n_neg_syn > 0 and len(neg) > 1:
        # Empirical default: mix only already-negative samples for this
        # one-vs-rest head. This was stronger in the local 5-case single-view run.
        i = torch.randint(0, len(neg), (cfg.n_neg_syn,))
        j = torch.randint(0, len(neg), (cfg.n_neg_syn,))
        lam = torch.empty(cfg.n_neg_syn, 1).uniform_(cfg.ood_mix_low, cfg.ood_mix_high)
        neg = torch.cat([neg, lam * neg[i] + (1 - lam) * neg[j]], 0)
    return pos, neg


def train_inspector(pos_real, neg_real, cfg, device, all_real=None, all_labels=None):
    """Train one (class, view) inspector. Returns (module, prototype, sigma)."""
    insp = Inspector(pos_real.size(1), cfg.inspector_hidden,
                     cfg.inspector_latent,
                     dropout=getattr(cfg, "inspector_dropout", 0.1)).to(device)
    opt = torch.optim.Adam(insp.parameters(), lr=cfg.lr)
    pos, neg = _augment(pos_real, neg_real, cfg, all_real, all_labels)
    X = torch.cat([pos, neg], 0).to(device)
    z = torch.cat([torch.zeros(len(pos)), torch.ones(len(neg))]).to(device)  # 0=pos
    pos_real_d = pos_real.to(device)
    for _ in range(cfg.inspector_epochs):
        insp.eval()
        with torch.no_grad():                          # prototype from real pos
            phi = insp(pos_real_d).mean(0)             # deterministic (no dropout)
        insp.train()
        perm = torch.randperm(len(X), device=device)
        for s in range(0, len(X), cfg.batch_size_insp):
            idx = perm[s:s + cfg.batch_size_insp]
            l = insp(X[idx])
            d = torch.norm(l - phi, dim=1)
            p = torch.exp(-d).clamp(1e-6, 1 - 1e-6)
            zi = z[idx]
            loss = -((1 - zi) * torch.log(p) + zi * torch.log(1 - p)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    insp.eval()
    with torch.no_grad():
        lp = insp(pos_real_d)
        phi = lp.mean(0)
        sigma = torch.sqrt(((lp - phi) ** 2).sum(1).mean()).item()
    return insp, phi.detach(), sigma


def train_inspectors(model, train_ds, cfg, n_known, device, verbose=True):
    F, _, y = extract(model, train_ds, cfg, device)        # F: (N,V,feat)
    V = F.size(1)
    bank = {}
    pbar = tqdm(total=n_known * V, desc="Stage-2 inspectors",
                disable=not verbose, dynamic_ncols=True)
    labels = torch.as_tensor(y)
    for k in range(n_known):
        for v in range(V):
            Fv = F[:, v]
            bank[(k, v)] = train_inspector(Fv[y == k], Fv[y != k], cfg, device,
                                           all_real=Fv, all_labels=labels)
            pbar.update(1)
    pbar.close()
    return bank, V


@torch.no_grad()
def anomaly_scores(bank, view_indices, F_test, preds, cfg, device, mode="combined"):
    """Multi-view anomaly score per sample (high => unknown).

    `view_indices` (E2): which trained-view inspectors to aggregate over.
        F_test is shaped (N, V', feat) with V' = len(view_indices).
    `mode` (E4 ablation):
        combined  -> d_mean/s_bar + beta*var/s_bar^2 (default)
        mean_only -> d_mean/s_bar                   (single-view-style signal)
        var_only  -> var/s_bar^2                    (cross-view disagreement)

    When V'==1, var is zero -> var_only is degenerate. Reported as 0 so it is
    visibly weak in the table without crashing.
    """
    preds = np.asarray(preds)
    scores = np.zeros(len(preds))
    V_p = len(view_indices)
    for k in np.unique(preds):
        idx = np.where(preds == k)[0]
        Fk = F_test[idx].to(device)                         # (n, V', feat)
        dvs, svs = [], []
        for i, v in enumerate(view_indices):
            insp, phi, sigma = bank[(int(k), v)]
            d = torch.norm(insp(Fk[:, i]) - phi, dim=1).cpu().numpy()
            dvs.append(d); svs.append(sigma)
        dvs = np.stack(dvs, 1)                              # (n, V')
        svs = np.array(svs)
        d_mean = dvs.mean(1)
        s_bar = svs.mean() + 1e-6
        var = dvs.var(1) if V_p > 1 else np.zeros_like(d_mean)
        if mode == "mean_only":
            scores[idx] = d_mean / s_bar
        elif mode == "var_only":
            scores[idx] = var / (s_bar ** 2)
        else:                                                # combined
            scores[idx] = d_mean / s_bar + cfg.osr_beta * var / (s_bar ** 2)
    return scores


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def evaluate_inspector(cfg, model, train_ds, known_ds, unk_ds, n_known, device,
                       view_subsets=None, score_modes=None, eta_sweep=None,
                       return_scores=False):
    """Stage-2 OSR eval. By default returns one inspector head + MSP baseline.

    E2 (missing-view robustness): pass `view_subsets`, a list of
        (label, [view_indices]) tuples. Each subset re-forwards the test data
        through the same trained model with that view subset, producing its
        own metrics. The trained bank stays as is (it was learned on all
        views); we just sum/var over a subset of the (k,v) inspectors.
    E4 (score-component ablation): pass `score_modes`, a subset of
        {"combined", "mean_only", "var_only"}. Each mode produces a separate
        OSR head in the result dict.
    A7 (heuristic eta threshold, Eom Fig 10): pass `eta_sweep`, a list of
        floats. Returns per-eta acc/f1 in result["eta_sweep"] = {eta -> {acc, f1}}.
        Threshold rule: predict unknown if anomaly_score > eta. Reported only
        for the main (vname=="all"/single subset, smode=="combined") head to
        keep the table compact.

    Head names in the returned dict:
        single subset & single mode  -> "inspector", "msp"
        multi subsets, single mode   -> "insp_<sub>", "msp_<sub>"
        single subset, multi modes   -> "insp_<mode>", "msp"
        multi subsets, multi modes   -> "insp_<sub>_<mode>", "msp_<sub>"
    """
    bank, V = train_inspectors(model, train_ds, cfg, n_known, device)
    if view_subsets is None:
        view_subsets = [("all", list(range(V)))]
    if score_modes is None:
        score_modes = ["combined"]
    multi_sub = len(view_subsets) > 1
    multi_mode = len(score_modes) > 1

    results = {}
    main_closed = None
    eta_records = None
    for vname, vindices in view_subsets:
        Fk, Lk, yk = extract(model, known_ds, cfg, device, view_indices=vindices)
        Fu, Lu, _ = extract(model, unk_ds, cfg, device, view_indices=vindices)
        pred_k = Lk.argmax(1)
        pred_u = Lu.argmax(1)
        closed_acc = float((pred_k == yk).mean())
        if main_closed is None or vname == "all":
            main_closed = closed_acc

        # ---- inspector head(s) ----
        for smode in score_modes:
            sk = anomaly_scores(bank, vindices, Fk, pred_k, cfg, device, mode=smode)
            su = anomaly_scores(bank, vindices, Fu, pred_u, cfg, device, mode=smode)
            conf_k, conf_u = -sk, -su
            is_known = np.concatenate([np.ones(len(conf_k)), np.zeros(len(conf_u))])
            au = auroc(np.concatenate([conf_k, conf_u]), is_known)
            oc = oscr(conf_k, (pred_k == yk).astype(float), conf_u)
            acc, f1, thr = best_threshold_acc_f1(pred_k, conf_k, yk, conf_u,
                                                 n_known, pred_u=pred_u)
            if multi_sub and multi_mode:
                hname = f"insp_{vname}_{smode}"
            elif multi_sub:
                hname = f"insp_{vname}"
            elif multi_mode:
                hname = f"insp_{smode}"
            else:
                hname = "inspector"
            results[hname] = dict(auroc=au, oscr=oc, acc=acc, f1=f1, thr=float(thr),
                                  closed_acc=closed_acc)
            # A7: heuristic eta threshold (paper Fig 10 style). Score from
            # anomaly_scores is already roughly d/sigma; threshold at eta means
            # "more than eta sigmas from prototype -> unknown". Only computed
            # for the main head (combined, full views) to keep tables compact.
            if (eta_sweep is not None and smode == "combined"
                    and (vname == "all" or len(view_subsets) == 1)):
                per_eta = {}
                for eta in eta_sweep:
                    a, f, _ = fixed_eta_acc_f1(pred_k, sk, yk, su, n_known, eta,
                                               pred_u=pred_u)
                    per_eta[float(eta)] = dict(acc=a, f1=f)
                if eta_records is None:
                    eta_records = per_eta

        # B2 fusion hook: expose the mean_only anomaly score of the primary
        # subset so pretrain.recon_osr_heads can fuse e_recon with it.
        if return_scores and (vname == "all" or len(view_subsets) == 1):
            sk_m = anomaly_scores(bank, vindices, Fk, pred_k, cfg, device,
                                  mode="mean_only")
            su_m = anomaly_scores(bank, vindices, Fu, pred_u, cfg, device,
                                  mode="mean_only")
            results["_mean_scores"] = (sk_m, su_m)

        # ---- MSP baseline (per view subset) ----
        ck, cu = _softmax(Lk).max(1), _softmax(Lu).max(1)
        is_known = np.concatenate([np.ones(len(ck)), np.zeros(len(cu))])
        msp_name = f"msp_{vname}" if multi_sub else "msp"
        results[msp_name] = dict(
            auroc=auroc(np.concatenate([ck, cu]), is_known),
            oscr=oscr(ck, (pred_k == yk).astype(float), cu),
            **dict(zip(("acc", "f1", "thr"),
                       best_threshold_acc_f1(pred_k, ck, yk, cu, n_known,
                                             pred_u=pred_u))),
            closed_acc=closed_acc)

    results["closed_acc"] = main_closed if main_closed is not None else closed_acc
    if eta_records is not None:
        results["eta_sweep"] = eta_records
    return results
