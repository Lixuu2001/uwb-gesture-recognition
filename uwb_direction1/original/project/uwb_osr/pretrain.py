"""B2 Stage-0: cross-view masked-reconstruction pretraining.

Mask ONE radar view per batch, encode the visible views with the shared
dual-path encoder, and reconstruct the masked view from them. To do this the
encoder must learn the geometry that ties the three views to one 3D hand motion
-- exactly what the additive physical bias (E5) failed to capture.

Two products, both opt-in and reported honestly (lambda_recon_osr defaults 0):
  1. a pretrained encoder state_dict that initialises Stage-1 (init-only benefit);
  2. a reconstruction-error OOD signal e_recon: at test time we mask each view,
     reconstruct it from the others, and measure the error. Known gestures stay
     geometrically self-consistent (low error); unknown gestures do not (high).

Open-set hygiene: pretraining sees train_ds only (known-train), never unknown
classes or test samples, so e_recon stays a valid OSR signal.

Caveat (mae_target='token'): the token target is produced by a stop-grad copy of
the *training* encoder (no EMA in v1), so it carries a representation-collapse
risk. 'raw' (predict the echo cube) is the grounded default.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import SharedDualPathEncoder
from .metrics import auroc, oscr, best_threshold_acc_f1


class MaskedViewDecoder(nn.Module):
    """Reconstruct one masked view from the visible views' temporal tokens.

    memory = visible views' temporal tokens (B, Vvis*T, D)
    output = (B, T, out_dim) where out_dim = M (raw echo) or D (token target).
    """

    def __init__(self, cfg):
        super().__init__()
        self.target = cfg.mae_target
        out_dim = cfg.M if self.target == "raw" else cfg.dim
        self.query = nn.Parameter(torch.randn(1, cfg.T, cfg.dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.dim, nhead=cfg.heads,
            dim_feedforward=int(cfg.dim * cfg.mlp_ratio), dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=2)
        self.head = nn.Linear(cfg.dim, out_dim)

    def forward(self, memory):                       # memory: (B, Vvis*T, D)
        q = self.query.expand(memory.size(0), -1, -1)
        return self.head(self.dec(q, memory))        # (B, T, out_dim)


def _target_for(enc, x_masked_view, cfg, slots=None):
    """Reconstruction target for the masked view. x_masked_view: (B,1,T,M).
    `slots`: trained-view slot of the masked view (matters for the F2 xmv
    encoder, whose token-level view embedding depends on the slot)."""
    if cfg.mae_target == "raw":
        return x_masked_view[:, 0]                   # (B,T,M)
    with torch.no_grad():                            # stop-grad token target
        _, t_tok, _ = enc(x_masked_view, view_indices=slots)
    return t_tok[:, 0]                               # (B,T,D)


def _encode_visible(enc, x_vis, emb_slots):
    """Encode already-sliced visible views x_vis (B,Vvis,T,M). `emb_slots` are the
    trained-view rows of view_emb that x_vis corresponds to. Returns memory
    (B, Vvis*T, D)."""
    _, t_tok, _ = enc(x_vis, view_indices=emb_slots)         # (B,Vvis,T,D)
    B, Vvis, T, D = t_tok.shape
    return t_tok.reshape(B, Vvis * T, D)


def pretrain_encoder(cfg, train_ds, device, verbose=True):
    """Stage-0 pretrain. Returns (encoder_state_dict, encoder, decoder).

    encoder/decoder are returned frozen-ready for the e_recon OOD signal; the
    state_dict is loaded into the (separately fine-tuned) Stage-1 model.
    """
    V = cfg.n_views
    enc = SharedDualPathEncoder(cfg).to(device)
    if V < 2:                                        # single view: nothing to mask
        return enc.state_dict(), enc, None
    dec = MaskedViewDecoder(cfg).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    pin = device == "cuda"
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=pin)
    pbar = tqdm(range(cfg.mae_epochs), desc="Stage-0 MAE", disable=not verbose,
                dynamic_ncols=True)
    for _ in pbar:
        enc.train(); dec.train()
        running = 0.0
        for batch in loader:                         # tolerate the 3-tuple that
            x = batch[0].to(device, non_blocking=pin)  # a subj-adv train_ds yields
            vstar = int(torch.randint(0, V, (1,)).item())
            vis = [i for i in range(V) if i != vstar]
            mem = _encode_visible(enc, x[:, vis], vis)
            pred = dec(mem)
            tgt = _target_for(enc, x[:, [vstar]], cfg, slots=[vstar])
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        pbar.set_postfix(mse=f"{running/len(loader):.4f}")
    return enc.state_dict(), enc, dec


@torch.no_grad()
def recon_error(enc, dec, ds, cfg, device, view_indices=None,
                per_view=False):
    """Per-sample cross-view reconstruction error (high => unknown).

    For each present view, mask it, reconstruct from the others, MSE; average
    over the masked views. `view_indices` are the trained-view slots the data
    occupies (E2). Default = all trained views.

    per_view=True returns the raw per-masked-view MSE matrix (N, Vp) instead
    of the mean -- the m_v of the G3 reliability definition (c-lite
    pre-registration 2026-07-19). The default keeps the historical mean
    interface bit-identical for every existing caller.
    """
    enc.eval(); dec.eval()
    if view_indices is None:
        view_indices = list(range(cfg.n_views))
    Vp = len(view_indices)
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=cfg.num_workers)
    out = []
    for x, _ in loader:
        x = x.to(device)                             # (B, Vp, T, M)
        pv = []
        for j, v in enumerate(view_indices):
            vis_local = [k for k in range(Vp) if k != j]      # positions in x
            vis_slots = [view_indices[k] for k in vis_local]  # trained-view slots
            mem = _encode_visible(enc, x[:, vis_local], vis_slots)
            pred = dec(mem)
            tgt = _target_for(enc, x[:, [j]], cfg, slots=[view_indices[j]])
            pv.append(((pred - tgt) ** 2).mean(dim=(1, 2)).cpu().numpy())
        out.append(np.stack(pv, 1))                  # (B, Vp) raw m_v
    M = np.concatenate(out)
    return M if per_view else M.mean(1)


@torch.no_grad()
def _logits(model, ds, cfg, device):
    model.eval()
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=cfg.num_workers)
    L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(device))["logits"].cpu())
        Y.append(y)
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def _zscore(ref, *arrs):
    """Standardise each array by the mean/std of `ref` (the known set)."""
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-8)
    return [(a - mu) / sd for a in arrs]


def recon_osr_heads(model, enc, dec, known_ds, unk_ds, n_known, cfg, device,
                    insp_scores=None):
    """Build the e_recon OSR head(s) + the known/unknown early-screen number.

    Returns (heads_dict, screen_str). insp_scores, if given, is
    (sk_anom, su_anom) of the inspector mean_only anomaly score (high=unknown);
    used to build fused heads at swept lambda (only when lambda_recon_osr>0).
    """
    if dec is None:
        return {}, "recon: skipped (single view)"
    Lk, yk = _logits(model, known_ds, cfg, device)
    Lu, _ = _logits(model, unk_ds, cfg, device)
    pred_k = Lk.argmax(1)
    correct_k = (pred_k == yk).astype(float)

    ek = recon_error(enc, dec, known_ds, cfg, device)     # anomaly (high=unknown)
    eu = recon_error(enc, dec, unk_ds, cfg, device)

    heads = {}

    def _head(anom_k, anom_u):
        conf_k, conf_u = -anom_k, -anom_u
        is_known = np.concatenate([np.ones(len(conf_k)), np.zeros(len(conf_u))])
        au = auroc(np.concatenate([conf_k, conf_u]), is_known)
        oc = oscr(conf_k, correct_k, conf_u)
        acc, f1, thr = best_threshold_acc_f1(pred_k, conf_k, yk, conf_u, n_known)
        return dict(auroc=au, oscr=oc, acc=acc, f1=f1, thr=float(thr),
                    closed_acc=float(correct_k.mean()))

    heads["recon"] = _head(ek, eu)                        # standalone e_recon

    # Early-screen: AUROC of recon error alone == "is there any known/unknown
    # separation". If this is near 0.5, e_recon is hopeless before full OSR.
    screen = (f"recon early-screen  AUROC={heads['recon']['auroc']*100:5.2f}  "
              f"OSCR={heads['recon']['oscr']*100:5.2f}  "
              f"(known MSE={ek.mean():.4g}  unknown MSE={eu.mean():.4g})")

    # Opt-in fusion with inspector mean_only score (z-normalised, swept lambda).
    if insp_scores is not None and cfg.lambda_recon_osr > 0:
        sk, su = insp_scores                              # inspector anomaly
        ekz, euz = _zscore(ek, ek, eu)
        skz, suz = _zscore(sk, sk, su)
        for lam in (0.3, 1.0, 3.0):
            heads[f"recon+mean_l{lam}"] = _head(skz + lam * ekz, suz + lam * euz)
    return heads, screen
