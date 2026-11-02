"""S1 -- attribute linear-probe early screen (compositional pillar go/no-go).

Question: does the learned gesture representation encode the two motion axes
(horizontal, vertical) as *separable, composable* attributes? If so, a linear
probe trained only on single-axis swipes should still read BOTH axes correctly
on the diagonal swipes it has NEVER seen -- the standard compositional
zero-shot (CZSL) test (RESEARCH_PLAN.md S1 / paper-2 pillar two).

Protocol
--------
* Encoder: a normal closed-set MultiViewOSRNet trained on 8 SEED classes
  (4 orthogonal swipes + 2 rotations + push + empty). The 4 DIAGONAL swipes
  are held out from the encoder entirely (strict compositional split).
* Probe: freeze c_fused, train two linear heads (horizontal 3-way, vertical
  3-way) on the orthogonal-swipe TRAIN split, where each swipe has exactly one
  active axis. Test on:
    - in-dist sanity : orthogonal-swipe TEST split (held-out repetitions);
    - the screen     : the 4 diagonals (unseen combinations) -- here the true
                       label is NEVER "none", so the probe must detect the
                       component it only ever saw in isolation.

go criterion: diagonal per-axis accuracy clearly above chance (>70%); means the
representation composes -> paper 2 pillar two is alive.

Run
---
  python probe_attributes.py                       # real data, paper config
  python probe_attributes.py --smoke --device cpu  # pipeline smoke (synthetic;
                                                     attributes are meaningless,
                                                     only checks it runs)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, VIEW_ORDER
from uwb_osr.data import load_cache, make_synthetic, MVDataset, osr_split
from uwb_osr.engine import train_one_case, get_device
from uwb_osr.model import MultiViewOSRNet
from uwb_osr.pretrain import pretrain_encoder

# Gesture index (0..11) -> motion attributes. Indices follow config.GESTURE_NAMES
# (== G-number - 1). H/V codes: 0/1 are the two directions, 2 = "no motion on
# this axis". Rotations/push/empty carry no clean swipe axis -> not probed.
SEED = [0, 1, 2, 3, 8, 9, 10, 11]      # G1-4 swipes + G9,G10 rot + G11 push + G12 empty
DIAG = [4, 5, 6, 7]                    # G5-G8 diagonal swipes = unseen combinations
ORTHO = [0, 1, 2, 3]                   # single-axis swipes (probe is trained here)

# horizontal: LR=0 (left->right), RL=1 (right->left), none=2
H_ATTR = {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 0, 6: 1, 7: 1}
# vertical: UD=0 (up->down), DU=1 (down->up), none=2
V_ATTR = {0: 2, 1: 2, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}
H_NAME = {0: "L->R", 1: "R->L", 2: "none"}
V_NAME = {0: "U->D", 1: "D->U", 2: "none"}


@torch.no_grad()
def extract_feats(model, X, y, idx, cfg, device):
    """Frozen features for the given indices. Returns:
      cfused : (N, feat)        the multi-view fused closed-set readout
      fviews : (N, V, feat)     per-view feature codes (before fusion)
    The per-view set lets us tell a real compositional failure apart from a
    fusion-cancellation artifact (e.g. horizontal direction sign-flips across
    L/R radars and may average out in cfused but survive in a single view)."""
    view_idx = [VIEW_ORDER.index(v) for v in cfg.views]
    ds = MVDataset(X, y, idx, view_idx, label_map=None, normalize=cfg.normalize)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=cfg.num_workers)
    model.eval()
    cf, fv = [], []
    for x, _ in loader:
        out = model(x.to(device))
        cf.append(out["c_fused"].cpu())
        fv.append(out["f_views"].cpu())
    return torch.cat(cf).numpy(), torch.cat(fv).numpy()


def train_linear_probe(Xtr, ytr, n_cls, device, epochs=300, lr=1e-3, wd=1e-3):
    """A plain linear head on frozen features (logistic regression via Adam)."""
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    head = nn.Linear(Xt.size(1), n_cls).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(Xt), yt)
        loss.backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def probe_acc(head, Xte, yte, device):
    pred = head(torch.tensor(Xte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    return float((pred == yte).mean()), pred


def attrs_for(y_subset):
    h = np.array([H_ATTR[int(g)] for g in y_subset])
    v = np.array([V_ATTR[int(g)] for g in y_subset])
    return h, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--encoder", choices=["closed", "b2"], default="closed",
                    help=("how the probed encoder is trained: 'closed' = "
                          "discriminative closed-set CE on the 8 seed classes "
                          "(S1 baseline); 'b2' = self-supervised cross-view "
                          "masked reconstruction, NO labels (S2 -- tests whether "
                          "the generative objective composes the horizontal axis "
                          "the discriminative encoder failed at)."))
    ap.add_argument("--mae-epochs", type=int, default=-1,
                    help="B2 Stage-0 pretrain epochs (default from config, 100)")
    ap.add_argument("--mae-target", choices=["raw", "token"], default=None,
                    help="B2 reconstruction target (default raw, no collapse)")
    ap.add_argument("--epochs", type=int, default=-1, help="encoder Stage-1 epochs")
    ap.add_argument("--pool", choices=["mean", "flatten"], default="flatten")
    ap.add_argument("--normalize",
                    choices=["none", "abs", "per_view_std", "abs_std"],
                    default="abs_std")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(seed=args.seed)
    cfg.pool = args.pool
    cfg.normalize = args.normalize
    # Probe pure encoder features (no cross-view fusion module), so the only
    # variable between encoder=closed and encoder=b2 is the training objective.
    cfg.use_fusion = False
    if args.device:
        cfg.device = args.device
    if args.epochs > 0:
        cfg.epochs = args.epochs
    if args.mae_epochs > 0:
        cfg.mae_epochs = args.mae_epochs
    if args.mae_target:
        cfg.mae_target = args.mae_target

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = get_device(cfg)
    tag = "S2/b2" if args.encoder == "b2" else "S1/closed"
    print(f"[{tag}] device={device} pool={cfg.pool} normalize={cfg.normalize} "
          f"encoder={args.encoder} epochs={cfg.epochs} "
          f"mae_epochs={cfg.mae_epochs} mae_target={cfg.mae_target}")

    if args.smoke:
        cfg.train_per_known, cfg.test_per_known = 50, 20
        cfg.n_unknown_test = 200
        if args.epochs <= 0:
            cfg.epochs = 3
        cfg.mae_epochs = 3
        X, y, _ = make_synthetic(cfg, n_per_class=80, seed=cfg.seed)
        print(f"[smoke] synthetic X{X.shape} (attributes meaningless here)")
    else:
        X, y, _ = load_cache(cfg, mmap=False)
        print(f"[data] X{X.shape} y{y.shape}")

    # --- 1) train the encoder on the 8 SEED classes (diagonals fully held out).
    # osr_split: SEED as "known", DIAG as "unknown" -> no diagonal ever seen.
    sp = osr_split(cfg, y, known=SEED, unknown=DIAG, seed=cfg.seed)
    view_idx = [VIEW_ORDER.index(v) for v in cfg.views]
    train_ds = MVDataset(X, y, sp["train_idx"], view_idx, sp["label_map"], cfg.normalize)
    print(f"[{tag}] training encoder on {len(SEED)} seed classes "
          f"(train units={len(sp['train_idx'])}), diagonals fully held out")
    if args.encoder == "b2":
        # Self-supervised cross-view masked reconstruction (NO labels). The
        # resulting encoder is loaded into a fusion-free model purely to read
        # out c_fused / f_views for probing.
        mae_state, _, _ = pretrain_encoder(cfg, train_ds, device, verbose=True)
        model = MultiViewOSRNet(cfg, len(SEED)).to(device)
        model.encoder.load_state_dict(mae_state)
    else:
        model = train_one_case(cfg, train_ds, sp["n_known"], device,
                               desc=f"{tag} encoder")

    # --- 2) build probe-train / probe-test index sets
    # probe-train: orthogonal-swipe samples the encoder saw (its train split)
    tr_idx = sp["train_idx"]
    ortho_tr = tr_idx[np.isin(y[tr_idx], ORTHO)]
    # in-dist sanity: orthogonal-swipe TEST split (held-out repetitions)
    te_idx = sp["test_known_idx"]
    ortho_te = te_idx[np.isin(y[te_idx], ORTHO)]
    # the screen: all diagonal samples (encoder never saw any)
    diag_idx = np.where(np.isin(y, DIAG))[0]

    print(f"[S1] probe-train(ortho seen)={len(ortho_tr)}  "
          f"sanity(ortho held-out reps)={len(ortho_te)}  "
          f"screen(diagonals)={len(diag_idx)}")

    # --- 3) extract frozen features (fused + per-view)
    CFtr, FVtr = extract_feats(model, X, y, ortho_tr, cfg, device)
    CFsan, FVsan = extract_feats(model, X, y, ortho_te, cfg, device)
    CFdia, FVdia = extract_feats(model, X, y, diag_idx, cfg, device)

    htr, vtr = attrs_for(y[ortho_tr])
    hsan, vsan = attrs_for(y[ortho_te])
    hdia, vdia = attrs_for(y[diag_idx])

    V = FVtr.shape[1]
    # feature sources to probe: the fused readout, then each single view.
    sources = [("fused", CFtr, CFsan, CFdia)]
    for v in range(V):
        sources.append((f"view-{cfg.views[v]}", FVtr[:, v], FVsan[:, v], FVdia[:, v]))

    # --- 4) train + eval the two linear probes on every feature source
    print("\n================ S1 attribute probe ================")
    print("(diagonals = unseen combinations; >70% per axis => that axis composes)\n")
    summary = {}
    for src_name, Ftr, Fsan, Fdia in sources:
        print(f"---- feature source: {src_name} ----")
        for name, ytr_a, ysan_a, ydia_a, namemap in [
            ("horizontal", htr, hsan, hdia, H_NAME),
            ("vertical",   vtr, vsan, vdia, V_NAME),
        ]:
            head = train_linear_probe(Ftr, ytr_a, 3, device)
            a_tr, _ = probe_acc(head, Ftr, ytr_a, device)
            a_san, _ = probe_acc(head, Fsan, ysan_a, device)
            a_dia, pred_dia = probe_acc(head, Fdia, ydia_a, device)
            summary[(src_name, name)] = a_dia
            print(f"  [{name:10s}] train={a_tr*100:5.2f} sanity={a_san*100:5.2f} "
                  f"** diag={a_dia*100:5.2f} **")
            for g in DIAG:
                m = y[diag_idx] == g
                if m.sum() == 0:
                    continue
                true = namemap[int(ydia_a[m][0])]
                pr = pred_dia[m]
                vals, cnts = np.unique(pr, return_counts=True)
                top = namemap[int(vals[cnts.argmax()])]
                acc = float((pr == ydia_a[m]).mean())
                print(f"      G{g+1}(true {true:5s}): mode={top:5s} acc={acc*100:5.1f}")
        print()

    print("================ summary: diagonal acc (%) ================")
    print(f"{'source':14s} {'horizontal':>12s} {'vertical':>12s}")
    for src_name, *_ in sources:
        print(f"{src_name:14s} {summary[(src_name,'horizontal')]*100:>11.2f} "
              f"{summary[(src_name,'vertical')]*100:>11.2f}")
    print("\n[read] horizontal high in a single view but low in fused => fusion "
          "cancels the sign-flipped horizontal direction (artifact, fixable).\n"
          "       horizontal low everywhere => discriminative encoder genuinely "
          "fails to compose it (root cause -> generative spine should help).")


if __name__ == "__main__":
    main()
