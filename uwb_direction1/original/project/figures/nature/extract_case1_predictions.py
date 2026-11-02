"""Re-train case 1 with the project's standard pipeline and save the
per-sample predictions / OSR scores / ground-truth labels needed for the
Eom-style confusion-matrix figure (Fig. 9 / 10 of Eom et al. 2025).

Run twice -- once for single-view (--single), once for three-view
(default). Output files:

    results/case1_1view.npz
    results/case1_3view.npz

with keys:
    y_known            (N_k,)   true gesture labels (in [0, n_known))
    pred_known         (N_k,)   closed-set predictions
    conf_known         (N_k,)   inspector confidence (-anomaly score)
    msp_known          (N_k,)   MSP confidence for the predicted class
    y_unknown          (N_u,)   true gesture labels (original 0..11)
    pred_unknown       (N_u,)   closed-set predictions (forced into known)
    conf_unknown       (N_u,)   inspector confidence
    msp_unknown        (N_u,)   MSP confidence
    known_classes      (n_known,)  original class IDs for the known gestures
    case_idx           ()   the case index (here 0 = Case 1)
    acc, f1, auroc, oscr, thr_best  scalar per-case metrics (inspector head)

These files are then read by fig5_confmat.py to draw confusion matrices at
two thresholds (best-acc and 3-sigma-like) -- exactly the layout of Eom
Fig. 9 / 10.
"""
import argparse
import os
import sys

import numpy as np
import torch

# allow `from uwb_osr...` imports while running from this folder
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# this file lives at D:/uwb_gestures/figures/nature/, ROOT = D:/uwb_gestures
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # so the data loader finds cache/X*.npy with its relative path

from config import Config, OSR_TEST_CASES, VIEW_ORDER         # noqa: E402
from uwb_osr.data import load_cache, MVDataset, osr_split     # noqa: E402
from uwb_osr.engine import train_one_case, get_device         # noqa: E402
from uwb_osr.inspector import (train_inspectors, anomaly_scores,
                                extract)                       # noqa: E402
from uwb_osr.metrics import auroc, oscr, best_threshold_acc_f1  # noqa: E402


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="single-view (Left only) instead of three-view")
    ap.add_argument("--case", type=int, default=0, help="case index 0..4")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "1view" if args.single else "3view"
    out_path = os.path.join(args.out_dir, f"case{args.case + 1}_{tag}.npz")
    print(f"== extract_case1_predictions ==  out={out_path}")

    cfg = Config(seed=args.seed)
    if args.single:
        cfg.views = ["Left"]
    cfg.pool = "flatten"
    cfg.normalize = "abs_std"
    cfg.osr_head = "inspector"
    cfg.epochs = args.epochs
    cfg.batch_size_insp = 256

    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = get_device(cfg)
    print(f"device={device}; views={cfg.views}; pool={cfg.pool}; "
          f"norm={cfg.normalize}; epochs={cfg.epochs}")

    X, y, _ = load_cache(cfg, mmap=False)
    print(f"[data] X{X.shape} y{y.shape}")
    case = OSR_TEST_CASES[args.case]
    print(f"--- Case {args.case+1}: known={case['known']}  "
          f"unknown={case['unknown']} ---")

    sp = osr_split(cfg, y, case["known"], case["unknown"], cfg.seed)
    view_idx = [VIEW_ORDER.index(v) for v in cfg.views]
    nrm = cfg.normalize
    train_ds = MVDataset(X, y, sp["train_idx"], view_idx,
                          sp["label_map"], nrm)
    known_ds = MVDataset(X, y, sp["test_known_idx"], view_idx,
                          sp["label_map"], nrm)
    unk_ds = MVDataset(X, y, sp["test_unknown_idx"], view_idx, None, nrm)

    model = train_one_case(cfg, train_ds, sp["n_known"], device, verbose=True,
                            desc=f"case{args.case+1} Stage-1")
    bank, V = train_inspectors(model, train_ds, cfg, sp["n_known"], device)

    Fk, Lk, yk = extract(model, known_ds, cfg, device)
    Fu, Lu, _  = extract(model, unk_ds, cfg, device)
    pred_k = Lk.argmax(1)
    pred_u = Lu.argmax(1)
    closed_acc = float((pred_k == yk).mean())

    sk = anomaly_scores(bank, V, Fk, pred_k, cfg, device)
    su = anomaly_scores(bank, V, Fu, pred_u, cfg, device)
    conf_k, conf_u = -sk, -su
    is_known = np.concatenate([np.ones(len(conf_k)),
                                np.zeros(len(conf_u))])
    au = auroc(np.concatenate([conf_k, conf_u]), is_known)
    oc = oscr(conf_k, (pred_k == yk).astype(float), conf_u)
    acc, f1, thr = best_threshold_acc_f1(pred_k, conf_k, yk, conf_u,
                                          sp["n_known"])

    # MSP confidences too -- only the max-prob across known classes.
    msp_k = softmax_np(Lk).max(1)
    msp_u = softmax_np(Lu).max(1)

    # The dataset stores original 0..11 IDs in y for unknown samples; map
    # known IDs through label_map (case-specific 0..n_known-1).
    inv_map = {v: k for k, v in sp["label_map"].items()}
    known_classes = np.array(sorted(inv_map.keys(),
                                     key=lambda k: inv_map[k]))
    y_unknown_orig = y[sp["test_unknown_idx"]]

    np.savez(out_path,
             y_known=yk,
             pred_known=pred_k,
             conf_known=conf_k,
             msp_known=msp_k,
             y_unknown=y_unknown_orig,
             pred_unknown=pred_u,
             conf_unknown=conf_u,
             msp_unknown=msp_u,
             known_classes=known_classes,
             unknown_classes=np.array(case["unknown"]),
             case_idx=np.array(args.case),
             closed_acc=np.array(closed_acc),
             acc=np.array(acc),
             f1=np.array(f1),
             auroc=np.array(au),
             oscr=np.array(oc),
             thr_best=np.array(thr),
             view_tag=np.array(tag))
    print(f"saved {out_path}")
    print(f"closed_acc={closed_acc*100:.2f}  "
           f"INSPECTOR oscr={oc*100:.2f} auroc={au*100:.2f}  "
           f"acc={acc*100:.2f} f1={f1*100:.2f} thr={thr:.4f}")


if __name__ == "__main__":
    main()
