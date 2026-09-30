"""Single-radar observability pilot (2026-09-02).

Question 1 (geometry -> recognisability).  A monostatic radar at P_v only measures
r_v(t) = ||p(t) - P_v||.  With the nominal UWB-Gestures geometry
(Left (-0.55,0,0), Top (0,0.55,0), Right (0.55,0,0), hand plane z ~ 0.45 m):
  * Left/Right: r^2 = (x -/+ 0.55)^2 + y^2 + z^2  -> even in y  -> any two gestures
    that differ only by a sign flip of y(t) give IDENTICAL range tracks:
        G3<->G4 (up-down / down-up), G5<->G6, G7<->G8.
  * Top: r^2 = x^2 + (y-0.55)^2 + z^2 -> even in x -> G1<->G2, G5<->G7, G6<->G8.
  * Circles: CW/CCW collapse for a view iff the start point lies on that view's
    symmetry axis (start at 12 o'clock -> Top blind, Left/Right not).
Prediction: cross-subject errors of a range-only single-view classifier concentrate
on these pairs, far above the 1/11 uniform-error baseline.

Question 2 (privileged trajectory -> single-view student).  Generalised distillation
(Lopez-Paz et al. 2016): teacher sees the 3-view-fitted 3-D trajectory (privileged,
training only); student sees one view's range track.  Does matching the teacher's
softened posterior improve LOSO accuracy over hard labels, and where (ambiguous pairs
vs. unambiguous classes)?

Everything here is a WEAK-FEATURE pilot (multinomial logistic regression, L-BFGS,
no GPU).  Numbers bound the mechanism, they are not the headline of any paper.
Run:  python proposal_v2/pilot/sv_observability_pilot.py  [--reps 20]
"""
import argparse
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, ".")
from uwb_osr.trajectory import viterbi_track, fit_trajectory, bspline_basis  # noqa: E402

NAMES = ["G1 L>R", "G2 R>L", "G3 U>D", "G4 D>U", "G5 LR-UD", "G6 LR-DU",
         "G7 RL-UD", "G8 RL-DU", "G9 CW", "G10 CCW", "G11 push", "G12 empty"]
# predicted blind pairs per view (0-based class ids); circles listed separately
PAIRS = {0: [(2, 3), (4, 5), (6, 7)],          # Left  : even in y
         1: [(0, 1), (4, 6), (5, 7)],          # Top   : even in x
         2: [(2, 3), (4, 5), (6, 7)]}          # Right : even in y
CIRC = (8, 9)
VIEWS = ["Left", "Top", "Right"]


# ---------------------------------------------------------------------------
def softmax_fit(X, Q, l2=1e-2, W0=None):
    """Multinomial logistic regression with SOFT targets Q (n,C); returns (W,b)."""
    n, d = X.shape
    C = Q.shape[1]
    Xb = np.hstack([X, np.ones((n, 1))])

    def f(theta):
        W = theta.reshape(d + 1, C)
        Z = Xb @ W
        lse = logsumexp(Z, axis=1, keepdims=True)
        P = np.exp(Z - lse)
        loss = -(Q * (Z - lse)).sum() / n + 0.5 * l2 * (W[:-1] ** 2).sum()
        G = Xb.T @ (P - Q) / n
        G[:-1] += l2 * W[:-1]
        return loss, G.ravel()

    x0 = np.zeros((d + 1) * C) if W0 is None else W0.ravel()
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x.reshape(d + 1, C)


def predict_logits(W, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ W


def standardise(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def onehot(y, C=12):
    Q = np.zeros((len(y), C))
    Q[np.arange(len(y)), y] = 1.0
    return Q


# ---------------------------------------------------------------------------
def build_features(reps, seed=0):
    X = np.load("cache/X_clutterremoved.npy", mmap_mode="r")
    y = np.load("cache/y_clutterremoved.npy")
    s = np.load("cache/subj_clutterremoved.npy")
    idx = []
    for sub in np.unique(s):
        for g in range(12):
            ii = np.where((s == sub) & (y == g))[0][:reps]
            idx.extend(ii.tolist())
    idx = np.array(idx)
    T = X.shape[2]
    B = bspline_basis(T=T, K=30)
    R = np.zeros((len(idx), 3, T))          # range tracks (m)
    A = np.zeros((len(idx), 3, T))          # echo-strength confidence
    P = np.zeros((len(idx), T, 3))          # fitted trajectory (m)
    t0 = time.time()
    for k, i in enumerate(idx):
        fr = np.asarray(X[i])               # (3, T, M)
        for v in range(3):
            r, c = viterbi_track(fr[v])
            R[k, v], A[k, v] = r, c
        P[k] = fit_trajectory(R[k], A[k], B=B, lam=0.05, mu=0.05)["p"]
        if k % 400 == 0:
            print(f"  feat {k}/{len(idx)}  {time.time()-t0:.0f}s", flush=True)
    return R, A, P, y[idx], s[idx]


def feats_single(R, v):
    r = R[:, v] - R[:, v].mean(1, keepdims=True)         # centred range track
    dr = np.diff(R[:, v], axis=1)                         # range rate
    return np.hstack([r, dr])


def feats_traj(P):
    p = P - P.mean(1, keepdims=True)                      # centred (T,3)
    dp = np.diff(P, axis=1)
    return np.hstack([p.reshape(len(P), -1), dp.reshape(len(P), -1)])


# ---------------------------------------------------------------------------
def loso(Xf, y, s, teacher_X=None, lam=0.0, temp=1.0, l2=1e-2):
    """Return predictions (n,) under leave-one-subject-out.
    If teacher_X is given, a teacher is trained on it in-fold and the student
    minimises (1-lam)*CE(hard) + lam*CE(teacher soft @ temp)."""
    pred = np.zeros(len(y), dtype=int)
    for sub in np.unique(s):
        tr, te = s != sub, s == sub
        Xtr, Xte = standardise(Xf[tr], Xf[te])
        Q = onehot(y[tr])
        if teacher_X is not None and lam > 0:
            Ttr, _ = standardise(teacher_X[tr], teacher_X[te])
            Wt = softmax_fit(Ttr, onehot(y[tr]), l2=l2)
            Zt = predict_logits(Wt, Ttr) / temp
            St = np.exp(Zt - logsumexp(Zt, axis=1, keepdims=True))
            Q = (1 - lam) * Q + lam * St
        W = softmax_fit(Xtr, Q, l2=l2)
        pred[te] = predict_logits(W, Xte).argmax(1)
    return pred


def pair_stats(pred, y, pairs):
    """Fraction of errors that land on the predicted blind partner."""
    err = pred != y
    n_err = err.sum()
    partner = {}
    for a, b in pairs:
        partner[a], partner[b] = b, a
    in_pair = sum(1 for i in np.where(err)[0] if y[i] in partner and pred[i] == partner[y[i]])
    covered = np.isin(y, list(partner)) & err
    return n_err, in_pair, covered.sum()


def per_class_acc(pred, y):
    return np.array([(pred[y == c] == c).mean() for c in range(12)])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="logs/sv_observability_pilot.npz")
    args = ap.parse_args()

    print("[1/3] Viterbi range tracks + 3-view trajectory fit ...", flush=True)
    R, A, P, y, s = build_features(args.reps)
    print(f"      n={len(y)}  subjects={np.unique(s).tolist()}")

    Ftraj = feats_traj(P)
    F3 = np.hstack([feats_single(R, v) for v in range(3)])
    out = {"y": y, "s": s}

    print("[2/3] LOSO, hard labels ------------------------------------------")
    pred_traj = loso(Ftraj, y, s)
    pred_3v = loso(F3, y, s)
    print(f"  3-view fitted trajectory : acc {100*(pred_traj==y).mean():.1f}")
    print(f"  3-view range tracks      : acc {100*(pred_3v==y).mean():.1f}")
    out["pred_traj"], out["pred_3v"] = pred_traj, pred_3v

    preds_hard = {}
    for v in range(3):
        pv = loso(feats_single(R, v), y, s)
        preds_hard[v] = pv
        n_err, in_pair, cov = pair_stats(pv, y, PAIRS[v])
        pc = per_class_acc(pv, y)
        circ = int(((y == 8) & (pv == 9)).sum() + ((y == 9) & (pv == 8)).sum())
        print(f"  {VIEWS[v]:5s} single view      : acc {100*(pv==y).mean():.1f} | "
              f"errors on predicted blind partner {in_pair}/{cov} "
              f"= {100*in_pair/max(cov,1):.1f}% of errors of covered classes "
              f"(uniform baseline 9.1%) | CW<->CCW confusions {circ}")
        print("        per-class acc: " + " ".join(f"{NAMES[c]}={100*pc[c]:.0f}" for c in range(12)))
        out[f"pred_hard_{v}"] = pv

    print("[3/3] LOSO, generalised distillation from the trajectory teacher ---")
    for v in range(3):
        base = preds_hard[v]
        line = []
        for lam, temp in [(0.5, 2.0), (1.0, 2.0), (1.0, 5.0)]:
            pv = loso(feats_single(R, v), y, s, teacher_X=Ftraj, lam=lam, temp=temp)
            out[f"pred_kd_{v}_l{lam}_t{temp}"] = pv
            pc_b, pc_k = per_class_acc(base, y), per_class_acc(pv, y)
            blind = sorted({c for pr in PAIRS[v] for c in pr})
            clear = [c for c in range(12) if c not in blind]
            line.append(f"lam={lam},T={temp}: acc {100*(pv==y).mean():.1f} "
                        f"(hard {100*(base==y).mean():.1f}) | "
                        f"blind-pair classes {100*pc_b[blind].mean():.1f}->{100*pc_k[blind].mean():.1f}, "
                        f"other classes {100*pc_b[clear].mean():.1f}->{100*pc_k[clear].mean():.1f}")
        print(f"  {VIEWS[v]:5s}: " + "\n         ".join(line))

    np.savez(args.out, **out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
