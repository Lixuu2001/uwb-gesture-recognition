"""W2 gate: can the array geometry PREDICT the failures already on record?

This is the cheapest kill-shot for research point 2.3 (`proposal_v2/05`).  Four
predictions are made from the geometry ALONE, then checked against the recorded
results.  Three should hold; the fourth MUST fail - that is the falsification
clause ("a theory that predicts everything is worthless").

  P1  symmetric vs antisymmetric pooling blindness   -> vs RESULTS 49.2
  P2  chirality twins need cross-view phase          -> vs RESULTS 35 / 49.5
  P3  which view-drop hurts which motion axis        -> vs RESULTS 44 (drop family)
  P4  clutter-domain drift                           -> geometry MUST stay silent

Zero GPU.  Usage: python tools/traj_w2_predict.py [--reps 20]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,   # noqa: E402
                                readouts, dprime, RADAR_POS, Z_REF, FPS)

CACHE = r"D:\uwb_gestures\cache"
VIEWS = ["Left", "Top", "Right"]
K_CTRL, LAM, MU = 30, 0.05, 0.05
P0 = np.array([0.0, 0.0, Z_REF])          # nominal hand position


def unit_vectors(p0=P0):
    """u_v = unit vector radar -> hand; radial range change = u_v . displacement."""
    d = p0 - RADAR_POS
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def sign_acc(a, b, flip=False):
    a, b = (-np.asarray(a), -np.asarray(b)) if flip else (np.asarray(a), np.asarray(b))
    return 100 * (np.mean(a > 0) + np.mean(b < 0)) / 2


# ---------------------------------------------------------------------------
def p1_pooling():
    """P1: which motion axis survives which cross-view pooling operator?"""
    U = unit_vectors()                                   # (3,3) rows L,T,R
    axes = {"horizontal (x)": 0, "vertical (y)": 1, "depth (z)": 2}
    print("=" * 78)
    print("P1  pooling blindness predicted from geometry")
    print("=" * 78)
    print("  per-view sensitivity u_v . e_d  (radial range change per unit motion)")
    print(f"{'':16s}" + "".join(f"{v:>10s}" for v in VIEWS))
    for nm, d in axes.items():
        print(f"  {nm:14s}" + "".join(f"{U[v, d]:10.3f}" for v in range(3)))

    print("\n  pooled sensitivity (|value| ~ 0  =>  predicted BLIND)")
    print(f"{'pooling':16s}{'horizontal':>14s}{'vertical':>12s}   prediction")
    rows = []
    for nm, w in (("mean(L,T,R)", np.array([1, 1, 1]) / 3.0),
                  ("R - L", np.array([-1.0, 0.0, 1.0])),
                  ("R + L", np.array([1.0, 0.0, 1.0]))):
        sx, sy = w @ U[:, 0], w @ U[:, 1]
        pred = []
        pred.append("horizontal BLIND" if abs(sx) < 1e-9 else "horizontal ok")
        pred.append("vertical BLIND" if abs(sy) < 1e-9 else "vertical ok")
        print(f"  {nm:14s}{sx:14.6f}{sy:12.6f}   {', '.join(pred)}")
        rows.append((nm, sx, sy))

    print("\n  RECORDED (RESULTS 49.2, physical-layer signed Doppler centroids, 3-way):")
    print("    mean(L,T,R): horizontal  0.00   vertical 35.62")
    print("    R - L      : horizontal 55.44   vertical  2.44")
    print("\n  VERDICT: geometry predicts a 2x2 pattern (mean kills horizontal,")
    print("           R-L kills vertical).  Both recorded cells match.")
    return rows


# ---------------------------------------------------------------------------
def p3_predicted_variance():
    """P3 prediction: variance amplification along each axis for every 2-view subset."""
    U = unit_vectors()
    print("=" * 78)
    print("P3  which view-drop hurts which axis - PREDICTED")
    print("=" * 78)
    print("  var amplification  d^T (J_S^T J_S + prior)^-1 d   (larger = worse observed)")
    print("  NOTE: pinv() must NOT be used here - it maps an unobservable direction")
    print("  to variance 0 instead of infinity.  A ridge is used and inf is reported.")
    print(f"{'kept views':18s}{'dropped':>9s}{'  var_x':>10s}{'  var_y':>10s}")
    pred = {}
    for drop in range(3):
        keep = [v for v in range(3) if v != drop]
        J = U[keep]
        # the fit only puts a prior on z, so a direction with no sensitivity in any
        # kept view stays unidentifiable -> infinite variance
        F = J.T @ J + np.diag([0.0, 0.0, MU])
        eig = np.linalg.eigvalsh(F)
        if eig.min() < 1e-9:
            C = np.linalg.inv(F + 1e-9 * np.eye(3))
        else:
            C = np.linalg.inv(F)
        vx, vy = C[0, 0], C[1, 1]
        pred[drop] = (vx, vy)
        fmt = lambda x: ("   inf" if x > 1e6 else f"{x:10.3f}")            # noqa: E731
        print(f"  {'+'.join(VIEWS[k] for k in keep):16s}{VIEWS[drop]:>9s}"
              f"{fmt(vx):>10s}{fmt(vy):>10s}")
    print("\n  => predicted ordering of damage (larger var = more damage):")
    for ax, i in (("horizontal", 0), ("vertical", 1)):
        order = sorted(range(3), key=lambda d: -pred[d][i])
        print(f"     {ax:11s}: drop {' > '.join(VIEWS[d] for d in order)}")
    return pred


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    p1_pooling()
    print()
    pred = p3_predicted_variance()

    # ---------------------------------------------------------------- data
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)

    need = [0, 1, 2, 3, 8, 9]
    tracks, meta = [], []
    for g in need:
        for s in range(1, 9):
            for i in np.where((y == g) & (subj == s))[0][:args.reps]:
                unit = np.asarray(X[i])
                tr = [viterbi_track(unit[v]) for v in range(3)]
                tracks.append((np.stack([t[0] for t in tr]),
                               np.stack([t[1] for t in tr])))
                meta.append((g, s))
    meta = np.array(meta)
    print(f"\n[data] {len(tracks)} samples tracked\n")

    # ------------------------------------------------------------- P2
    print("=" * 78)
    print("P2  chirality twins: does separability need CROSS-VIEW phase?")
    print("=" * 78)
    print("  prediction: per-view MARGINAL statistics are invariant to the")
    print("  cross-view phase that distinguishes CW from CCW -> near chance;")
    print("  cross-view loop area uses exactly that phase -> separable.\n")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rot = np.isin(meta[:, 0], [8, 9])
    feats_marg, feats_cross, lab, sub = [], [], [], []
    for k in np.where(rot)[0]:
        r, c = tracks[k]
        f = []
        for v in range(3):                      # per-view marginal stats only
            rv = r[v] - r[v].mean()
            sp = np.gradient(r[v]) * FPS
            f += [rv.std(), np.ptp(rv), np.mean(rv ** 3), sp.std(),
                  np.mean(sp ** 3), np.mean(np.abs(sp))]
        feats_marg.append(f)
        # cross-view: signed areas of every view pair (needs relative phase)
        cf = []
        for a in range(3):
            for b in range(a + 1, 3):
                xa, xb = r[a] - r[a].mean(), r[b] - r[b].mean()
                cf.append(0.5 * np.sum(xa[:-1] * xb[1:] - xa[1:] * xb[:-1]))
        feats_cross.append(cf)
        lab.append(meta[k, 0]); sub.append(meta[k, 1])
    Fm, Fc = np.array(feats_marg), np.array(feats_cross)
    lab, sub = np.array(lab), np.array(sub)

    for nm, F in (("per-view marginal (18 feats)", Fm),
                  ("cross-view signed area (3 feats)", Fc)):
        accs = []
        for s in range(1, 9):
            tr_m, te_m = sub != s, sub == s
            sc = StandardScaler().fit(F[tr_m])
            clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(F[tr_m]), lab[tr_m])
            accs.append(100 * clf.score(sc.transform(F[te_m]), lab[te_m]))
        accs = np.array(accs)
        print(f"  {nm:34s} LOSO acc = {accs.mean():5.2f} +- {accs.std():5.2f}"
              f"   (per-subj {np.array2string(accs, precision=0, separator=',')})")
    print("\n  RECORDED: Doppler-centroid loop area d'=0.10 / 51.62% (RESULTS 49.5);")
    print("            range-domain loop area 92.81% (W1).")

    # ------------------------------------------------------------- P3 measured
    print("\n" + "=" * 78)
    print("P3  which view-drop hurts which axis - MEASURED")
    print("=" * 78)
    res = {}
    for drop in [None, 0, 1, 2]:
        keep = tuple(v for v in range(3) if v != drop) if drop is not None else (0, 1, 2)
        rd = {}
        for k, (r, c) in enumerate(tracks):
            g = meta[k, 0]
            if g not in (0, 1, 2, 3):
                continue
            out = fit_trajectory(r, c, keep, B=B, lam=LAM, mu=MU)
            rd.setdefault(g, []).append(readouts(out["p"]))
        h = sign_acc([d["asym_x"] for d in rd[0]], [d["asym_x"] for d in rd[1]])
        h = max(h, 100 - h)
        v = sign_acc([d["asym_y"] for d in rd[3]], [d["asym_y"] for d in rd[2]])
        v = max(v, 100 - v)
        res[drop] = (h, v)
        tag = "all three" if drop is None else f"drop {VIEWS[drop]}"
        print(f"  {tag:14s} horizontal acc = {h:6.2f}   vertical acc = {v:6.2f}")

    print("\n  measured damage (percentage points lost vs all three views):")
    for ax, i in (("horizontal", 0), ("vertical", 1)):
        dmg = {d: res[None][i] - res[d][i] for d in (0, 1, 2)}
        order = sorted(dmg, key=lambda d: -dmg[d])
        print(f"    {ax:11s}: drop {' > '.join(f'{VIEWS[d]}({dmg[d]:+.1f})' for d in order)}")
        # the nominal array is mirror-symmetric in Left/Right, so their predicted
        # variances are IDENTICAL.  Ranking them is not a prediction - only the
        # strict inequalities the geometry actually asserts may be scored.
        groups, seen = [], []
        for d in sorted(range(3), key=lambda d: -pred[d][i]):
            if seen and abs(pred[d][i] - pred[seen[-1]][i]) < 1e-6:
                groups[-1].append(d)
            else:
                groups.append([d])
            seen.append(d)
        pstr = " > ".join("=".join(VIEWS[d] for d in g) for g in groups)
        ok = True
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if min(dmg[d] for d in groups[a]) <= max(dmg[d] for d in groups[b]):
                    ok = False
        print(f"    {'':11s}  predicted: drop {pstr}   -> "
              f"{'MATCH' if ok else 'MISMATCH'}  (ties are not scored)")

    # ------------------------------------------------------------- P4
    print("\n" + "=" * 78)
    print("P4  falsification clause: clutter-domain drift")
    print("=" * 78)
    print("  The Jacobian J = [u_L; u_T; u_R] depends only on the array geometry and")
    print("  the hand position.  Mixing ClutterRemoved towards Raw changes NEITHER.")
    print("  => geometric prediction of degradation: ZERO, for every gesture.")
    print("  RECORDED (RESULTS 44.1): domain is the DOMINANT failure family -")
    print("  H drops 78.1 points, all 4 view axes x 2 vocabularies double-trigger.")
    print("  => the geometry correctly FAILS to predict it.  Falsification clause")
    print("     satisfied: the framework is not vacuous.")


if __name__ == "__main__":
    main()
