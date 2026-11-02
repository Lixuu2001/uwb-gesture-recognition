"""Phase 1a: per-view range-sensitivity matrix and its spectrum.

The observable is rho_v(p) = ||p - P_v||, so the sensitivity of view v to a motion
component is d rho_v / d p = (p - P_v)/||p - P_v|| -- the unit line-of-sight vector.
Stacking the three views gives a 3x3 matrix J whose spectrum says which motion
directions are well measured and which are not.

This produces the falsifiable predictions that tools/w3_band.py must reproduce:
  P1  the least-observable direction at the nominal standoff
  P2  the std condition number (widest / narrowest posterior direction)
  P3  drop-one-view: the null direction vs the held-out line of sight

Usage:  python tools/w3_sens.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from uwb_osr.trajectory import RADAR_POS, Z_REF          # noqa: E402

VIEWS = ["Left", "Top", "Right"]
AXES = ["x (horizontal)", "y (vertical)", "z (depth)"]


def jac_at(p, views=(0, 1, 2), pos=RADAR_POS):
    """(V,3) matrix of d rho_v / d p at position p."""
    p = np.asarray(p, dtype=float)
    d = p[None, :] - pos[list(views)]
    r = np.linalg.norm(d, axis=1, keepdims=True)
    return d / r


def spectrum(J):
    G = J.T @ J
    w, V = np.linalg.eigh(G)                      # ascending
    return w[::-1], V[:, ::-1]                    # descending


def numeric_jac(p, views=(0, 1, 2), pos=RADAR_POS, h=1e-6):
    p = np.asarray(p, dtype=float)
    out = np.zeros((len(views), 3))
    for k in range(3):
        ep = np.zeros(3)
        ep[k] = h
        rp = np.linalg.norm(p + ep - pos[list(views)], axis=1)
        rm = np.linalg.norm(p - ep - pos[list(views)], axis=1)
        out[:, k] = (rp - rm) / (2 * h)
    return out


def null_dir(J2):
    """Unit null direction of a full-rank 2x3 matrix."""
    _, _, Vt = np.linalg.svd(J2)
    return Vt[-1]


def main():
    p0 = np.array([0.0, 0.0, Z_REF])
    print(f"nominal target      : {p0}")
    print(f"radar positions     : L{RADAR_POS[0]} T{RADAR_POS[1]} R{RADAR_POS[2]}")

    J = jac_at(p0)
    Jn = numeric_jac(p0)
    print(f"\nanalytic vs finite-difference max abs diff: {np.abs(J - Jn).max():.2e}")

    print("\n=== A. sensitivity matrix  d rho_v / d p  (dimensionless) ===")
    print(f"{'':8s}" + "".join(f"{a:>18s}" for a in AXES))
    for v in range(3):
        print(f"{VIEWS[v]:8s}" + "".join(f"{J[v, k]:18.3f}" for k in range(3)))
    zeros = [(VIEWS[v], AXES[k]) for v in range(3) for k in range(3)
             if abs(J[v, k]) < 1e-12]
    print(f"exact zeros ({len(zeros)}): " + ", ".join(f"{a}/{b.split()[0]}" for a, b in zeros))

    w, V = spectrum(J)
    print("\n=== B. spectrum of J^T J (three views) ===")
    for i in range(3):
        print(f"  eig{i+1} = {w[i]:7.4f}   dir = ({V[0,i]:+.3f}, {V[1,i]:+.3f}, {V[2,i]:+.3f})")
    least = V[:, -1] * np.sign(V[np.argmax(np.abs(V[:, -1])), -1])
    print(f"\n  P1 least-observable direction : ({least[0]:+.3f}, {least[1]:+.3f}, {least[2]:+.3f})")
    print(f"  P2 std condition number       : {np.sqrt(w[0] / w[-1]):.3f}")
    print(f"     rank                       : {np.linalg.matrix_rank(J)} "
          f"(full rank => all 3 DOF observable with all three views)")

    print("\n=== C. drop-one-view: null direction vs held-out line of sight ===")
    print("  (this is the 'blind direction' claim in 00 section 4.4)")
    for v in range(3):
        keep = tuple(u for u in range(3) if u != v)
        n = null_dir(jac_at(p0, keep))
        n = n * np.sign(n[np.argmax(np.abs(n))])
        u_out = jac_at(p0, (v,))[0]
        cos = abs(float(n @ u_out))
        print(f"  drop {VIEWS[v]:>5s}: null=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})  "
              f"|cos with held-out LOS| = {cos:.3f}")

    print("\n=== D. pooled readouts: the two exact zeros in 00 section 4.4 ===")
    for name, wts in [("mean(L,T,R)", np.array([1, 1, 1]) / 3.0),
                      ("R - L", np.array([-1.0, 0.0, 1.0]))]:
        s = wts @ J
        print(f"  {name:12s} sensitivity = "
              f"(x {s[0]:+.6f}, y {s[1]:+.6f}, z {s[2]:+.6f})")

    print("\n=== E. robustness: how much do P1/P2 move over the working volume? ===")
    grid = []
    for x in np.linspace(-0.30, 0.30, 7):
        for y in np.linspace(-0.30, 0.30, 7):
            for z in (0.41, 0.45, 0.48):
                Jg = jac_at([x, y, z])
                wg, Vg = spectrum(Jg)
                d = Vg[:, -1] * np.sign(Vg[np.argmax(np.abs(Vg[:, -1])), -1])
                grid.append((np.sqrt(wg[0] / wg[-1]), abs(float(d @ least))))
    grid = np.array(grid)
    print(f"  std condition number : median {np.median(grid[:,0]):.3f}  "
          f"range [{grid[:,0].min():.3f}, {grid[:,0].max():.3f}]  (n={len(grid)})")
    print(f"  |cos| of local least-obs dir with the nominal one : "
          f"median {np.median(grid[:,1]):.3f}  min {grid[:,1].min():.3f}")

    np.savez("logs/w3_sens.npz", J=J, eig=w, eigvec=V, least=least,
             cond_std=float(np.sqrt(w[0] / w[-1])), grid=grid)
    print("\nsaved -> logs/w3_sens.npz")


if __name__ == "__main__":
    main()
