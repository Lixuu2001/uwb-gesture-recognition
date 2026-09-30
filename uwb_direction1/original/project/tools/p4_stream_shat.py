"""s_hat on the controlled stream: does calib-select survive drift? (P4 SS12)

Pure post-processing on the stream runner's own dumps, no forward passes:

  * logs/p4_stream_c{ci}_calib_s{ts}.npz supply the calibration rows
    (R_cal / R_cal_reg) from the SAME per-fold-seeded bank as the stream
    scores (the n3 dumps are a different RNG trajectory = different bank,
    q vectors ~0.2 apart -- do not mix);
  * logs/p4_stream_c{ci}_{cond}_s{ts}.npz supply per-event unfused R for
    every (family x view) drift condition.

Per fold: s_cal = SS42 calib-select (H-max on the 1.0..3.0 grid subject to
calib base-acc drop <= 2 pts vs s=1). Per condition: stream routing
accuracy at s in {1, 1.5, s_cal, s_orc}, where s_orc re-runs the same
selection on the condition's own stream events (evaluator-side oracle,
labels never feed anything online). The question: does a scale chosen on
CLEAN calibration data stay near the drift-段 optimum, or does compound
drift move the optimal routing scale?

Pairing guard: the hybrid q vector recomputed from the n3 dump's calib rows
must match the q stored in the stream dump (same bank ⇒ same thresholds);
folds failing the check abort loudly rather than mixing banks.

Usage:  python tools/p4_stream_shat.py            # both cases
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from p4_g1_verdict import _acc, _select          # noqa: E402
from p4_stream import G0_FAMILIES, _cond_sfx     # noqa: E402
from uwb_osr.registration import _conformal_q    # noqa: E402

CAL_PREFIX = {0: "logs/p4_stream_c0_calib",
              1: "logs/p4_stream_c1_calib"}
VIEW_TOKENS = (0, 1, 2, "all")
ALPHA, LAM = 0.1, 1.0


def _check_pairing(cal, sd):
    """Recompute the stream runner's hybrid q from the n3 calib rows; a
    mismatch means the two dumps come from different banks."""
    nb, nr = int(sd["nb"]), int(sd["nr"])
    own = cal["R_cal"][np.arange(len(cal["R_cal"])), cal["y_cal"]]
    mu_r, sd_r = float(own.mean()), float(own.std() + 1e-8)
    E = cal["E_cal"]
    mu_e, sd_e = float(E.mean()), float(E.std() + 1e-8)
    q = np.full(nb + nr, _conformal_q((own - mu_r) / sd_r, ALPHA))
    q[:nb] = _conformal_q((own - mu_r) / sd_r
                          + LAM * (E - mu_e) / sd_e, ALPHA)
    Rcr, ycr = cal["R_cal_reg"], cal["y_cal_reg"]
    own_n = (Rcr[np.arange(len(Rcr)), ycr] - mu_r) / sd_r
    for c in range(nb, nb + nr):
        qc = _conformal_q(own_n[ycr == c], ALPHA)
        if np.isfinite(qc):
            q[c] = qc
    return float(np.abs(q - sd["q"]).max())


def main():
    for ci in (0, 1):
        conds = ["clean"] + [f + _cond_sfx(vt) for vt in VIEW_TOKENS
                             for f in G0_FAMILIES]
        rows = {}                                # cond -> list of fold dicts
        scals = []
        for ts in range(1, 9):
            calf = f"{CAL_PREFIX[ci]}_s{ts}.npz"
            strf = f"logs/p4_stream_c{ci}_clean_s{ts}.npz"
            if not (os.path.exists(calf) and os.path.exists(strf)):
                continue
            cal = np.load(calf)
            nb = int(cal["nb"])
            s_cal, _ = _select(cal["R_cal"], cal["y_cal"],
                               cal["R_cal_reg"], cal["y_cal_reg"], nb)
            err = _check_pairing(cal, np.load(strf))
            if err > 1e-3:
                raise SystemExit(f"pairing check FAILED c{ci} s{ts}: "
                                 f"max|q_n3 - q_stream| = {err:.4g}")
            scals.append(s_cal)
            for cond in conds:
                f = f"logs/p4_stream_c{ci}_{cond}_s{ts}.npz"
                if not os.path.exists(f):
                    continue
                d = np.load(f)
                kind, yt, R = d["kind"], d["ytrue"], d["R"]
                b, r = kind == 0, kind == 1
                Rb, yb = R[b], yt[b]
                Rr, yr = R[r], yt[r]
                s_orc, _ = _select(Rb, yb, Rr, yr, nb)
                e = {"s_orc": s_orc}
                for tag, s in [("1", 1.0), ("15", 1.5), ("cal", s_cal),
                               ("orc", s_orc)]:
                    ab, an = _acc(Rb, yb, nb, s), _acc(Rr, yr, nb, s)
                    e[f"H{tag}"] = 2 * ab * an / (ab + an + 1e-9)
                    e[f"b{tag}"], e[f"n{tag}"] = ab, an
                rows.setdefault(cond, []).append(e)
        if not scals:
            print(f"case {ci}: no paired dumps, skipped")
            continue

        def m(cond, k):
            return 100 * float(np.mean([e[k] for e in rows[cond]]))

        print(f"\n== s_hat on stream, case {ci} ({len(scals)} folds; "
              f"s_cal mean {np.mean(scals):.2f}) ==")
        print(f"  {'condition':13s} {'H@1':>6s} {'H@1.5':>6s} {'H@cal':>6s} "
              f"{'H@orc':>6s} {'gap_cal':>8s} {'cal-1.5':>8s} {'s_orc':>6s}")
        for cond in conds:
            if cond not in rows:
                continue
            so = float(np.mean([e["s_orc"] for e in rows[cond]]))
            gap = m(cond, "Horc") - m(cond, "Hcal")
            d15 = m(cond, "Hcal") - m(cond, "H15")
            mark = " *" if d15 < -1.0 else ""
            print(f"  {cond:13s} {m(cond, 'H1'):6.1f} {m(cond, 'H15'):6.1f} "
                  f"{m(cond, 'Hcal'):6.1f} {m(cond, 'Horc'):6.1f} "
                  f"{gap:8.1f} {d15:+8.1f} {so:6.2f}{mark}")
        print("  gap_cal = oracle - calib-select on that condition; "
              "cal-1.5 = calib-select vs fixed 1.5 (* = loses by >1); "
              "s_orc = mean per-condition oracle scale.")


if __name__ == "__main__":
    main()
