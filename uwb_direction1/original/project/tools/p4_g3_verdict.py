"""G3/H4 c-lite verdicts: six S0 systems + S1 localisation, pure post-hoc.

All systems are numpy re-aggregations of tools/p4_g3_stream.py's raw dumps
(D per-view distances, M per-view recon MSEs, calib rows). Each system
recomputes its OWN mu/sigma/q and per-fold s_cal from its own calibration
scores (pre-registration v0.2 discipline; spec: docs/superpowers/specs/
2026-07-19-g3-h4-clite-preregistration.md).

Systems:
  v1      mean(D)/mean(sigma), +lam*z(E) on base cols   (replay of SS44/45)
  L1      same as v1 but routed at its own s_cal        (legacy baseline)
  L0      mean(D)/mean(sigma), NO additive penalty      (root-cause probe)
  uni     mean_v(D_v/sigma_v), no penalty               (aggregation control)
  rel     sum_v w_v (D_v/sigma_v), reliability weights  (G3 candidate)
  orc     known-polluted view masked (single-view cells; evaluator ceiling)

Gates (run as the matching dumps land):
  --gate eng        v1 vs the SS44 stream dumps: max|dR|,|dE| <= 1e-5 and
                    identical pred/sets/covered vs the G2 B=0 logs
  --gate rootcause  L0 vs L1 on stream {drop, occlude_vall, domain_vall}:
                    cov_b +5 pts in >=2/3, >=6/8 users positive per passing
                    condition, clean H drop <= 1
  --gate s1         polluted-view localisation at p50 single-view cells
  --gate reach      orc vs uni severity-AUC (>=4/5 non-atten families)
  --gate g3         rel vs uni severity-AUC + clean guard + rejU safety

Usage:
  python tools/p4_g3_verdict.py --gate eng
  python tools/p4_g3_verdict.py --gate rootcause --cases 0,1
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from p4_g1_verdict import _select                    # noqa: E402
from p4_stream import G0_FAMILIES                    # noqa: E402
from uwb_osr.registration import _conformal_q        # noqa: E402
from uwb_osr.reliability import (ReliabilityCalib,   # noqa: E402
                                 aggregate, oracle_mask_weights,
                                 uniform_weights)

LAM, ALPHA = 1.0, 0.1
STREAM_CELLS = ("clean", "drop", "occlude_vall", "domain_vall")


def _cal(ci, ts):
    f = f"logs/p4_g3_c{ci}_calib_s{ts}.npz"
    return np.load(f) if os.path.exists(f) else None


def _cell(ci, cell, ts):
    f = f"logs/p4_g3_c{ci}_{cell}_s{ts}.npz"
    return np.load(f) if os.path.exists(f) else None


def _legacy_R(D, S):
    return D.mean(2) / (S.mean(1)[None] + 1e-6)


def _weights(system, M, rc, V, polluted=None):
    n = len(M)
    if system == "uni":
        return uniform_weights(n, V)
    if system == "rel":
        return rc.weights(M)[0]
    if system == "orc":
        return (uniform_weights(n, V) if polluted is None
                else oracle_mask_weights(n, V, polluted))
    return None                                       # legacy aggregation


class System:
    """One S0 system's frozen per-fold state, built from calib raw rows."""

    def __init__(self, name, cal, s_route=None):
        self.name = name
        self.nb, self.nr = int(cal["nb"]), int(cal["nr"])
        self.S = cal["S"]
        self.V = self.S.shape[1]
        self.fuse = name in ("v1", "L1")
        self.rc = (ReliabilityCalib(cal["M_cal"]) if name == "rel" else None)
        Wc = _weights(name, cal["M_cal"], self.rc, self.V)
        Rc = (aggregate(cal["D_cal"], self.S, Wc) if Wc is not None
              else _legacy_R(cal["D_cal"], self.S))
        yc = cal["y_cal"]
        own = Rc[np.arange(len(Rc)), yc]
        self.mu_r, self.sd_r = float(own.mean()), float(own.std() + 1e-8)
        E_cal = cal["M_cal"].mean(1)
        self.mu_e, self.sd_e = float(E_cal.mean()), float(E_cal.std() + 1e-8)
        zb = (own - self.mu_r) / self.sd_r
        fused = zb + (LAM * (E_cal - self.mu_e) / self.sd_e
                      if self.fuse else 0.0)
        C = self.nb + self.nr
        self.q = np.full(C, _conformal_q(zb, ALPHA))
        self.q[:self.nb] = _conformal_q(fused, ALPHA)
        Wcr = (_weights(name, cal["M_cal_reg"], self.rc, self.V)
               if len(cal["M_cal_reg"]) else None)
        Rcr = (np.zeros((0, C)) if not len(cal["D_cal_reg"]) else
               aggregate(cal["D_cal_reg"], self.S, Wcr)
               if Wcr is not None else _legacy_R(cal["D_cal_reg"], self.S))
        ycr = cal["y_cal_reg"]
        if len(Rcr):
            own_n = (Rcr[np.arange(len(Rcr)), ycr] - self.mu_r) / self.sd_r
            for c in range(self.nb, C):
                qc = _conformal_q(own_n[ycr == c], ALPHA)
                if np.isfinite(qc):
                    self.q[c] = qc
        # per-fold routing scale from OWN calib scores (frozen across cells);
        # v1 replays the deployed fixed 1.5 (SS45 honesty note)
        if s_route is not None:
            self.s = s_route
        else:
            self.s, _ = _select(Rc, yc, Rcr, ycr, self.nb) if len(Rcr) \
                else (1.0, None)

    def score(self, D, M, polluted=None, W_override=None):
        W = (W_override if W_override is not None
             else _weights(self.name, M, self.rc, self.V, polluted))
        R = (aggregate(D, self.S, W) if W is not None
             else _legacy_R(D, self.S))
        Z = (R - self.mu_r) / self.sd_r
        if self.fuse:
            E = M.mean(1)
            Z[:, :self.nb] += LAM * ((E - self.mu_e) / self.sd_e)[:, None]
        sets = Z <= self.q[None, :]
        Rs = R.copy()
        Rs[:, :self.nb] *= self.s
        return dict(R=R, sets=sets, pred=Rs.argmin(1))

    def metrics(self, out, kind, ytrue):
        b, r, u = kind == 0, kind == 1, kind == 2
        cov_row = np.zeros(len(kind), bool)
        known = b | r
        cov_row[known] = out["sets"][known, ytrue[known]]
        acc_b = float((out["pred"][b] == ytrue[b]).mean())
        acc_n = float((out["pred"][r] == ytrue[r]).mean())
        return dict(acc_b=acc_b, acc_n=acc_n,
                    H=2 * acc_b * acc_n / (acc_b + acc_n + 1e-9),
                    cov_b=float(cov_row[b].mean()),
                    cov_n=float(cov_row[r].mean()),
                    cov_k=float(cov_row[known].mean()),
                    rej_u=float((out["sets"][u].sum(1) == 0).mean()))


def _fold_iter(ci, cells, systems, s_route_v1=1.5):
    for ts in range(1, 9):
        cal = _cal(ci, ts)
        if cal is None:
            continue
        kind, ytrue = cal["kind"], cal["ytrue"]
        sys_objs = {n: System(n, cal, s_route=(s_route_v1 if n == "v1"
                                               else None))
                    for n in systems}
        row = {}
        for cell in cells:
            d = _cell(ci, cell, ts)
            if d is None:
                continue
            polluted = None
            if "_v" in cell:
                vt = cell.split("_v")[1].split("_")[0]
                polluted = None if vt == "all" else int(vt)
            row[cell] = {n: so.metrics(so.score(d["D"], d["M"],
                                                polluted=polluted),
                                       kind, ytrue)
                         for n, so in sys_objs.items()}
        yield ts, row


def gate_eng(cases):
    """Amended v0.2.1 (evidence: 2026-07-19 diagnosis): the literal 1e-5
    score bar is unattainable against dumps born in ANOTHER process --
    cudnn.benchmark autotune picks different inspector-matmul kernels per
    process (uniform ~1e-5..3e-4 wobble on D; E bit-stable; calib rows and
    q bit-identical; an in-process rescore matches the G3 dumps at 0.0).
    Gate = scores <=5e-4 vs legacy dumps AND decision-level EXACT equality
    (pred / prediction sets / covered) vs the G2 B=0 logs."""
    print("== gate ENG: v1 replay vs SS44/45 dumps (v0.2.1) ==")
    ok = True
    for ci in cases:
        for ts in range(1, 9):
            cal = _cal(ci, ts)
            if cal is None:
                continue
            v1 = System("v1", cal, s_route=1.5)
            kind, ytrue = cal["kind"], cal["ytrue"]
            for cell in STREAM_CELLS:
                d = _cell(ci, cell, ts)
                g0f = f"logs/p4_stream_c{ci}_{cell}_s{ts}.npz"
                if d is None or not os.path.exists(g0f):
                    continue
                g0 = np.load(g0f)
                R = _legacy_R(d["D"], d["S"])
                E = d["M"].mean(1)
                dR = float(np.abs(R - g0["R"]).max())
                dE = float(np.abs(E - g0["E"]).max())
                if dR > 5e-4 or dE > 5e-4:
                    ok = False
                    print(f"  FAIL c{ci} s{ts} {cell}: dR={dR:.2e} "
                          f"dE={dE:.2e} (score tol 5e-4)")
                g2f = f"logs/p4_g2_c{ci}_{cell}_B0_s{ts}.npz"
                if not os.path.exists(g2f):
                    continue
                g2 = np.load(g2f, allow_pickle=True)
                out = v1.score(d["D"], d["M"])
                known = kind <= 1
                cov = np.zeros(len(kind), bool)
                cov[known] = out["sets"][known, ytrue[known]]
                same_pred = np.array_equal(out["pred"], g2["pred"])
                same_cov = np.array_equal(cov, np.asarray(g2["covered"],
                                                          bool))
                same_sz = np.array_equal(out["sets"].sum(1), g2["setsz"])
                if not (same_pred and same_cov and same_sz):
                    ok = False
                    print(f"  FAIL c{ci} s{ts} {cell}: decisions diverge "
                          f"(pred={same_pred} cov={same_cov} sz={same_sz})")
    print(f"  ENG gate: {'PASS' if ok else 'FAIL'} "
          "(scores<=5e-4 + exact decisions)")
    return ok


def gate_rootcause(cases):
    print("== gate ROOT-CAUSE: legacy-L0 vs legacy-L1 (stream cells) ==")
    overall = True
    for ci in cases:
        rows = {}
        for ts, row in _fold_iter(ci, STREAM_CELLS, ("L0", "L1")):
            for cell, mm in row.items():
                rows.setdefault(cell, []).append(
                    (mm["L0"], mm["L1"]))
        if not rows:
            print(f"  case {ci}: no dumps")
            continue
        n_pass = 0
        for cell in ("drop", "occlude_vall", "domain_vall"):
            if cell not in rows:
                continue
            d_cov = [100 * (a["cov_b"] - b["cov_b"]) for a, b in rows[cell]]
            gain = float(np.mean(d_cov))
            pos = sum(x > 0 for x in d_cov)
            hit = gain >= 5 and pos >= 6
            n_pass += hit
            d_rej = float(np.mean([100 * (b["rej_u"] - a["rej_u"])
                                   for a, b in rows[cell]]))
            print(f"  c{ci} {cell:13s} d_cov_b={gain:+5.1f} pos={pos}/8 "
                  f"rejU_drop={d_rej:+5.1f} {'PASS' if hit else 'fail'}")
        dH = float(np.mean([100 * (a["H"] - b["H"])
                            for a, b in rows.get("clean", [])]))
        clean_ok = dH >= -1
        print(f"  c{ci} clean dH(L0-L1)={dH:+.1f} "
              f"({'ok' if clean_ok else 'FAIL'})  -> "
              f"{n_pass}/3 conditions (need >=2)")
        overall &= (n_pass >= 2 and clean_ok)
    print(f"  ROOT-CAUSE gate: {'PASS' if overall else 'FAIL'}")
    return overall


def gate_s1(cases):
    print("== gate S1: polluted-view localisation (p50 single-view) ==")
    from sklearn.metrics import roc_auc_score
    fam_pass = {}
    for ci in cases:
        per_fam = {}
        alert_clean = []
        for ts in range(1, 9):
            cal = _cal(ci, ts)
            if cal is None:
                continue
            rc = ReliabilityCalib(cal["M_cal"])
            cl = _cell(ci, "clean", ts)
            if cl is not None:
                alert_clean.append(float(rc.weights(cl["M"])[1].mean()))
            for fam in G0_FAMILIES:
                for v in (0, 1, 2):
                    sfx = "" if v == 0 else f"_v{v}"
                    d = _cell(ci, f"{fam}{sfx}_p50", ts)
                    if d is None:
                        continue
                    E = d["M"] / rc.med
                    lab = np.zeros((len(E), 3), int)
                    lab[:, v] = 1
                    auc = roc_auc_score(lab.ravel(), E.ravel())
                    top1 = float((E.argmax(1) == v).mean())
                    per_fam.setdefault(fam, []).append((auc, top1))
        print(f"  case {ci}: clean all-view alert "
              f"{100 * np.mean(alert_clean) if alert_clean else -1:.1f}% "
              f"(gate <=5%)")
        for fam, v in per_fam.items():
            auc = float(np.mean([a for a, _ in v]))
            t1 = float(np.mean([t for _, t in v]))
            hit = auc >= 0.75 and t1 >= 0.60
            fam_pass.setdefault(fam, {})[ci] = hit
            print(f"  c{ci} {fam:8s} AUROC={auc:.3f} top1={t1:.2f} "
                  f"{'PASS' if hit else 'fail'}")
    if fam_pass:
        union = [f for f, d in fam_pass.items() if any(d.values())]
        per_case_ok = all(
            sum(d.get(ci, False) for d in fam_pass.values()) >= 3
            for ci in cases)
        ok = len(union) >= 4 and per_case_ok
        print(f"  S1 gate: {len(union)}/6 families, per-case>=3: "
              f"{per_case_ok} -> {'PASS' if ok else 'FAIL'}")


def _auc_by_family(ci, sys_a, sys_b):
    """Severity-AUC (trapezoid over {0,.25,.5,.75,1}) per family, macro
    over single views and folds; returns {fam: mean dAUC (a-b)}."""
    sevs = (0.0, 0.25, 0.5, 0.75, 1.0)
    out = {}
    for fam in G0_FAMILIES:
        vals = []
        for ts in range(1, 9):
            cal = _cal(ci, ts)
            if cal is None:
                continue
            kind, ytrue = cal["kind"], cal["ytrue"]
            so = {n: System(n, cal) for n in (sys_a, sys_b)}
            for v in (0, 1, 2):
                sfx = "" if v == 0 else f"_v{v}"
                Hs = {sys_a: [], sys_b: []}
                for sv in sevs:
                    d = (_cell(ci, "clean", ts) if sv == 0 else
                         _cell(ci, f"{fam}{sfx}_p{int(sv * 100)}", ts))
                    if d is None:
                        Hs = None
                        break
                    for n in (sys_a, sys_b):
                        m = so[n].metrics(
                            so[n].score(d["D"], d["M"], polluted=v),
                            kind, ytrue)
                        Hs[n].append(m["H"])
                if Hs is None:
                    continue
                a = np.trapz(Hs[sys_a], sevs)
                b = np.trapz(Hs[sys_b], sevs)
                vals.append(100 * (a - b))
        if vals:
            out[fam] = (float(np.mean(vals)), len(vals))
    return out


def gate_e0a(cases):
    """E0-a oracle-reachability ceiling on the EXISTING Phase A drop dumps
    (amendment 2026-07-20; zero GPU). oracle semantics: the evaluator
    knows per event which view is polluted AND that sev>0 -- clean events
    keep uniform weights so the ceiling is not diluted. Gates (locked):
    oracle dH >= +3 in BOTH vocabs and >= 6/8 users positive per vocab."""
    print("== gate E0-a: oracle-mask ceiling on stream 'drop' (v0) ==")
    ok = True
    for ci in cases:
        rows = []
        for ts in range(1, 9):
            cal = _cal(ci, ts)
            d = _cell(ci, "drop", ts)
            if cal is None or d is None:
                continue
            kind, ytrue = cal["kind"], cal["ytrue"]
            sev = d["sev"]
            V = cal["S"].shape[1]
            uni = System("uni", cal)
            n = len(kind)
            W_orc = uniform_weights(n, V)
            hot = sev > 0
            W_orc[hot] = oracle_mask_weights(int(hot.sum()), V, 0)
            m_u = uni.metrics(uni.score(d["D"], d["M"]), kind, ytrue)
            m_o = uni.metrics(uni.score(d["D"], d["M"], W_override=W_orc),
                              kind, ytrue)
            e = {"dH": 100 * (m_o["H"] - m_u["H"]), "ts": ts}
            for sv in (0.25, 0.5, 0.75, 1.0):     # directional diagnostic
                sel = sev == sv
                if sel.sum() < 10:
                    continue
                ks = kind[sel]
                acc = {}
                for tag, W in (("u", None), ("o", W_orc)):
                    out = uni.score(d["D"][sel], d["M"][sel],
                                    W_override=None if W is None
                                    else W[sel])
                    pred = out["pred"]
                    b, r = ks == 0, ks == 1
                    ab = float((pred[b] == ytrue[sel][b]).mean())
                    an = float((pred[r] == ytrue[sel][r]).mean())
                    acc[tag] = 2 * ab * an / (ab + an + 1e-9)
                e[f"dH_p{int(sv * 100)}"] = 100 * (acc["o"] - acc["u"])
            rows.append(e)
        if not rows:
            print(f"  case {ci}: no dumps")
            ok = False
            continue
        dH = float(np.mean([e["dH"] for e in rows]))
        pos = sum(e["dH"] > 0 for e in rows)
        strat = {k: float(np.mean([e[k] for e in rows if k in e]))
                 for k in ("dH_p25", "dH_p50", "dH_p75", "dH_p100")}
        hit = dH >= 3 and pos >= 6
        ok &= hit
        print(f"  c{ci}: oracle dH={dH:+.1f} pos={pos}/8 "
              f"by-sev {' '.join(f'{k[4:]}:{v:+.1f}' for k, v in strat.items())} "
              f"-> {'PASS' if hit else 'FAIL'}")
        for e in rows:
            print(f"    fold {e['ts']}: dH={e['dH']:+.1f}")
    print(f"  E0-a gate: {'PASS -> E0-b may be requested' if ok else 'FAIL -> route a'}")
    return ok


def gate_reach(cases):
    print("== gate REACH: oracle-mask vs uniform-rv severity-AUC ==")
    _auc_gate(cases, "orc", "uni", "REACH")


def gate_g3(cases):
    print("== gate G3: reliability vs uniform-rv severity-AUC ==")
    _auc_gate(cases, "rel", "uni", "G3")


def _auc_gate(cases, a, b, tag):
    fam_hits = {}
    for ci in cases:
        for fam, (d, n) in _auc_by_family(ci, a, b).items():
            fam_hits.setdefault(fam, []).append(d)
            print(f"  c{ci} {fam:8s} dAUC_H={d:+6.2f} (n={n})")
    if fam_hits:
        won = [f for f, v in fam_hits.items()
               if f != "atten" and np.mean(v) > 0]
        print(f"  {tag}: non-atten families improved {len(won)}/5 "
              f"(need >=4): {'PASS' if len(won) >= 4 else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True,
                    choices=["eng", "rootcause", "e0a", "s1", "reach", "g3"])
    ap.add_argument("--cases", default="0,1")
    args = ap.parse_args()
    cases = [int(c) for c in args.cases.split(",")]
    {"eng": gate_eng, "rootcause": gate_rootcause, "e0a": gate_e0a,
     "s1": gate_s1, "reach": gate_reach, "g3": gate_g3}[args.gate](cases)


if __name__ == "__main__":
    main()
