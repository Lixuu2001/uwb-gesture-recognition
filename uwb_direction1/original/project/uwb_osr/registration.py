"""Paper-3 few-shot gesture registration (N-series, docs/RESEARCH_PLAN.md SS4).

The deployed open-set system (frozen Stage-1 encoder + per-base-class inspectors)
gains a new ability: register a previously-unknown gesture from a few examples so
it becomes a recognised class, without retraining the encoder and without
disturbing the base classes (per-class heads => structurally forget-free).

The one design change versus the OSR pipeline: routing. The OSR head first picks
a base class with the closed-set classifier (argmax over B logits), which can
never output a newly registered class. Here we instead classify by the *nearest
prototype across all classes* (base + registered): for class k the normalised
distance is r_k = mean_v d_k^v / mean_v sigma_k^v ("how many class-k std-devs
away"); predict argmin_k r_k, and reject as unknown when even the nearest is too
far (min_k r_k > tau).

N1 (evaluate_registration) sets that decision with a FIXED tau = eta = 3 (the A7
3-sigma heuristic). N1 showed a single fixed threshold cannot serve both well-
estimated base classes and few-shot new classes, nor a trained-inspector vs an
NCM score scale. N2 (register_and_score + conformal_report) replaces eta with a
split-conformal threshold calibrated on held-out base data, giving a coverage
guarantee and a set-valued prediction (empty=reject, singleton=confident,
multi=uncertain).

Two registration methods are compared (the few-shot ablation, SS4.3):
  A (trained):  a per-(class,view) MLP inspector trained few-shot, same family as
                the base inspectors (reuses train_inspector + its augmentation).
  B (NCM):      no MLP; the prototype is the mean support feature in the frozen
                feature space. For a fair cross-class comparison method B rebuilds
                the base prototypes the same way, so the whole bank lives in one
                space.
"""
import numpy as np
import torch

from .data import MVDataset
from .inspector import extract, train_inspector, train_inspectors
from .metrics import auroc, oscr


@torch.no_grad()
def _class_dists(bank, k, F, V, device):
    """Raw per-view distances (N, V) + per-view sigmas (V,) of class k.

    d^v = ||h_k^v(f_v) - phi_k^v||, where h_k^v is the class-k inspector
    (identity for NCM, when insp is None)."""
    dvs, svs = [], []
    for v in range(V):
        insp, phi, sigma = bank[(int(k), v)]
        fv = F[:, v].to(device)
        lat = insp(fv) if insp is not None else fv
        d = torch.norm(lat - phi.to(device), dim=1).cpu().numpy()
        dvs.append(d)
        svs.append(sigma)
    return np.stack(dvs, 1), np.asarray(svs, float)


@torch.no_grad()
def proto_dist_tensor(bank, classes, F, V, device):
    """Raw distance tensor D (N, C, V) + sigma matrix S (C, V).

    Keeping the per-view axis lets post-hoc analysis re-aggregate (per-view
    p-value merging) without re-running the encoder; the scalar r used by
    routing is D.mean(view) / S.mean(view)."""
    Ds, Ss = [], []
    for k in classes:
        d, s = _class_dists(bank, k, F, V, device)
        Ds.append(d)
        Ss.append(s)
    return np.stack(Ds, 1), np.stack(Ss, 0)


@torch.no_grad()
def proto_dist_matrix(bank, classes, F, V, device):
    """Full normalised-distance matrix R (N, C): r_k for every class k.

    r_k = mean_v d_k^v / mean_v sigma_k^v (unchanged aggregation)."""
    D, S = proto_dist_tensor(bank, classes, F, V, device)
    return D.mean(2) / (S.mean(1)[None, :] + 1e-6)


@torch.no_grad()
def nearest_proto(bank, classes, F, V, device):
    """Nearest-prototype classify + reject score.

    Returns (preds, min_r): preds in the mapped label space (argmin over
    `classes`), min_r the distance to the chosen prototype (high => likely
    unknown; confidence-of-known = -min_r)."""
    R = proto_dist_matrix(bank, classes, F, V, device)
    col = R.argmin(1)
    preds = np.asarray(classes)[col]
    min_r = R[np.arange(len(R)), col]
    return preds, min_r


def _subject_deltas(Ft, yt, tsubj, class_ids):
    """Per-subject feature offsets, estimated on classes with full data.

    delta[s] (V, feat) = mean over `class_ids` of (subject-s class centroid -
    population class centroid): the first-order direction 'how subject s
    renders a gesture relative to everyone'. The base classes have every
    training subject's data, so this subject-shift dictionary comes for free --
    it is exactly the cross-person spread a few support shots cannot span
    (the mechanism behind the balanced-support gain, amplified without new
    data, and applied at registration time so the encoder geometry that the
    sigma-normalised routing depends on is never touched)."""
    yt_t = torch.as_tensor(yt)
    st_t = torch.as_tensor(np.asarray(tsubj))
    class_mu = {}
    for k in class_ids:
        mk = yt_t == k
        if mk.any():
            class_mu[k] = Ft[mk].mean(0)                    # (V, feat)
    deltas = {}
    for s in sorted(set(int(v) for v in np.asarray(tsubj))):
        offs = [Ft[(yt_t == k) & (st_t == s)].mean(0) - mu
                for k, mu in class_mu.items()
                if bool(((yt_t == k) & (st_t == s)).any())]
        if offs:
            deltas[s] = torch.stack(offs).mean(0)           # (V, feat)
    return deltas


def _delta_translate(Fs_c, s_of, deltas, v):
    """Synthetic cross-subject positives for one view: each support feature is
    shifted by (delta[target] - delta[own]) for every other training subject,
    hallucinating 'this gesture as the others would render it'. When the
    support subject has no delta (personalization: test subject), the raw
    target delta is used (support treated as population-centred)."""
    out = []
    for i in range(Fs_c.size(0)):
        s0 = int(s_of[i])
        own = deltas.get(s0)
        for s, dlt in deltas.items():
            if s == s0:
                continue
            shift = dlt[v] - own[v] if own is not None else dlt[v]
            out.append(Fs_c[i, v] + shift)
    if not out:
        return Fs_c.new_empty((0, Fs_c.size(2)))
    return torch.stack(out)


def _register_method_a(model, train_ds, support_ds, cfg, device, n_base,
                       train_subj=None, support_subj=None, delta_aug=False):
    """Method A: trained few-shot inspectors, same family as base heads.

    delta_aug (needs train_subj/support_subj aligned with the two datasets):
    augment each new class's support with subject-delta translated copies
    before training its inspector, so the prototype/sigma capture cross-person
    spread instead of the few support renderings."""
    bank, V = train_inspectors(model, train_ds, cfg, n_base, device, verbose=False)
    Fs, _, ys = extract(model, support_ds, cfg, device)     # support: labels B..B+R-1
    Ft, _, yt = extract(model, train_ds, cfg, device)       # base features as negatives
    yt_t = torch.as_tensor(yt)
    deltas = None
    if delta_aug:
        if train_subj is None or support_subj is None:
            raise ValueError("delta_aug needs train_subj/support_subj")
        deltas = _subject_deltas(Ft, yt, train_subj, range(n_base))
    support_subj = None if support_subj is None else np.asarray(support_subj)
    for c in np.unique(ys):
        sel = ys == c
        for v in range(V):
            pos = Fs[sel, v]
            if deltas:
                pos = torch.cat(
                    [pos, _delta_translate(Fs[sel], support_subj[sel], deltas, v)], 0)
            neg = Ft[:, v]
            bank[(int(c), v)] = train_inspector(pos, neg, cfg, device,
                                                all_real=Ft[:, v], all_labels=yt_t)
    # 1-shot safety (mirrors method B's floor): sigma estimated from a single
    # support collapses to ~0, r = d/(sigma+1e-6) explodes, and the class can
    # never be predicted nor covered. Floor a degenerate new-class sigma with
    # the same view's mean base sigma so its column stays on the shared scale.
    for v in range(V):
        base_sig = float(np.mean([bank[(k, v)][2] for k in range(n_base)]))
        for c in np.unique(ys):
            insp, phi, s = bank[(int(c), v)]
            if s < 1e-3:
                bank[(int(c), v)] = (insp, phi, base_sig)
    return bank, V


def _ncm_proto(pos, device):
    """Mean prototype + RMS spread of a class's features (raw feature space)."""
    phi = pos.mean(0).to(device)
    if len(pos) > 1:
        sigma = torch.sqrt(((pos.to(device) - phi) ** 2).sum(1).mean()).item()
    else:
        sigma = 0.0                                          # 1-shot: filled later
    return phi.detach(), sigma


def _register_method_b(model, train_ds, support_ds, cfg, device, n_base):
    """Method B: nearest-class-mean prototypes for base AND new (one space)."""
    Ft, _, yt = extract(model, train_ds, cfg, device)
    Fs, _, ys = extract(model, support_ds, cfg, device)
    V = Ft.size(1)
    bank = {}
    for k in range(n_base):
        for v in range(V):
            bank[(k, v)] = (None, *_ncm_proto(Ft[yt == k, v], device))
    for c in np.unique(ys):
        for v in range(V):
            bank[(int(c), v)] = (None, *_ncm_proto(Fs[ys == c, v], device))
    # 1-shot safety: any sigma that collapsed to ~0 borrows the mean base sigma so
    # its normalised distance stays on the same scale as the others.
    base_sig = [bank[(k, v)][2] for k in range(n_base) for v in range(V)]
    floor = float(np.mean(base_sig)) if base_sig else 1.0
    for key, (insp, phi, s) in bank.items():
        if s < 1e-3:
            bank[key] = (insp, phi, floor)
    return bank, V


def evaluate_registration(cfg, model, split, X, y, view_idx, device,
                          method="A", verbose=False):
    """Run one few-shot registration evaluation (one LOSO fold).

    Trains base inspectors, registers the new classes few-shot, then scores the
    held-out subject's base / registered / unknown samples by nearest prototype.
    Returns threshold-free metrics (closed accuracies, H, AUROC, OSCR) plus an
    operating point at tau = 95th percentile of base-train distances (N1 stand-in
    for the N2 conformal threshold)."""
    nrm = cfg.normalize
    train_ds = MVDataset(X, y, split["train_idx"], view_idx, split["label_map"], nrm)
    support_ds = MVDataset(X, y, split["support_idx"], view_idx, split["label_map"], nrm)
    base_ds = MVDataset(X, y, split["test_base_idx"], view_idx, split["label_map"], nrm)
    reg_ds = MVDataset(X, y, split["test_reg_idx"], view_idx, split["label_map"], nrm)
    unk_ds = MVDataset(X, y, split["test_unknown_idx"], view_idx, None, nrm)
    n_base, n_reg = split["n_base"], split["n_register"]
    classes = list(range(n_base + n_reg))

    if method.upper() == "B":
        bank, V = _register_method_b(model, train_ds, support_ds, cfg, device, n_base)
    else:
        bank, V = _register_method_a(model, train_ds, support_ds, cfg, device, n_base)

    Fb, _, yb = extract(model, base_ds, cfg, device)        # base test, labels 0..B-1
    Fr, _, yr = extract(model, reg_ds, cfg, device)         # registered test, B..B+R-1
    Fu, _, _ = extract(model, unk_ds, cfg, device)          # true unknown

    pred_b, r_b = nearest_proto(bank, classes, Fb, V, device)
    pred_r, r_r = nearest_proto(bank, classes, Fr, V, device)
    pred_u, r_u = nearest_proto(bank, classes, Fu, V, device)

    ok_b = (pred_b == yb)
    ok_r = (pred_r == yr)
    acc_base = float(ok_b.mean()) if len(ok_b) else 0.0
    acc_new = float(ok_r.mean()) if len(ok_r) else 0.0
    H = 2 * acc_base * acc_new / (acc_base + acc_new + 1e-9)

    # threshold-free open-set: confidence-of-known = -min_r
    conf_known = np.concatenate([-r_b, -r_r])
    correct_known = np.concatenate([ok_b.astype(float), ok_r.astype(float)])
    is_known = np.concatenate([np.ones(len(conf_known)), np.zeros(len(r_u))])
    au = auroc(np.concatenate([conf_known, -r_u]), is_known)
    oc = oscr(conf_known, correct_known, -r_u)

    # operating point: the A7 heuristic 3-sigma threshold (reject when the
    # nearest prototype is > eta normalised std-devs away). It is deliberately a
    # *fixed* rule here: N1 shows whether a single global threshold can serve
    # both well-estimated base classes and few-shot new classes (it cannot, the
    # new-class sigma from a handful of shots underestimates spread), which is
    # precisely the gap the N2 conformal calibration closes.
    tau = float(getattr(cfg, "osr_eta", 3.0))
    acc_base_op = float((ok_b & (r_b <= tau)).mean()) if len(ok_b) else 0.0
    acc_new_op = float((ok_r & (r_r <= tau)).mean()) if len(ok_r) else 0.0
    rej_unk = float((r_u > tau).mean()) if len(r_u) else 0.0
    H_op = 2 * acc_base_op * acc_new_op / (acc_base_op + acc_new_op + 1e-9)

    return dict(acc_base=acc_base, acc_new=acc_new, H=H, auroc=au, oscr=oc,
                acc_base_op=acc_base_op, acc_new_op=acc_new_op,
                rej_unknown=rej_unk, H_op=H_op, tau=tau)


def _balanced_shots(rows, row_subj, n_shot, rng):
    """Pick n_shot row indices, each from a different subject when possible
    (mirrors register_split_conf's subject-balanced support draw)."""
    subjects = list(sorted(set(int(s) for s in row_subj)))
    rng.shuffle(subjects)
    picks, i = [], 0
    while len(picks) < n_shot and i < n_shot * max(1, len(subjects)):
        s = subjects[i % len(subjects)]
        cand = rows[np.asarray(row_subj) == s]
        cand = np.setdiff1d(cand, picks)
        if len(cand) > 0:
            picks.append(int(rng.choice(cand)))
        i += 1
    return np.array(picks[:n_shot], dtype=int)


def simulate_fewshot_calib(cfg, model, proto_ds, calib_ds, proto_subj, n_base,
                           n_shot, device, delta_aug=False, seed=0):
    """Simulated-task conformal, the data-free replacement for the falsified
    sigma-shrinkage: rehearse the few-shot registration ON THE BASE CLASSES.

    For each base class k: draw n_shot subject-balanced support shots from the
    training subjects, train a few-shot inspector exactly like a real
    registration (negatives = the OTHER base classes, as if k were new; same
    augmentation, same delta_aug setting), then score the calibration
    subject's class-k samples against it. The pooled own-r distribution
    measures the few-shot head's cross-person distance inflation without any
    new-class calibration data, so its conformal quantile transfers to the
    real new-class columns (cf. few-shot conformal with auxiliary tasks,
    Fisch et al. ICML'21 -- check the boundary before writing).

    Returns (sim_r, sim_y): calib samples' r to their own class's simulated
    few-shot head, and the aligned base labels."""
    rng = np.random.RandomState(seed)
    Ft, _, yt = extract(model, proto_ds, cfg, device)
    Fc, _, yc = extract(model, calib_ds, cfg, device)
    V = Ft.size(1)
    proto_subj = np.asarray(proto_subj)
    sim_r, sim_y = [], []
    for k in range(n_base):
        rows_k = np.where(yt == k)[0]
        if len(rows_k) == 0 or not (yc == k).any():
            continue
        shots = _balanced_shots(rows_k, proto_subj[rows_k], n_shot, rng)
        neg_mask = yt != k
        deltas = None
        if delta_aug:
            # deltas from the other classes only: class k plays "new", so its
            # own subject structure must not leak into the augmentation.
            others = [c for c in range(n_base) if c != k]
            deltas = _subject_deltas(Ft, yt, proto_subj, others)
        yt_neg = torch.as_tensor(yt[neg_mask])
        head = {}
        for v in range(V):
            pos = Ft[shots, v]
            if deltas:
                pos = torch.cat(
                    [pos, _delta_translate(Ft[shots], proto_subj[shots], deltas, v)], 0)
            head[(0, v)] = train_inspector(pos, Ft[neg_mask, v], cfg, device,
                                           all_real=Ft[neg_mask, v],
                                           all_labels=yt_neg)
        r = proto_dist_matrix(head, [0], Fc[yc == k], V, device)[:, 0]
        sim_r.append(r)
        sim_y.append(np.full(len(r), k, np.int64))
    if not sim_r:
        return np.zeros(0), np.zeros(0, np.int64)
    return np.concatenate(sim_r), np.concatenate(sim_y)


# ---------------------------------------------------------------------------
# N2: conformal calibration of the accept/reject threshold (paper-3 SS4.4)
# ---------------------------------------------------------------------------
def register_and_score(cfg, model, split, X, y, view_idx, device,
                       method="A", calib_frac=0.25, seed=0,
                       subj=None, delta_aug=False, sim_task=False, n_shot=5):
    """Train base inspectors + register new classes, then return the per-sample
    normalised-distance matrices needed for conformal calibration (one fold).

    Calibration source (the key to a valid guarantee under LOSO):
      * if `split` carries calib_idx (register_split_conf, a held-out subject)
        the prototypes use all of train_idx and that held-out subject is the
        calibration set -> calibration and test are both unseen users, i.e.
        exchangeable, so the coverage guarantee transfers to a new user;
      * otherwise (register_split) the base-train is split into prototype and
        calibration parts. This is the flawed variant: calibration comes from the
        *training* subjects whose prototype distances are far smaller than an
        unseen user's, so it under-covers; kept only for the negative control.

    P3 extensions (all opt-in, additive to the dump):
      * subj (full per-sample subject array) enables delta_aug and sim_task;
      * delta_aug: subject-delta translated support augmentation (method A);
      * sim_task: rehearse few-shot registration on the base classes and dump
        the calib subject's own-r distribution (sim_r_cal / sim_y_cal);
      * per-view raw distances D_* (N, C, V) + sigma_mat (C, V) are always
        dumped so per-view aggregation is a pure post-hoc choice.
    Returns a dict of R matrices (calib / base / reg / unknown test) + labels."""
    nrm = cfg.normalize
    rng = np.random.RandomState(seed)
    n_base, n_reg = split["n_base"], split["n_register"]
    classes = list(range(n_base + n_reg))

    if "calib_idx" in split:                       # held-out-subject calibration
        proto_idx = np.asarray(split["train_idx"])
        calib_idx = np.asarray(split["calib_idx"])
    else:                                          # legacy within-train split (control)
        tr = np.asarray(split["train_idx"])
        base_lab = np.array([split["label_map"][int(y[i])] for i in tr])
        proto_idx, calib_idx = [], []
        for k in range(n_base):
            idxk = tr[base_lab == k]
            rng.shuffle(idxk)
            ncal = max(1, int(round(len(idxk) * calib_frac)))
            calib_idx.append(idxk[:ncal])
            proto_idx.append(idxk[ncal:])
        proto_idx = np.concatenate(proto_idx)
        calib_idx = np.concatenate(calib_idx)

    def ds(idx, lm):
        return MVDataset(X, y, idx, view_idx, lm, nrm)
    proto_ds = ds(proto_idx, split["label_map"])
    support_ds = ds(split["support_idx"], split["label_map"])
    proto_subj = None if subj is None else np.asarray(subj)[proto_idx]
    support_subj = (None if subj is None
                    else np.asarray(subj)[np.asarray(split["support_idx"], int)])

    if method.upper() == "B":
        bank, V = _register_method_b(model, proto_ds, support_ds, cfg, device, n_base)
    else:
        bank, V = _register_method_a(model, proto_ds, support_ds, cfg, device,
                                     n_base, train_subj=proto_subj,
                                     support_subj=support_subj,
                                     delta_aug=delta_aug)

    def Rof(idx, lm):
        if len(idx) == 0:
            return (np.zeros((0, len(classes))),
                    np.zeros((0, len(classes), V)), np.zeros(0, np.int64))
        F, _, yy = extract(model, ds(idx, lm), cfg, device)
        D, S = proto_dist_tensor(bank, classes, F, V, device)
        return D.mean(2) / (S.mean(1)[None, :] + 1e-6), D, yy
    R_cal, D_cal, y_cal = Rof(calib_idx, split["label_map"])    # base, 0..B-1
    # N3: held-out-subject calibration for the registered classes (B..B+R-1)
    R_cal_reg, D_cal_reg, y_cal_reg = Rof(
        split.get("calib_reg_idx", np.array([], int)), split["label_map"])
    R_base, D_base, y_base = Rof(split["test_base_idx"], split["label_map"])
    R_reg, D_reg, y_reg = Rof(split["test_reg_idx"], split["label_map"])  # B..B+R-1
    R_unk, D_unk, _ = Rof(split["test_unknown_idx"], None)
    # per-class mean sigma over views: lets P2 sigma-shrinkage be a pure post-hoc
    # rescale of the dumped distances (r* = r * sigma / sigma*), no retraining.
    sigma_mat = np.array([[bank[(k, v)][2] for v in range(V)] for k in classes])
    sigma_bar = sigma_mat.mean(1)

    sim_r_cal = np.zeros(0)
    sim_y_cal = np.zeros(0, np.int64)
    if sim_task:
        if proto_subj is None:
            raise ValueError("sim_task needs the subj array")
        sim_r_cal, sim_y_cal = simulate_fewshot_calib(
            cfg, model, proto_ds, ds(calib_idx, split["label_map"]), proto_subj,
            n_base, n_shot, device, delta_aug=delta_aug, seed=seed)

    return dict(R_cal=R_cal, y_cal=y_cal, R_cal_reg=R_cal_reg, y_cal_reg=y_cal_reg,
                R_base=R_base, y_base=y_base,
                R_reg=R_reg, y_reg=y_reg, R_unk=R_unk, sigma_bar=sigma_bar,
                D_cal=D_cal, D_cal_reg=D_cal_reg, D_base=D_base, D_reg=D_reg,
                D_unk=D_unk, sigma_mat=sigma_mat,
                sim_r_cal=sim_r_cal, sim_y_cal=sim_y_cal,
                delta_aug=int(delta_aug), n_shot=int(n_shot),
                n_base=n_base, n_reg=n_reg)


def _conformal_q(scores, alpha):
    """Split-conformal upper threshold: smallest q with empirical
    P(score <= q) >= 1-alpha under the finite-sample (n+1) correction. Returns
    +inf when n calibration points cannot guarantee 1-alpha (e.g. few-shot)."""
    scores = np.asarray(scores, float)
    n = len(scores)
    if n == 0:
        return np.inf
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return np.inf                       # too few calib points for this 1-alpha
    return float(np.sort(scores)[rank - 1])


def conformal_q_vector(rd, alpha, mode, eta=3.0):
    """Per-class accept thresholds q (length n_base+n_reg).

    Calibration scores = each calib sample's distance to its own true class
    prototype: base from R_cal (held-out-subject base), new from R_cal_reg
    (held-out-subject registered, N3). Modes:
      'global'   : one pooled q from base calib, applied to every class (N2).
      'mondrian' : per-class q for ALL classes from their own held-out-subject
                   calib (N3: new classes use R_cal_reg instead of borrowing q).
      'hybrid'   : pooled base q for base classes (stable), per-class held-out q
                   for new classes (fixes the 5-shot scale mismatch). N3 main.
      'eta'      : fixed eta for every class (the N1 heuristic, for contrast)."""
    C = rd["n_base"] + rd["n_reg"]
    if mode == "eta":
        return np.full(C, float(eta))
    own_b = rd["R_cal"][np.arange(len(rd["R_cal"])), rd["y_cal"]]   # base r-to-true
    q_global = _conformal_q(own_b, alpha)                          # pooled base q
    q = np.full(C, q_global, float)

    # new-class held-out calibration scores (N3), if present
    Rcr = rd.get("R_cal_reg")
    ycr = rd.get("y_cal_reg")
    has_new_calib = Rcr is not None and len(Rcr) > 0
    own_n = (Rcr[np.arange(len(Rcr)), ycr] if has_new_calib else None)

    if mode == "mondrian":
        for k in range(rd["n_base"]):
            qk = _conformal_q(own_b[rd["y_cal"] == k], alpha)
            q[k] = qk if np.isfinite(qk) else q_global
        for c in range(rd["n_base"], C):
            if has_new_calib:
                qc = _conformal_q(own_n[ycr == c], alpha)
                q[c] = qc if np.isfinite(qc) else q_global
    elif mode == "hybrid":
        for c in range(rd["n_base"], C):
            if has_new_calib:
                qc = _conformal_q(own_n[ycr == c], alpha)
                q[c] = qc if np.isfinite(qc) else q_global
    return q


def conformal_report(rd, alpha, mode, eta=3.0):
    """Conformal metrics for one fold under threshold vector `mode`."""
    q = conformal_q_vector(rd, alpha, mode, eta)

    def grp(R, ytrue):
        sets = R <= q[None, :]
        sizes = sets.sum(1)
        if ytrue is None:
            return dict(empty=float((sizes == 0).mean()), size=float(sizes.mean()))
        cov = float(sets[np.arange(len(R)), ytrue].mean())
        sing = sizes == 1
        if sing.sum() > 0:
            sing_pred = sets[sing].argmax(1)            # single member's column
            sing_acc = float((sing_pred == ytrue[sing]).mean())
        else:
            sing_acc = 0.0
        return dict(cov=cov, size=float(sizes.mean()),
                    empty=float((sizes == 0).mean()),
                    sing_rate=float(sing.mean()), sing_acc=sing_acc)

    b = grp(rd["R_base"], rd["y_base"])
    r = grp(rd["R_reg"], rd["y_reg"])
    u = grp(rd["R_unk"], None)
    return dict(cov_base=b["cov"], cov_new=r["cov"],
                size_base=b["size"], size_new=r["size"],
                empty_base=b["empty"], empty_new=r["empty"],
                sing_acc_base=b["sing_acc"], sing_acc_new=r["sing_acc"],
                sing_rate_base=b["sing_rate"], sing_rate_new=r["sing_rate"],
                reject_unknown=u["empty"], size_unknown=u["size"])
