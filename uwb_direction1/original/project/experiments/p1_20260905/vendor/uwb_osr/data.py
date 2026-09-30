"""Data layer for the UWB-Gestures dataset (Figshare 12652592).

Key facts verified against the raw files
----------------------------------------
* Each ClutterRemoved CSV is (9000, 189): 100 repetitions x 90 frames, 189 bins.
* Folder names are INCONSISTENT across participants, so files are located by
  globbing + regex on the *file name* (which IS consistent), never by path.
* The three radars (Left / Top / Right) for one (participant, gesture) are
  soft-synchronised: repetition i across the three files is the same physical
  gesture event -> we pair them by sample index to build a multi-view unit.
* View amplitudes differ by ~10x across radars -> per-view standardisation.

Produces, after `build_cache`:
    X    : (9600, 3, 90, 189) float32   (units, views[L,T,R], T, M)
    y    : (9600,)            int64      gesture label 0..11
    subj : (9600,)            int64      participant 1..8
"""
import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from config import VIEW_ORDER

_FNAME_RE = re.compile(
    r"HV(\d+)_G(\d+)_Radar(Left|Top|Right)_(ClutterRemoved|Raw)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------
def _index_files(data_root: str, data_type: str):
    """Return {(participant, gesture, view): filepath} for the given data type."""
    csvs = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
    index = {}
    for c in csvs:
        m = _FNAME_RE.search(os.path.basename(c))
        if not m:
            continue
        p, g, view, dtype = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        if dtype.lower() != data_type.lower():
            continue
        index[(p, g, view.capitalize())] = c
    return index


def build_cache(cfg, force: bool = False):
    """Read every CSV once and persist a single packed array. Idempotent."""
    import pandas as pd

    os.makedirs(cfg.cache_dir, exist_ok=True)
    tag = cfg.data_type.lower()
    xp = os.path.join(cfg.cache_dir, f"X_{tag}.npy")
    yp = os.path.join(cfg.cache_dir, f"y_{tag}.npy")
    sp = os.path.join(cfg.cache_dir, f"subj_{tag}.npy")
    if (not force) and all(os.path.exists(p) for p in (xp, yp, sp)):
        return xp, yp, sp

    index = _index_files(cfg.data_root, cfg.data_type)
    participants = sorted({k[0] for k in index})
    gestures = sorted({k[1] for k in index})
    print(f"[cache] participants={participants} gestures={gestures} "
          f"views={VIEW_ORDER} files={len(index)}")

    X_list, y_list, s_list = [], [], []
    for p in participants:
        for g in gestures:
            views = []
            for v in VIEW_ORDER:                      # fixed slot order L,T,R
                f = index[(p, g, v)]
                d = pd.read_csv(f, header=None).values.astype(np.float32)
                d = d.reshape(-1, cfg.T, cfg.M)       # (100, 90, 189)
                views.append(d)
            n = min(v.shape[0] for v in views)
            vv = np.stack([v[:n] for v in views], axis=1)  # (100, 3, 90, 189)
            X_list.append(vv)
            y_list.append(np.full(n, g - 1, dtype=np.int64))   # G1->0
            s_list.append(np.full(n, p, dtype=np.int64))
        print(f"[cache] participant HV{p} done")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    s = np.concatenate(s_list, axis=0)
    np.save(xp, X); np.save(yp, y); np.save(sp, s)
    print(f"[cache] saved X{X.shape} -> {xp}  ({X.nbytes/1e9:.2f} GB)")
    return xp, yp, sp


def load_cache(cfg, mmap: bool = True):
    tag = cfg.data_type.lower()
    xp = os.path.join(cfg.cache_dir, f"X_{tag}.npy")
    yp = os.path.join(cfg.cache_dir, f"y_{tag}.npy")
    sp = os.path.join(cfg.cache_dir, f"subj_{tag}.npy")
    if not all(os.path.exists(p) for p in (xp, yp, sp)):
        build_cache(cfg)
    X = np.load(xp, mmap_mode="r" if mmap else None)
    y = np.load(yp); s = np.load(sp)
    return X, y, s


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _analytic(x):
    """Analytic signal along the fast-time axis (numpy-only Hilbert)."""
    M = x.shape[-1]
    h = np.zeros(M)
    h[0] = 1.0
    if M % 2 == 0:
        h[M // 2] = 1.0
        h[1:M // 2] = 2.0
    else:
        h[1:(M + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


def doppler_map(x):
    """Signed radial-velocity map from the RF frames. x: (V,T,M) -> (V,T,M).

    The frame-to-frame phase increment of the analytic signal is proportional
    to radial velocity, so its SIGN says whether the hand approaches or recedes
    from that radar. Weighted by echo strength so that empty range bins, where
    the phase is pure noise, stay near zero. The static component is removed
    first (it sits at zero Doppler and would otherwise dominate the phase)."""
    z = _analytic(x.astype(np.float64))
    z = z - z.mean(axis=1, keepdims=True)
    d = z[:, 1:] * np.conj(z[:, :-1])                  # (V,T-1,M)
    v = np.angle(d) * np.sqrt(np.abs(d))
    return np.concatenate([v[:, :1], v], axis=1).astype(np.float32)


class MVDataset(Dataset):
    """Indexes into the packed cache. `indices` selects units; `view_idx`
    selects which view slots (subset of [0,1,2]). `label_map` remaps original
    gesture labels (0..11) to contiguous class ids for an OSR test case."""

    def __init__(self, X, y, indices, view_idx=(0, 1, 2), label_map=None,
                 normalize="per_view_std", subj_labels=None, doppler="off"):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices)
        self.view_idx = list(view_idx)
        self.label_map = label_map
        self.normalize = normalize
        self.doppler = doppler
        # N4c: optional per-row subject ids (full-X aligned, already remapped to
        # 0..n_subj-1). When set, __getitem__ yields a third element so the
        # subject-adversarial head can train; eval datasets leave this None and
        # keep the 2-tuple contract.
        self.subj_labels = subj_labels

    def __len__(self):
        return len(self.indices)

    # ========================================================================
    #   NORMALIZE SWITCH  (cfg.normalize ; CLI --normalize {none, abs,
    #                                                       per_view_std, abs_std})
    #   This is the ONLY place input preprocessing branches on the mode. It
    #   shapes the geometry the inspector sees in feature space; abs_std =
    #   |x| then per-sample per-view std-normalise is the empirical winner and
    #   matches Eom's "amplitude-based recognition" phrasing.
    # ========================================================================
    def _norm(self, x):  # x: (V, T, M)
        # Options to disentangle the gap to Eom:
        #   none           -> raw signed clutter-removed signal (no scaling)
        #   abs            -> |x|, matching the paper's "amplitude" wording
        #   per_view_std   -> per-sample per-view zero-mean unit-std (our default,
        #                     but this kills absolute amplitude cues that the
        #                     OSR distance geometry probably needs)
        #   abs_std        -> |x| then per-sample per-view std normalisation
        if self.normalize == "abs":
            return np.abs(x)
        if self.normalize == "abs_std":
            x = np.abs(x)
        if self.normalize in ("per_view_std", "abs_std"):
            mu = x.mean(axis=(1, 2), keepdims=True)
            sd = x.std(axis=(1, 2), keepdims=True) + 1e-6
            x = (x - mu) / sd
        return x

    # ========================================================================
    #   DOPPLER SWITCH  (cfg.doppler ; CLI --doppler {off, only, concat})
    #   Runs on the signed RF frames BEFORE `_norm` takes |x|, which is the
    #   whole point: the direction of motion lives in the phase and abs() is
    #   where it gets destroyed.
    # ========================================================================
    def _doppler(self, x):  # x: (V, T, M) signed, pre-normalisation
        v = doppler_map(x)
        mu = v.mean(axis=(1, 2), keepdims=True)
        sd = v.std(axis=(1, 2), keepdims=True) + 1e-6
        return (v - mu) / sd

    def __getitem__(self, i):
        idx = self.indices[i]
        x = np.asarray(self.X[idx])[self.view_idx].astype(np.float32)   # (V,T,M)
        if self.doppler == "only":
            x = self._doppler(x)
        elif self.doppler == "concat":
            x = np.concatenate([self._norm(x), self._doppler(x)], axis=-1)
        else:
            x = self._norm(x)
        raw_label = int(self.y[idx])
        label = self.label_map[raw_label] if self.label_map is not None else raw_label
        if self.subj_labels is not None:
            return torch.from_numpy(x), label, int(self.subj_labels[idx])
        return torch.from_numpy(x), label


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def osr_split(cfg, y, known, unknown, seed=0):
    """Build train/test index sets for one open-set case.

    Returns dict with: train_idx, test_known_idx, test_unknown_idx, label_map.
    Known gestures are remapped to 0..K-1; unknowns keep their original ids only
    for bookkeeping (their test label becomes the sentinel K downstream).
    """
    rng = np.random.RandomState(seed)
    label_map = {g: i for i, g in enumerate(known)}

    train_idx, test_known_idx = [], []
    for g in known:
        gi = np.where(y == g)[0]
        rng.shuffle(gi)
        n_tr = cfg.train_per_known
        n_te = cfg.test_per_known
        train_idx.append(gi[:n_tr])
        test_known_idx.append(gi[n_tr:n_tr + n_te])
    train_idx = np.concatenate(train_idx)
    test_known_idx = np.concatenate(test_known_idx)

    unk_pool = np.where(np.isin(y, unknown))[0]
    rng.shuffle(unk_pool)
    test_unknown_idx = unk_pool[:cfg.n_unknown_test]

    return dict(train_idx=train_idx, test_known_idx=test_known_idx,
                test_unknown_idx=test_unknown_idx, label_map=label_map,
                n_known=len(known))


def loso_split(cfg, y, subj, known, unknown, test_subject, seed=0):
    """Leave-One-Subject-Out variant: train on all but `test_subject`."""
    rng = np.random.RandomState(seed)
    label_map = {g: i for i, g in enumerate(known)}
    is_known = np.isin(y, known)
    is_unk = np.isin(y, unknown)
    tr_mask = is_known & (subj != test_subject)
    te_k_mask = is_known & (subj == test_subject)
    te_u_mask = is_unk & (subj == test_subject)
    test_unknown_idx = np.where(te_u_mask)[0]
    rng.shuffle(test_unknown_idx)
    return dict(train_idx=np.where(tr_mask)[0],
                test_known_idx=np.where(te_k_mask)[0],
                test_unknown_idx=test_unknown_idx[:cfg.n_unknown_test],
                label_map=label_map, n_known=len(known))


def register_split(cfg, y, subj, base, register, unknown, test_subject,
                   n_shot, seed=0):
    """Paper-3 few-shot registration split under LOSO (docs/RESEARCH_PLAN.md SS4).

    The deployed system is trained on `base` classes (training subjects only).
    `register` classes are added few-shot at registration time from `n_shot`
    training-subject samples each; `unknown` classes are never registered and
    must still be rejected. Evaluation is on the held-out `test_subject`.

    label_map sends base -> 0..B-1 and register -> B..B+R-1, so the full known
    label space after registration is 0..B+R-1.

    Returns dict with:
        train_idx        base-class, training subjects (encoder + base inspectors)
        support_idx      few-shot register samples, training subjects
        support_y        their mapped labels (B..B+R-1), aligned to support_idx
        test_base_idx    base-class, test subject
        test_reg_idx     register-class, test subject
        test_unknown_idx unknown-class, test subject (capped at n_unknown_test)
        test_reg_y       mapped labels for test_reg_idx (B..B+R-1)
        label_map, n_base, n_register
    """
    rng = np.random.RandomState(seed)
    label_map = {g: i for i, g in enumerate(base)}
    for j, g in enumerate(register):
        label_map[g] = len(base) + j

    is_train_subj = subj != test_subject
    train_idx = np.where(np.isin(y, base) & is_train_subj)[0]

    support_idx, support_y = [], []
    for g in register:
        pool = np.where((y == g) & is_train_subj)[0]
        rng.shuffle(pool)
        picks = pool[:n_shot]
        support_idx.append(picks)
        support_y.append(np.full(len(picks), label_map[g], dtype=np.int64))
    support_idx = np.concatenate(support_idx) if support_idx else np.array([], int)
    support_y = np.concatenate(support_y) if support_y else np.array([], np.int64)

    is_test_subj = subj == test_subject
    test_base_idx = np.where(np.isin(y, base) & is_test_subj)[0]
    test_reg_idx = np.where(np.isin(y, register) & is_test_subj)[0]
    test_reg_y = np.array([label_map[int(y[i])] for i in test_reg_idx], np.int64)
    test_unknown_idx = np.where(np.isin(y, unknown) & is_test_subj)[0]
    rng.shuffle(test_unknown_idx)

    return dict(train_idx=train_idx,
                support_idx=support_idx, support_y=support_y,
                test_base_idx=test_base_idx,
                test_reg_idx=test_reg_idx, test_reg_y=test_reg_y,
                test_unknown_idx=test_unknown_idx[:cfg.n_unknown_test],
                label_map=label_map,
                n_base=len(base), n_register=len(register))


def register_split_conf(cfg, y, subj, base, register, unknown, test_subject,
                        calib_subject, n_shot, seed=0, subject_balanced=False,
                        support_personal=False):
    """Registration split for N2 conformal under LOSO (docs/RESEARCH_PLAN.md SS4).

    Leave-TWO-subjects-out: `test_subject` is evaluated, `calib_subject` supplies
    the conformal calibration set, and the remaining subjects train the encoder,
    base prototypes and few-shot registration. Crucially both the calibration and
    the test subject are unseen by the score function, so calibration and test are
    exchangeable (held-out users) and the coverage guarantee transfers to a new
    user. This fixes the failure of calibrating on training subjects, whose
    prototype distances are far smaller than an unseen user's.

    support_personal (N4 diagnostic) draws the support shots from the TEST
    subject instead of the training subjects, excluding them from the test set:
    the personalization scenario ("the user registers the gesture on their own
    device"). Against the cross-person runs it decomposes the new-class error
    into subject shift vs class hardness. Overrides subject_balanced.

    calib_subject may be one subject or a list (P3 SS38: a single calibration
    subject is a "scale lottery" -- their personal distance scale sets every
    threshold and per-fold coverage swings 61-100; pooling several held-out
    calibration subjects averages the scale variance away, at the cost of one
    encoder training subject each).

    Same keys as register_split plus calib_idx (base samples of calib_subject)."""
    rng = np.random.RandomState(seed)
    label_map = {g: i for i, g in enumerate(base)}
    for j, g in enumerate(register):
        label_map[g] = len(base) + j

    calib_subjects = (list(calib_subject)
                      if isinstance(calib_subject, (list, tuple, np.ndarray))
                      else [calib_subject])
    is_calib = np.isin(subj, calib_subjects)
    is_enc = (subj != test_subject) & ~is_calib     # encoder + prototypes
    train_idx = np.where(np.isin(y, base) & is_enc)[0]
    calib_idx = np.where(np.isin(y, base) & is_calib)[0]
    # N3: the calibration subject performed every gesture, so it also supplies
    # held-out calibration for the registered classes (their few-shot prototype
    # comes only from the training-subject shots; this just calibrates the
    # threshold). This gives new classes their own exchangeable calibration
    # instead of borrowing the base threshold, fixing the 5-shot scale mismatch.
    calib_reg_idx = np.where(np.isin(y, register) & is_calib)[0]
    calib_reg_y = np.array([label_map[int(y[i])] for i in calib_reg_idx], np.int64)

    # support shots for each new class. subject_balanced (P1.5) draws each shot
    # from a different encoder subject, so the 5-shot prototype/sigma capture
    # cross-person spread instead of one person's narrow variance (which is a
    # root cause of the underestimated new-class sigma). Random is the N1/N2 mode.
    # support_personal (N4) draws them from the test subject (removed from the
    # test set below).
    is_test = subj == test_subject
    enc_subjs = sorted(set(subj[is_enc].tolist()))
    support_idx, support_y = [], []
    for g in register:
        if support_personal:
            pool = np.where((y == g) & is_test)[0]
            rng.shuffle(pool)
            picks = pool[:n_shot]
        elif subject_balanced and enc_subjs:
            order = list(enc_subjs)
            rng.shuffle(order)
            picks = []
            i = 0
            while len(picks) < n_shot and i < n_shot * max(1, len(order)):
                s = order[i % len(order)]
                cand = np.where((y == g) & (subj == s))[0]
                if len(cand) > 0:
                    picks.append(int(rng.choice(cand)))
                i += 1
            picks = np.array(picks[:n_shot], dtype=int)
        else:
            pool = np.where((y == g) & is_enc)[0]
            rng.shuffle(pool)
            picks = pool[:n_shot]
        support_idx.append(picks)
        support_y.append(np.full(len(picks), label_map[g], dtype=np.int64))
    support_idx = np.concatenate(support_idx) if support_idx else np.array([], int)
    support_y = np.concatenate(support_y) if support_y else np.array([], np.int64)

    test_base_idx = np.where(np.isin(y, base) & is_test)[0]
    test_reg_idx = np.where(np.isin(y, register) & is_test)[0]
    if support_personal and len(support_idx):
        # the test subject's support shots must not also be scored as test items
        test_reg_idx = np.setdiff1d(test_reg_idx, support_idx)
    test_reg_y = np.array([label_map[int(y[i])] for i in test_reg_idx], np.int64)
    test_unknown_idx = np.where(np.isin(y, unknown) & is_test)[0]
    rng.shuffle(test_unknown_idx)

    return dict(train_idx=train_idx, calib_idx=calib_idx,
                calib_reg_idx=calib_reg_idx, calib_reg_y=calib_reg_y,
                support_idx=support_idx, support_y=support_y,
                test_base_idx=test_base_idx,
                test_reg_idx=test_reg_idx, test_reg_y=test_reg_y,
                test_unknown_idx=test_unknown_idx[:cfg.n_unknown_test],
                label_map=label_map,
                n_base=len(base), n_register=len(register))


# ---------------------------------------------------------------------------
# Synthetic data (smoke test, no real files needed)
# ---------------------------------------------------------------------------
def make_synthetic(cfg, n_per_class=120, n_classes=12, seed=0):
    """Class-structured multi-view signals so closed-set is learnable and
    unknown classes are separable. Returns X (N,3,T,M), y, subj."""
    rng = np.random.RandomState(seed)
    T, M, V = cfg.T, cfg.M, 3
    # each class has a distinct moving blob; each view sees a shifted/scaled copy
    centers = rng.uniform(0.2, 0.8, size=(n_classes, V, 2))   # (start_bin, speed)
    scales = rng.uniform(0.5, 1.5, size=(n_classes, V))
    X, y, s = [], [], []
    tt = np.linspace(0, 1, T)[:, None]
    bins = np.linspace(0, 1, M)[None, :]
    for c in range(n_classes):
        for _ in range(n_per_class):
            sample = np.zeros((V, T, M), np.float32)
            for v in range(V):
                start, speed = centers[c, v]
                mu = start + speed * tt * 0.4                  # blob center over time
                blob = np.exp(-((bins - mu) ** 2) / (2 * 0.02 ** 2))
                sample[v] = (scales[c, v] * blob).astype(np.float32)
            sample += rng.normal(0, 0.05, sample.shape).astype(np.float32)
            X.append(sample); y.append(c); s.append(rng.randint(1, 9))
    X = np.stack(X); y = np.array(y, np.int64); s = np.array(s, np.int64)
    return X, y, s
