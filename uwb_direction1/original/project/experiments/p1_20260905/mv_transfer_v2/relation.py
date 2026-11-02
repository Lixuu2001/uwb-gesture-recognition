"""Method A losses: cross-user supervised contrast and teacher relation transfer."""
import torch
from torch.nn import functional as F


def cross_user_masks(labels, subjects):
    """Neighbours are other people only; positives are their same-class samples."""
    allowed = subjects[:, None] != subjects[None, :]
    positive = allowed & (labels[:, None] == labels[None, :])
    return allowed, positive


def cross_user_relation_losses(student, teacher, labels, subjects, tau_student=0.2,
                               tau_teacher=0.2, want_relation=True, want_contrast=True):
    """student, teacher: B,D of any two widths. Teacher targets are detached.

    The relation term matches the cross-user neighbour distribution of the
    student to that of the teacher. The contrast term uses labels and user ids
    only, so it is reported as its own control and never folded into the
    relation claim.
    """
    student = F.normalize(student.float(), dim=-1)
    allowed, positive = cross_user_masks(labels, subjects)
    if not bool(allowed.any(1).all()) or not bool(positive.any(1).all()):
        raise ValueError('batch lacks cross-user neighbours or positives')
    student_logits = (student @ student.T / tau_student).masked_fill(~allowed, -1e4)
    log_s = F.log_softmax(student_logits, dim=1)
    zero = student.sum() * 0
    relation = zero
    if want_relation:
        teacher = F.normalize(teacher.detach().float(), dim=-1)
        teacher_logits = (teacher @ teacher.T / tau_teacher).masked_fill(~allowed, -1e4)
        log_t = F.log_softmax(teacher_logits, dim=1)
        prob_t = log_t.exp()
        relation = (prob_t * (log_t - log_s) * allowed).sum(1).mean()
    contrast = zero
    if want_contrast:
        contrast = -((log_s * positive).sum(1) / positive.sum(1)).mean()
    return relation, contrast


def rkd_distance_loss(student, teacher, labels, subjects):
    """Ordinary RKD-D control: mean-normalised pairwise distances, Huber matched."""
    student = student.float(); teacher = teacher.detach().float()
    allowed, _ = cross_user_masks(labels, subjects)
    ds = torch.cdist(student, student)
    dt = torch.cdist(teacher, teacher)
    ds = ds / ds[allowed].mean().clamp_min(1e-8)
    dt = dt / dt[allowed].mean().clamp_min(1e-8)
    return F.smooth_l1_loss(ds[allowed], dt[allowed])


def global_within_class_derangement(y, subj, pool, seed):
    """Sample id -> target id lookup, deranged inside every (class, user) cell.

    This is the correct instance-mismatch control. The earlier microbatch-local
    variant below could only swap when a cell happened to have two members in
    the same 32-sample microbatch, which left 68.4% of training exposures
    carrying their own target and therefore could not test instance
    correspondence at all. Building the derangement once over the whole
    training pool removes that dependence on batch composition.

    Returns the lookup and the fraction of pooled rows whose target changed.
    """
    import numpy as np
    y = np.asarray(y); subj = np.asarray(subj); pool = np.asarray(pool)
    rng = np.random.default_rng(seed)
    lookup = np.arange(len(y), dtype=np.int64)
    changed = 0
    for cls in np.unique(y[pool]):
        for user in np.unique(subj[pool]):
            cell = pool[(y[pool] == cls) & (subj[pool] == user)]
            if len(cell) < 2:
                continue
            perm = rng.permutation(len(cell))
            for _ in range(64):
                if not (perm == np.arange(len(cell))).any():
                    break
                perm = rng.permutation(len(cell))
            fixed = perm == np.arange(len(cell))
            if fixed.any() and len(cell) > 1:        # last resort: rotate the strays
                perm[fixed] = np.roll(perm[fixed], 1) if fixed.sum() > 1 else \
                    perm[(np.nonzero(fixed)[0][0] + 1) % len(cell)]
            lookup[cell] = cell[perm]
            changed += int((lookup[cell] != cell).sum())
    return lookup, float(changed / max(len(pool), 1))


def shuffle_within_class(ids, labels, subjects, generator):
    """SUPERSEDED microbatch-local instance mismatch. Kept only so the already
    finished B-SHUF / A-SHUF runs remain reproducible from their saved config.

    It swaps targets only inside the current microbatch and skips any
    (class, user) cell with a single member there, so most exposures keep their
    own target. Use global_within_class_derangement for any new control.
    """
    out = ids.clone()
    for cls in torch.unique(labels):
        for user in torch.unique(subjects[labels == cls]):
            pos = torch.nonzero((labels == cls) & (subjects == user), as_tuple=True)[0]
            if len(pos) < 2:
                continue
            fixed = torch.arange(len(pos), device=ids.device)
            perm = torch.randperm(len(pos), generator=generator, device=ids.device)
            for _ in range(8):
                if not bool((perm == fixed).any()):
                    break
                perm = torch.randperm(len(pos), generator=generator, device=ids.device)
            out[pos] = ids[pos[perm]]
    return out
