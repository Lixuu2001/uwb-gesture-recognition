"""Open-set metrics in pure numpy (no sklearn dependency).

Convention: `score` is a confidence-of-KNOWN value (higher => more known).
For the evidential head use s = 1 - uncertainty; for MSP use s = max softmax.
"""
import numpy as np


def auroc(scores, is_known):
    """AUROC for the binary task known(1) vs unknown(0)."""
    scores = np.asarray(scores, float)
    y = np.asarray(is_known, float)
    order = np.argsort(-scores)
    y = y[order]
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return float("nan")
    tpr = np.concatenate([[0], np.cumsum(y) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / N])
    return float(np.trapz(tpr, fpr))


def oscr(score_k, correct_k, score_u):
    """Open-Set Classification Rate (area under CCR-FPR curve), Dhamija 2018.

    score_k/correct_k : known-test confidence and whether classified correctly.
    score_u           : unknown-test confidence.
    """
    score_k = np.asarray(score_k, float)
    correct_k = np.asarray(correct_k, float)
    score_u = np.asarray(score_u, float)
    m_k, m_u = len(score_k), len(score_u)
    thr = np.sort(np.unique(np.concatenate([score_k, score_u])))[::-1]
    ccr, fpr = [0.0], [0.0]
    for t in thr:
        ccr.append(((score_k > t) & (correct_k > 0)).sum() / m_k)
        fpr.append((score_u > t).sum() / m_u)
    ccr, fpr = np.array(ccr), np.array(fpr)
    idx = np.argsort(fpr)
    return float(np.trapz(ccr[idx], fpr[idx]))


def macro_f1(y_true, y_pred, n_labels):
    """Macro-F1 over `n_labels` classes (include the unknown sentinel)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(n_labels):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def fixed_eta_acc_f1(pred_k, score_k, label_k, score_u, K, eta, pred_u=None):
    """A7: heuristic-threshold eval at a fixed `eta` (paper Fig 10 style).

    Anomaly scores from inspector.anomaly_scores are roughly d/sigma, so a
    sample is judged unknown when score > eta (more than eta sigmas from the
    class prototype). Convention here mirrors best_threshold_acc_f1: score is
    *anomaly* (high=unknown). label_k in 0..K-1, unknown is sentinel K.
    `pred_u` is the classifier's predicted class for unknown-test samples; an
    accepted unknown keeps that class (Eom Fig.9 spreads UG over the known
    gestures). A class-0 fallback would concentrate the false positives on one
    class and bias macro-F1 down.
    """
    pred_k = np.asarray(pred_k); score_k = np.asarray(score_k)
    label_k = np.asarray(label_k); score_u = np.asarray(score_u)
    pred_u = (np.zeros(len(score_u), int) if pred_u is None
              else np.asarray(pred_u))
    y_true = np.concatenate([label_k, np.full(len(score_u), K)])
    # Known-test: keep predicted class if score<=eta else mark unknown.
    pred_known = np.where(score_k <= eta, pred_k, K)
    # Unknown-test: label as the classifier's prediction if score<=eta.
    pred_unknown = np.where(score_u <= eta, pred_u, K)
    pred = np.concatenate([pred_known, pred_unknown])
    acc = float((pred == y_true).mean())
    f1 = macro_f1(y_true, pred, K + 1)
    return acc, f1, float(eta)


def best_threshold_acc_f1(pred_k, score_k, label_k, score_u, K, pred_u=None):
    """Sweep the unknown threshold, return (acc, f1) at the best-accuracy point.
    Known true label in 0..K-1; unknown samples get sentinel label K.
    `pred_u`: classifier predictions for unknown-test samples (see
    fixed_eta_acc_f1) so accepted unknowns spread over the known classes."""
    pred_k = np.asarray(pred_k); score_k = np.asarray(score_k)
    label_k = np.asarray(label_k); score_u = np.asarray(score_u)
    pred_u = (np.zeros(len(score_u), int) if pred_u is None
              else np.asarray(pred_u))
    all_scores = np.concatenate([score_k, score_u])
    y_true = np.concatenate([label_k, np.full(len(score_u), K)])
    cand = np.unique(all_scores)
    if len(cand) > 500:
        cand = np.quantile(cand, np.linspace(0, 1, 500))
    best = (-1.0, 0.0, 0.0)
    for t in cand:
        pred = np.concatenate([
            np.where(score_k > t, pred_k, K),
            np.where(score_u > t, pred_u, K),
        ])
        acc = (pred == y_true).mean()
        if acc > best[0]:
            best = (acc, macro_f1(y_true, pred, K + 1), t)
    return best[0], best[1], best[2]
