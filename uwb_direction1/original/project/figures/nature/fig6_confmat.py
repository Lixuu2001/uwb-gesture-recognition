"""Figure 6 -- Eom Fig 9 / Fig 10 style confusion matrices.

Layout (matches Eom Fig 9 + Fig 10 stacked):
    Two row-pairs of confusion matrices, side by side.
        (a)  Single-view, best-acc threshold      |  (a)  Three-view, best-acc threshold
        (b)  Single-view, heuristic threshold     |  (b)  Three-view, heuristic threshold

Each matrix has rows = true gesture, columns = predicted gesture, with the
6 known gestures plus an "UG" (unknown gesture) column for samples
rejected by the OSR head.

Data source
-----------
results/case1_1view.npz and results/case1_3view.npz produced by
extract_case1_predictions.py.

For each matrix we:
  1. Convert OSR confidence (`conf_known`, `conf_unknown`) into the binary
     known/unknown decision using a threshold.
  2. Samples with conf < threshold are rejected -> column "UG".
     Samples with conf >= threshold use the closed-set predicted class.
  3. Row-normalise (each true-gesture row sums to 1.0).

Thresholds:
  * best-acc:   the threshold that maximises sample accuracy (saved in npz)
  * heuristic:  mean(conf_known) - 3 * std(conf_known) -- the 3-sigma rule
    of Eom's deep-inspector (their Fig 10 uses sigma = 3).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _style import PALETTE, save_pub

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
from config import OSR_TEST_CASES  # noqa: E402

# Eom gesture labels a..l for 0..11
GESTURE_LABEL = list("abcdefghijkl")


def _load(view_tag):
    p = os.path.join(RESULTS, f"case1_{view_tag}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p, allow_pickle=True)


def _build_confmat(d, threshold):
    """Build a (n_known + 1, n_known + 1) row-normalised confusion matrix.

    Rows = true gestures: the n_known known gestures in their canonical
    order (from `known_classes`) then the UG row (all unknown samples
    treated as a single 'true UG' row).
    Cols = predicted gestures: the n_known known gestures + UG column.
    """
    # NB: extract_case1_predictions.py wrote local 0..n_known-1 IDs into
    # the `known_classes` field. The canonical Eom-style gesture letters
    # come from OSR_TEST_CASES[case_idx]["known"], which is the list of
    # ORIGINAL class IDs in the same local-class order.
    case_idx = int(d["case_idx"])
    known_cls = np.array(OSR_TEST_CASES[case_idx]["known"])  # original IDs
    n_known = len(known_cls)
    y_known = d["y_known"]                               # mapped 0..n_known-1
    pred_known = d["pred_known"]
    conf_known = d["conf_known"]
    pred_unknown = d["pred_unknown"]
    conf_unknown = d["conf_unknown"]

    # ---- known-test samples: each row is a known gesture ----
    cm = np.zeros((n_known + 1, n_known + 1), dtype=np.float64)
    for k in range(n_known):
        idx = np.where(y_known == k)[0]
        n = len(idx)
        if n == 0:
            continue
        for i in idx:
            if conf_known[i] >= threshold:
                cm[k, pred_known[i]] += 1
            else:
                cm[k, n_known] += 1
        cm[k] /= n

    # ---- unknown-test samples: collapsed into a single 'true UG' row ----
    n_u = len(conf_unknown)
    for i in range(n_u):
        if conf_unknown[i] >= threshold:
            cm[n_known, pred_unknown[i]] += 1
        else:
            cm[n_known, n_known] += 1
    cm[n_known] /= max(n_u, 1)
    return cm, known_cls


def _draw_cm(ax, cm, known_cls, title):
    """Draw one confusion-matrix heatmap in the Eom style (blue cmap)."""
    n_known = len(known_cls)
    labels = [GESTURE_LABEL[int(c)] for c in known_cls] + ["UG"]

    # blue cmap matching Eom Fig 9/10
    cmap = LinearSegmentedColormap.from_list(
        "eom_blue",
        [(0.0, "#FFFFFF"), (0.25, "#DDE6F2"), (0.6, "#3D6BB0"),
         (1.0, "#0B3163")],
        N=256,
    )
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1.0, aspect="equal")

    # tick labels
    ax.set_xticks(np.arange(n_known + 1))
    ax.set_yticks(np.arange(n_known + 1))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted gesture", fontsize=7)
    ax.set_ylabel("True gesture", fontsize=7)

    # numeric annotations
    for i in range(n_known + 1):
        for j in range(n_known + 1):
            v = cm[i, j]
            color = "white" if v > 0.55 else PALETTE["text"]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.0, color=color)

    # thin grid lines between cells (white)
    for k in range(n_known + 2):
        ax.axhline(k - 0.5, color="white", lw=0.5)
        ax.axvline(k - 0.5, color="white", lw=0.5)

    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def heuristic_threshold(d):
    """3-sigma-like rule: mean - 3*std of known confidence."""
    c = d["conf_known"]
    return float(c.mean() - 3.0 * c.std())


def main():
    one = _load("1view")
    three = _load("3view")

    thresholds = {
        "1view_best": float(one["thr_best"]),
        "3view_best": float(three["thr_best"]),
        "1view_heu":  heuristic_threshold(one),
        "3view_heu":  heuristic_threshold(three),
    }
    print("thresholds:", thresholds)

    # build the four confusion matrices
    cm_1_best, kc1 = _build_confmat(one,   thresholds["1view_best"])
    cm_3_best, kc3 = _build_confmat(three, thresholds["3view_best"])
    cm_1_heu,  _   = _build_confmat(one,   thresholds["1view_heu"])
    cm_3_heu,  _   = _build_confmat(three, thresholds["3view_heu"])

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.3),
                              gridspec_kw=dict(wspace=0.30, hspace=0.40))

    panels = [
        (axes[0, 0], cm_1_best, kc1,
         f"(a)  Single-view, best-acc threshold (θ = {thresholds['1view_best']:.3f})"),
        (axes[0, 1], cm_3_best, kc3,
         f"(b)  Three-view,  best-acc threshold (θ = {thresholds['3view_best']:.3f})"),
        (axes[1, 0], cm_1_heu, kc1,
         f"(c)  Single-view, 3σ heuristic threshold (θ = {thresholds['1view_heu']:.3f})"),
        (axes[1, 1], cm_3_heu, kc3,
         f"(d)  Three-view,  3σ heuristic threshold (θ = {thresholds['3view_heu']:.3f})"),
    ]
    last_im = None
    for ax, cm, kc, title in panels:
        last_im = _draw_cm(ax, cm, kc, title)

    # shared colorbar on the right
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.03,
                         orientation="vertical")
    cbar.set_label("Row-normalised frequency", rotation=90)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "Confusion matrices on Eom Case 1 "
        "(known = b, e, f, h, i, l; unknown = a, c, d, g, j, k)",
        fontsize=8.5, fontweight="bold", y=0.995,
    )

    save_pub(fig, os.path.join(HERE, "fig6_confmat"))
    print("[fig6] wrote fig6_confmat.{png,pdf,svg}")


if __name__ == "__main__":
    main()
