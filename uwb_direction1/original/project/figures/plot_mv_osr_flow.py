import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.collections import LineCollection


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})


COL = {
    "ink": "#263238",
    "muted": "#60717A",
    "line": "#43515A",
    "neutral": "#F4F6F7",
    "neutral_edge": "#C9D2D8",
    "input": "#DCEEF7",
    "input_edge": "#78AFC8",
    "amp": "#E8F4E8",
    "amp_edge": "#78B67A",
    "encoder": "#D9E6F5",
    "encoder_edge": "#5F8BB8",
    "fusion": "#E4E0F1",
    "fusion_edge": "#8A79AF",
    "inspector": "#F3DFE7",
    "inspector_edge": "#B77C93",
    "decision": "#FFF2CC",
    "decision_edge": "#D5A93D",
    "accept": "#E5F4E6",
    "reject": "#F8E4E1",
}


def box(ax, x, y, w, h, text, fc, ec, lw=1.0, r=0.016, fs=7.6,
        weight="normal", ha="center"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha=ha, va="center", fontsize=fs, color=COL["ink"],
        fontweight=weight, linespacing=1.15,
    )
    return p


def arrow(ax, start, end, color=None, lw=1.05, ms=9, style="-|>",
          connectionstyle="arc3,rad=0"):
    arr = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color or COL["line"],
        connectionstyle=connectionstyle, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)
    return arr


def mini_cir(ax, x, y, w, h, label, seed, edge=COL["input_edge"]):
    rng = np.random.default_rng(seed)
    xx = np.linspace(0, 1, 42)
    yy = np.linspace(0, 1, 22)
    X, Y = np.meshgrid(xx, yy)
    center = 0.23 + 0.13 * seed
    trail = np.exp(-((X - (center + 0.35 * Y)) ** 2) / 0.004)
    burst = 0.35 * np.exp(-((X - 0.20) ** 2) / 0.0009)
    img = trail + burst + 0.05 * rng.normal(size=X.shape)
    ax.imshow(img, extent=(x, x + w, y, y + h), origin="lower",
              cmap="viridis", aspect="auto", zorder=1)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none",
                           edgecolor=edge, linewidth=0.9, zorder=2))
    ax.text(x + w / 2, y + h + 0.012, label, ha="center", va="bottom",
            fontsize=7.2, color=COL["ink"], fontweight="bold")


def stack_tokens(ax, x, y, w, h, n, color, edge, label):
    for i in range(n):
        off = i * min(w, h) * 0.11
        ax.add_patch(Rectangle((x + off, y + off), w, h,
                               facecolor=color, edgecolor=edge,
                               linewidth=0.6))
    ax.text(x + w / 2 + (n - 1) * min(w, h) * 0.055,
            y - 0.015, label, ha="center", va="top",
            fontsize=6.6, color=COL["muted"])


def draw_encoder(ax, x, y, w, h):
    box(ax, x, y, w, h, "", COL["encoder"], COL["encoder_edge"], lw=1.1, r=0.018)
    ax.text(x + w / 2, y + h - 0.028,
            "Shared dual-path Transformer encoder",
            ha="center", va="center", fontsize=8.3,
            color=COL["ink"], fontweight="bold")
    ax.text(x + w / 2, y + h - 0.052,
            "same weights for Left, Top and Right views",
            ha="center", va="center", fontsize=6.4, color=COL["muted"])

    lane_h = 0.066
    stack_tokens(ax, x + 0.024, y + 0.118, 0.046, 0.032, 5,
                 "#F8FBFD", COL["encoder_edge"], "90 temporal tokens")
    stack_tokens(ax, x + 0.024, y + 0.030, 0.046, 0.032, 5,
                 "#F8FBFD", COL["encoder_edge"], "189 spatial tokens")

    box(ax, x + 0.094, y + 0.105, 0.100, lane_h,
        "Patch +\nposition\nembedding", "#FFF3E4", "#D79B58", fs=6.7, r=0.010)
    box(ax, x + 0.094, y + 0.017, 0.100, lane_h,
        "Patch +\nposition\nembedding", "#FFF3E4", "#D79B58", fs=6.7, r=0.010)

    box(ax, x + 0.218, y + 0.105, 0.096, lane_h,
        "Temporal\nTransformer", "#CFE0F2", COL["encoder_edge"], fs=7.0, r=0.010)
    box(ax, x + 0.218, y + 0.017, 0.096, lane_h,
        "Spatial\nTransformer", "#CFE0F2", COL["encoder_edge"], fs=7.0, r=0.010)

    circ = Circle((x + 0.342, y + 0.102), 0.017,
                  facecolor="white", edgecolor=COL["encoder_edge"], linewidth=1.0)
    ax.add_patch(circ)
    ax.text(x + 0.342, y + 0.102, "+", ha="center", va="center",
            fontsize=10, color=COL["encoder_edge"], fontweight="bold")
    ax.text(x + 0.342, y + 0.047, "concat", ha="center", va="center",
            fontsize=6.5, color=COL["muted"])

    for yy in [y + 0.138, y + 0.050]:
        arrow(ax, (x + 0.075, yy), (x + 0.092, yy), lw=0.8, ms=7)
        arrow(ax, (x + 0.195, yy), (x + 0.216, yy), lw=0.8, ms=7)
    arrow(ax, (x + 0.315, y + 0.138), (x + 0.330, y + 0.111), lw=0.8, ms=7)
    arrow(ax, (x + 0.315, y + 0.050), (x + 0.330, y + 0.093), lw=0.8, ms=7)

    box(ax, x + w - 0.112, y + 0.070, 0.084, 0.060,
        "feature\ncode $f_v$\n(8928)", "#F8FBFD", COL["encoder_edge"], fs=6.8, r=0.010)
    arrow(ax, (x + 0.360, y + 0.102), (x + w - 0.114, y + 0.100), lw=0.8, ms=7)


def main():
    out_dir = r"D:\uwb_gestures\figures"
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(9.4, 5.65), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.030, 0.965, "Multi-view UWB open-set gesture recognition",
            ha="left", va="top", fontsize=11.5, fontweight="bold", color=COL["ink"])
    ax.text(0.030, 0.925,
            "Three synchronized CIR views are encoded with a shared dual-path Transformer and inspected with class-specific per-view prototypes.",
            ha="left", va="top", fontsize=7.3, color=COL["muted"])

    # Inputs
    ax.text(0.043, 0.855, "Synchronized radar views", ha="left", va="center",
            fontsize=8.2, fontweight="bold", color=COL["ink"])
    ys = [0.740, 0.610, 0.480]
    labels = ["Left", "Top", "Right"]
    for i, (yy, lab) in enumerate(zip(ys, labels), 1):
        mini_cir(ax, 0.042, yy, 0.118, 0.080, lab, i)

    box(ax, 0.205, 0.572, 0.118, 0.122,
        "Amplitude\npreprocessing\n$|x|$ + std\n(abs_std)",
        COL["amp"], COL["amp_edge"], fs=7.3, weight="bold")
    for yy in ys:
        arrow(ax, (0.162, yy + 0.040), (0.203, 0.633),
              connectionstyle="arc3,rad=0.0", lw=0.85, ms=8)

    draw_encoder(ax, 0.365, 0.485, 0.440, 0.255)
    arrow(ax, (0.324, 0.633), (0.363, 0.613), lw=1.0, ms=9)

    # Feature code copies
    f_y = [0.392, 0.326, 0.260]
    f_labels = [r"$f_{\mathrm{Left}}$", r"$f_{\mathrm{Top}}$", r"$f_{\mathrm{Right}}$"]
    for yy, lab, c in zip(f_y, f_labels, ["#DCEEF7", "#E6F2F8", "#DCEEF7"]):
        box(ax, 0.402, yy, 0.100, 0.043, lab, c, COL["input_edge"], fs=7.7, r=0.009)
    ax.text(0.452, 0.455, "per-view feature codes", ha="center",
            va="center", fontsize=7.0, color=COL["muted"])
    # Compact feature bus: avoids visually over-weighting repeated arrows.
    bus_x = 0.526
    ax.plot([bus_x, bus_x], [f_y[-1] + 0.022, f_y[0] + 0.022],
            color=COL["line"], lw=0.9)
    for yy in f_y:
        ax.plot([0.502, bus_x], [yy + 0.022, yy + 0.022],
                color=COL["line"], lw=0.9)

    # Closed-set branch
    ax.text(0.704, 0.420, "Stage 1: closed-set prediction", ha="center",
            va="center", fontsize=8.2, fontweight="bold", color=COL["ink"])
    box(ax, 0.545, 0.325, 0.136, 0.070,
        "Cross-view\nattention fusion", COL["fusion"], COL["fusion_edge"],
        fs=7.2, weight="bold", r=0.012)
    box(ax, 0.717, 0.325, 0.112, 0.070,
        "Known-gesture\nclassifier", COL["fusion"], COL["fusion_edge"],
        fs=7.2, r=0.012)
    box(ax, 0.862, 0.325, 0.092, 0.070,
        "predicted\nclass $\\hat{k}$", COL["decision"], COL["decision_edge"],
        fs=7.4, weight="bold", r=0.012)
    arrow(ax, (bus_x, 0.360), (0.543, 0.360), lw=0.9, ms=8)
    arrow(ax, (0.682, 0.360), (0.715, 0.360), lw=0.9, ms=8)
    arrow(ax, (0.830, 0.360), (0.860, 0.360), lw=0.9, ms=8)

    # Inspector branch
    ax.text(0.585, 0.205, "Stage 2: multi-view deep inspection", ha="center",
            va="center", fontsize=8.2, fontweight="bold", color=COL["ink"])
    box(ax, 0.545, 0.100, 0.154, 0.085,
        "Per-class, per-view\ninspectors\n$\\pi^{v}_{I,\\hat{k}}(f_v)$",
        COL["inspector"], COL["inspector_edge"], fs=7.0, weight="bold", r=0.012)
    box(ax, 0.733, 0.100, 0.122, 0.085,
        "Prototype\ndistances\n$d_v / \\sigma_v$",
        COL["inspector"], COL["inspector_edge"], fs=7.0, r=0.012)
    box(ax, 0.882, 0.100, 0.090, 0.085,
        "anomaly\nscore $s$", COL["decision"], COL["decision_edge"],
        fs=7.2, weight="bold", r=0.012)
    arrow(ax, (bus_x, 0.292), (0.543, 0.142),
          connectionstyle="arc3,rad=-0.12", lw=0.80, ms=8)
    arrow(ax, (0.908, 0.325), (0.910, 0.187), lw=0.85, ms=8,
          connectionstyle="arc3,rad=0.0")
    ax.text(0.925, 0.250, "select\ninspectors\nfor $\\hat{k}$",
            ha="left", va="center", fontsize=6.4, color=COL["muted"])
    arrow(ax, (0.700, 0.142), (0.731, 0.142), lw=0.9, ms=8)
    arrow(ax, (0.856, 0.142), (0.880, 0.142), lw=0.9, ms=8)

    # Equation and final decision
    ax.text(0.731, 0.055,
            r"$s=\mathrm{mean}_v(d_v)/\bar{\sigma}+\beta\,\mathrm{var}_v(d_v)/\bar{\sigma}^{2}$",
            ha="center", va="center", fontsize=8.0, color=COL["ink"])
    ax.text(0.731, 0.029,
            "mean distance = common mismatch; variance = cross-view disagreement",
            ha="center", va="center", fontsize=6.4, color=COL["muted"])

    box(ax, 0.872, 0.020, 0.052, 0.046, "accept\nknown", COL["accept"],
        "#80B783", fs=6.6, r=0.010)
    box(ax, 0.932, 0.020, 0.052, 0.046, "reject\nunknown", COL["reject"],
        "#C98982", fs=6.6, r=0.010)
    arrow(ax, (0.926, 0.100), (0.904, 0.067), lw=0.75, ms=7)
    arrow(ax, (0.932, 0.100), (0.957, 0.067), lw=0.75, ms=7)

    # Bottom note comparing with the original single-view path.
    box(ax, 0.045, 0.080, 0.275, 0.070,
        "Relation to Eom et al.\nSingle-view paper path is recovered with V=1:\n"
        "the variance term vanishes and one inspector is used per class.",
        COL["neutral"], COL["neutral_edge"], fs=6.8, r=0.012)

    # Subtle grouping labels
    ax.text(0.047, 0.430, "Input", fontsize=7.0, fontweight="bold",
            color=COL["muted"], ha="left")
    ax.text(0.370, 0.760, "Feature extraction", fontsize=7.0,
            fontweight="bold", color=COL["muted"], ha="left")

    # Lightweight separators.
    for x in [0.345, 0.525]:
        ax.plot([x, x], [0.075, 0.865], color="#E5EAED", lw=0.8, zorder=0)

    fig.savefig(os.path.join(out_dir, "mv_osr_three_view_flow.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "mv_osr_three_view_flow.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "mv_osr_three_view_flow.png"), dpi=450, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "mv_osr_three_view_flow.tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
