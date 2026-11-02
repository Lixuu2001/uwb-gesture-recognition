"""Shared style + palette for the MV-OSR figure set.

The four figures (F1 per-case OSCR, F2 pool*normalize heatmap, F3 normalization
diagnostic, F4 closed-vs-OSR gap + seed stability) share this style so they read
as one cohesive figure set when placed side by side in a manuscript.

Style is NMI-pastel-leaning: low-saturation neutrals for the comparison, one
teal accent for "our three-view headline", one salmon hue for the Eom baseline,
and a muted green only for directional Δ-gain annotations.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

# Unified palette across all four figures.
PALETTE = {
    "single":  "#BFC8D0",   # neutral gray-blue (single-view, less prominent)
    "single_dk": "#5A6D80", # darker variant for outlines / text
    "three":   "#2E7F8E",   # teal accent (three-view headline)
    "three_lt":"#7FB4BD",   # lighter teal for backgrounds / shaded bands
    "eom":     "#E07A5F",   # salmon for the Eom 88.42 reference line
    "gain":    "#5A9367",   # muted green for ΔOSCR gain
    "msp":     "#C5A572",   # muted ochre for MSP baseline
    "grid":    "#E5E5E5",
    "text":    "#1F1F1F",
}

# Figure widths in inches following standard 1-col (89 mm) / 1.5-col (120 mm) /
# 2-col (183 mm) print widths.
MM = 1.0 / 25.4
W1, W15, W2 = 89 * MM, 120 * MM, 183 * MM


def save_pub(fig, stem, dpi_png=300):
    """Write the figure to PNG (300 dpi), PDF (vector), SVG (editable)."""
    fig.savefig(f"{stem}.png", dpi=dpi_png)
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)
