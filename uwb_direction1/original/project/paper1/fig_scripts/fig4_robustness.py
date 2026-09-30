# -*- coding: utf-8 -*-
"""论文图 4:缺视角鲁棒性(哑铃图)。数据源 RESULTS.md §11.1(基线 E2)与 §19.5
(view dropout 0.5)。输出文件名 = fig3_robustness(与 mvosr_paper.tex 一致)。
运行:  D:/D_conda/envs/pytorch/python.exe fig4_robustness.py
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 110,
})

C_SINGLE = "#8491B4"   # 基线:灰蓝
C_THREE  = "#00A087"   # 丢视角训练:青绿
C_RED    = "#E64B35"   # 单视角直训参考线

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {name}.pdf/png")


def main():
    # (标签, 视角数, base, base_std, vd, vd_std);RESULTS §11.1 + §19.5
    rows = [
        ("全视角 L+T+R", 3, 95.33, 0.79, 94.88, 0.94),
        ("上失效(L+R)", 2, 86.22, 2.63, 90.81, 2.28),
        ("右失效(L+T)", 2, 80.21, 2.62, 91.97, 0.93),
        ("左失效(T+R)", 2, 71.40, 9.75, 92.35, 1.07),
        ("仅剩左 L", 1, 54.71, 6.46, 83.61, 3.21),
        ("仅剩上 T", 1, 53.37, 7.11, 72.12, 3.94),
        ("仅剩右 R", 1, 43.35, 7.51, 81.30, 1.36),
    ]
    sv_ref = 86.74   # 单视角直训(Left)5-case,Raw 忠实复现

    y = np.arange(len(rows))[::-1]   # 自上而下
    fig, ax = plt.subplots(figsize=(6.4, 3.05))

    for yi, (lab, nv, b, bs, v, vs) in zip(y, rows):
        ax.plot([b, v], [yi, yi], color="0.78", lw=1.6, zorder=2,
                solid_capstyle="round")
        ax.errorbar(b, yi, xerr=bs, fmt="o", ms=4.6, color=C_SINGLE,
                    ecolor=C_SINGLE, elinewidth=0.9, capsize=2, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.errorbar(v, yi, xerr=vs, fmt="o", ms=4.6, color=C_THREE,
                    ecolor=C_THREE, elinewidth=0.9, capsize=2, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.5)
        d = v - b
        if abs(d) > 2:
            ax.annotate(f"+{d:.1f}", (v + vs + 1.3, yi), va="center",
                        fontsize=6.5, color="#00695C", fontweight="bold")
        else:
            ax.annotate(f"{d:+.2f}(噪声内)", (min(b, v) - bs - 1.3, yi),
                        va="center", ha="right", fontsize=6.3, color="0.45")

    # 单视角直训参考线
    ax.axvline(sv_ref, ls=(0, (5, 3)), lw=0.9, color=C_RED, zorder=1)
    ax.text(sv_ref, len(rows) - 0.62, " 单视角直训 86.74", color=C_RED,
            fontsize=6.3, ha="left", va="bottom")

    # 视角数分组分隔线
    ax.axhline(5.5, color="0.88", lw=0.6)
    ax.axhline(2.5, color="0.88", lw=0.6)
    for yy, t in ((6.0, "3 路"), (4.0, "2 路"), (1.0, "1 路")):
        ax.text(101.5, yy, t, fontsize=6.5, color="0.45", va="center",
                rotation=90)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("OSCR (%,5-case 均值,误差棒为标准差)")
    ax.set_xlim(33, 103)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.xaxis.grid(True, lw=0.4, color="0.9")
    ax.set_axisbelow(True)

    hB = mpl.lines.Line2D([], [], marker="o", ls="", ms=4.6, color=C_SINGLE,
                          markeredgecolor="white", label="朴素三视角训练(基线)")
    hV = mpl.lines.Line2D([], [], marker="o", ls="", ms=4.6, color=C_THREE,
                          markeredgecolor="white",
                          label="随机丢视角训练($p$=0.5)")
    ax.legend(handles=[hB, hV], loc="upper left", bbox_to_anchor=(0.005, 0.62))
    save(fig, "fig3_robustness")


if __name__ == "__main__":
    print("rendering fig4 (robustness) ->", os.path.normpath(OUT))
    main()
    print("done.")
