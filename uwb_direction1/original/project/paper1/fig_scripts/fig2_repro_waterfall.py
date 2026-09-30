# -*- coding: utf-8 -*-
"""图 2:单视角复现因素分解(瀑布图)。数据源 RESULTS.md §20.1(D2,case 4)。
运行:  D:/D_conda/envs/pytorch/python.exe fig2_repro_waterfall.py
输出:  ../figs/fig2_main_per_case_repro.{pdf,png}  (文件名见 save())
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- 全局样式
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei"],  # 中文首位防缺字
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

# NPG 科研配色
C_SINGLE = "#8491B4"   # 灰蓝(起始)
C_THREE  = "#00A087"   # 青绿(正增益)
C_DARK   = "#3C5488"   # 深蓝(终值)
C_RED    = "#E64B35"   # 参考线
C_GRAY   = "#9AA5B1"   # 伪因素

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {name}.pdf/png")


def main():
    # RESULTS.md §20.1(D2,case 4 单视角)
    levels = [73.18, 75.38, 75.39, 77.84, 88.35]
    deltas = ["+2.20", "+0.01", "+2.46", "+10.51"]
    names = ["基线\n(杂波对消)", "+ 式(12)\nmixup 负样本", "+ 检察器\nbatch=16",
             "+ 忠实\n$D_{key}$=32", "+ 原始回波\n(Raw)输入", "复现结果"]
    paper = 87.98

    fig, ax = plt.subplots(figsize=(5.5, 2.75))
    base_y = 70.0
    x = np.arange(6, dtype=float)

    # 起始柱与终值柱
    ax.bar(x[0], levels[0] - base_y, 0.56, bottom=base_y, color=C_SINGLE,
           zorder=3)
    ax.bar(x[5], levels[-1] - base_y, 0.56, bottom=base_y, color=C_DARK,
           zorder=3)
    ax.annotate(f"{levels[0]:.2f}", (x[0], levels[0] + 0.55), ha="center",
                fontsize=6.6)
    ax.annotate(f"{levels[-1]:.2f}", (x[5], levels[-1] + 0.55), ha="center",
                fontsize=6.6, fontweight="bold", color=C_DARK)

    # 增量浮柱
    for i in range(4):
        lo, hi = levels[i], levels[i + 1]
        col = C_GRAY if (hi - lo) < 0.1 else C_THREE
        ax.bar(x[i + 1], hi - lo if hi > lo else 0.08, 0.56, bottom=lo,
               color=col, zorder=3)
        ax.annotate(deltas[i], (x[i + 1], hi + 0.55), ha="center",
                    fontsize=6.6, color="#00695C" if col == C_THREE else "0.45",
                    fontweight="bold" if col == C_THREE else "normal")
        ax.plot([x[i] + 0.28, x[i + 1] - 0.28], [lo, lo], color="0.65",
                lw=0.7, ls=":", zorder=2)
    ax.plot([x[4] + 0.28, x[5] - 0.28], [levels[-1], levels[-1]], color="0.65",
            lw=0.7, ls=":", zorder=2)

    # 论文参考线
    ax.axhline(paper, ls=(0, (5, 3)), lw=0.9, color=C_RED, zorder=2)
    ax.text(-0.42, paper + 0.4, f"论文 case-4:{paper:.2f}", color=C_RED,
            fontsize=6.4, ha="left")
    ax.annotate("伪因素", (x[2], 73.3), ha="center", fontsize=6.0, color="0.45")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.4)
    ax.set_ylabel("case-4 OSCR (%)")
    ax.set_ylim(base_y, 91.5)
    ax.yaxis.grid(True, lw=0.4, color="0.88")
    ax.set_axisbelow(True)
    save(fig, "fig5_repro_waterfall")


if __name__ == "__main__":
    print("rendering fig2 (waterfall) ->", os.path.normpath(OUT))
    main()
    print("done.")
