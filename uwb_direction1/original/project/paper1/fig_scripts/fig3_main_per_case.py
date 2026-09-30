# -*- coding: utf-8 -*-
"""论文图:主结果 单视角 vs 三视角逐 case OSCR(分组柱状图)。
数据源 RESULTS.md §3。输出文件名 = fig2_main_per_case(与 mvosr_paper.tex 一致)。
运行:  D:/D_conda/envs/pytorch/python.exe fig3_main_per_case.py

设计(Nature quantitative grid,单 hero 面板):
  · 统一蓝色族:浅蓝=单视角基线,深蓝=三视角(本文,hero);
    绿色只留给增益 Δ,红色只留给文献参考线 —— 颜色即语义。
  · 增益标注统一为单行 +X.X,置于 hero 柱顶固定偏移处;
    "最难 case" 用 case 4 背后的浅色高亮带表达,避免标签错位。
  · 均值数值水平排版、白底托底,置于误差棒上方(不再竖排)。
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",                 # SVG 文本保持可编辑
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 110,
})

# ---- 统一蓝色族 + 语义强调色 ------------------------------------------------
C_SINGLE = "#A9B8D6"   # 单视角 / 基线:浅蓝
C_THREE  = "#2E5E8C"   # 三视角 / 本文方法(hero):深蓝
C_GAIN   = "#1F9D55"   # 多视角增益 Δ:绿
C_REF    = "#C0453B"   # 文献参考线:红
C_BAND   = "#F4EFE6"   # case 4 高亮带:暖浅色
C_GRID   = "#E8E8E8"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {name}.pdf/png/svg")


def main():
    # 单视角 = 忠实复现配置(Raw + D_key=32 + 式12 mixup,RESULTS §17/复现);
    # 三视角 = 主结果配置(ClutterRemoved,RESULTS §3)。各为其视角数下最优配置。
    single = [87.81, 84.53, 87.39, 87.18, 86.81]   # mean 86.74 ± 1.29(忠实复现)
    three  = [96.19, 94.80, 94.72, 94.04, 95.95]   # mean 95.14 ± 0.81
    gains  = [8.38, 10.27, 7.33, 6.86, 9.14]        # mean +8.40,逐 case 一致
    s_mean, s_std = 86.74, 1.29
    t_mean, t_std = 95.14, 0.81
    eom = 88.42

    x = np.arange(5, dtype=float)
    xm = 5.95                                       # “均值”组的位置
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.6, 3.0))

    # 2) 逐 case 柱
    bar_kw = dict(width=w, zorder=3, linewidth=0)
    ax.bar(x - w / 2, single, color=C_SINGLE, label="单视角(Left)", **bar_kw)
    ax.bar(x + w / 2, three,  color=C_THREE,  label="三视角(L+T+R)", **bar_kw)

    # 3) 均值组(带误差棒)
    err = dict(lw=1.0, capsize=3, capthick=1.0, ecolor="0.35", zorder=4)
    ax.bar(xm - w / 2, s_mean, yerr=s_std, color=C_SINGLE, error_kw=err, **bar_kw)
    ax.bar(xm + w / 2, t_mean, yerr=t_std, color=C_THREE,  error_kw=err, **bar_kw)

    # 4) Eom 文献单视角参考线(我们的忠实单视角复现几乎贴在此线上)
    ax.axhline(eom, ls=(0, (5, 3)), lw=1.0, color=C_REF, zorder=2)
    ax.text(0.05, eom + 0.45, f"Eom 2025(单视角){eom:.2f}", color=C_REF,
            fontsize=6.8, va="bottom", ha="left", fontweight="bold")

    # 5) 增益 Δ:统一单行,置于 hero 柱顶固定偏移处
    for xi, t, g in zip(x, three, gains):
        ax.annotate(f"+{g:.1f}", (xi + w / 2, t + 0.5), ha="center",
                    va="bottom", fontsize=7.5, color=C_GAIN, fontweight="bold")

    # 7) 均值数值:水平排版、白底托底,置于误差棒上方
    bbox = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95)
    ax.annotate(f"{s_mean:.2f}", (xm - w / 2, s_mean + s_std + 0.6),
                ha="center", va="bottom", fontsize=7.5, color=C_SINGLE,
                fontweight="bold", bbox=bbox, zorder=6)
    ax.annotate(f"{t_mean:.2f}", (xm + w / 2, t_mean + t_std + 0.6),
                ha="center", va="bottom", fontsize=7.5, color=C_THREE,
                fontweight="bold", bbox=bbox, zorder=6)

    # 8) 坐标轴 / 网格 / 分隔
    ax.axvline(5.0, color="0.85", lw=0.8, zorder=1)
    ax.set_xticks(list(x) + [xm])
    ax.set_xticklabels(["case 1", "case 2", "case 3", "case 4", "case 5",
                        "5-case 均值"])
    ax.set_ylabel("OSCR (%)")
    ax.set_ylim(60, 101)
    ax.set_xlim(-0.62, 7.25)
    ax.set_yticks(np.arange(60, 101, 10))
    ax.yaxis.grid(True, lw=0.5, color=C_GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.005),
              columnspacing=1.8, handlelength=1.1, handletextpad=0.5)
    save(fig, "fig2_main_per_case")


if __name__ == "__main__":
    print("rendering main per-case ->", os.path.normpath(OUT))
    main()
    print("done.")
