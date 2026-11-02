# -*- coding: utf-8 -*-
"""论文图 5:消融总览(双面板)。
(a) 归一化×池化 分组柱:RESULTS §10.4。
(b) 架构级融合负结果链 森林图:RESULTS §20-23(F0/F1/F2/F6)+ §12(E4)/§14(E5)。
输出文件名 = fig4_ablation(与 mvosr_paper.tex 一致)。
运行:  D:/D_conda/envs/pytorch/python.exe fig5_ablation.py
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

C_SINGLE = "#8491B4"   # 单视角:灰蓝
C_THREE  = "#00A087"   # 三视角:青绿
C_RED    = "#E64B35"   # 负增益/有害
C_GRAY   = "#9AA5B1"   # 近零
C_BAND   = "#E8EAEE"   # 噪声带底色

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {name}.pdf/png")


def main():
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(6.7, 2.85), gridspec_kw=dict(width_ratios=[1.0, 1.12],
                                                    wspace=0.58))

    # ---- (a) 归一化 × 池化(RESULTS §10.4)------------------------------
    cfgs = ["mean\nper-view-std$^\\dagger$", "flatten\nper-view-std",
            "mean\nabs-std", "flatten\nabs-std$\\,\\bigstar$"]
    sv  = [69.71, 72.42, 83.34, 82.90]
    sve = [np.nan, 4.73, 3.34, 4.83]
    tv  = [92.56, 90.63, 93.63, 95.14]
    tve = [np.nan, 2.26, 2.05, 0.81]

    x = np.arange(4, dtype=float)
    w = 0.36
    for i in range(4):
        hatch = "////" if i == 0 else None           # case-0 一次性结果
        a.bar(x[i] - w / 2, sv[i], w, color=C_SINGLE, hatch=hatch,
              edgecolor="white", linewidth=0.4, zorder=3,
              yerr=None if np.isnan(sve[i]) else sve[i],
              error_kw=dict(lw=0.8, capsize=2, ecolor="0.25"))
        a.bar(x[i] + w / 2, tv[i], w, color=C_THREE, hatch=hatch,
              edgecolor="white", linewidth=0.4, zorder=3,
              yerr=None if np.isnan(tve[i]) else tve[i],
              error_kw=dict(lw=0.8, capsize=2, ecolor="0.25"))
    for xi, v, e in zip(x, tv, tve):
        a.annotate(f"{v:.1f}", (xi + w / 2, v + (0 if np.isnan(e) else e) + 0.9),
                   ha="center", fontsize=6.2, color="#00695C")
    for xi, v, e in zip(x, sv, sve):
        a.annotate(f"{v:.1f}", (xi - w / 2, v + (0 if np.isnan(e) else e) + 0.9),
                   ha="center", fontsize=6.2, color="#5B6B94")

    a.set_xticks(x)
    a.set_xticklabels(cfgs, fontsize=6.6)
    a.set_ylabel("OSCR (%)")
    a.set_ylim(60, 103)
    a.set_yticks([60, 70, 80, 90, 100])
    a.yaxis.grid(True, lw=0.4, color="0.88")
    a.set_axisbelow(True)
    a.set_title("(a) 归一化是首因,池化是次因", fontsize=7.5, pad=4)
    a.legend(handles=[mpl.patches.Patch(color=C_SINGLE, label="单视角"),
                      mpl.patches.Patch(color=C_THREE, label="三视角")],
             loc="upper left", handlelength=1.2, handleheight=0.9,
             borderaxespad=0.15)
    a.text(0.0, -0.365, "$^\\dagger$仅 case-0;其余为 5-case 均值±std",
           transform=a.transAxes, fontsize=5.8, color="0.45")

    # ---- (b) 架构级融合负结果链(F0/F1/F2/F6 + E5/E4)--------------------
    items = [  # (标签, ΔOSCR, 协议)
        ("F0 移除跨视角注意力",       +0.05, "随机·5-case"),
        ("F2 瓶颈中层融合(最优)",   -0.03, "随机·case-0"),
        ("E5 物理先验偏置",           -0.04, "随机·5-case"),
        ("E4 分歧分量 β=1",           -0.57, "随机·5-case"),
        ("F1 融合特征喂检察器",       -0.69, "随机·case-0"),
        ("F6 瓶颈融合 @ LOSO",        -1.79, "LOSO·case-0"),
    ]
    yy = np.arange(len(items))[::-1]
    vals = [it[1] for it in items]
    cols = [C_GRAY if abs(v) < 0.1 else C_RED for v in vals]

    b.axvspan(-0.81, 0.81, color=C_BAND, zorder=0)
    b.text(-0.77, -0.43, "主结果 ±1 std", fontsize=5.8, color="0.45",
           ha="left", va="bottom")
    b.barh(yy, vals, 0.58, color=cols, zorder=3)
    b.axvline(0, color="0.3", lw=0.8, zorder=4)
    for yi, (lab, v, proto) in zip(yy, items):
        b.annotate(f"{v:+.2f}", (v - 0.06 if v < 0 else v + 0.06, yi),
                   va="center", ha="right" if v < 0 else "left", fontsize=6.4,
                   color="0.2")
        # 协议标签放坐标轴外侧右缘(轴坐标 x>1),避免与噪声带/数值重叠
        b.text(1.03, yi, proto, transform=b.get_yaxis_transform(),
               va="center", ha="left", fontsize=5.8, color="0.5")
    b.set_yticks(yy)
    b.set_yticklabels([it[0] for it in items], fontsize=6.8)
    b.set_xlabel("$\\Delta$OSCR(相对无融合基线,%)")
    b.set_xlim(-2.2, 1.0)
    b.set_ylim(-0.55, len(items) - 0.45)
    b.xaxis.grid(True, lw=0.4, color="0.9")
    b.set_axisbelow(True)
    b.set_title("(b) 架构级融合:零增益或有害", fontsize=7.5, pad=4)

    save(fig, "fig4_ablation")


if __name__ == "__main__":
    print("rendering fig5 (ablation) ->", os.path.normpath(OUT))
    main()
    print("done.")
