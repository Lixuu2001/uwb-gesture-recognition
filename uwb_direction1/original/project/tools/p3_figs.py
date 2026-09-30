# -*- coding: utf-8 -*-
"""论文三正文图:shot 曲线(fig_shot_curve)与自我个性化闭环(fig_loop)。

数据取自 docs/RESULTS.md §37.1 / §39.1(8 折 LOSO 人均值,主划分)。
用法:python tools/p3_figs.py   → 输出 paper3/figs/*.pdf + *.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- 主题(浅色印刷面) -------------------------------------------------
BLUE   = "#2a78d6"   # 主系列
BLUE_L = "#cde2fb"   # 蓝 100(上界柱填充)
AMBER  = "#c98500"   # 对照系列(蓝-黄轴,CVD 安全)
GRAY_B = "#c9c8c1"   # 基线柱
INK    = "#0b0b0b"   # 主文字
INK2   = "#52514e"   # 次文字
MUTED  = "#898781"   # 坐标轴标签
GRID   = "#e1e0d9"   # 网格
AXIS   = "#c3c2b7"   # 轴线
GREEN  = "#006300"   # 增量(向好)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 9.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, linestyle=(0, (1, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


# ======================================================================
# 图 1:shot 曲线(两面板)
# ======================================================================
K = [1, 5, 10, 20]
acc, acc_sd = [20.2, 75.5, 83.8, 86.2], [12.9, 11.2, 7.9, 4.3]
cov_free = [13.1, 65.7, 85.0, 88.9]          # 无 oracle
K_or, cov_or = [5, 10, 20], [82.1, 87.0, 89.2]  # oracle(K=1 为退化伪象,不画)

fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.95))
fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.16, wspace=0.26)

# ---- (a) 准确率 ----
a.axhline(80, color=AXIS, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
a.text(1.0, 81.5, "部署线 80", color=MUTED, fontsize=8.5, va="bottom")
a.errorbar(K, acc, yerr=acc_sd, color=BLUE, linewidth=2,
           marker="o", markersize=6, markerfacecolor=BLUE,
           markeredgecolor="white", markeredgewidth=1.2,
           ecolor=BLUE, elinewidth=1.1, capsize=2.5, alpha=None, zorder=3)
for x, y, lab, (dx, dy) in zip(K, acc, ["20.2", "75.5", "83.8", "86.2"],
                               [(10, 4), (7, -13), (7, 5), (7, 5)]):
    a.annotate(lab, (x, y), textcoords="offset points",
               xytext=(dx, dy), fontsize=8.5, color=INK, ha="left")
a.set_xscale("log")
a.set_xticks(K); a.set_xticklabels([str(k) for k in K])
a.minorticks_off()
a.set_xlim(0.82, 30)
a.set_ylim(0, 100)
a.set_xlabel("每类注册样本数 $K$(对数轴)")
a.set_ylabel("新类识别准确率(%)")
a.set_title("(a) 准确率随 $K$ 单调跨线", fontsize=10, color=INK, pad=8)
style_ax(a)

# ---- (b) 覆盖率 ----
b.axhline(90, color=AXIS, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
b.text(1.0, 91.5, "目标覆盖 90", color=MUTED, fontsize=8.5, va="bottom")
b.plot(K_or, cov_or, color=AMBER, linewidth=1.8, linestyle=(0, (5, 3)),
       marker="s", markersize=5.5, markerfacecolor="white",
       markeredgecolor=AMBER, markeredgewidth=1.4, zorder=3,
       label="oracle 上界(校准队列含新类)")
b.plot(K, cov_free, color=BLUE, linewidth=2,
       marker="o", markersize=6, markerfacecolor=BLUE,
       markeredgecolor="white", markeredgewidth=1.2, zorder=4,
       label="无 oracle(可直接部署)")
# 直接标签:蓝在下、琥珀在上,错开避免重合
for x, y, lab in zip(K, cov_free, ["13.1", "65.7", "85.0", "88.9"]):
    b.annotate(lab, (x, y), textcoords="offset points",
               xytext=(4, -13), fontsize=8.5, color=BLUE, ha="left")
for x, y, lab in zip(K_or, cov_or, ["82.1", "87.0", "89.2"]):
    b.annotate(lab, (x, y), textcoords="offset points",
               xytext=(-4, 7), fontsize=8.5, color=AMBER, ha="right")
b.set_xscale("log")
b.set_xticks(K); b.set_xticklabels([str(k) for k in K])
b.minorticks_off()
b.set_xlim(0.82, 30)
b.set_ylim(0, 100)
b.set_xlabel("每类注册样本数 $K$(对数轴)")
b.set_ylabel("新类经验覆盖率(%)")
b.set_title("(b) $K{=}20$ 覆盖率追平 oracle", fontsize=10, color=INK, pad=8)
b.legend(loc="lower right", fontsize=8, frameon=False,
         handlelength=2.2, borderaxespad=0.2)
style_ax(b)

fig.savefig("paper3/figs/fig_shot_curve.pdf")
fig.savefig("paper3/figs/fig_shot_curve.png", dpi=220)
plt.close(fig)

# ======================================================================
# 图 2:自我个性化闭环(分组柱)
# ======================================================================
groups = ["$m{=}5$", "$m{=}10$"]
cold = [72.9, 73.7]
warm = [79.9, 81.1]
orac = [91.2, 93.2]
delta = ["+7.0", "+7.4"]

fig, ax = plt.subplots(figsize=(4.8, 3.15))
fig.subplots_adjust(left=0.105, right=0.98, top=0.97, bottom=0.20)

w, gap = 0.24, 0.03            # 柱宽与柱间空隙
xs = [0, 1.08]
for i, x in enumerate(xs):
    ax.bar(x - w - gap, cold[i], w, color=GRAY_B, zorder=3)
    ax.bar(x, warm[i], w, color=BLUE, zorder=3)
    ax.bar(x + w + gap, orac[i], w, facecolor=BLUE_L, edgecolor=BLUE,
           linewidth=0.9, hatch="//", zorder=3)
    for dx, v in [(-w - gap, cold[i]), (0, warm[i]), (w + gap, orac[i])]:
        ax.annotate(f"{v:.1f}", (x + dx, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=9, color=INK)
    ax.annotate(delta[i] + "\n免标注", (x - w / 2, warm[i]),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=8.5, color=GREEN, linespacing=1.15)

ax.set_xticks(xs)
ax.set_xticklabels([f"采纳门槛 {g}" for g in groups], fontsize=10, color=INK2)
ax.set_ylim(0, 104)
ax.set_ylabel("新类识别准确率(%)")
style_ax(ax)
ax.tick_params(axis="x", length=0)

legend_items = [
    Patch(facecolor=GRAY_B, label="冷启动(跨人 5 样本)"),
    Patch(facecolor=BLUE, label="静默自学(共形门控,免标注)"),
    Patch(facecolor=BLUE_L, edgecolor=BLUE, hatch="//",
          label="真标签个性化(上界)"),
]
ax.legend(handles=legend_items, loc="upper center",
          bbox_to_anchor=(0.5, -0.115), ncol=3, fontsize=7.6,
          frameon=False, handlelength=1.4, columnspacing=1.0,
          handletextpad=0.5)

fig.savefig("paper3/figs/fig_loop.pdf", bbox_inches="tight")
fig.savefig("paper3/figs/fig_loop.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("done")
