import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 字体设置
# =========================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =========================
# 2. 数据
# =========================
cases = ["case 1", "case 2", "case 3", "case 4", "case 5", "5-case\n均值"]

single_view = np.array([87.04, 84.35, 85.99, 73.54, 83.59, 82.90])
multi_view  = np.array([96.19, 94.80, 94.72, 94.04, 95.95, 95.14])
gains = multi_view - single_view

single_err = np.array([0, 0, 0, 0, 0, 4.83])
multi_err  = np.array([0, 0, 0, 0, 0, 0.81])

# =========================
# 3. 配色：更稳的论文风格
# =========================
color_single = "#9E9E9E"   # 中性灰
color_multi  = "#2F5597"   # 学术深蓝
color_gain   = "#1B4F72"   # 深蓝标注
color_ref    = "#E64B35"   # 红色参考线
color_sep    = "#CCCCCC"   # 均值组分隔线
color_grid   = "#E6E6E6"

# =========================
# 4. 画布与柱子
# =========================
fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)

x = np.arange(len(cases))
bar_width = 0.24

bars_single = ax.bar(
    x - bar_width / 2,
    single_view,
    width=bar_width,
    color=color_single,
    edgecolor="none",
    label="单视角(Left)",
    zorder=3
)

bars_multi = ax.bar(
    x + bar_width / 2,
    multi_view,
    width=bar_width,
    color=color_multi,
    edgecolor="none",
    label="三视角(L+T+R)",
    zorder=3
)

# =========================
# 5. 均值组误差棒
# =========================
mean_idx = len(cases) - 1

ax.errorbar(
    x[mean_idx] - bar_width / 2,
    single_view[mean_idx],
    yerr=single_err[mean_idx],
    fmt="none",
    ecolor="black",
    elinewidth=1.1,
    capsize=4,
    capthick=1.1,
    zorder=5
)

ax.errorbar(
    x[mean_idx] + bar_width / 2,
    multi_view[mean_idx],
    yerr=multi_err[mean_idx],
    fmt="none",
    ecolor="black",
    elinewidth=1.1,
    capsize=4,
    capthick=1.1,
    zorder=5
)

# =========================
# 6. 三视角增益标注
# =========================
for i, bar in enumerate(bars_multi):
    height = bar.get_height()
    cx = bar.get_x() + bar.get_width() / 2

    if i == 3:
        ax.text(
            cx,
            height + 1.8,
            f"+{gains[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color_gain
        )
        ax.text(
            cx,
            height + 0.7,
            "(最难 case)",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=color_gain
        )
    else:
        ax.text(
            cx,
            height + 1.0,
            f"+{gains[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color_gain
        )

# =========================
# 7. 均值组柱内竖排白字
# =========================
ax.text(
    x[mean_idx] - bar_width / 2,
    single_view[mean_idx] - 7.0,
    "82.90",
    ha="center",
    va="center",
    rotation=90,
    fontsize=9.5,
    fontweight="bold",
    color="white"
)

ax.text(
    x[mean_idx] + bar_width / 2,
    multi_view[mean_idx] - 6.0,
    "95.14",
    ha="center",
    va="center",
    rotation=90,
    fontsize=9.5,
    fontweight="bold",
    color="white"
)

# =========================
# 8. 均值组前浅灰分隔线
# =========================
ax.axvline(
    x=4.5,
    color=color_sep,
    linestyle="--",
    linewidth=1.0,
    zorder=1
)

# =========================
# 9. Eom 2025 参考线
# =========================
ref_y = 88.42

ax.axhline(
    y=ref_y,
    color=color_ref,
    linestyle=(0, (5, 5)),
    linewidth=1.2,
    zorder=2
)

ax.text(
    5.52,
    ref_y + 0.25,
    "Eom 2025 单视角\n88.42",
    ha="left",
    va="center",
    fontsize=9.5,
    color=color_ref,
    fontweight="bold"
)

# =========================
# 10. 坐标轴与网格
# =========================
ax.set_ylabel("OSCR (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(cases, fontsize=10)

ax.set_ylim(60, 100)
ax.set_yticks(np.arange(60, 101, 5))
ax.tick_params(axis="y", labelsize=10)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    color=color_grid,
    zorder=0
)

ax.set_axisbelow(True)

# =========================
# 11. 图例：放到图外，不压柱子
# =========================
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=2,
    frameon=False,
    fontsize=10,
    handlelength=2.2,
    columnspacing=2.2
)

# =========================
# 12. 边框优化
# =========================
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

# 右侧留空间给参考线文字
ax.set_xlim(-0.55, 6.15)

plt.subplots_adjust(top=0.82, right=0.88, bottom=0.16, left=0.09)

# =========================
# 13. 保存
# =========================
plt.savefig("oscr_single_vs_multi_clean.png", dpi=600, bbox_inches="tight")
plt.savefig("oscr_single_vs_multi_clean.svg", bbox_inches="tight")
plt.savefig("oscr_single_vs_multi_clean.pdf", bbox_inches="tight")

plt.show()