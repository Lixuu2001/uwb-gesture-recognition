# -*- coding: utf-8 -*-
"""Thesis figures. Run:  python make_figs.py   (from thesis/figs/; CPU only)
All numbers are copied from the finalized papers / RESULTS (regenerated
final dumps); the signal figure reads the real cache.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
C1, C2, C3, C4 = "#4878A8", "#E08214", "#7FBF7B", "#B2ABD2"
GRAY = "#666666"


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def box(ax, x, y, w, h, text, fc="#EAF0F6", ec="#4878A8", fs=10):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                   fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2))


# --------------------------------------------------------------- 1 roadmap
def fig_roadmap():
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)
    box(ax, 0.2, 3.3, 3.0, 1.0,
        "第3章 多视角开集识别\n未知拒识 + 传感器容错", "#EAF0F6", C1)
    box(ax, 3.6, 3.3, 3.0, 1.0,
        "第4章 少样本注册\n共形接受 + 路由尺度", "#FDEBD3", C2)
    box(ax, 7.0, 3.3, 2.8, 1.0,
        "第5章 开放世界\n漂移审计 + 尺度对齐", "#E8F4E8", C3)
    arrow(ax, 3.2, 3.8, 3.6, 3.8); arrow(ax, 6.6, 3.8, 7.0, 3.8)
    for x, t in [(0.2, "问题:词表外动作误触发\n传感器失效"),
                 (3.6, "问题:词表不可扩展\n接受无错误率保证"),
                 (7.0, "问题:部署期复合漂移\n人工常数不可迁移")]:
        ax.text(x + 1.45, 2.75, t, ha="center", va="center",
                fontsize=8.5, color=GRAY)
    box(ax, 1.8, 0.9, 6.2, 0.9,
        "冻结基座:双路径 Transformer 编码器 + 逐类逐视角深度检查器 + 决策级聚合",
        "#F5F5F5", GRAY, 9.5)
    for x in (1.7, 5.1, 8.4):
        arrow(ax, x, 2.4, x, 1.9)
    ax.text(5, 0.35, "留人协议 / 预注册判据 / 负结果如实报告",
            ha="center", fontsize=9, color=GRAY, style="italic")
    save(fig, "fig_roadmap.pdf")


# --------------------------------------------------------------- 2 signal
def fig_signal():
    X = np.load(os.path.join(ROOT, "cache", "X_clutterremoved.npy"),
                mmap_mode="r")
    y = np.load(os.path.join(ROOT, "cache", "y_clutterremoved.npy"))
    idx = int(np.where(y == 1)[0][10])          # 一条示例样本
    x = np.abs(np.asarray(X[idx], np.float32))
    names = ["左视角", "上视角", "右视角"]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6), sharey=True)
    for v, ax in enumerate(axes):
        im = ax.imshow(x[v], aspect="auto", origin="lower", cmap="viridis",
                       extent=[0, x.shape[2], 0, x.shape[1]])
        ax.set_title(names[v], fontsize=10)
        ax.set_xlabel("距离单元(快时间)")
        if v == 0:
            ax.set_ylabel("帧(慢时间)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="幅值")
    save(fig, "fig_signal.pdf")


# --------------------------------------------------------------- 3 arch
def fig_arch():
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    for i, t in enumerate(["左视角\n距离-慢时间矩阵", "上视角", "右视角"]):
        box(ax, 0.2, 3.6 - 1.5 * i, 1.9, 1.1, t, "#F5F5F5", GRAY, 8.5)
    box(ax, 2.7, 1.2, 2.1, 3.5,
        "权重共享\n双路径 Transformer\n编码器\n(随机丢视角训练)", "#EAF0F6", C1, 9)
    for i in range(3):
        arrow(ax, 2.1, 4.15 - 1.5 * i, 2.7, 3.6 - 0.7 * i)
    for i, t in enumerate(["视角1 检查器组\n$d_k^1,\\ \\sigma_k^1$",
                           "视角2 检查器组", "视角3 检查器组"]):
        box(ax, 5.3, 3.6 - 1.5 * i, 1.9, 1.1, t, "#FDEBD3", C2, 8.5)
        arrow(ax, 4.8, 3.6 - 0.7 * i, 5.3, 4.15 - 1.5 * i)
    box(ax, 7.7, 2.4, 2.1, 1.4,
        "决策级聚合\n$r_k=\\frac{\\mathrm{mean}_v d_k^v}{\\mathrm{mean}_v \\sigma_k^v}$",
        "#E8F4E8", C3, 9)
    for i in range(3):
        arrow(ax, 7.2, 4.15 - 1.5 * i, 7.7, 3.1)
    ax.text(8.75, 1.6, "$\\min_k r_k$ 分类\n$r$ 过大 $\\Rightarrow$ 拒识",
            ha="center", fontsize=9, color=GRAY)
    save(fig, "fig_arch.pdf")


# --------------------------------------------------------------- 4 dropout
def fig_dropout():
    states = ["全部在位", "上失效", "右失效", "左失效", "仅剩左", "仅剩上", "仅剩右"]
    naive = [95.33, 86.22, 80.21, 71.40, 54.71, 53.37, 43.35]
    drop = [94.88, 90.81, 91.97, 92.35, 83.61, 72.12, 81.30]
    x = np.arange(len(states)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.bar(x - w / 2, naive, w, label="朴素三视角训练", color=C4)
    ax.bar(x + w / 2, drop, w, label="随机丢视角训练", color=C1)
    ax.axhline(86.74, ls="--", lw=1, color=GRAY)
    ax.text(6.45, 87.3, "单视角基线 86.74", fontsize=8, color=GRAY, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(states, fontsize=9)
    ax.set_ylabel("OSCR (%)"); ax.set_ylim(35, 100)
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_dropout.pdf")


# --------------------------------------------------------------- 5 splits
def fig_splits():
    single = [87.81, 84.53, 87.39, 87.18, 86.81]
    multi = [96.19, 94.80, 94.72, 94.04, 95.95]
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - w / 2, single, w, label="单视角(复现文献)", color=C4)
    ax.bar(x + w / 2, multi, w, label="三视角决策级聚合", color=C1)
    for i in range(5):
        ax.text(x[i] + w / 2, multi[i] + 0.3, f"+{multi[i]-single[i]:.1f}",
                ha="center", fontsize=8, color=C1)
    ax.set_xticks(x); ax.set_xticklabels([f"划分{i+1}" for i in range(5)])
    ax.set_ylabel("OSCR (%)"); ax.set_ylim(80, 100)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_splits.pdf")


# --------------------------------------------------------------- 6 stream
def fig_stream():
    fig, ax = plt.subplots(figsize=(7.8, 2.4))
    segs = [("冷启动", 0.15, "#F5F5F5"), ("渐变退化", 0.20, "#FDEBD3"),
            ("类别到来", 0.20, "#EAF0F6"), ("复合漂移", 0.30, "#F6DADA"),
            ("恢复", 0.15, "#E8F4E8")]
    x0 = 0
    for name, frac, col in segs:
        ax.axvspan(x0, x0 + frac, color=col)
        ax.text(x0 + frac / 2, 1.12, name, ha="center", fontsize=9)
        x0 += frac
    t = np.linspace(0, 1, 400)
    sev = np.piecewise(
        t, [t < .15, (t >= .15) & (t < .35), (t >= .35) & (t < .75),
            t >= .75],
        [0, lambda u: (u - .15) / .20, 1, lambda u: np.clip(1 - (u - .75) / .15, 0, 1)])
    grid = np.round(sev * 4) / 4
    ax.step(t, grid, where="mid", color=C2, lw=1.8, label="严重度(五档网格)")
    ax.plot(t, sev, color=GRAY, lw=0.8, ls=":", label="连续严重度")
    ax.set_ylim(-0.06, 1.3); ax.set_xlim(0, 1)
    ax.set_xlabel("流位置(归一化)"); ax.set_ylabel("退化严重度")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_stream.pdf")


# --------------------------------------------------------------- 7 audit
def fig_audit():
    fams = ["幅值衰减", "加性噪声", "距离轴平移", "慢时间遮挡", "视角掉落", "处理域漂移"]
    axes_n = ["左视角", "上视角", "右视角", "全视角"]
    dH = np.array([                 # 主词表 dH(全流),终版判决
        [0.0, -0.8, -0.7, -0.7],
        [-0.1, 0.4, -0.6, 1.7],
        [-0.8, 4.7, 1.4, 5.9],
        [4.5, 5.0, 4.3, 15.6],
        [10.3, 17.6, 8.2, 56.1],
        [28.4, 46.4, 26.0, 78.1]])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    im = ax.imshow(dH, cmap="OrRd", vmin=0, vmax=80, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(axes_n, fontsize=9)
    ax.set_yticks(range(6)); ax.set_yticklabels(fams, fontsize=9)
    for i in range(6):
        for j in range(4):
            ax.text(j, i, f"{dH[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if dH[i, j] > 40 else "black")
    fig.colorbar(im, shrink=0.85, label="调和精度降幅(百分点)")
    save(fig, "fig_audit.pdf")


# --------------------------------------------------------------- 8 s_orc
def fig_sorc():
    fams = ["干净流", "加性噪声", "距离轴平移", "慢时间遮挡", "视角掉落", "处理域漂移"]
    c0 = [1.92, 1.65, 1.65, 1.25, 1.36, 1.09]
    c1 = [1.82, 1.69, 1.60, 1.26, 1.39, 1.01]
    x = np.arange(len(fams)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x - w / 2, c0, w, label="主词表", color=C1)
    ax.bar(x + w / 2, c1, w, label="交换词表", color=C2)
    ax.axhline(1.5, ls="--", lw=1, color=GRAY)
    ax.text(5.4, 1.53, "固定常数 1.5", fontsize=8, color=GRAY, ha="right")
    ax.axhline(1.0, ls=":", lw=1, color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=8.5)
    ax.set_ylabel("逐条件最优路由尺度"); ax.set_ylim(0.9, 2.05)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_sorc.pdf")


# --------------------------------------------------------------- 9 ablate
def fig_ablate():
    conds = ["干净流", "视角掉落", "全视角噪声"]
    data = {"严格采纳门": [0.3, 0.5, -1.6], "仅确认": [1.5, 1.9, 0.4],
            "有界池": [1.3, 2.2, 0.3], "无门自训练": [0.3, -0.2, -2.0]}
    cols = [C4, C1, C3, "#D6604D"]
    x = np.arange(len(conds)); w = 0.19
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for i, (k, v) in enumerate(data.items()):
        ax.bar(x + (i - 1.5) * w, v, w, label=k, color=cols[i])
    ax.axhline(0, lw=0.8, color="black")
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("相对冻结系统的调和精度变化\n(百分点)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_ablate.pdf")


if __name__ == "__main__":
    for f in (fig_roadmap, fig_signal, fig_arch, fig_dropout, fig_splits,
              fig_stream, fig_audit, fig_sorc, fig_ablate):
        f()
