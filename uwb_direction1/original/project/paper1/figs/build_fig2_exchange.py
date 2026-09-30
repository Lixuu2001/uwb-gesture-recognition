# -*- coding: utf-8 -*-
"""论文一（中文版）图 2 —— 跨视角信息交换的位置、对象与带宽。

    python paper1/figs/build_fig2_exchange.py       # 产出 pptx / svg / html

设计依据：`paper1/refs/fig2_method_reference_library/README_CN.md` 第五节。

版面结构（2026-08-10 重排，解决"重复多 / 线乱 / 不对齐"）
--------------------------------------------------------
    ┌ 共用前端：三视角 CIR → abs-std → 共享编码器 E_θ 前段 ┐   ← 只画一次
    └──────────────┬──────────────┬──────────────┬──────┘
                   ↓              ↓              ↓
              ┌ (a) ┐        ┌ (b) ┐        ┌ (c) ┐

上一版把"三列完全相同"的前端（9 张 CIR、3 个编码器容器、36 个编码带、18 个
token 堆）在每列各画一遍，既冗余又把版面挤乱。既然三列的定义就是"只有一处
不同"，相同的部分就只画一次，三列从"编码器末段"起分叉。

每列内部三条视角泳道是**从头贯通到底的直线**，模块盖在泳道上；泳道之间的
连接用"上一模块底 → 下一模块顶"的整段箭头，不留缝。三个交换槽位统一画成
**横跨三条泳道的整条横杠**，唯一的差别是横杠出现在哪一行：

    槽位① 编码器内    (c) 共享瓶颈 token 总线
    槽位② 编码器后    (b) 视角维多头自注意力
    槽位③ 距离证据    三列都有，只有 (a) 以它为唯一交换点故点亮

未被使用的槽位画成浅色虚线占位框，泳道直穿而过（表示"此处无操作"）。
斜线、梯子、汇流钩全部去掉——所有连接只有水平或垂直两种走向。

配色纪律（README §5，勿回退）
----------------------------
三列共有结构一律低饱和灰蓝，**只用一种强调色**标交换位置。不要给三种策略
各配一套主色，否则读者会误以为除交换层级外还有别的架构差异。视角色 L/T/R
跨三列完全一致，编码的是视角身份而非策略。

术语（README §1 核准，勿改回旧称）
--------------------------------
  决策级距离聚合 / 编码后特征级融合·视角维自注意力 / MBT 式瓶颈中层融合
  共享编码器 E_θ（不要写 Siamese Network，也不能写成完全复现 MBT）

代码核对（uwb_osr/model.py）
--------------------------
  * CrossViewFusion 把三视角 token 排成长度 V 的序列做多头自注意力，
    Q/K/V 同源且未屏蔽对角线；只作用于时间路径，距离路径独立。
  * BottleneckFusionLayer 逐视角把 [H_v; B] 送进同一 block，层末对各视角
    产生的瓶颈副本取平均再回写，故少量瓶颈 token 是唯一跨视角通道。
  * 决策级聚合合并的是原型距离 d_v（标量证据），不是特征向量。
  * 交换行③三列都要画：特征级与瓶颈式同样在末端按式(2)聚合。

版式：1280 × 828 units（1 unit = 0.01 in）。\\textwidth = 16.2 cm = 6.378 in
→ 打印缩放约 0.498，即幻灯片字号 × 0.5 ≈ 打印磅值，图内最小 12pt（打印 6pt）。
线宽统一抬到 1.5~2.0pt（打印 0.75~1.0pt），圆角 rx ≥ 5。

维护提示：本文件是 UTF-8 中文源码，只用编辑器/Edit 工具修改；
用 PowerShell 的 Get-Content/Set-Content 重写会按 ANSI 解码而毁掉全部中文。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figkit import Canvas                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# ---------------------------------------------------------------- palette
INK, MUTED, FAINT = "1A2430", "6E7C8A", "9AA7B4"
LANE = "7C8B99"                        # 泳道线
GHOST = "D9E0E7"                       # 未使用槽位的占位虚线
PANEL_S, PANEL_F = "D7E0E8", "FCFDFE"
BASE_S, BASE_F = "76899B", "EAF0F5"    # 三列共有结构：低饱和灰蓝
HEAD_F = "EDF2F6"                      # 面板标题条
ACC, ACC_F = "6A4C9C", "EDE6F7"        # 唯一强调色：跨视角交换

VIEWS = [("L", "左", "4E7CAE", "E1EBF5"),
         ("T", "上", "5E9484", "E2EFEA"),
         ("R", "右", "B5795B", "F4E9E1")]

# 线宽 / 圆角（打印后约为一半，故整体抬高）
SW_MOD, SW_LANE, SW_BAR, SW_CONT = 1.6, 1.7, 2.0, 1.5
RX_MOD, RX_BAR, RX_CONT = 6, 8, 10

W, H = 1280, 828

# 深度轴
AX_X, AX_LAB = 88, 78

# 共用前端
BAND_X0, BAND_X1 = 124, 1264
BAND_Y0, BAND_Y1 = 76, 212
BUS_Y = 228                            # 前端 → 三列的总线

# 三个面板（严格等宽等距；所有纵向坐标三列共用，保证逐像素对齐）
PAN_X = [124, 508, 892]
PAN_W = 372
PAN_Y0, PAN_Y1 = 244, 814
LANE_DX = [93, 186, 279]               # 泳道相对面板左边的 x
MOD_W = 84                             # 模块宽（泳道居中）
BAR_X0, BAR_W = 36, 300                # 交换横杠相对面板左边的 x 与宽

Y_HEAD0, Y_HEAD1 = 246, 292
Y_TITLE, Y_SUB = 262, 281
Y_LANELAB = 306
Y_COMB = 316
Y_ENC, H_ENC = 328, 34
Y_S1A, Y_S1B = 370, 416                # 槽位①：编码器内
Y_TOK, H_TOK = 424, 34
Y_S2A, Y_S2B = 466, 526                # 槽位②：编码器后（含三节点全连接）
Y_INSP, H_INSP = 534, 32
Y_D, H_D = 574, 26
Y_S3A, Y_S3B = 608, 654                # 槽位③：距离证据
Y_S, H_S = 662, 32
Y_RULE = 710
Y_FOOT = [722, 740, 758]
Y_CHIP, H_CHIP = 770, 34

TICKS = [("编码器内", (Y_S1A + Y_S1B) / 2),
         ("编码器后", (Y_S2A + Y_S2B) / 2),
         ("距离证据", (Y_S3A + Y_S3B) / 2)]

c = Canvas(W, H, ink=INK, lane=LANE)


# ================================================================ 页眉
def header():
    c.text(W / 2, 30, "跨视角信息交换：位置 × 对象 × 带宽", size=17, bold=True)
    x = 300
    for v, zh, s, f in VIEWS:
        c.rect(x, 49, 15, 15, fill=f, stroke=s, sw=SW_MOD, rx=3)
        c.text(x + 21, 56, [(zh + " ", "zh"), (v, "it")], size=12,
               color=MUTED, align="left")
        x += 74
    x += 20
    c.rect(x, 49, 15, 15, fill=ACC_F, stroke=ACC, sw=SW_BAR, rx=3)
    c.text(x + 21, 56, "跨视角交换", size=12, color=ACC, align="left",
           bold=True)
    x += 128
    c.rect(x, 49, 15, 15, fill="FFFFFF", stroke=GHOST, sw=SW_MOD, rx=3,
           dash="3,3")
    c.text(x + 21, 56, "该策略不使用的交换槽位", size=12, color=FAINT,
           align="left")


# ================================================================ 共用前端
def front_end():
    """三列完全相同的部分，只画一次。"""
    c.rect(BAND_X0, BAND_Y0, BAND_X1 - BAND_X0, BAND_Y1 - BAND_Y0,
           fill=PANEL_F, stroke=PANEL_S, sw=SW_CONT, rx=RX_CONT, z=-9)
    c.chip(BAND_X0 + 12, BAND_Y0 + 10, 196, 26, "共用前端 · 三列完全相同",
           size=13, fill=HEAD_F, color=INK)

    for (v, zh, s, f), cx in zip(VIEWS, [216, 372, 528]):
        c.rect(cx - 28, 118, 56, 22, fill=f, stroke=s, sw=SW_MOD, rx=5)
        c.text(cx, 129, [(zh + " ", "zh"), (v, "it")], size=12, bold=True)
        c.rect(cx - 64, 146, 128, 46, fill=None, stroke=s, sw=SW_MOD, rx=5)
        c.image(cx - 64, 146, 128, 46, os.path.join(ASSETS, "cir_%s.png" % v))
    c.text(376, 200, [("三视角 CIR　", "zh"), ("X", "it"), ("v", "itsub"),
                      (" ∈ ℝ", "zh"), ("90×189", "sup")], size=12,
           color=MUTED)

    c.arrow(592, 169, 616, 169, color=LANE, sw=SW_LANE)
    c.box(616, 146, 82, 46, ["幅度标准化", [("abs-std", "en")]],
          stroke=BASE_S, sw=SW_MOD, rx=RX_MOD, fill="FFFFFF")
    c.arrow(698, 169, 722, 169, color=LANE, sw=SW_LANE)
    c.box(722, 140, 366, 58,
          [[("共享编码器 ", "zh"), ("E", "it"), ("θ", "itsub"),
            (" · 前 4−", "zh"), ("K", "it"), (" 层", "zh")],
           "L / T / R 权重共享，逐视角独立前向"],
          stroke=BASE_S, sw=SW_MOD, rx=RX_MOD, fill=BASE_F)
    c.arrow(1088, 169, 1112, 169, color=LANE, sw=SW_LANE)
    c.lines(1188, 162, ["输出三视角 token",
                        "上：时序　下：距离"], size=12, color=MUTED)

    # 前端 → 三列：从编码器输出引一条竖线到总线，再三支等长下行（无斜线）
    mids = [x0 + PAN_W / 2 for x0 in PAN_X]
    c.line(905, BAND_Y1, 905, BUS_Y, color=LANE, sw=SW_LANE)
    c.line(mids[0], BUS_Y, mids[2], BUS_Y, color=LANE, sw=SW_LANE)
    for mx in mids:
        c.arrow(mx, BUS_Y, mx, PAN_Y0, color=LANE, sw=SW_LANE)


def depth_axis():
    """左侧网络深度轴：交换横杠越靠下 = 交换发生得越晚。"""
    c.line(AX_X, Y_COMB, AX_X, Y_S3B + 8, color=BASE_S, sw=SW_LANE,
           arrow=True)
    c.text(AX_X, Y_COMB - 14, "网络深度", size=12, color=MUTED, bold=True)
    for lab, y in TICKS:
        c.line(AX_X - 5, y, AX_X + 5, y, color=BASE_S, sw=SW_MOD)
        c.text(AX_LAB, y, lab, size=12, color=MUTED, align="right")
    c.text(AX_X, Y_S3B + 26, "越晚", size=12, color=FAINT)


# ================================================================ 面板
def panel(i, kind, title, sub, foot, concl):
    """kind ∈ {A 决策级, B 特征级, C 瓶颈}；三列骨架完全一致。"""
    x0 = PAN_X[i]
    cxs = [x0 + dx for dx in LANE_DX]
    bx0, bx1 = x0 + BAR_X0, x0 + BAR_X0 + BAR_W

    c.rect(x0, PAN_Y0, PAN_W, PAN_Y1 - PAN_Y0, fill=PANEL_F, stroke=PANEL_S,
           sw=SW_CONT, rx=RX_CONT, z=-9)
    c.rect(x0 + 8, Y_HEAD0, PAN_W - 16, Y_HEAD1 - Y_HEAD0, fill=HEAD_F,
           stroke=None, rx=RX_MOD, z=-8)
    c.text(x0 + PAN_W / 2, Y_TITLE, title, size=13, bold=True)
    c.text(x0 + PAN_W / 2, Y_SUB, sub, size=12, color=ACC)

    # ---- 三条贯通泳道（画在最底层，模块盖在上面，天然对齐）
    for cx in cxs:
        c.line(cx, Y_COMB, cx, Y_S3A, color=LANE, sw=SW_LANE, z=-4)

    # ---- 入口梳状分叉：一进三
    c.line(x0 + PAN_W / 2, PAN_Y0, x0 + PAN_W / 2, Y_COMB, color=LANE,
           sw=SW_LANE, z=-4)
    c.line(cxs[0], Y_COMB, cxs[2], Y_COMB, color=LANE, sw=SW_LANE, z=-4)
    for (v, zh, s, f), cx in zip(VIEWS, cxs):
        c.text(cx, Y_LANELAB, [(zh + " ", "zh"), (v, "it")], size=12,
               color=s, bold=True)

    # ---- 编码器末段
    enc_row = ([[("瓶颈层 ×", "zh"), ("K", "it")]] if kind == "C"
               else [[("编码层 ×", "zh"), ("K", "it")]])
    for (v, zh, s, f), cx in zip(VIEWS, cxs):
        c.box(cx - MOD_W / 2, Y_ENC, MOD_W, H_ENC, enc_row,
              fill="FFFFFF", stroke=ACC if kind == "C" else BASE_S,
              sw=SW_MOD, rx=RX_MOD, size=12)

    # ---- 槽位①：编码器内（仅 (c) 使用）
    _slot(kind == "C", bx0, bx1, Y_S1A, Y_S1B, cxs,
          Y_ENC + H_ENC, Y_TOK, _bottleneck)

    # ---- 编码后 token（原子模块：上半时序、下半距离）
    for (v, zh, s, f), cx in zip(VIEWS, cxs):
        c.rect(cx - MOD_W / 2, Y_TOK, MOD_W, H_TOK, fill="FFFFFF",
               stroke=s, sw=SW_MOD, rx=RX_MOD)
        c.rect(cx - MOD_W / 2 + 8, Y_TOK + 6, MOD_W - 16, 9, fill=f,
               stroke=s, sw=1.0, rx=3)
        c.rect(cx - MOD_W / 2 + 8, Y_TOK + 19, MOD_W - 16, 9, fill="FFFFFF",
               stroke=s, sw=1.0, rx=3)

    # ---- 槽位②：编码器后（仅 (b) 使用）
    _slot(kind == "B", bx0, bx1, Y_S2A, Y_S2B, cxs,
          Y_TOK + H_TOK, Y_INSP, _view_attention)

    # ---- 逐视角检查器 → 原型距离
    for (v, zh, s, f), cx in zip(VIEWS, cxs):
        c.box(cx - MOD_W / 2, Y_INSP, MOD_W, H_INSP,
              [[("检查器 ", "zh"), ("η", "it"), ("k̂", "itsub"),
                ("(" + v + ")", "sup")]], fill="FFFFFF", stroke=BASE_S,
              sw=SW_MOD, rx=RX_MOD, size=12)
        c.arrow(cx, Y_INSP + H_INSP, cx, Y_D, color=LANE, sw=SW_LANE)
        c.box(cx - 30, Y_D, 60, H_D, [[("d", "it"), (v, "itsub")]],
              fill=f, stroke=s, sw=SW_MOD, rx=5, size=13)
        c.arrow(cx, Y_D + H_D, cx, Y_S3A, color=LANE, sw=SW_LANE)

    # ---- 槽位③：距离证据（三列都有；(a) 以它为唯一交换点故点亮）
    _distance_agg(bx0, bx1, kind == "A", x0)

    # ---- 位置 / 对象 / 带宽 + 结论
    c.line(x0 + 24, Y_RULE, x0 + PAN_W - 24, Y_RULE, color=PANEL_S, sw=SW_MOD)
    for row, y in zip(foot, Y_FOOT):
        c.text(x0 + 24, y, row, size=12, color=MUTED, align="left")
    c.box(x0 + 24, Y_CHIP, PAN_W - 48, H_CHIP, [concl], fill="FFFFFF",
          stroke=ACC if kind == "A" else BASE_S, sw=SW_MOD, rx=RX_MOD,
          size=12, bold=True)


def _slot(active, bx0, bx1, ya, yb, cxs, y_prev, y_next, draw):
    """交换槽位：用则画横杠并分段接线，不用则画占位虚线、泳道直穿。"""
    if active:
        draw(bx0, bx1, ya, yb, cxs)
        for cx in cxs:
            c.arrow(cx, y_prev, cx, ya, color=ACC, sw=SW_LANE)
            c.arrow(cx, yb, cx, y_next, color=ACC, sw=SW_LANE)
    else:
        c.rect(bx0, ya, bx1 - bx0, yb - ya, fill=None, stroke=GHOST,
               sw=SW_MOD, rx=RX_BAR, dash="4,4", z=-5)
        c.text((bx0 + bx1) / 2, (ya + yb) / 2 - 15, "此处无跨视角交换",
               size=12, color=FAINT)
        for cx in cxs:
            c.arrow(cx, y_prev, cx, y_next, color=LANE, sw=SW_LANE)


def _bottleneck(bx0, bx1, ya, yb, cxs):
    """槽位①：少量共享瓶颈 token 构成的总线，横跨三条泳道。"""
    c.rect(bx0, ya, bx1 - bx0, yb - ya, fill=ACC_F, stroke=ACC, sw=SW_BAR,
           rx=RX_BAR, z=-2)
    c.text((bx0 + bx1) / 2, ya + 14,
           [("共享瓶颈 token（", "zh"), ("B", "it"), (" = 4）", "zh")],
           size=12, color=ACC, bold=True)
    ybus = yb - 16
    c.line(cxs[0], ybus, cxs[2], ybus, color=ACC, sw=SW_BAR)
    mid = (cxs[0] + cxs[2]) / 2
    for k in range(4):
        c.circ(mid - 51 + k * 34, ybus, 7, fill=ACC, stroke=None)


def _view_attention(bx0, bx1, ya, yb, cxs):
    """槽位②：视角维多头自注意力，三节点全连接（含未屏蔽的自连接）。"""
    c.rect(bx0, ya, bx1 - bx0, yb - ya, fill=ACC_F, stroke=ACC, sw=SW_BAR,
           rx=RX_BAR, z=-2)
    c.text((bx0 + bx1) / 2, ya + 13, "视角维多头自注意力", size=12,
           color=ACC, bold=True)
    ny = ya + 32
    c.line(cxs[0], ny, cxs[1], ny, color=ACC, sw=SW_BAR, both=True,
           arrow=True)
    c.line(cxs[1], ny, cxs[2], ny, color=ACC, sw=SW_BAR, both=True,
           arrow=True)
    yb2 = yb - 10                                 # L↔R：绕到节点下方走直角
    c.line(cxs[0], ny + 10, cxs[0], yb2, color=ACC, sw=SW_MOD)
    c.line(cxs[0], yb2, cxs[2], yb2, color=ACC, sw=SW_MOD)
    c.line(cxs[2], yb2, cxs[2], ny + 10, color=ACC, sw=SW_MOD)
    for (v, zh, s, f), cx in zip(VIEWS, cxs):
        c.circ(cx, ny, 9.5, fill=f, stroke=s, sw=SW_MOD)
        c.text(cx, ny, [(v, "it")], size=12, bold=True)


def _distance_agg(bx0, bx1, accent, x0):
    """槽位③：按式(2)聚合标量距离证据；三列都做，(a) 以此为唯一交换点。"""
    col = ACC if accent else BASE_S
    c.rect(bx0, Y_S3A, bx1 - bx0, Y_S3B - Y_S3A, fill=ACC_F if accent
           else "FFFFFF", stroke=col, sw=SW_BAR if accent else SW_MOD,
           rx=RX_BAR, z=-2)
    c.text((bx0 + bx1) / 2, Y_S3A + 15,
           [("mean", "en"), ("v", "itsub"), ("(", "en"), ("d", "it"),
            ("v", "itsub"), (") + β·var", "en"), ("v", "itsub"),
            ("(", "en"), ("d", "it"), ("v", "itsub"), (")", "en")],
           size=12, color=INK)
    c.text((bx0 + bx1) / 2, Y_S3B - 14, "式(2) 跨视角距离聚合", size=12,
           color=col, bold=True)
    mx = x0 + PAN_W / 2
    c.arrow(mx, Y_S3B, mx, Y_S, color=col, sw=SW_LANE)
    c.box(mx - 78, Y_S, 156, H_S, [[("异常分数 ", "zh"), ("s", "it")]],
          fill="FFFFFF", stroke=col, sw=SW_MOD, rx=RX_MOD, size=13)


# ================================================================ build
def main():
    header()
    front_end()
    depth_axis()
    panel(0, "A", "(a) 决策级距离聚合（本文采用）", "交换发生在逐视角检查器之后",
          ["位置：逐视角检查器之后",
           [("对象：原型距离 ", "zh"), ("d", "it"), ("v", "itsub"),
            ("（每视角 1 个标量）", "zh")],
           "带宽：3 个标量"],
          "特征全程独立，仅聚合距离证据")
    panel(1, "B", "(b) 编码后特征级融合", "Q/K/V 同源，自连接未屏蔽",
          ["位置：编码器之后、检查器之前",
           "对象：对齐慢时间位置上的时序 token",
           "带宽：全带宽；距离路径不参与"],
          "时序 token 全带宽交互，距离路径独立")
    panel(2, "C", "(c) MBT 式瓶颈中层融合", "瓶颈 token 交换 · 编码器内部后 K 层",
          ["位置：编码器内部（后 K 层）",
           [("对象：视角 token ↔ 共享瓶颈 token", "zh")],
           [("带宽：", "zh"), ("B", "it"), (" = 4 个瓶颈 token", "zh")]],
          "少量瓶颈 token 构成唯一跨视角通道")
    n = c.write_all(HERE, "fig2_exchange",
                    caption="<b>图 2</b>&nbsp;&nbsp;3 种跨视角信息交换策略的"
                            "结构对比：位置、对象与带宽")
    print("shapes:", n)
    for e in ("svg", "html", "pptx"):
        print(" ->", os.path.join(HERE, "fig2_exchange." + e))


if __name__ == "__main__":
    main()
