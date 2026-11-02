# -*- coding: utf-8 -*-
"""论文一（中文版）图 2 —— 多视角开集手势识别总体框架与跨视角融合策略对比。

    python paper1/figs/build_fig2_mvosr_cvpr.py     # 产出 pptx / svg / html

内容以用户提供的 GPT-Image 草图为蓝本，逐项对正文与代码核准后重绘：

  ① 总体框架：采集几何 → 三视角 CIR → abs-std → 共享权重双路径编码器
     → 两路展平后分叉（闭集读出跨视角取均值 / 检查器保持逐视角 + 视角嵌入）
     → 阶段一闭集分类 → 阶段二逐类×逐视角检查器 → 式(2) 聚合 → 阈值判决
  ② 三种跨视角交换插入位置的对比：瓶颈中层 / 特征级 / 决策级（本文）

相对草图修正的事实性问题（依据 mvosr_paper_cn.tex 与 uwb_osr/model.py）：
  * 闭集头输入是「两路 token 各自展平 → 对 3 视角取均值 → 拼接」的 8928 维
    （model.py:356-359），而非三路特征码并列送入；检查器分支则保持逐视角并
    加视角嵌入 f_v = enc(X_v) + e_v（model.py:221-225）。草图把两支画成一支。
  * 采集几何补上真实尺寸：左右基线 1.1 m、上雷达位于水平中点上方 0.55 m，
    并用箭头标出各雷达指向操作区的径向，草图中三雷达朝向是随意的。
  * 特征级融合按正文置于特征码之上、检查器之前；代码中仅时间路径参与
    （model.py:355、364-368），图中据实标注。
  * 输入缩略图改用数据集真实 CIR（G5，clutter-removed），非示意纹理。

版式：画布 1280 x 980 units（1 unit = 0.01 in），\\textwidth = 16.2 cm = 6.378 in
→ 打印缩放约 0.498，即幻灯片字号 x 0.5 ≈ 打印磅值，故图内最小字号 12pt（6pt）。
"""
import base64
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# ============================================================== palette
INK, MUTED, FAINT = "1A2430", "6E7C8A", "9AA7B4"
STROKE, LANE, HAIR = "AEBBC7", "8FA0AF", "E6ECF1"
CONT_F, CONT_S, CHIP_F = "FBFCFD", "E1E8EE", "EDF1F5"
BOX_F = "FFFFFF"

TIME_S, TIME_F = "D9553C", "FCEBE6"      # 时间路径
RANG_S, RANG_F = "3D7EB8", "E6F0F9"      # 距离路径
EXCH_S, EXCH_F = "6E4FA3", "ECE5F7"      # 跨视角交换（编码阶段）
OSR_S,  OSR_F = "00A087", "E1F3EF"       # 开集判决 / 式(2) / 决策级
CLS_S,  CLS_F = "7C8896", "F0F3F6"       # 闭集分类

W, H = 1280, 980
P = []


# ============================================================== primitives
def rect(x, y, w, h, fill=None, stroke=None, sw=1.1, rx=4, dash=None, z=0):
    P.append(dict(k="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
                  sw=sw, rx=rx, dash=dash, z=z))


def circ(cx, cy, r, fill=None, stroke=None, sw=1.1, dash=None, z=0):
    P.append(dict(k="circ", cx=cx, cy=cy, r=r, fill=fill, stroke=stroke,
                  sw=sw, dash=dash, z=z))


def line(x1, y1, x2, y2, color=LANE, sw=1.1, arrow=False, dash=None, z=0,
         both=False):
    P.append(dict(k="line", x1=x1, y1=y1, x2=x2, y2=y2, color=color, sw=sw,
                  arrow=arrow, dash=dash, z=z, both=both))


def text(cx, cy, runs, size=12, color=INK, align="center", bold=False, z=5):
    """runs: str 或 [(片段, 类型)]，类型 ∈ zh/en/it/sub/sup/itsub。"""
    if isinstance(runs, str):
        runs = [(runs, "zh")]
    P.append(dict(k="text", cx=cx, cy=cy, runs=runs, size=size, color=color,
                  align=align, bold=bold, z=z))


def image(x, y, w, h, path, z=0):
    P.append(dict(k="image", x=x, y=y, w=w, h=h, path=path, z=z))


def stack_box(x, y, w, h, runs, stroke=STROKE, fill=BOX_F, size=12, n=2):
    """带叠层阴影的模块，暗示"若干层"。"""
    for i in range(n, 0, -1):
        rect(x + i * 2.6, y - i * 2.6, w, h, fill="F1F4F7" if i > 1 else "F8FAFC",
             stroke="D3DCE4", sw=0.85, z=-3 - i)
    rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.15)
    _multiline(x + w / 2, y + h / 2, runs, size)


def box(x, y, w, h, runs, stroke=STROKE, fill=BOX_F, size=12, sw=1.15):
    rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw)
    _multiline(x + w / 2, y + h / 2, runs, size)


def _multiline(cx, cy, runs, size, color=INK, bold=False):
    """runs 为 list-of-lines（每行本身是 runs 列表或字符串）时纵向排布。"""
    if isinstance(runs, str) or (runs and isinstance(runs[0], tuple)):
        text(cx, cy, runs, size=size, color=color, bold=bold)
        return
    lh = size * 1.45
    y0 = cy - (len(runs) - 1) * lh / 2
    for i, ln in enumerate(runs):
        text(cx, y0 + i * lh, ln, size=size, color=color, bold=bold)


def chip(x, y, w, h, s, size=13, fill=CHIP_F, color=INK):
    rect(x, y, w, h, fill=fill, stroke=None, rx=4)
    text(x + w / 2, y + h / 2, s, size=size, color=color, bold=True)


def arrow(x1, y1, x2, y2, color=LANE, sw=1.25, dash=None):
    line(x1, y1, x2, y2, color=color, sw=sw, arrow=True, dash=dash)


# ============================================================== section 1
SEC1_Y0, SEC1_Y1 = 58, 600
ENC_X0, ENC_X1 = 412, 1046
ENC_Y0, ENC_Y1 = 82, 316
T_CY, R_CY = 166, 254                    # 两条路径的中心线


def section1():
    text(26, 38, "① 总体框架｜三视角权重共享主干与两阶段开集判决", size=16,
         color=INK, align="left", bold=True)
    rect(16, SEC1_Y0, 1248, SEC1_Y1 - SEC1_Y0, fill=CONT_F, stroke=CONT_S,
         rx=10, z=-9)
    _geometry()
    _inputs()
    _encoder()
    _stage1()
    _stage2()
    _legend()


# ---------------------------------------------------------------- 采集几何
def _geometry():
    hx, hy, hr = 100, 178, 26            # 操作区中心
    rad = [("L", 40, 246), ("T", 100, 100), ("R", 160, 246)]
    for name, x, y in rad:
        rect(x - 15, y - 8, 30, 16, fill="FFFFFF", stroke=INK, sw=1.2, rx=3)
        rect(x - 9, y - 4, 7, 8, fill=CLS_F, stroke=CLS_S, sw=0.8, rx=1)
        circ(x + 5, y, 3.4, fill=None, stroke=INK, sw=1.0)
        # 指向操作区的径向
        dx, dy = hx - x, hy - y
        d = (dx * dx + dy * dy) ** 0.5
        sx, sy = x + dx / d * 12, y + dy / d * 12
        ex, ey = hx - dx / d * (hr + 4), hy - dy / d * (hr + 4)
        arrow(sx, sy, ex, ey, color=FAINT, sw=1.0)
    text(40, 226, [("L", "it"), (" 左", "zh")], size=12, color=INK, bold=True)
    text(160, 226, [("R", "it"), (" 右", "zh")], size=12, color=INK, bold=True)
    text(100, 80, [("T", "it"), (" 上", "zh")], size=12, color=INK, bold=True)

    circ(hx, hy, hr, fill="FFFFFF", stroke=FAINT, sw=1.0, dash="4,3")
    _multiline(hx, hy, ["手势", "操作区"], 12, color=MUTED)

    # 尺寸标注
    line(40, 268, 160, 268, color=FAINT, sw=0.9)
    for x in (40, 160):
        line(x, 264, x, 272, color=FAINT, sw=0.9)
    text(100, 280, [("1.1 m", "en")], size=12, color=MUTED)
    line(134, 108, 134, 246, color=FAINT, sw=0.9, dash="3,3")
    for y in (108, 246):
        line(130, y, 138, y, color=FAINT, sw=0.9)
    text(142, 177, [("0.55 m", "en")], size=12, color=MUTED, align="left")

    text(114, 300, "三视角采集几何", size=12, color=MUTED)


# ---------------------------------------------------------------- 输入
def _inputs():
    ys = [100, 176, 252]
    for (v, zh), y in zip([("L", "左"), ("T", "上"), ("R", "右")], ys):
        text(212, y - 9, [("X", "it"), (v, "itsub"), ("（%s）" % zh, "zh")],
             size=12, color=INK, align="left")
        rect(212, y, 118, 56, fill=None, stroke=STROKE, sw=1.0, rx=2)
        image(212, y, 118, 56, os.path.join(ASSETS, "cir_%s.png" % v))
    text(212, 324, [("纵 慢时间 90　横 距离 189", "zh")], size=12,
         color=MUTED, align="left")

    rect(342, 96, 58, 214, fill=CLS_F, stroke=CLS_S, sw=1.1, rx=5)
    _multiline(371, 203, ["幅度", "标准化", [("abs-std", "en")]], 12,
               color=INK)
    for y in [128, 204, 280]:
        arrow(330, y, 342, y)
        line(400, y, 406, y, color=LANE, sw=1.25)
    line(406, 128, 406, 280, color=LANE, sw=1.25)      # 三视角共用同一套权重
    arrow(406, 204, ENC_X0, 204, color=LANE)


# ---------------------------------------------------------------- 编码器
def _encoder():
    rect(ENC_X0, ENC_Y0, ENC_X1 - ENC_X0, ENC_Y1 - ENC_Y0, fill="FFFFFF",
         stroke=CONT_S, sw=1.2, rx=8, z=-6)
    chip(422, 88, 614, 24,
         "共享权重双路径编码器｜L / T / R 三视角共用同一套参数", size=13)

    paths = [("时间", T_CY, TIME_S, TIME_F, True,
              [("时间路径：按行切片 → ", "zh"), ("90", "en"),
               (" 个 token（每个 ", "zh"), ("189", "en"), (" 维）", "zh")],
              [("Z", "it"), ("t", "itsub"), (" ∈ ℝ", "zh"), ("90×32", "sup")]),
             ("距离", R_CY, RANG_S, RANG_F, False,
              [("距离路径：按列切片 → ", "zh"), ("189", "en"),
               (" 个 token（每个 ", "zh"), ("90", "en"), (" 维）", "zh")],
              [("Z", "it"), ("r", "itsub"), (" ∈ ℝ", "zh"), ("189×32", "sup")])]

    for _, cy, cs, cf, row_wise, lab, zlab in paths:
        text(424, cy - 40, lab, size=12, color=cs, align="left", bold=True)
        _slice_glyph(424, cy, 64, 52, cs, cf, row_wise)
        arrow(488, cy, 500, cy)
        _token_strip(502, cy, cs, cf)
        arrow(576, cy, 588, cy)
        box(588, cy - 22, 84, 44, ["线性投影", [("D", "it"), (" = 32", "en")]],
            stroke=cs, fill=cf)
        arrow(672, cy, 684, cy)
        stack_box(684, cy - 23, 98, 46,
                  [[("Transformer", "en")], "编码层 × 4"], stroke=cs, fill=cf)
        arrow(782, cy, 794, cy)
        box(794, cy - 20, 84, 40, zlab, stroke=cs)
        arrow(878, cy, 890, cy)

    box(890, 146, 70, 128, ["两路", "token", "展平"], stroke=STROKE)
    # 分叉：两支都吃两路展平结果，差别在「是否对视角取均值 / 是否加视角嵌入」
    line(960, 210, 966, 210, color=STROKE, sw=1.25)
    line(966, 167, 966, 255, color=STROKE, sw=1.25)
    arrow(966, 167, 972, 167, color=OSR_S)
    arrow(966, 255, 972, 255, color=CLS_S)
    box(972, 144, 64, 46, ["拼接", [("+ e", "it"), ("v", "itsub")]],
        stroke=OSR_S, fill=OSR_F)
    box(972, 232, 64, 46, ["跨视角", "取均值"], stroke=CLS_S, fill=CLS_F)
    text(729, 298,
         [("每层 4 头自注意力 + 两层 MLP（GELU，dropout 0.1）", "zh")],
         size=12, color=MUTED)

    # 逐视角特征码
    for i, v in enumerate(["L", "T", "R"]):
        y = 122 + i * 40
        box(1060, y, 74, 30, [[("f", "it"), (v, "itsub")]], stroke=OSR_S,
            size=13)
        line(1036, 167, 1050, 167, color=OSR_S, sw=1.25)
        arrow(1050, y + 15, 1060, y + 15, color=OSR_S)
    line(1050, 137, 1050, 227, color=OSR_S, sw=1.25)
    text(1060, 248, "逐视角特征码", size=12, color=OSR_S, align="left")
    text(1060, 264, [("f", "it"), ("v", "itsub"),
                     (" = enc(", "it"), ("X", "it"), ("v", "itsub"),
                     (") + e", "it"), ("v", "itsub")], size=12, color=MUTED,
         align="left")


def _slice_glyph(x, cy, w, h, cs, cf, row_wise):
    """切片示意：时间路径按行取 token，距离路径按列取 token。"""
    y = cy - h / 2
    rect(x, y, w, h, fill="EFF3F7", stroke=STROKE, sw=1.0, rx=2)
    if row_wise:
        for k in range(4):
            yy = y + 6 + k * 11
            rect(x + 3, yy, w - 6, 6, fill=cf, stroke=cs, sw=0.8, rx=1)
    else:
        for k in range(4):
            xx = x + 6 + k * 13
            rect(xx, y + 3, 6, h - 6, fill=cf, stroke=cs, sw=0.8, rx=1)


def _token_strip(x, cy, cs, cf):
    for k in range(3):
        rect(x + k * 15, cy - 5.5, 11, 11, fill=cf, stroke=cs, sw=0.9, rx=2)
    text(x + 50, cy, [("…", "en")], size=12, color=cs)
    rect(x + 62, cy - 5.5, 11, 11, fill=cf, stroke=cs, sw=0.9, rx=2)


# ---------------------------------------------------------------- 阶段一
S2_Y0, S2_Y1 = 362, 588


def _stage1():
    rect(420, S2_Y0, 344, S2_Y1 - S2_Y0, fill="FFFFFF", stroke=CONT_S, sw=1.2,
         rx=8, z=-6)
    chip(430, S2_Y0 + 8, 324, 24, "阶段一｜闭集分类", size=13)
    arrow(1004, 278, 1004, 340, color=CLS_S)
    line(1004, 340, 592, 340, color=CLS_S, sw=1.25)
    arrow(592, 340, 592, S2_Y0 + 42, color=CLS_S)
    text(996, 328, "跨视角平均后的 8928 维读出", size=12, color=CLS_S,
         align="right")

    box(446, S2_Y0 + 52, 292, 52,
        ["闭集分类头（交叉熵训练）",
         [("8928 → 2048 → 256 → |𝒦|", "en")]], stroke=CLS_S, fill=CLS_F)
    arrow(592, S2_Y0 + 104, 592, S2_Y0 + 126, color=CLS_S)
    box(596, S2_Y0 + 126, 120, 32, [[("预测类 ", "zh"), ("k̂", "it")]],
        stroke=CLS_S)
    text(656, S2_Y0 + 178, [("按 ", "zh"), ("k̂", "it"), (" 选定检查器", "zh")],
         size=12, color=CLS_S)
    text(592, S2_Y0 + 202, "仅见已知类训练样本，第二阶段冻结", size=12,
         color=MUTED)


# ---------------------------------------------------------------- 阶段二
def _stage2():
    x0 = 778
    rect(x0, S2_Y0, 1254 - x0, S2_Y1 - S2_Y0, fill="FFFFFF", stroke=CONT_S,
         sw=1.2, rx=8, z=-6)
    chip(x0 + 10, S2_Y0 + 8, 1244 - x0, 24,
         "阶段二｜逐类 × 逐视角深度检查器与决策级聚合", size=13)

    lanes = [S2_Y0 + 74, S2_Y0 + 122, S2_Y0 + 170]
    ix0, ix1 = x0 + 16, x0 + 152
    dx0, dx1 = x0 + 166, x0 + 212
    bar_cx = x0 + 240
    s_cx = x0 + 296

    for v, ly in zip(["L", "T", "R"], lanes):
        arrow(x0 - 14, ly, ix0, ly, color=OSR_S)
        box(ix0, ly - 17, ix1 - ix0, 34,
            [[("检查器 ", "zh"), ("η", "it"), ("k̂", "itsub"),
              ("(" + v + ")", "sup")]], stroke=OSR_S, size=13)
        arrow(ix1, ly, dx0, ly, color=OSR_S)
        box(dx0, ly - 15, dx1 - dx0, 30, [[("d", "it"), (v, "itsub")]],
            stroke=OSR_S, size=13)
        arrow(dx1, ly, bar_cx - 12, ly, color=OSR_S)

    top, bot = lanes[0] - 20, lanes[-1] + 20
    rect(bar_cx - 12, top, 24, bot - top, fill=OSR_F, stroke=OSR_S, sw=1.3,
         rx=6)
    cy = lanes[1]
    line(bar_cx, lanes[0] + 6, bar_cx, cy - 9, color="FFFFFF", sw=1.5,
         arrow=True)
    line(bar_cx, lanes[2] - 6, bar_cx, cy + 9, color="FFFFFF", sw=1.5,
         arrow=True)
    circ(bar_cx, cy, 4.6, fill="FFFFFF")
    text(bar_cx, top - 14, "式(2)", size=12, color=OSR_S, bold=True)
    text(bar_cx, bot + 14, "跨视角聚合", size=12, color=OSR_S, bold=True)

    arrow(bar_cx + 12, cy, s_cx - 21, cy, color=OSR_S, sw=1.5)
    circ(s_cx, cy, 21, fill="FFFFFF", stroke=OSR_S, sw=1.5)
    text(s_cx, cy, [("s", "it")], size=16, color=INK)
    text(s_cx, cy + 36, "异常分数", size=12, color=MUTED)

    box(x0 + 336, cy - 42, 130, 32,
        [[("s", "it"), (" ≤ τ：判为已知", "zh")]], stroke=OSR_S, fill=OSR_F)
    box(x0 + 336, cy + 10, 130, 32,
        [[("s", "it"), (" > τ：拒识未知", "zh")]], stroke=CLS_S, fill=CLS_F)
    arrow(s_cx + 21, cy - 8, x0 + 336, cy - 26, color=OSR_S)
    arrow(s_cx + 21, cy + 8, x0 + 336, cy + 26, color=CLS_S)

    # k̂ 选定该用哪一组检查器
    line(716, S2_Y0 + 142, 752, S2_Y0 + 142, color=CLS_S, sw=1.0, dash="4,3")
    line(752, S2_Y0 + 142, 752, lanes[1], color=CLS_S, sw=1.0, dash="4,3")
    arrow(752, lanes[1], ix0, lanes[1], color=CLS_S, dash="4,3")


def _legend():
    x, y = 32, 372
    items = [(TIME_S, TIME_F, "时间路径"), (RANG_S, RANG_F, "距离路径"),
             (OSR_S, OSR_F, "开集判决通路"), (CLS_S, CLS_F, "闭集分类通路")]
    for i, (s, f, lab) in enumerate(items):
        yy = y + i * 26
        rect(x, yy - 8, 16, 16, fill=f, stroke=s, sw=1.1, rx=3)
        text(x + 24, yy, lab, size=12, color=MUTED, align="left")
    yy = y + 4 * 26 + 4
    arrow(x, yy, x + 22, yy, color=LANE)
    text(x + 30, yy, "数据流", size=12, color=MUTED, align="left")
    line(x, yy + 24, x + 22, yy + 24, color=CLS_S, sw=1.0, dash="4,3")
    text(x + 30, yy + 24, "选择关系", size=12, color=MUTED, align="left")
    text(x, yy + 56, "三视角样本是同一次手势", size=12, color=FAINT,
         align="left")
    text(x, yy + 72, "在 3 台雷达下的径向投影", size=12, color=FAINT,
         align="left")


# ============================================================== section 2
SEC2_Y0, SEC2_Y1 = 644, 968
PAN_Y0, PAN_H = 656, 292


def section2():
    text(26, 624, "② 跨视角交换的插入位置对比｜仅编码路径不同，主干与下游完全一致",
         size=16, color=INK, align="left", bold=True)
    rect(16, SEC2_Y0, 1248, SEC2_Y1 - SEC2_Y0, fill=CONT_F, stroke=CONT_S,
         rx=10, z=-9)
    _panel(28, "（a）瓶颈中层融合", "交换在编码器内部（后 K 层）", "A",
           "两路各设共享瓶颈 token（B = 4），总深度不变")
    _panel(440, "（b）特征级融合", "交换在编码器之后、检查器之前", "B",
           "以其余视角为键值交互一次（仅时间路径）")
    _panel(852, "（c）决策级聚合", "编码路径无任何跨视角交换", "C",
           "3 路特征码全程独立，只在式(2)处汇合")


def _panel(x0, title, sub, kind, note):
    w = 400
    acc = OSR_S if kind == "C" else EXCH_S
    rect(x0, PAN_Y0, w, PAN_H, fill="FFFFFF",
         stroke=OSR_S if kind == "C" else CONT_S, sw=1.4 if kind == "C" else 1.2,
         rx=8, z=-6)
    text(x0 + 16, PAN_Y0 + 20, title, size=14, color=INK, align="left",
         bold=True)
    text(x0 + 16, PAN_Y0 + 42, sub, size=12, color=acc, align="left")
    if kind == "C":
        rect(x0 + w - 90, PAN_Y0 + 10, 76, 21, fill=OSR_S, rx=4)
        text(x0 + w - 52, PAN_Y0 + 20.5, "本文方法", size=12, color="FFFFFF",
             bold=True)

    lanes = [PAN_Y0 + 92, PAN_Y0 + 134, PAN_Y0 + 176]
    ex0, ex1 = x0 + 34, x0 + 228     # 编码器总跨度三面板一致 = 总深度不变
    fx0 = x0 + 276
    for v, ly in zip(["L", "T", "R"], lanes):
        text(x0 + 16, ly, [(v, "it")], size=13, color=MUTED, bold=True,
             align="left")
        if kind == "A":
            box(ex0, ly - 15, 88, 30,
                [[("编码 ×(4−", "zh"), ("K", "it"), (")", "zh")]], size=12)
            arrow(ex0 + 88, ly, x0 + 130, ly, color=LANE)
            arrow(x0 + 154, ly, x0 + 162, ly, color=EXCH_S)
            box(x0 + 162, ly - 15, ex1 - x0 - 162, 30,
                [[("融合 ×", "zh"), ("K", "it")]], stroke=EXCH_S, size=12)
            arrow(ex1, ly, fx0, ly, color=EXCH_S)
        elif kind == "B":
            box(ex0, ly - 15, ex1 - ex0, 30, [[("编码层 × 4", "zh")]], size=12)
            arrow(ex1, ly, x0 + 236, ly, color=LANE)
            arrow(x0 + 260, ly, fx0, ly, color=EXCH_S)
        else:
            box(ex0, ly - 15, ex1 - ex0, 30, [[("编码层 × 4", "zh")]], size=12)
            arrow(ex1, ly, fx0, ly, color=LANE)
        fl = [("f̃" if kind == "B" else "f", "it"), (v, "itsub")]
        box(fx0, ly - 15, 48, 30, [fl],
            stroke=OSR_S if kind == "C" else EXCH_S, size=13)

    if kind == "A":
        _capsule(x0 + 142, lanes, EXCH_S, EXCH_F, "token")
        text(x0 + 142, PAN_Y0 + 64, "共享瓶颈", size=12, color=EXCH_S,
             bold=True)
    elif kind == "B":
        _capsule(x0 + 248, lanes, EXCH_S, EXCH_F, "attn")
        text(x0 + 248, PAN_Y0 + 64, "跨视角注意力", size=12, color=EXCH_S,
             bold=True)

    # 共用下游
    ty = PAN_Y0 + 212
    rect(x0 + 34, ty, 332, 32, fill=OSR_F if kind == "C" else "F5F7F9",
         stroke=OSR_S if kind == "C" else STROKE, sw=1.1, rx=5)
    text(x0 + 200, ty + 16,
         [("共用下游：检查器 → ", "zh"), ("d", "it"), ("v", "itsub"),
          (" → 式(2) 聚合 → ", "zh"), ("s", "it")], size=12,
         color=OSR_S if kind == "C" else MUTED, bold=(kind == "C"))
    line(fx0 + 24, lanes[2] + 15, fx0 + 24, ty - 10, color=HAIR, sw=0.9)
    arrow(fx0 + 24, ty - 10, fx0 + 24, ty,
          color=OSR_S if kind == "C" else LANE)
    text(x0 + 16, PAN_Y0 + 266, note, size=12, color=MUTED, align="left")


def _capsule(cx, lanes, col, fill, motif):
    top, bot = lanes[0] - 18, lanes[-1] + 18
    rect(cx - 12, top, 24, bot - top, fill=fill, stroke=col, sw=1.3, rx=6)
    if motif == "token":
        for k in range(4):
            yy = top + 13 + k * (bot - top - 26) / 3
            rect(cx - 7.5, yy - 4.5, 15, 9, fill="FFFFFF", stroke=col, sw=0.9,
                 rx=2)
    else:
        for a, b in [(0, 1), (1, 2)]:
            line(cx, lanes[a] + 7, cx, lanes[b] - 7, color="FFFFFF", sw=1.5,
                 arrow=True, both=True)
        for ly in lanes:
            circ(cx, ly, 3.4, fill="FFFFFF")


def build():
    P.clear()
    section1()
    section2()
    return P


# ============================================================== SVG
ZH_F = "'Microsoft YaHei','PingFang SC',sans-serif"
EN_F = "'Arial','Helvetica Neue',sans-serif"
IT_F = "'Times New Roman','Cambria',serif"
PT = 100 / 72.0


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _run_svg(r, size):
    s, kind = r
    px = size * PT
    if kind in ("zh", "en", "it"):
        fam = {"zh": ZH_F, "en": EN_F, "it": IT_F}[kind]
        it = ' font-style="italic"' if kind == "it" else ""
        return f'<tspan font-family="{fam}"{it}>{_esc(s)}</tspan>'
    fam = IT_F if kind == "itsub" else EN_F
    it = ' font-style="italic"' if kind == "itsub" else ""
    dy = -px * 0.34 if kind == "sup" else px * 0.22
    return (f'<tspan font-family="{fam}"{it} font-size="{px*0.72:.2f}" '
            f'dy="{dy:.2f}">{_esc(s)}</tspan>'
            f'<tspan dy="{-dy:.2f}" font-size="0"> </tspan>')


def render_svg():
    prims = build()
    o = io.StringIO()
    o.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" '
            f'height="{H}" viewBox="0 0 {W} {H}" font-family="{ZH_F}">\n<defs>\n')
    for col in {p.get("color") for p in prims if p["k"] == "line"} | {LANE}:
        o.write(f'<marker id="a{col}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="6" markerHeight="6" '
                f'orient="auto-start-reverse">'
                f'<path d="M0,1.6 L10,5 L0,8.4 z" fill="#{col}"/></marker>\n')
    o.write(f'</defs>\n<rect width="{W}" height="{H}" fill="#fff"/>\n')

    for p in sorted(prims, key=lambda d: d["z"]):
        k = p["k"]
        if k == "rect":
            f = f'#{p["fill"]}' if p["fill"] else "none"
            s = f'#{p["stroke"]}' if p["stroke"] else "none"
            d = f' stroke-dasharray="{p["dash"]}"' if p["dash"] else ""
            o.write(f'<rect x="{p["x"]:.2f}" y="{p["y"]:.2f}" '
                    f'width="{p["w"]:.2f}" height="{p["h"]:.2f}" rx="{p["rx"]}" '
                    f'fill="{f}" stroke="{s}" '
                    f'stroke-width="{p["sw"]*PT:.2f}"{d}/>\n')
        elif k == "circ":
            f = f'#{p["fill"]}' if p["fill"] else "none"
            s = f'#{p["stroke"]}' if p["stroke"] else "none"
            d = f' stroke-dasharray="{p["dash"]}"' if p["dash"] else ""
            o.write(f'<circle cx="{p["cx"]:.2f}" cy="{p["cy"]:.2f}" '
                    f'r="{p["r"]:.2f}" fill="{f}" stroke="{s}" '
                    f'stroke-width="{p["sw"]*PT:.2f}"{d}/>\n')
        elif k == "line":
            m = "a" + p["color"]
            a = f' marker-end="url(#{m})"' if p["arrow"] else ""
            if p.get("both"):
                a += f' marker-start="url(#{m})"'
            d = f' stroke-dasharray="{p["dash"]}"' if p["dash"] else ""
            o.write(f'<line x1="{p["x1"]:.2f}" y1="{p["y1"]:.2f}" '
                    f'x2="{p["x2"]:.2f}" y2="{p["y2"]:.2f}" '
                    f'stroke="#{p["color"]}" stroke-width="{p["sw"]*PT:.2f}" '
                    f'stroke-linecap="round"{a}{d}/>\n')
        elif k == "image":
            b64 = base64.b64encode(open(p["path"], "rb").read()).decode()
            o.write(f'<image x="{p["x"]:.2f}" y="{p["y"]:.2f}" '
                    f'width="{p["w"]:.2f}" height="{p["h"]:.2f}" '
                    f'preserveAspectRatio="none" '
                    f'xlink:href="data:image/png;base64,{b64}"/>\n')
        else:
            anc = {"center": "middle", "left": "start",
                   "right": "end"}[p["align"]]
            wt = ' font-weight="600"' if p["bold"] else ""
            body = "".join(_run_svg(r, p["size"]) for r in p["runs"])
            o.write(f'<text xml:space="preserve" x="{p["cx"]:.2f}" '
                    f'y="{p["cy"]:.2f}" text-anchor="{anc}" '
                    f'dominant-baseline="central" '
                    f'font-size="{p["size"]*PT:.2f}" fill="#{p["color"]}"'
                    f'{wt}>{body}</text>\n')
    o.write("</svg>\n")
    return o.getvalue()


# ============================================================== PPTX
def render_pptx(path):
    from pptx import Presentation
    from pptx.util import Emu, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.oxml.ns import qn

    U = 9144                                  # 1 unit (0.01 in) -> EMU
    prims = build()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(W * U), Emu(H * U)
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    shapes = sl.shapes

    def rgb(h):
        return RGBColor.from_string(h)

    def style(sh, fill, stroke, sw, dash=None):
        if fill:
            sh.fill.solid()
            sh.fill.fore_color.rgb = rgb(fill)
        else:
            sh.fill.background()
        if stroke:
            sh.line.color.rgb = rgb(stroke)
            sh.line.width = Pt(sw)
            if dash:
                ln = sh.line._get_or_add_ln()
                ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
        else:
            sh.line.fill.background()
        sh.shadow.inherit = False

    def connector(p):
        sh = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(p["x1"] * U)),
                                  Emu(int(p["y1"] * U)), Emu(int(p["x2"] * U)),
                                  Emu(int(p["y2"] * U)))
        ln = sh.line._get_or_add_ln()
        sh.line.color.rgb = rgb(p["color"])
        sh.line.width = Pt(p["sw"])
        if p["dash"]:
            ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
        # headEnd / tailEnd 必须排在 <a:ln> 末尾，否则 PowerPoint 静默丢箭头
        if p.get("both"):
            ln.append(ln.makeelement(qn("a:headEnd"), {
                "type": "triangle", "w": "med", "len": "med"}))
        if p["arrow"]:
            ln.append(ln.makeelement(qn("a:tailEnd"), {
                "type": "triangle", "w": "med", "len": "med"}))

    FONTS = {"zh": ("Microsoft YaHei", False), "en": ("Arial", False),
             "it": ("Times New Roman", True), "itsub": ("Times New Roman", True),
             "sub": ("Arial", False), "sup": ("Arial", False)}

    def est_w(runs, size):
        n = 0.0
        for s, kind in runs:
            f = 0.72 if kind in ("sub", "sup", "itsub") else 1.0
            for ch in s:
                n += (1.0 if ord(ch) > 0x2E7F else 0.53) * f
        return n * size * PT

    for p in sorted(prims, key=lambda d: d["z"]):
        k = p["k"]
        if k == "rect":
            shp = (MSO_SHAPE.ROUNDED_RECTANGLE if p["rx"]
                   else MSO_SHAPE.RECTANGLE)
            sh = shapes.add_shape(shp, Emu(int(p["x"] * U)),
                                  Emu(int(p["y"] * U)), Emu(int(p["w"] * U)),
                                  Emu(int(p["h"] * U)))
            if p["rx"]:
                sh.adjustments[0] = min(0.5, p["rx"] / min(p["w"], p["h"]))
            style(sh, p["fill"], p["stroke"], p["sw"], p["dash"])
        elif k == "circ":
            r = p["r"]
            sh = shapes.add_shape(MSO_SHAPE.OVAL, Emu(int((p["cx"] - r) * U)),
                                  Emu(int((p["cy"] - r) * U)),
                                  Emu(int(2 * r * U)), Emu(int(2 * r * U)))
            style(sh, p["fill"], p["stroke"], p["sw"], p["dash"])
        elif k == "line":
            connector(p)
        elif k == "image":
            shapes.add_picture(p["path"], Emu(int(p["x"] * U)),
                               Emu(int(p["y"] * U)), Emu(int(p["w"] * U)),
                               Emu(int(p["h"] * U)))
        else:
            wpx = max(36.0, est_w(p["runs"], p["size"]) * 1.22 + 20)
            hpx = p["size"] * PT * 2.0
            # margin 已置 0，锚点即文本边界，不能再加补偿偏移
            x = {"center": p["cx"] - wpx / 2, "left": p["cx"],
                 "right": p["cx"] - wpx}[p["align"]]
            tb = shapes.add_textbox(Emu(int(x * U)),
                                    Emu(int((p["cy"] - hpx / 2) * U)),
                                    Emu(int(wpx * U)), Emu(int(hpx * U)))
            tf = tb.text_frame
            tf.word_wrap = False
            tf.margin_left = tf.margin_right = 0
            tf.margin_top = tf.margin_bottom = 0
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = tf.paragraphs[0]
            para.alignment = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT,
                              "right": PP_ALIGN.RIGHT}[p["align"]]
            for s, kind in p["runs"]:
                run = para.add_run()
                run.text = s
                name, italic = FONTS[kind]
                sub = kind in ("sub", "sup", "itsub")
                run.font.size = Pt(p["size"] * (0.72 if sub else 1.0))
                run.font.name = name
                run.font.bold = p["bold"]
                run.font.italic = italic
                run.font.color.rgb = rgb(p["color"])
                rPr = run.font._rPr
                rPr.append(rPr.makeelement(qn("a:ea"), {"typeface": name}))
                if kind in ("sub", "itsub"):
                    rPr.set("baseline", "-25000")
                elif kind == "sup":
                    rPr.set("baseline", "30000")

    prs.save(path)
    return path


def main():
    svg = render_svg()
    open(os.path.join(HERE, "fig2_mvosr_cvpr.svg"), "w",
         encoding="utf-8").write(svg)
    html = ('<meta charset="utf-8"><title>论文一 图2 效果图</title><style>'
            'body{margin:0;background:#eceff2;font-family:' + ZH_F + '}'
            '.wrap{padding:24px}.card{background:#fff;border-radius:10px;'
            'box-shadow:0 2px 20px rgba(20,35,50,.12);padding:16px;'
            'max-width:1360px;margin:0 auto}.cap{max-width:1360px;'
            'margin:14px auto 0;color:#3d4a58;font-size:14px;line-height:1.7}'
            'svg{width:100%;height:auto;display:block}</style>'
            '<div class="wrap"><div class="card">' + svg + '</div>'
            '<div class="cap"><b>图 2</b>&nbsp;&nbsp;多视角开集手势识别总体框架'
            '与3种跨视角融合策略的结构对比</div></div>')
    open(os.path.join(HERE, "fig2_mvosr_cvpr.html"), "w",
         encoding="utf-8").write(html)
    render_pptx(os.path.join(HERE, "fig2_mvosr_cvpr.pptx"))
    print("shapes:", len(build()))
    for e in ("svg", "html", "pptx"):
        print(" ->", os.path.join(HERE, "fig2_mvosr_cvpr." + e))


if __name__ == "__main__":
    main()
