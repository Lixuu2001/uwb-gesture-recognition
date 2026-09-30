#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper-1 Figure 2 redesign mock.

Grammar: ONE shared backbone strip (input -> shared encoder -> tokens ->
feature code -> closed head -> per-view checker -> score) drawn once, carrying
three numbered insertion points. Below it, three panels each show only the
mechanism inserted at its own point, in the same visual language as the
reference (token strips, attention grid, bottleneck tokens, green verdict pill).
"""
import base64
import pathlib

HERE = pathlib.Path(__file__).parent
MEDIA = HERE / "r4x" / "ppt" / "media"
OUT = HERE / "fig2_redesign_mock.svg"

W, H = 1760, 1124
M = 30

INK, SUB, FAINT, RULE = "#1F2937", "#5B6472", "#9AA3AF", "#D6DAE0"
PANEL_BG = "#FFFFFF"
BAND_BG = "#FAFBFC"
NF, NS = "#F2F4F6", "#94A0B0"
L_F, L_S = "#FCF1CF", "#BE8C12"
T_F, T_S = "#DCE9F6", "#2E75B6"
R_F, R_S = "#FADFE2", "#C0504D"
G_F, G_S = "#E7F5EC", "#2E8B57"          # cross-view exchange
P_F, P_S = "#E9DEF6", "#7A5AA8"          # bottleneck tokens

CJK = "'Microsoft YaHei','Source Han Sans SC','Noto Sans SC',sans-serif"
SER = "'Cambria','Times New Roman',serif"

VIEW = [("左 L", L_F, L_S), ("上 T", T_F, T_S), ("右 R", R_F, R_S)]
o = []
add = o.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(x, y, w, h, fill, stroke, rx=8, sw=1.3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"'
        f' fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def txt(x, y, s, size=13, fill=INK, anchor="middle", weight="400", fam=CJK,
        style="normal", op=1.0):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}"'
        f' fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"'
        f' font-style="{style}" opacity="{op}">{esc(s)}</text>')


def rows(x, y, ls, size=13, fill=INK, anchor="middle", weight="400", lh=None,
         fam=CJK):
    lh = lh or size * 1.5
    y0 = y - (len(ls) - 1) * lh / 2 + size * 0.35
    for i, s in enumerate(ls):
        txt(x, y0 + i * lh, s, size, fill, anchor, weight, fam)


def mvar(x, y, parts, size=16, fill=INK, anchor="middle", weight="400"):
    """parts: list of (text, kind) kind in {i,r,sub,sup}"""
    frag = []
    for s, k in parts:
        if k == "i":
            frag.append(f'<tspan font-style="italic">{esc(s)}</tspan>')
        elif k == "r":
            frag.append(f'<tspan>{esc(s)}</tspan>')
        elif k == "sub":
            frag.append(f'<tspan font-size="{size*0.68:.1f}" dy="{size*0.24:.1f}"'
                        f' font-style="italic">{esc(s)}</tspan>'
                        f'<tspan font-size="{size:.1f}" dy="{-size*0.24:.1f}"></tspan>')
        elif k == "sup":
            frag.append(f'<tspan font-size="{size*0.68:.1f}" dy="{-size*0.42:.1f}">'
                        f'{esc(s)}</tspan>'
                        f'<tspan font-size="{size:.1f}" dy="{size*0.42:.1f}"></tspan>')
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SER}" font-size="{size}"'
        f' fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
        + "".join(frag) + "</text>")


def arr(x1, y1, x2, y2, stroke=INK, sw=1.6, mk="arw", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
        f' stroke="{stroke}" stroke-width="{sw}" marker-end="url(#{mk})"{d}/>')


def biarr(x1, y1, x2, y2, stroke=P_S, sw=1.2, dash="4 3"):
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
        f' stroke="{stroke}" stroke-width="{sw}" stroke-dasharray="{dash}"'
        f' marker-end="url(#arwP)" marker-start="url(#arwP)"/>')


def poly(pts, stroke=INK, sw=1.5, mk="arw", dash=None):
    p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{mk})"' if mk else ""
    add(f'<polyline points="{p}" fill="none" stroke="{stroke}" stroke-width="{sw}"'
        f' stroke-linejoin="round"{m}{d}/>')


def strip(x, y, n, fill, stroke, cw=15, ch=17, gap=2.5, ell=True):
    """horizontal token strip; returns right edge"""
    cx = x
    for _ in range(n):
        rect(cx, y - ch / 2, cw, ch, fill, stroke, rx=3, sw=1.0)
        cx += cw + gap
    if ell:
        txt(cx + 8, y + 4, "…", 14, SUB)
        cx += 18
        rect(cx, y - ch / 2, cw, ch, fill, stroke, rx=3, sw=1.0)
        cx += cw
    return cx


def img(p, x, y, w, h):
    b64 = base64.b64encode(p.read_bytes()).decode()
    add(f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"'
        f' preserveAspectRatio="none" xlink:href="data:image/png;base64,{b64}"/>')
    rect(x, y, w, h, "none", FAINT, rx=2, sw=0.9)


def marker_dot(x, y, label, active, r=13):
    f, s = (G_F, G_S) if active else ("#FFFFFF", RULE)
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{f}" stroke="{s}"'
        f' stroke-width="{1.8 if active else 1.2}"/>')
    txt(x, y + 4.5, label, 12.5, G_S if active else FAINT, weight="700")


# =========================================================================== #
add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
    f' width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add("<defs>")
for nm, c, sz in [("arw", INK, 7), ("arwS", SUB, 6), ("arwP", P_S, 5),
                  ("arwG", G_S, 7)]:
    add(f'<marker id="{nm}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="{sz}"'
        f' markerHeight="{sz}" orient="auto-start-reverse">'
        f'<path d="M0,1 L10,5 L0,9 z" fill="{c}"/></marker>')
add("</defs>")
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

txt(W / 2, 48, "跨视角信息交换插在管线的哪一层", 27, INK, weight="700")
txt(W / 2, 78, "三种策略的输入、共享编码器与最终判决式完全相同；唯一自变量是交换的插入深度",
    15, SUB)

# ------------------------------------------------------------ shared strip --
BY0, BY1 = 100, 316
rect(M, BY0, W - 2 * M, BY1 - BY0, BAND_BG, RULE, rx=12, sw=1.3, dash="7 5")
rect(M + 16, BY0 - 12, 268, 26, "#FFFFFF", RULE, rx=13, sw=1.2)
txt(M + 150, BY0 + 5, "共享主干 · 三栏相同，只画一次", 13.5, SUB, weight="700")

# CIR stack
for i, (nm, f, s) in enumerate(VIEW):
    yy = 140 + i * 46
    rect(52, yy + 6, 44, 28, f, s, rx=5, sw=1.2)
    txt(74, yy + 25, nm, 13, INK, weight="700")
    img(MEDIA / ["image.png", "image2.png", "image3.png"][i], 104, yy, 100, 40)
    poly([(206, yy + 20), (228, yy + 20), (228, 210)], SUB, 1.4, mk=None)
txt(154, 300, "三视角 CIR  90×189 · abs-std 归一", 12, SUB)
arr(228, 210, 244, 210, SUB, 1.8)

BW, GAP = 172, 36
BX = [246 + i * (BW + GAP) for i in range(6)]
BOX_Y0, BOX_H = 142, 136
for i in range(5):
    arr(BX[i] + BW, 210, BX[i + 1], 210, INK, 1.8)

rect(BX[0], BOX_Y0, BW, BOX_H, "#FFFFFF", NS, rx=9, sw=1.4)
txt(BX[0] + BW / 2, BOX_Y0 + 26, "共享编码器", 15, INK, weight="700")
mvar(BX[0] + BW / 2, BOX_Y0 + 50, [("E", "i"), ("θ", "sub")], 17)
rows(BX[0] + BW / 2, BOX_Y0 + 96,
     ["双路径 · 每路径 4 层", "pre-LN Transformer", "三视角权重共享"], 11.5, SUB)

rect(BX[1], BOX_Y0, BW, BOX_H, "#FFFFFF", NS, rx=9, sw=1.4)
txt(BX[1] + BW / 2, BOX_Y0 + 24, "每视角两条 token 序列", 13, INK, weight="700")
strip(BX[1] + 22, BOX_Y0 + 52, 3, "#FFF8E3", L_S, cw=13, ch=15)
txt(BX[1] + BW / 2 + 4, BOX_Y0 + 74, "时序 token   t = 1…T (90)", 11, SUB)
strip(BX[1] + 22, BOX_Y0 + 98, 3, "#EDF4FB", T_S, cw=13, ch=15)
txt(BX[1] + BW / 2 + 4, BOX_Y0 + 120, "距离 token   r = 1…M (189)", 11, SUB)

rect(BX[2], BOX_Y0, BW, BOX_H, "#FFFFFF", NS, rx=9, sw=1.4)
txt(BX[2] + BW / 2, BOX_Y0 + 26, "逐视角特征码", 15, INK, weight="700")
mvar(BX[2] + BW / 2, BOX_Y0 + 54, [("f", "i"), ("v", "sub")], 17)
rows(BX[2] + BW / 2, BOX_Y0 + 98,
     ["flatten(时序) ⊕ flatten(距离)", "＋ 视角嵌入  e_v", "dim = 8928"], 11.5, SUB)

rect(BX[3], BOX_Y0, BW, BOX_H, "#FFFFFF", NS, rx=9, sw=1.4)
txt(BX[3] + BW / 2, BOX_Y0 + 26, "闭集分类头", 15, INK, weight="700")
rows(BX[3] + BW / 2, BOX_Y0 + 62, ["跨视角平均读出", "(闭集已饱和 97%+)"], 11.5, SUB)
rect(BX[3] + 40, BOX_Y0 + 88, BW - 80, 30, NF, NS, rx=7, sw=1.2)
mvar(BX[3] + BW / 2, BOX_Y0 + 109, [("→  k̂", "i")], 16)

rect(BX[4], BOX_Y0, BW, BOX_H, "#FFFFFF", NS, rx=9, sw=1.4)
txt(BX[4] + BW / 2, BOX_Y0 + 26, "逐视角检查器", 15, INK, weight="700")
mvar(BX[4] + BW / 2, BOX_Y0 + 52, [("η", "i"), ("k̂", "sub"), ("(v)", "sup")], 17)
rows(BX[4] + BW / 2, BOX_Y0 + 96,
     ["由 k̂ 选定；每 (类, 视角) 一个", "原型距离 d_v = ‖η(f_v) − φ‖"], 11.5, SUB)

rect(BX[5], BOX_Y0, BW, BOX_H, "#FFFFFF", NS, rx=9, sw=1.4)
txt(BX[5] + BW / 2, BOX_Y0 + 26, "开集异常分数", 15, INK, weight="700")
mvar(BX[5] + BW / 2, BOX_Y0 + 56, [("s", "i"), ("(X)", "r")], 17)
rows(BX[5] + BW / 2, BOX_Y0 + 98, ["三栏共用同一判决式", "(式 12，见 (c) 栏)"], 11.5, SUB)

# insertion markers on the backbone
for mx, lab, cap in [(BX[0] + BW / 2, "①", "编码器最后一层内部"),
                     (BX[1] + BW + GAP / 2, "②", "编码后 token 上"),
                     (BX[4] + BW + GAP / 2, "③", "距离聚合处")]:
    marker_dot(mx, BOX_Y0 + BOX_H, lab, True)
    txt(mx, BOX_Y0 + BOX_H + 32, cap, 11.5, G_S, weight="700")

txt(W / 2, BY1 + 26,
    "① ② ③ 为三条互斥的设计路线，每次只启用一个；下方三栏分别放大对应机制",
    14, SUB)

# ---------------------------------------------------------------- panels ---
PY0 = 360
PW = (W - 2 * M - 2 * 26) / 3
PH = 730
PANELS = [
    dict(tag="(a)", name="瓶颈中层融合（MBT 式）", pt=0,
         verdict="编码器内部窄通道交换",
         nums="随机切分 OSCR 95.96   ·   LOSO 83.43 ± 8.45", ours=False),
    dict(tag="(b)", name="编码后特征级融合", pt=1,
         verdict="编码后时序 token 全带宽交互",
         nums="随机切分 OSCR 95.30   ·   LOSO 已落后，未纳入", ours=False),
    dict(tag="(c)", name="决策级距离聚合（本文）", pt=2,
         verdict="三路全程独立，仅在决策端汇合",
         nums="随机切分 OSCR 95.99   ·   LOSO 85.22 ± 5.55", ours=True),
]

for pi, P in enumerate(PANELS):
    px = M + pi * (PW + 26)
    ours = P["ours"]
    rect(px, PY0, PW, PH, PANEL_BG, G_S if ours else RULE, rx=12,
         sw=2.0 if ours else 1.3)
    txt(px + PW / 2, PY0 + 38, f'{P["tag"]} {P["name"]}', 19, INK, weight="700")

    # ---- mini backbone bar with this panel's insertion point ----
    bx0, bw = px + 26, PW - 52
    cw = (bw - 4 * 11) / 5
    ybar = PY0 + 66
    cells = ["输入", "编码器", "特征码", "检查器", "分数"]
    xs = []
    for i, c in enumerate(cells):
        cx = bx0 + i * (cw + 11)
        xs.append(cx)
        rect(cx, ybar, cw, 30, NF, RULE, rx=6, sw=1.0)
        txt(cx + cw / 2, ybar + 20, c, 12, SUB)
        if i < 4:
            add(f'<line x1="{cx+cw:.1f}" y1="{ybar+15}" x2="{cx+cw+11:.1f}"'
                f' y2="{ybar+15}" stroke="{RULE}" stroke-width="1.2"/>')
    pts = [(xs[1] + cw / 2, "①"), (xs[1] + cw + 5.5, "②"), (xs[3] + cw + 5.5, "③")]
    for k, (mx, lab) in enumerate(pts):
        marker_dot(mx, ybar - 7, lab, k == P["pt"], r=10)

    my = PY0 + 108           # mechanism area top
    cxm = px + PW / 2

    # ===================================================== (a) bottleneck ==
    if P["pt"] == 0:
        gx, gy, gw, gh = px + 26, my + 6, PW - 52, 306
        rect(gx, gy, gw, gh, "#FBFDFB", G_S, rx=10, sw=1.6, dash="6 4")
        txt(cxm, gy + 24, "编码器第 4 层（最后一层）内部", 14, G_S, weight="700")
        btx = gx + gw - 104
        txt(btx, gy + 50, "共享瓶颈 token", 12, P_S, weight="700")
        lane_y = [gy + 78, gy + 128, gy + 178]
        for i, (nm, f, s) in enumerate(VIEW):
            rect(gx + 16, lane_y[i] - 13, 32, 26, f, s, rx=5, sw=1.1)
            txt(gx + 32, lane_y[i] + 5, nm.split()[1], 12.5, INK, weight="700")
            strip(gx + 56, lane_y[i], 3, f, s, cw=14, ch=16)
        for k in range(4):
            add(f'<circle cx="{btx - 51 + k * 34}" cy="{gy + 80}" r="13"'
                f' fill="{P_F}" stroke="{P_S}" stroke-width="1.3"/>')
        mvar(btx, gy + 116, [("N", "i"), ("B", "sub"), (" = 4", "r")], 13, P_S)
        for i in range(3):
            biarr(gx + 148, lane_y[i], btx - 74, gy + 96 + i * 4)
        txt(gx + 150, gy + 214, "每视角 token 只与瓶颈互相读写", 11.5, P_S, "start")
        arr(btx, gy + 130, btx, gy + 152, P_S, 1.3, mk="arwP", dash="4 3")
        rect(btx - 72, gy + 154, 144, 30, P_F, P_S, rx=7, sw=1.2)
        txt(btx, gy + 174, "瓶颈跨视角取平均", 12.5, INK, weight="600")
        poly([(btx - 72, gy + 169), (gx + 132, gy + 169), (gx + 132, gy + 196)],
             P_S, 1.2, mk="arwP", dash="4 3")
        txt(cxm, gy + 246, "瓶颈是唯一的跨视角通道，无两两稠密注意力",
            12.5, G_S, weight="600")
        txt(cxm, gy + 270, "K = 1（仅最后一层）· 总深度不变", 11.5, SUB)
        txt(cxm, gy + 290, "交换前先在 token 上注入视角身份", 11.5, SUB)

        yy = gy + gh + 30
        arr(cxm, gy + gh, cxm, yy - 4, INK, 1.7)
        for i, (nm, f, s) in enumerate(VIEW):
            strip(px + 52 + i * ((PW - 104) / 3), yy + 20, 3, f, s, cw=14, ch=16)
        txt(cxm, yy + 54, "更新后的三路 token 仍分开读出 → 逐视角特征码", 12.5, SUB)
        rect(px + 44, yy + 68, PW - 88, 60, "#FFFFFF", NS, rx=8, sw=1.2, dash="5 4")
        txt(cxm, yy + 88, "时序路径与距离路径各有一组瓶颈", 12, INK, weight="600")
        txt(cxm, yy + 110, "两条路径都参与交换（与 (b) 栏不同）", 11.5, SUB)

    # =================================================== (b) post-encoder ==
    elif P["pt"] == 1:
        txt(cxm, my + 16, "编码后的 时序 token，按慢时间索引 t 对齐", 13, SUB)
        ty = my + 48
        for i, (nm, f, s) in enumerate(VIEW):
            lx = px + 52 + i * ((PW - 104) / 3)
            rect(lx, ty - 13, 30, 26, f, s, rx=5, sw=1.1)
            txt(lx + 15, ty + 5, nm.split()[1], 12.5, INK, weight="700")
            strip(lx + 38, ty, 2, f, s, cw=13, ch=15)
            arr(lx + 40, ty + 20, lx + 40, ty + 46, s, 1.6)

        gx, gy, gw, gh = px + 26, my + 96, PW - 52, 244
        rect(gx, gy, gw, gh, "#FBFDFB", G_S, rx=10, sw=1.6)
        txt(cxm, gy + 26, "视角维自注意力", 15, G_S, weight="700")
        txt(cxm, gy + 46, "对每个慢时间索引 t，在 3 个视角之间做自注意力", 11.5, SUB)
        gs = 42
        g0x, g0y = gx + 96, gy + 86
        txt(g0x + gs, gy + 68, "键 / 值视角", 11, SUB)
        add(f'<text x="{g0x - 46:.1f}" y="{g0y + gs:.1f}" font-family="{CJK}"'
            f' font-size="11" fill="{SUB}" text-anchor="middle"'
            f' transform="rotate(-90 {g0x - 46:.1f} {g0y + gs:.1f})">查询视角</text>')
        for k in range(3):
            txt(g0x - 24, g0y + k * gs + 5, VIEW[k][0].split()[1], 13, INK,
                weight="700")
            txt(g0x + k * gs, g0y + 3 * gs - 6, VIEW[k][0].split()[1], 13, INK,
                weight="700")
        for r in range(3):
            for c in range(3):
                cxx, cyy = g0x + c * gs, g0y + r * gs
                if r == c:
                    add(f'<circle cx="{cxx}" cy="{cyy}" r="13" fill="{VIEW[c][1]}"'
                        f' stroke="{VIEW[c][2]}" stroke-width="2.0"/>')
                else:
                    add(f'<circle cx="{cxx}" cy="{cyy}" r="13" fill="#FFFFFF"'
                        f' stroke="{G_S}" stroke-width="1.3"'
                        f' stroke-dasharray="3 2.5"/>')
        lgx = gx + gw - 128
        add(f'<circle cx="{lgx}" cy="{gy + 96}" r="9" fill="{T_F}"'
            f' stroke="{T_S}" stroke-width="2.0"/>')
        txt(lgx + 16, gy + 100, "自注意（对角）", 11.5, SUB, "start")
        add(f'<circle cx="{lgx}" cy="{gy + 124}" r="9" fill="#FFFFFF"'
            f' stroke="{G_S}" stroke-width="1.3" stroke-dasharray="3 2.5"/>')
        txt(lgx + 16, gy + 128, "跨视角（非对角）", 11.5, SUB, "start")
        txt(cxm, gy + gh - 14, "9 个位置全部有权重：全带宽两两交互，无信息瓶颈",
            12, G_S, weight="600")

        yy = gy + gh + 26
        for i, (nm, f, s) in enumerate(VIEW):
            lx = px + 52 + i * ((PW - 104) / 3)
            arr(lx + 40, yy - 22, lx + 40, yy - 2, s, 1.6)
            strip(lx + 8, yy + 18, 3, f, s, cw=14, ch=16)
        txt(cxm, yy + 50, "已耦合的时序 token → 特征码，同时喂闭集头与检查器",
            12.5, SUB)
        rect(px + 40, yy + 66, PW - 80, 62, "#FFFFFF", NS, rx=8, sw=1.2, dash="5 4")
        txt(cxm, yy + 88, "距离路径 token 不经过任何跨视角阶段", 12, INK,
            weight="600")
        txt(cxm, yy + 110, "原样进入特征码 → 两条路径在此不对称", 11.5, SUB)

    # ================================================== (c) decision-level ==
    else:
        txt(cxm, my + 16, "三路检查器彼此独立，各自算出原型距离", 13, SUB)
        lane_x = [px + 62 + i * ((PW - 124) / 2) for i in range(3)]
        ey = my + 54
        for i, (nm, f, s) in enumerate(VIEW):
            rect(lane_x[i] - 46, ey - 22, 92, 44, f, s, rx=7, sw=1.3)
            mvar(lane_x[i], ey + 6,
                 [("η", "i"), ("k̂", "sub"), (f"({nm.split()[1]})", "sup")], 16)
            arr(lane_x[i], ey + 24, lane_x[i], ey + 62, s, 1.6)
            rect(lane_x[i] - 34, ey + 64, 68, 40, f, s, rx=7, sw=1.3)
            mvar(lane_x[i], ey + 90, [("d", "i"), (nm.split()[1], "sub")], 18)
        txt(cxm, ey + 128, "三条通路在此之前从不相连", 12, SUB)

        my2 = ey + 150
        for i in range(3):
            poly([(lane_x[i], ey + 106), (lane_x[i], my2), (cxm, my2)],
                 SUB, 1.6, mk=None)
        arr(cxm, my2, cxm, my2 + 34, SUB, 1.9)

        sy = my2 + 36
        rect(px + 22, sy, PW - 44, 92, G_F, G_S, rx=10, sw=1.8)
        txt(cxm, sy + 24, "跨视角距离聚合（式 12）", 13.5, G_S, weight="700")
        mvar(cxm, sy + 62,
             [("s", "i"), ("(X) = mean", "r"), ("v", "sub"), ("(", "r"),
              ("d", "i"), ("v", "sub"), (") / σ̄  +  β · var", "r"),
              ("v", "sub"), ("(", "r"), ("d", "i"), ("v", "sub"),
              (") / σ̄", "r"), ("2", "sup")], 17)
        txt(cxm, sy + 84, "σ̄ = 三视角类内尺度均值", 11.5, SUB)

        by = sy + 116
        rect(px + 26, by, (PW - 66) / 2, 62, "#FFFFFF", NS, rx=8, sw=1.2)
        rows(px + 26 + (PW - 66) / 4, by + 31,
             ["第一项 · 多视角平均", "几何互补 + 一致性去噪"], 11.5, SUB)
        rect(px + 40 + (PW - 66) / 2, by, (PW - 66) / 2, 62, "#FFFFFF", NS,
             rx=8, sw=1.2)
        rows(px + 40 + 3 * (PW - 66) / 4, by + 31,
             ["第二项 · 跨视角分歧", "三路应同时贴近各自原型"], 11.5, SUB)
        txt(cxm, by + 92, "β = 0 时退化为纯平均；V = 1 时退化为单视角判决",
            12, SUB)

    # ---- verdict pill + numbers ----
    vy = PY0 + PH - 88
    rect(px + 28, vy, PW - 56, 42, G_F if ours else "#FFFFFF",
         G_S if ours else NS, rx=10, sw=1.5)
    txt(px + PW / 2, vy + 27, P["verdict"], 15, INK, weight="700")
    txt(px + PW / 2, vy + 66, P["nums"], 12.5, SUB)

add("</svg>")
OUT.write_text("\n".join(o), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size)
