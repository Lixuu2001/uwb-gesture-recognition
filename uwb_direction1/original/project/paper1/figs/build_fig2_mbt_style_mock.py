#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper-1 Figure 2, MBT Fig.1 grammar: abstract token grid.

Vocabulary (CIR, dual-path encoder, f_v, k-hat, checker, score) is already
established by Figure 1, so this figure draws ONLY what differs: the cross-view
connection topology. One legend, one depth axis, three identical skeletons.
"""
import pathlib

OUT = pathlib.Path(__file__).parent / "fig2_mbt_style_mock.svg"

W, H = 1580, 726
M = 28

INK, SUB, FAINT, RULE = "#1F2937", "#5B6472", "#9AA3AF", "#D6DAE0"
WIRE = "#9AA3AF"
L_F, L_S = "#F6D97A", "#B8860B"
T_F, T_S = "#9EC5E8", "#2E75B6"
R_F, R_S = "#F0A6AC", "#BE4B48"
G_F, G_S = "#E7F5EC", "#2E8B57"
P_F, P_S = "#B99FE0", "#6A4CA0"

CJK = "'Microsoft YaHei','Source Han Sans SC','Noto Sans SC',sans-serif"
SER = "'Cambria','Times New Roman',serif"

VIEW = [("左 L", L_F, L_S), ("上 T", T_F, T_S), ("右 R", R_F, R_S)]
o = []
add = o.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(x, y, w, h, fill, stroke, rx=8, sw=1.3, dash=None, op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"'
        f' fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>')


def txt(x, y, s, size=13, fill=INK, anchor="middle", weight="400", fam=CJK):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}"'
        f' fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
        f'{esc(s)}</text>')


def mvar(x, y, parts, size=15, fill=INK, anchor="middle"):
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
        f' fill="{fill}" text-anchor="{anchor}">' + "".join(frag) + "</text>")


def line(x1, y1, x2, y2, stroke=WIRE, sw=0.6, op=0.55, dash=None, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
        f' stroke="{stroke}" stroke-width="{sw}" opacity="{op}"'
        f' stroke-linecap="{cap}"{d}/>')


def dot(x, y, f, s, r=7.5, sw=1.1):
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{f}" stroke="{s}"'
        f' stroke-width="{sw}"/>')


def chip(x, y, s, size=12.5, fill=G_S, pad=8):
    """text on an opaque chip so it stays legible over wire bundles"""
    w = len(s) * size * 0.92 + 2 * pad
    rect(x - w / 2, y - size * 0.92, w, size * 1.75, "#FFFFFF", "none", rx=6,
         sw=0)
    txt(x, y + size * 0.36, s, size, fill, weight="700")


def arrow(x1, y1, x2, y2, stroke=INK, sw=1.6):
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
        f' stroke="{stroke}" stroke-width="{sw}" marker-end="url(#arw)"/>')


# --------------------------------------------------------------------------- #
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"'
    f' viewBox="0 0 {W} {H}">')
add('<defs>')
add(f'<marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"'
    f' markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,1 L10,5 L0,9 z" fill="{INK}"/></marker>')
add(f'<marker id="arwG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"'
    f' markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,1 L10,5 L0,9 z" fill="{G_S}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

txt(W / 2, 40, "跨视角信息交换发生的层级", 24, INK, weight="700")

# ------------------------------------------------------------------ legend --
LX, LY, LW, LH = 300, 56, W - 600, 40
rect(LX, LY, LW, LH, "#FFFFFF", SUB, rx=8, sw=1.1, dash="5 3")
items = [("左 L", L_F, L_S), ("上 T", T_F, T_S), ("右 R", R_F, R_S),
         ("共享瓶颈", P_F, P_S)]
cx = LX + 26
txt(cx, LY + 25, "token 类型：", 13, INK, "start", "600")
cx += 82
for nm, f, s in items:
    dot(cx, LY + 20, f, s)
    txt(cx + 12, LY + 25, nm, 13, INK, "start")
    cx += 22 + len(nm) * 13.5
line(cx + 6, LY + 20, cx + 36, LY + 20, WIRE, 0.9, 0.8)
txt(cx + 42, LY + 25, "视角内注意力", 13, INK, "start")
cx += 42 + 6 * 13.5 + 14
line(cx + 6, LY + 20, cx + 36, LY + 20, G_S, 2.0, 1.0)
txt(cx + 42, LY + 25, "跨视角交换", 13, G_S, "start", "700")

txt(W / 2, LY + LH + 24,
    "输入 CIR、共享双路径编码器与判决式 s(X) 的完整定义见图 1；此处只画三种策略在跨视角连接上的差异",
    13, SUB)

# ------------------------------------------------------------------- layout --
AXW = 178
PX0 = M + AXW + 12
PW = 420
PGAP = 28
PY0, PY1 = 150, 690

ROW = {"l1": 214, "l2": 264, "l3": 314, "l4": 364,
       "f": 428, "d": 486, "s": 546}
AXIS = [("l1", ["第 1 层"]), ("l2", ["第 2 层"]), ("l3", ["第 3 层"]),
        ("l4", ["第 4 层"])]

# depth axis
add(f'<line x1="{M + AXW - 14}" y1="196" x2="{M + AXW - 14}" y2="{ROW["s"] + 24}"'
    f' stroke="{RULE}" stroke-width="2" marker-end="url(#arw)"/>')
txt(M + AXW - 30, 190, "管线深度", 13, SUB, "end", "700")
for key, labs in AXIS:
    y = ROW[key]
    add(f'<circle cx="{M + AXW - 14}" cy="{y}" r="3.5" fill="{RULE}"/>')
    txt(M + AXW - 26, y + 5, labs[0], 13, SUB, "end")
# encoder bracket
add(f'<path d="M{M + 34},{ROW["l1"] - 16} h-10 V{ROW["l4"] + 16} h10"'
    f' fill="none" stroke="{RULE}" stroke-width="1.4"/>')
add(f'<text x="{M + 18}" y="{(ROW["l1"] + ROW["l4"]) / 2}" font-family="{CJK}"'
    f' font-size="12.5" fill="{SUB}" text-anchor="middle"'
    f' transform="rotate(-90 {M + 18} {(ROW["l1"] + ROW["l4"]) / 2})">'
    f'共享编码器 · 4 层 × 2 路径</text>')
AXM = [("f", "特征码 ", [("f", "i"), ("v", "sub")]),
       ("d", "检查器 → ", [("d", "i"), ("v", "sub")]),
       ("s", "异常分数 ", [("s", "i"), ("(X)", "r")])]
for key, lab, parts in AXM:
    y = ROW[key]
    add(f'<circle cx="{M + AXW - 14}" cy="{y}" r="3.5" fill="{RULE}"/>')
    mvar(M + AXW - 26, y + 5, parts, 15, SUB, "end")
    txt(M + AXW - 26 - len(lab) * 4 - 22, y + 5, lab, 13, SUB, "end")

PANELS = [
    dict(tag="(a)", name="瓶颈中层融合（MBT 式）", mode="btl",
         note="时间 / 距离两条路径各有一组瓶颈，都参与交换",
         verdict="编码器内部窄通道交换",
         nums="随机切分 OSCR 95.96  ·  LOSO 83.43 ± 8.45", ours=False),
    dict(tag="(b)", name="编码后特征级融合", mode="attn",
         note="仅时间路径 token 参与，距离路径原样通过",
         verdict="编码后全带宽两两交互",
         nums="随机切分 OSCR 95.30  ·  LOSO 已落后，未纳入", ours=False),
    dict(tag="(c)", name="决策级距离聚合（本文）", mode="dec",
         note="三路检查器彼此独立，交换只发生在标量距离上",
         verdict="三路全程独立，仅在决策端汇合",
         nums="随机切分 OSCR 95.99  ·  LOSO 85.22 ± 5.55", ours=True),
]

# token x offsets inside a panel: 3 groups x 3 tokens
GOFF = [98, 120, 142, 188, 210, 232, 278, 300, 322]
GRP = [GOFF[0:3], GOFF[3:6], GOFF[6:9]]
GCEN = [120, 210, 300]

for pi, P in enumerate(PANELS):
    px = PX0 + pi * (PW + PGAP)
    ours = P["ours"]
    rect(px, PY0, PW, PY1 - PY0, "#FFFFFF", G_S if ours else RULE, rx=12,
         sw=2.0 if ours else 1.2)
    txt(px + PW / 2, PY0 + 32, f'{P["tag"]} {P["name"]}', 17, INK, weight="700")

    mode = P["mode"]

    # ---- highlight band for the inserted exchange -------------------------
    if mode == "btl":
        rect(px + 62, ROW["l4"] - 24, PW - 124, 78, G_F, G_S, rx=9, sw=1.4,
             dash="5 4")
    elif mode == "attn":
        rect(px + 40, ROW["l4"] + 18, PW - 80, 48, G_F, G_S, rx=9, sw=1.4,
             dash="5 4")

    # ---- within-view attention between consecutive encoder layers ---------
    layers = ["l1", "l2", "l3", "l4"]
    for a, b in zip(layers[:-1], layers[1:]):
        for g in GRP:
            for xa in g:
                for xb in g:
                    line(px + xa, ROW[a] + 8, px + xb, ROW[b] - 8)

    # ---- layer 4 -> feature code -----------------------------------------
    if mode == "attn":
        for gi, g in enumerate(GRP):
            for xa in g:
                for gj in range(3):
                    line(px + xa, ROW["l4"] + 8, px + GCEN[gj], ROW["f"] - 11,
                         G_S, 1.0, 0.55)
    else:
        for gi, g in enumerate(GRP):
            for xa in g:
                line(px + xa, ROW["l4"] + 8, px + GCEN[gi], ROW["f"] - 11)

    # ---- bottleneck tokens (mode btl) ------------------------------------
    if mode == "btl":
        by = ROW["l4"] + 32
        bxs = [px + 210 - 27 + k * 18 for k in range(4)]
        for gi in range(3):
            for bx in bxs:
                line(px + GCEN[gi], ROW["l4"] + 8, bx, by - 7, G_S, 1.0, 0.6)
        for bx in bxs:
            dot(bx, by, P_F, P_S, r=7)
        txt(px + 300, by + 4, "跨视角取平均", 11.5, P_S, "start")

    # ---- encoder layer tokens --------------------------------------------
    for key in layers:
        for gi, g in enumerate(GRP):
            for xo in g:
                dot(px + xo, ROW[key], VIEW[gi][1], VIEW[gi][2])

    # ---- band labels, drawn last so they stay legible over the wires ------
    if mode == "btl":
        chip(px + PW / 2, ROW["l4"] - 36, "第 4 层内插入共享瓶颈")
    elif mode == "attn":
        chip(px + PW / 2, ROW["l4"] + 40, "编码后跨视角自注意力")

    # ---- feature / distance nodes ----------------------------------------
    for gi in range(3):
        f, s = VIEW[gi][1], VIEW[gi][2]
        rect(px + GCEN[gi] - 26, ROW["f"] - 15, 52, 30, f, s, rx=7, sw=1.2)
        mvar(px + GCEN[gi], ROW["f"] + 5,
             [("f", "i"), (VIEW[gi][0].split()[1], "sub")], 15)
        line(px + GCEN[gi], ROW["f"] + 15, px + GCEN[gi], ROW["d"] - 15,
             s, 1.6, 1.0)
        rect(px + GCEN[gi] - 26, ROW["d"] - 15, 52, 30, f, s, rx=7, sw=1.2)
        mvar(px + GCEN[gi], ROW["d"] + 5,
             [("d", "i"), (VIEW[gi][0].split()[1], "sub")], 15)

    # ---- distance -> score ------------------------------------------------
    ex = (mode == "dec")
    col, wgt, opa = (G_S, 2.0, 1.0) if ex else (WIRE, 1.4, 0.75)
    my = ROW["d"] + 26
    for gi in range(3):
        line(px + GCEN[gi], ROW["d"] + 15, px + GCEN[gi], my, col, wgt, opa)
    line(px + GCEN[0], my, px + GCEN[2], my, col, wgt, opa)
    add(f'<line x1="{px + 210}" y1="{my}" x2="{px + 210}" y2="{ROW["s"] - 17}"'
        f' stroke="{col}" stroke-width="{wgt}" opacity="{opa}"'
        f' marker-end="url(#{"arwG" if ex else "arw"})"/>')
    rect(px + 210 - 92, ROW["s"] - 16, 184, 32, G_F if ex else "#FFFFFF",
         G_S if ex else FAINT, rx=8, sw=1.5 if ex else 1.1)
    mvar(px + 210, ROW["s"] + 5, [("s", "i"), ("(X)", "r")], 15)
    if not ex:
        txt(px + 210 + 100, ROW["s"] + 4, "三栏共有", 11, FAINT, "start")

    # ---- note / verdict / numbers ----------------------------------------
    txt(px + PW / 2, ROW["s"] + 44, P["note"], 12, SUB)
    rect(px + 30, PY1 - 92, PW - 60, 38, G_F if ours else "#FFFFFF",
         G_S if ours else FAINT, rx=9, sw=1.4)
    txt(px + PW / 2, PY1 - 67, P["verdict"], 14.5, INK, weight="700")
    txt(px + PW / 2, PY1 - 26, P["nums"], 12, SUB)

add("</svg>")
OUT.write_text("\n".join(o), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size)
