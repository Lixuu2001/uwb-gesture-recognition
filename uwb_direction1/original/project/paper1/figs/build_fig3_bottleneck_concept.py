# -*- coding: utf-8 -*-
"""Figure 3 concept sheet: MBT-style bottleneck mid-fusion (paper 1).

Emits a self-contained HTML file holding one inline 1600x940 SVG.

Design intent: graphite ink + a single low-saturation violet accent carries the
mechanism.  View identity (L/T/R) is reduced to a hairline rule plus a letter,
so the only saturated objects on the page are the bottleneck tokens and the
cross-view channel.  No isometric slabs, no glows, no gradient blobs.

    python paper1/figs/build_fig3_bottleneck_concept.py
"""
import os

W, H = 1600, 940

INK = "#161A1F"
INK2 = "#39404A"
MUTED = "#79828C"
FAINT = "#9AA2AB"
HAIR = "#DCE0E5"
PANEL = "#FCFCFD"
TOKS = "#C6CCD3"
TOKF = "#F2F4F6"
BTL = "#574AA0"
BTLF = "#EBE9F6"
BTLS = "#8B80C6"

LANES = [
    dict(k="L", zh="left view", cy=372, col="#A8762F", tint="#F8F2E6"),
    dict(k="T", zh="top view", cy=492, col="#2E6D8E", tint="#EAF1F6"),
    dict(k="R", zh="right view", cy=612, col="#99506A", tint="#F7EEF2"),
]
LANES[0]["zh"] = u"左视角"
LANES[1]["zh"] = u"上视角"
LANES[2]["zh"] = u"右视角"

BR_IN_L, BR_IN_R = 186, 474
GX_IN = 200
SEMI = 341
VX_IN = 358
BOX_X, BOX_W = 544, 272
BR_OUT_L = 874
GX_OUT = 888
SPLIT = 1030
VX_OUT = 1046
BR_OUT_R = 1162
BUS_X = 1240
NODE_X, NODE_Y = 1292, 492
TOK, GAP = 20, 6


def toks(x0, cy, n, fill, stroke):
    return "".join(
        '<rect x="%d" y="%d" width="%d" height="%d" rx="3" fill="%s" '
        'stroke="%s" stroke-width="1.1"/>'
        % (x0 + i * (TOK + GAP), cy - TOK // 2, TOK, TOK, fill, stroke)
        for i in range(n))


def bracket(x, cy, h, side, col=MUTED):
    t = 8 * side
    return ('<path d="M%d %d H%d V%d H%d" fill="none" stroke="%s" '
            'stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>'
            % (x + t, cy - h // 2, x, cy + h // 2, x + t, col))


def txt(x, y, s, cls="", anchor="start", extra=""):
    a = ' text-anchor="%s"' % anchor if anchor != "start" else ""
    c = ' class="%s"' % cls if cls else ""
    return '<text x="%s" y="%s"%s%s%s>%s</text>' % (x, y, c, a, extra, s)


def sub(s, size=13):
    return '<tspan baseline-shift="sub" font-size="%d">%s</tspan>' % (size, s)


def sup(s, size=13):
    return '<tspan baseline-shift="super" font-size="%d">%s</tspan>' % (size, s)


p = []
add = p.append

# ------------------------------------------------------------------ header --
add('<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, H))
add(txt(100, 64, "BOTTLENECK MID-FUSION&#160;&#160;·&#160;&#160;"
                 "INSIDE THE LAST ENCODER LAYER", "eyebrow"))
add(txt(100, 106, "三路 token 只与一组共享瓶颈 token 互读互写", "title"))
add(txt(100, 136, "跨视角信息被压进一条窄通道：瓶颈跨视角取平均，"
                  "是编码器里唯一的交换路径", "sub"))

add(txt(1500, 92, "[ z&#771;%s ; B&#771;%s ] = Block%s( [ z%s + e%s ; B%s ] )"
        % (sub("v"), sub("v"), sub("θ"), sub("v"), sub("v"), sup("(ℓ)")),
        "formula", "end"))
add(txt(1500, 128, "B%s = (1/V) Σ%s B&#771;%s"
        % (sup("(ℓ+1)"), sub("v"), sub("v")), "formula", "end"))
add(txt(1500, 152, "v ∈ { L, T, R }，三个视角依次通过同一个块", "tiny", "end"))
add('<path d="M100 174 H1500" stroke="%s" stroke-width="1"/>' % HAIR)

# -------------------------------------------------------------- depth rail --
add(txt(100, 206, "共享编码器 · 每路径 4 层 pre-LN Transformer", "tiny"))
for i in range(4):
    x = 100 + i * 87
    hot = (i == 3)
    add('<rect x="%d" y="214" width="78" height="38" rx="5" fill="%s" '
        'stroke="%s" stroke-width="%s"/>'
        % (x, BTL if hot else "#FFFFFF", BTL if hot else TOKS,
           "1.5" if hot else "1.1"))
    add(txt(x + 39, 238, "层 %d" % (i + 1),
            "railhot" if hot else "rail", "middle"))
add(txt(230, 272, "逐视角独立", "tiny", "middle"))
add(txt(408, 272, "瓶颈交换层 · 总深度不变", "tinyv", "middle"))

add('<rect x="1090" y="228" width="20" height="20" rx="3" fill="%s" '
    'stroke="%s" stroke-width="1.1"/>' % (TOKF, TOKS))
add(txt(1118, 243, "视角内 token", "tiny"))
add('<rect x="1230" y="228" width="20" height="20" rx="3" fill="%s" '
    'stroke="%s" stroke-width="1.3"/>' % (BTLF, BTL))
add(txt(1258, 243, "共享瓶颈 token", "tiny"))
add('<path d="M1385 238 H1419" stroke="%s" stroke-width="2.2" '
    'stroke-linecap="round"/>' % BTL)
add(txt(1427, 243, "跨视角通道", "tiny"))

add('<path d="M361 254 L100 300 M439 254 L1500 300" stroke="#CDD3DA" '
    'stroke-width="1" stroke-dasharray="5 5"/>')

# ------------------------------------------------------------------- panel --
add('<rect x="100" y="300" width="1400" height="456" rx="12" fill="%s" '
    'stroke="#C9CFD6" stroke-width="1.2" stroke-dasharray="7 6"/>' % PANEL)
add('<rect x="128" y="288" width="182" height="24" rx="12" fill="#FFFFFF" '
    'stroke="#C9CFD6" stroke-width="1"/>')
add(txt(219, 305, "瓶颈交换层内部（放大）", "chip", "middle"))

add('<path d="M350 362 V622" stroke="%s" stroke-width="1.2" '
    'stroke-dasharray="4 4"/>' % BTLS)

for ln in LANES:
    cy, col, k = ln["cy"], ln["col"], ln["k"]
    add('<rect x="112" y="%d" width="4" height="64" rx="2" fill="%s"/>'
        % (cy - 32, col))
    add(txt(128, cy - 2, k, "vsym", "start", ' fill="%s"' % col))
    add(txt(128, cy + 18, ln["zh"], "tiny"))

    add(bracket(BR_IN_L, cy, 60, +1))
    add(toks(GX_IN, cy, 5, "#FFFFFF", col))
    add(txt(SEMI, cy + 7, ";", "semi", "middle"))
    add(toks(VX_IN, cy, 4, BTLF, BTL))
    add(bracket(BR_IN_R, cy, 60, -1))
    add('<path d="M350 %d h8" stroke="%s" stroke-width="1.2"/>' % (cy, BTLS))
    add(txt(262, cy + 42, "z%s%s + e%s"
            % (sub(k, 12), sup("(ℓ)", 12), sub(k, 12)), "code", "middle"))
    add(txt(407, cy + 42, "B · N%s = 4" % sub("B", 11), "codev", "middle"))

    add('<path d="M482 %d H528" stroke="%s" stroke-width="1.4" '
        'marker-end="url(#ar)"/>' % (cy, INK2))
    add('<rect x="%d" y="%d" width="%d" height="68" rx="6" fill="#FFFFFF" '
        'stroke="%s" stroke-width="1.5"/>' % (BOX_X, cy - 34, BOX_W, INK))
    add(txt(BOX_X + BOX_W // 2, cy - 4, "共享 Transformer 块", "blk", "middle"))
    add(txt(BOX_X + BOX_W // 2, cy + 18,
            "在 [ token ; 瓶颈 ] 上自注意力，互读互写", "tiny", "middle"))

    add('<path d="M824 %d H858" stroke="%s" stroke-width="1.4" '
        'marker-end="url(#ar)"/>' % (cy, INK2))
    add(bracket(BR_OUT_L, cy, 60, +1))
    add(toks(GX_OUT, cy, 5, ln["tint"], col))
    add('<path d="M%d %d V%d" stroke="%s" stroke-width="1" '
        'stroke-dasharray="3 4"/>' % (SPLIT, cy - 22, cy + 22, FAINT))
    add(toks(VX_OUT, cy, 4, BTLF, BTL))
    add(bracket(BR_OUT_R, cy, 60, -1))
    add(txt(950, cy + 42, "z&#771;%s%s" % (sub(k, 12), sup("(ℓ)", 12)), "code", "middle"))
    add(txt(1095, cy + 42, "B&#771;%s" % sub(k, 12), "codev", "middle"))

    if cy == NODE_Y:
        d = "M1170 %d H1268" % cy
    else:
        s = 1 if cy < NODE_Y else -1
        d = ("M1170 %d H%d Q%d %d %d %d V%d Q%d %d %d %d H1268"
             % (cy, BUS_X - 14, BUS_X, cy, BUS_X, cy + 14 * s,
                NODE_Y - 14 * s, BUS_X, NODE_Y, BUS_X + 14, NODE_Y))
    add('<path d="%s" fill="none" stroke="%s" stroke-width="2" '
        'marker-end="url(#arv)"/>' % (d, BTL))

add('<circle cx="%d" cy="%d" r="22" fill="#FFFFFF" stroke="%s" '
    'stroke-width="1.8"/>' % (NODE_X, NODE_Y, BTL))
add(txt(NODE_X, NODE_Y + 8, "Σ", "sigma", "middle"))
add(txt(NODE_X, NODE_Y + 46, "跨视角取平均", "tinyv", "middle"))
add('<path d="M1316 %d H1344" stroke="%s" stroke-width="2" '
    'marker-end="url(#arv)"/>' % (NODE_Y, BTL))
add('<rect x="1350" y="460" width="140" height="64" rx="8" fill="%s" '
    'stroke="%s" stroke-width="1.5"/>' % (BTLF, BTL))
add(txt(1420, 488, "B%s" % sup("(ℓ+1)"), "btlbox", "middle"))
add(txt(1420, 510, "唯一的跨视角通道", "tinyv", "middle"))

add('<path d="M120 682 H1000" stroke="%s" stroke-width="1"/>' % HAIR)
add(txt(120, 708, "更新后的三路 token 仍分开读出 → 逐视角特征码 "
                  "f%s / f%s / f%s，检查器保持逐视角独立"
        % (sub("L", 11), sub("T", 11), sub("R", 11)), "note"))
add(txt(120, 734, "视角之间没有两两稠密注意力：每个视角只额外看到 4 个 token"
                  "（时序 90 → 4，距离 189 → 4）", "note2"))

# ------------------------------------------------------------------ footer --
add('<path d="M100 786 H980" stroke="%s" stroke-width="1"/>' % HAIR)
foot = [
    "时序路径（90 token）与距离路径（189 token）各有一组瓶颈，两条路径都参与交换",
    "交换前先在 token 上注入视角嵌入 e%s，打破视角置换对称" % sub("v", 11),
    "只有最后 K 层做交换，前 4 − K 层逐视角独立；总层数与无融合一致，容量不膨胀",
]
for i, s in enumerate(foot):
    y = 818 + i * 28
    add('<circle cx="104" cy="%d" r="2.4" fill="%s"/>' % (y - 5, FAINT))
    add(txt(118, y, s, "note2"))

# contrast inset: dense pairwise vs bottleneck
add('<rect x="1010" y="780" width="490" height="140" rx="10" fill="#FFFFFF" '
    'stroke="%s" stroke-width="1"/>' % HAIR)
add(txt(1030, 804, "跨视角连接方式对照", "tiny"))
add('<path d="M1215 812 V900" stroke="%s" stroke-width="1" '
    'stroke-dasharray="4 4"/>' % HAIR)

dense = [(1078, 830), (1078, 862), (1078, 894)]
for a in range(3):
    for b in range(a + 1, 3):
        (x1, y1), (x2, y2) = dense[a], dense[b]
        bulge = 26 if b - a == 1 else 46
        add('<path d="M%d %d Q%d %d %d %d" fill="none" stroke="%s" '
            'stroke-width="1.3"/>' % (x1, y1, x1 + bulge, (y1 + y2) // 2,
                                      x2, y2, "#A9B1B9"))
for (cx, cy), ln in zip(dense, LANES):
    add('<circle cx="%d" cy="%d" r="8" fill="#FFFFFF" stroke="%s" '
        'stroke-width="1.6"/>' % (cx, cy, ln["col"]))
add(txt(1112, 916, "两两稠密：3 条跨视角连边", "tiny", "middle"))

btlv = [(1272, 830), (1272, 862), (1272, 894)]
add('<rect x="1372" y="836" width="20" height="52" rx="4" fill="%s" '
    'stroke="%s" stroke-width="1.4"/>' % (BTLF, BTL))
for i in (1, 2, 3):
    add('<path d="M1372 %d h20" stroke="%s" stroke-width="1"/>'
        % (836 + i * 13, BTLS))
for (cx, cy) in btlv:
    add('<path d="M%d %d Q%d %d %d %d" fill="none" stroke="%s" '
        'stroke-width="1.5"/>' % (cx + 9, cy, 1330, cy, 1372, 862, BTL))
for (cx, cy), ln in zip(btlv, LANES):
    add('<circle cx="%d" cy="%d" r="8" fill="#FFFFFF" stroke="%s" '
        'stroke-width="1.6"/>' % (cx, cy, ln["col"]))
add(txt(1338, 916, "瓶颈：只经 4 个共享 token", "tinyv", "middle"))

svg = """<svg id="fig3" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="7.4" refY="4.5"
      orient="auto"><path d="M0 0 L8 4.5 L0 9 Z" fill="%s"/></marker>
    <marker id="arv" markerWidth="9" markerHeight="9" refX="7.4" refY="4.5"
      orient="auto"><path d="M0 0 L8 4.5 L0 9 Z" fill="%s"/></marker>
  </defs>
  %s
</svg>""" % (W, H, INK2, BTL, "\n  ".join(p))

CSS = """
  *{box-sizing:border-box}
  html,body{margin:0;background:#EDF0F3;color:__INK__;
    font-family:"Source Han Sans SC","Microsoft YaHei",Arial,sans-serif}
  body{display:flex;align-items:center;justify-content:center;padding:32px}
  .frame{width:__W__px;background:#fff;box-shadow:0 18px 50px rgba(22,26,31,.10)}
  svg{display:block;width:100%;height:auto}
  text{font-family:"Source Han Sans SC","Microsoft YaHei",Arial,sans-serif}
  .eyebrow{font-size:12.5px;font-weight:700;fill:#8B939C;letter-spacing:2.4px}
  .title{font-size:30px;font-weight:700;fill:__INK__;letter-spacing:.2px}
  .sub{font-size:15px;fill:__MUTED__}
  .formula{font-family:"Times New Roman",Georgia,serif;font-size:21px;
    font-style:italic;fill:__INK__}
  .rail{font-size:14px;font-weight:600;fill:__MUTED__}
  .railhot{font-size:14px;font-weight:700;fill:#fff}
  .chip{font-size:12.5px;font-weight:700;fill:__BTL__;letter-spacing:.4px}
  .vsym{font-family:"Times New Roman",Georgia,serif;font-size:22px;
    font-style:italic;font-weight:700}
  .blk{font-size:16.5px;font-weight:600;fill:__INK__}
  .code{font-family:"Times New Roman",Georgia,serif;font-size:14px;
    font-style:italic;fill:__INK2__}
  .codev{font-family:"Times New Roman",Georgia,serif;font-size:14px;
    font-style:italic;fill:__BTL__}
  .semi{font-size:19px;font-weight:600;fill:__MUTED__}
  .sigma{font-family:"Times New Roman",Georgia,serif;font-size:22px;
    font-style:italic;fill:__BTL__}
  .btlbox{font-family:"Times New Roman",Georgia,serif;font-size:19px;
    font-style:italic;font-weight:700;fill:__BTL__}
  .note{font-size:13.5px;font-weight:500;fill:__INK2__}
  .note2{font-size:13px;fill:__MUTED__}
  .tiny{font-size:12.5px;font-weight:500;fill:__MUTED__}
  .tinyv{font-size:12px;font-weight:600;fill:__BTL__}
"""
CSS = (CSS.replace("__INK2__", INK2).replace("__INK__", INK)
          .replace("__MUTED__", MUTED).replace("__BTL__", BTL)
          .replace("__W__", str(W)))

html = ("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Figure 3 — 瓶颈中层融合（概念图）</title>
<style>%s</style>
</head>
<body>
<div class="frame">
%s
</div>
</body>
</html>
""" % (CSS, svg))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fig3_bottleneck_concept.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out)
