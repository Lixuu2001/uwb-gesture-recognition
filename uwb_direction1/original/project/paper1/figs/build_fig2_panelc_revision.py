# -*- coding: utf-8 -*-
"""Panel (c) of fig2_three_fusion: original crop vs proposed revision.

Reads the exported slide PNG from the scratchpad, crops panel (c), and writes a
side-by-side comparison HTML in which the right half is a redrawn panel (c) in
the SAME 553x829 local coordinate system as the PPTX shapes, so every change
maps 1:1 onto a named shape.

    python paper1/figs/build_fig2_panelc_revision.py <scratchpad-dir>
"""
import base64
import os
import sys

SP = sys.argv[1] if len(sys.argv) > 1 else "."

INK = "#101621"
FRAME = "#303A45"
GRAY = "#45505E"
GREEN = "#28A86B"
BTL = "#805FA0"
BTLF = "#E7D9F2"
BTLBG = "#F4F1F8"

VIEWS = [
    dict(k="L", zh="左 L", chip=75, cir=44, arrow=112, enc=57,
         x1=36, x2=116, solid="#FFF0BF", ss="#C69A34",
         light="#FFF9E6", ls="#DDBE72"),
    dict(k="T", zh="上 T", chip=240, cir=209, arrow=277, enc=222,
         x1=218, x2=298, solid="#D9EFF8", ss="#4D99B8",
         light="#EDF7FC", ls="#8CBFD6"),
    dict(k="R", zh="右 R", chip=405, cir=374, arrow=442, enc=387,
         x1=400, x2=480, solid="#F8DDE2", ss="#C77782",
         light="#FCF0F3", ls="#DBA6AE"),
]

# funnel landing points on the bottleneck frame, in column order
LAND = [182, 220, 258, 296, 334, 372]
COLC = [56, 136, 238, 318, 420, 500]          # column centres
IN_ROWS = [316, 335, 373]
OUT_ROWS = [618, 637, 675]
IN_DOTS, OUT_DOTS = 354, 656

s = []
a = s.append


def cols():
    """(centre, left, solid?, view) for the six token columns, left to right."""
    out = []
    for v in VIEWS:
        out.append((v["x1"], True, v))
        out.append((v["x2"], False, v))
    return out


a('<rect x="2" y="1" width="549" height="827" rx="10" fill="#FFFFFF" '
  'stroke="%s" stroke-width="1.2"/>' % FRAME)
a('<text x="277" y="40" text-anchor="middle" class="t1">(c) MBT式瓶颈中层融合</text>')

for v in VIEWS:
    a('<rect x="%d" y="60" width="74" height="27" rx="5" fill="%s" stroke="%s" '
      'stroke-width="1.2"/>' % (v["chip"], v["solid"], v["ss"]))
    a('<text x="%d" y="78" text-anchor="middle" class="chip">%s</text>'
      % (v["chip"] + 37, v["zh"]))
    a('<rect x="%d" y="94" width="136" height="88" fill="#15101F" stroke="#252C36" '
      'stroke-width="1"/>' % v["cir"])
    for i, o in enumerate((28, 52, 76)):
        a('<path d="M%d 168 L%d 112" stroke="#E4A05C" stroke-width="%s" '
          'opacity="%s"/>' % (v["cir"] + o, v["cir"] + o + 26, 3 - i * 0.7,
                              0.75 - i * 0.2))
    a('<path d="M%d 184 V212" stroke="%s" stroke-width="1.2" '
      'marker-end="url(#agray)"/>' % (v["arrow"], GRAY))

a('<rect x="10" y="207" width="533" height="68" rx="8" fill="none" stroke="%s" '
  'stroke-width="1" stroke-dasharray="5 4"/>' % FRAME)
a('<text x="13" y="234" class="tiny">共享</text>')
a('<text x="16" y="248" class="tinyi">θ</text>')
for v in VIEWS:
    a('<rect x="%d" y="216" width="110" height="49" rx="5" fill="#F0F4EF" '
      'stroke="#59675C" stroke-width="1"/>' % v["enc"])
    a('<text x="%d" y="238" text-anchor="middle" class="enc">共享编码器</text>'
      % (v["enc"] + 55))
    a('<text x="%d" y="256" text-anchor="middle" class="enci">E<tspan '
      'baseline-shift="sub" font-size="8">θ</tspan></text>' % (v["enc"] + 55))

# --- view grouping rules + column captions ---------------------------------
for v in VIEWS:
    a('<path d="M%d 292 H%d" stroke="%s" stroke-width="3" stroke-linecap="round"/>'
      % (v["x1"], v["x2"] + 40, v["ss"]))
    a('<text x="%d" y="307" text-anchor="middle" class="cap">时序 90</text>'
      % (v["x1"] + 20))
    a('<text x="%d" y="307" text-anchor="middle" class="cap">距离 189</text>'
      % (v["x2"] + 20))

# --- input tokens -----------------------------------------------------------
for x, solid, v in cols():
    f, st = (v["solid"], v["ss"]) if solid else (v["light"], v["ls"])
    for y in IN_ROWS:
        a('<rect x="%d" y="%d" width="40" height="17" rx="2" fill="%s" '
          'stroke="%s" stroke-width="1"/>' % (x, y, f, st))
    a('<text x="%d" y="%d" text-anchor="middle" class="dots">⋯</text>'
      % (x + 20, IN_DOTS + 13))

# --- funnel in (bidirectional) ---------------------------------------------
for i, (x, solid, v) in enumerate(cols()):
    c, lx = COLC[i], LAND[i]
    a('<path d="M%d 394 C%d 424 %d 438 %d 466" fill="none" stroke="%s" '
      'stroke-width="1.3" marker-start="url(#a%s)" marker-end="url(#a%s)"/>'
      % (c, c, lx, lx, v["ss"], v["k"], v["k"]))

a('<text x="277" y="459" text-anchor="middle" class="btlcap" '
  'paint-order="stroke" stroke="#FFFFFF" stroke-width="3.4" '
  'stroke-linejoin="round">'
  '共享瓶颈 token · N<tspan baseline-shift="sub" font-size="7">B</tspan> = 4</text>')
a('<rect x="176" y="466" width="212" height="46" rx="7" fill="#FAF9FC" '
  'stroke="#B9A8CE" stroke-width="1.1"/>')
a('<rect x="170" y="472" width="212" height="46" rx="7" fill="%s" stroke="%s" '
  'stroke-width="1.6"/>' % (BTLBG, BTL))
for i in range(4):
    a('<rect x="%d" y="486" width="40" height="17" rx="2" fill="%s" stroke="%s" '
      'stroke-width="1.1"/>' % (181 + i * 48, BTLF, BTL))

a('<text x="396" y="481" class="btlnote">×2：时序 / 距离</text>')
a('<text x="396" y="493" class="btlnote">各一组瓶颈</text>')
a('<text x="396" y="512" class="btlmath">B ← (1/V) Σ<tspan baseline-shift="sub" '
  'font-size="8">v</tspan> B&#771;<tspan baseline-shift="sub" font-size="8">v</tspan></text>')
a('<text x="396" y="525" class="tiny2">三视角各写回一份副本</text>')

# --- funnel out -------------------------------------------------------------
for i, (x, solid, v) in enumerate(cols()):
    c, lx = COLC[i], LAND[i]
    a('<path d="M%d 520 C%d 548 %d 562 %d 590" fill="none" stroke="%s" '
      'stroke-width="1.3" marker-end="url(#a%s)"/>'
      % (lx, lx, c, c, v["ss"], v["k"]))

for v in VIEWS:
    a('<path d="M%d 600 H%d" stroke="%s" stroke-width="3" stroke-linecap="round"/>'
      % (v["x1"], v["x2"] + 40, v["ss"]))
    a('<text x="%d" y="613" text-anchor="middle" class="cap">更新后</text>'
      % (v["x1"] + 60))

for x, solid, v in cols():
    f, st = (v["solid"], v["ss"]) if solid else (v["light"], v["ls"])
    for y in OUT_ROWS:
        a('<rect x="%d" y="%d" width="40" height="17" rx="2" fill="%s" '
          'stroke="%s" stroke-width="1"/>' % (x, y, f, st))
    a('<text x="%d" y="%d" text-anchor="middle" class="dots">⋯</text>'
      % (x + 20, OUT_DOTS + 13))

a('<text x="277" y="712" text-anchor="middle" class="foot">'
  '三路 token 仍分开读出 → 逐视角特征码 f<tspan baseline-shift="sub" '
  'font-size="8">L</tspan> / f<tspan baseline-shift="sub" font-size="8">T</tspan>'
  ' / f<tspan baseline-shift="sub" font-size="8">R</tspan></text>')
a('<rect x="32" y="757" width="496" height="58" rx="10" fill="#FFFFFF" '
  'stroke="%s" stroke-width="1.4"/>' % GREEN)
a('<text x="280" y="793" text-anchor="middle" class="concl">'
  '少量 token 构成唯一跨视角通道</text>')

markers = "".join(
    '<marker id="a%s" markerWidth="8" markerHeight="8" refX="6.4" refY="4" '
    'orient="auto"><path d="M0 0L7.3 4L0 8Z" fill="%s"/></marker>'
    % (k, c) for k, c in [("L", "#C69A34"), ("T", "#4D99B8"),
                          ("R", "#C77782"), ("gray", GRAY)])

svg = ('<svg viewBox="0 0 553 829" xmlns="http://www.w3.org/2000/svg">'
       '<defs>%s</defs>%s</svg>' % (markers, "".join(s)))

with open(os.path.join(SP, "panel_c.png"), "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")

html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>图二 (c) 栏 — 原版 vs 修改版</title>
<style>
  *{box-sizing:border-box}
  html,body{margin:0;background:#EDF0F3;color:#101621;
    font-family:"Microsoft YaHei",Arial,sans-serif}
  body{display:flex;justify-content:center;padding:34px}
  .sheet{width:1120px;background:#fff;padding:38px 42px 30px;
    box-shadow:0 18px 50px rgba(16,22,33,.10)}
  h1{margin:0;font-size:21px}
  .lede{margin:8px 0 24px;font-size:13px;color:#6B7684;line-height:1.75}
  .row{display:flex;gap:26px;align-items:flex-start}
  .col{flex:1;min-width:0}
  .tag{font-size:11px;font-weight:700;letter-spacing:2px;color:#9AA2AB;
    margin-bottom:8px}
  .tag.now{color:#805FA0}
  .box{border:1px solid #DCE0E5;border-radius:10px;padding:10px;background:#fff}
  img,svg{display:block;width:100%%;height:auto}
  .cap2{font-size:12.5px;color:#6B7684;line-height:1.8;margin-top:12px}
  .cap2 b{color:#101621;font-weight:600}
  text{font-family:"Microsoft YaHei",Arial,sans-serif;fill:#101621}
  .t1{font-size:19px;font-weight:700}
  .chip{font-size:12px;font-weight:700}
  .enc{font-size:11px;font-weight:600}
  .enci{font-size:11px;font-family:"Times New Roman",serif;font-style:italic}
  .tiny{font-size:9px;fill:#45505E}
  .tinyi{font-size:10px;font-family:"Times New Roman",serif;font-style:italic;fill:#45505E}
  .tiny2{font-size:8px;fill:#45505E}
  .cap{font-size:8.5px;fill:#45505E}
  .dots{font-size:12px;fill:#45505E}
  .btlcap{font-size:9px;font-weight:700;fill:#805FA0}
  .btlnote{font-size:9.5px;font-weight:600;fill:#805FA0}
  .btlmath{font-size:12px;font-family:"Times New Roman",serif;font-style:italic;fill:#805FA0}
  .foot{font-size:9px;fill:#45505E}
  .concl{font-size:17px;font-weight:700}
</style></head><body><div class="sheet">
<h1>图二 (c) 栏 · 原版与修改版</h1>
<p class="lede">右侧沿用 553×829 的栏内坐标，每处改动都能对到 PPTX 里的形状名。
顶部三分之一（标题 / 视角胶囊 / CIR / 共享编码器）一个像素没动。</p>
<div class="row">
  <div class="col">
    <div class="tag">原版 · panel-(c)</div>
    <div class="box"><img src="data:image/png;base64,%s" alt="original panel c"></div>
    <p class="cap2"><b>问题</b>：token 列与瓶颈之间没有任何连线，机制的动词全丢了；
      瓶颈 token 是圆形、其余是矩形；瓶颈框用绿色而绿色已被结论条占用；
      六列距离 token 全部是 T 视角的蓝色。</p>
  </div>
  <div class="col">
    <div class="tag now">修改版</div>
    <div class="box">%s</div>
    <p class="cap2"><b>改动</b>：补双向漏斗连线（读写）与扇出连线；瓶颈 token 改矩形、
      尺寸与视角 token 一致；绿→紫，一色一义；距离 token 改为本视角浅色；
      上下各压掉一行腾出连线空间；瓶颈框叠一层表示"两条路径各一组"。</p>
  </div>
</div>
</div></body></html>
""" % (b64, svg)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fig2_panelc_revision.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out)
