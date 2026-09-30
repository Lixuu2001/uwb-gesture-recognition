# -*- coding: utf-8 -*-
"""fig6 v3: "融合深度轴"设计。
横轴 = 管线深度(输入→编码器→特征码→检查器→判决), 列头统一标注一次(含维度);
三行 = 三种融合策略, 各 3 条视角流: 灰线=视角内独立流, 线变彩色处=跨视角耦合开始;
彩色节点=交换位置(绿=判决层/红=特征层全带宽/橙=编码中层窄带), 判决处贝塞尔合流;
右侧结论徽章。颜色语义与 fig1 v2 一致。"""
import math
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN, MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

FT = "f̃"

def RGB(h): return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
PANEL   = RGB("F3F4F7")
STROKEN = RGB("9AA3B2"); HAIR = RGB("E0E3E8")
TXT     = RGB("1A1A1A"); SUBT = RGB("667085"); HDR = RGB("3A404A")
GREEN = RGB("00A087"); RED = RGB("E64B35"); ORANGE = RGB("E18727")
GREEN_D = RGB("00795F"); RED_D = RGB("B03A2A"); OR_D = RGB("A9671A")
XG_F = RGB("EAF6F2"); XR_F = RGB("FBE9E5"); XO_F = RGB("FCF3E3")
GRAYLN = "7B8494"; REDLN = "E64B35"; ORLN = "E18727"; GREENLN = "00A087"
EA = "Microsoft YaHei"; LA = "Times New Roman"

X_MIN, X_MAX = -2.75, 14.40
Y_MIN, Y_MAX = -2.55, 4.90
MARGIN = 0.35; SLIDE_W = 13.333
S = (SLIDE_W - 2*MARGIN)/(X_MAX - X_MIN)
SLIDE_H = MARGIN*2 + (Y_MAX - Y_MIN)*S
def X(xc): return MARGIN + (xc - X_MIN)*S
def Y(yc): return MARGIN + (Y_MAX - yc)*S

prs = Presentation()
prs.slide_width = Inches(SLIDE_W); prs.slide_height = Inches(SLIDE_H)
slide = prs.slides.add_slide(prs.slide_layouts[6])

def style_run(r, size, bold=False, italic=False, color=TXT, latin=LA, ea=EA, baseline=None):
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = color; r.font.name = latin
    rPr = r._r.get_or_add_rPr(); latin_el = rPr.find(qn('a:latin'))
    for tag, face in (('a:ea', ea), ('a:cs', latin)):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {}); latin_el.addnext(el)
        el.set('typeface', face)
    if baseline is not None: rPr.set('baseline', str(baseline))

def it(t):  return dict(t=t, it=True,  base='n', cn=False)
def up(t):  return dict(t=t, it=False, base='n', cn=False)
def cn(t):  return dict(t=t, it=False, base='n', cn=True)
def bcn(t, col=None): return dict(t=t, it=False, base='n', cn=True, b=True, col=col)
def sub(t, italic=True):  return dict(t=t, it=italic, base='sub', cn=False)
def sup(t, italic=False): return dict(t=t, it=italic, base='sup', cn=False)

def emit(para, ps, size, bold=False, color=TXT):
    for p in ps:
        r = para.add_run(); r.text = p['t']; sz = size; base = None
        if p['base'] == 'sub':  sz = size*0.72; base = -25000
        elif p['base'] == 'sup': sz = size*0.72; base = 30000
        style_run(r, sz, bold=p.get('b', bold), italic=p['it'],
                  color=p.get('col') or color,
                  latin=(EA if p['cn'] else LA), baseline=base)

def parts(c): return [cn(c)] if isinstance(c, str) else c

_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
def style_line(shape, color_hex, width_pt, dashed=False, arrow=False, head='med'):
    ln = shape.line._get_or_add_ln()
    for ch in list(ln): ln.remove(ch)
    ln.set('w', str(int(width_pt*12700))); ln.set('cap', 'rnd')
    ln.append(parse_xml('<a:solidFill %s><a:srgbClr val="%s"/></a:solidFill>' % (_NS, color_hex)))
    if dashed: ln.append(parse_xml('<a:prstDash %s val="dash"/>' % _NS))
    ln.append(parse_xml('<a:round %s/>' % _NS))
    if arrow: ln.append(parse_xml('<a:tailEnd %s type="triangle" w="%s" len="%s"/>' % (_NS, head, head)))

def band(x1, x2, y1, y2, fill=PANEL, radius=0.04):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(X(x1)), Inches(Y(y2)), Inches((x2-x1)*S), Inches((y2-y1)*S))
    try: shp.adjustments[0] = radius
    except Exception: pass
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.fill.background(); shp.shadow.inherit = False
    return shp

def box(cx, cy, w, h, line=STROKEN, fill=None, lw=0.8, radius=0.12,
        lines=None, size=8.5, bold=False, color=TXT, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    shp = slide.shapes.add_shape(shape,
            Inches(X(cx-w/2)), Inches(Y(cy+h/2)), Inches(w*S), Inches(h*S))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try: shp.adjustments[0] = radius
        except Exception: pass
    shp.fill.solid(); shp.fill.fore_color.rgb = fill if fill is not None else RGB("FFFFFF")
    shp.line.color.rgb = line; shp.line.width = Pt(lw); shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.01)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(lines or []):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        emit(p, parts(ln), size, bold=bold, color=color)
    return shp

def text(cx, cy, lines, size=8, color=TXT, bold=False, w=3.0, left=False, fill_white=False):
    left_in = X(cx) if left else X(cx - w/2)
    tb = slide.shapes.add_textbox(Inches(left_in), Inches(Y(cy)-0.24), Inches(w*S), Inches(0.48))
    tf = tb.text_frame; tf.word_wrap = not left
    if left: tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(0.01)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill_white:
        tb.fill.solid(); tb.fill.fore_color.rgb = RGB("FFFFFF")
        tb.line.fill.background()
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT if left else PP_ALIGN.CENTER
        emit(p, parts(ln), size, bold=bold, color=color)
    return tb

def seg(x1, y1, x2, y2, color=GRAYLN, arrow=False, width=1.0, dashed=False, head='med'):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Inches(X(x1)), Inches(Y(y1)), Inches(X(x2)), Inches(Y(y2)))
    c.shadow.inherit = False
    style_line(c, color, width, dashed=dashed, arrow=arrow, head=head); return c

def curve(p0, p1, color=GRAYLN, width=1.0, arrow=True, samples=28):
    """水平切线三次贝塞尔, freeform 折线逼近; 坐标以 EMU 整数传入避免截断"""
    x0,y0 = p0; x1,y1 = p1
    dx = (x1-x0)*0.55
    c0 = (x0+dx, y0); c1 = (x1-dx, y1)
    pts = []
    for i in range(samples+1):
        t = i/samples; mt = 1-t
        xx = mt**3*x0 + 3*mt*mt*t*c0[0] + 3*mt*t*t*c1[0] + t**3*x1
        yy = mt**3*y0 + 3*mt*mt*t*c0[1] + 3*mt*t*t*c1[1] + t**3*y1
        pts.append((int(X(xx)*914400), int(Y(yy)*914400)))
    fb = slide.shapes.build_freeform(pts[0][0], pts[0][1], scale=1.0)
    fb.add_line_segments(pts[1:], close=False)
    shp = fb.convert_to_shape()
    shp.fill.background(); shp.shadow.inherit = False
    style_line(shp, color, width, arrow=arrow)
    return shp

def sq_marker(x, y, color, side=0.10):
    sq = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
            Inches(X(x)-side/2), Inches(Y(y)-side/2), Inches(side), Inches(side))
    sq.fill.solid(); sq.fill.fore_color.rgb = color
    sq.line.fill.background(); sq.shadow.inherit = False
    return sq

# ==========================================================================
R1, R2, R3 = 3.10, 1.25, -0.60
LANE = 0.42
LANES = {"L": LANE, "T": 0.0, "R": -LANE}

# ---- 1. 列底纹(编码器/检查器两列打灰) + 表头分隔线 ----
band(1.6, 5.6, -2.0, 4.75)
band(8.0, 9.8, -2.0, 4.75)
seg(0.0, 3.92, 11.7, 3.92, color="D5D9E0", width=0.75)

# ---- 2. 表头: 阶段名 + 维度标注 ----
heads = [
    (0.80,  "输入",   [cn("三视角 CIR 90×189")]),
    (3.60,  "编码器", [cn("双路径 Transformer，90+189 token")]),
    (6.80,  "特征码", [it("f"), sub("v"), cn("，8928 维")]),
    (8.90,  "检查器", [cn("原型距离 "), it("d"), sub("v")]),
    (10.75, "判决",   [cn("分数 "), it("s"), up(" ≤ "), it("τ"), cn(" ?")]),
]
for hx, name, dim in heads:
    text(hx, 4.48, [[bcn(name, HDR)]], size=9.5, w=3.6)
    text(hx, 4.12, [dim], size=7, color=SUBT, w=3.9)

# ---- 3. 行标签(左) ----
def row_label(y, color, name, sub_):
    sq_marker(-2.62, y+0.16, color)
    text(-2.50, y+0.16, [[bcn(name, HDR)]], size=9.5, left=True, w=2.6)
    text(-2.50, y-0.22, [[cn(sub_)]], size=7.5, color=SUBT, left=True, w=2.6)
row_label(R1, GREEN,  "决策级聚合",   "本文")
row_label(R2, RED,    "特征级融合",   "F1")
row_label(R3, ORANGE, "瓶颈中层融合", "F2 · MBT 式")

# ---- 4. 行内容 ----
def lane_letters(ry):
    for v, off in LANES.items():
        text(0.35, ry+off, [[it(v)]], size=7.5, color=SUBT, w=0.5)

def enc_container(cx, ry, w):
    box(cx, ry, w, 1.6, line=HAIR, lw=0.8, radius=0.06)

def enc_label(cx, ry, label, w=2.4):
    text(cx, ry, [[cn(label)]], size=8, color=HDR, w=w, fill_white=True)

def chip(cx, ry, w, lines, size=8.5, color=TXT):
    return box(cx, ry, w, 0.44, lw=0.8, radius=0.25, lines=lines, size=size, color=color)

def snode(cx, ry, d=0.56, color_line=STROKEN, fill=None, label_col=TXT):
    return box(cx, ry, d, d, line=color_line, fill=fill, lw=1.0,
               lines=[[it("s")]], size=8.5, color=label_col, shape=MSO_SHAPE.OVAL)

def verdict(ry, txt_, line, fill, col):
    box(13.05, ry, 2.25, 0.60, line=line, fill=fill, lw=1.0, radius=0.5,
        lines=[[bcn(txt_, col)]], size=8)
    seg(10.98, ry, 11.86, ry, color=GRAYLN, arrow=True, width=1.0)

F_X, ETA_X, S_X = 5.95, 8.90, 10.65

# ---------- 行1: 决策级(本文) ----------
lane_letters(R1)
enc_container(3.6, R1, 3.0)
for v, off in LANES.items():
    y = R1+off
    chip(F_X, y, 0.72, [[it("f"), sub(v)]])
    chip(ETA_X, y, 0.70, [[it("η"), sup("("+v+")")]])
    seg(0.55, y, F_X-0.36, y)                      # 输入→编码→特征
    seg(F_X+0.36, y, ETA_X-0.35, y)                # 特征→检查器
enc_label(3.6, R1, "共享双路径编码器")
# 合流(平滑曲线) -> 绿色聚合节点(对角进圆)
node_d = 0.78
box(S_X, R1, node_d, node_d, line=GREEN, fill=XG_F, lw=1.4,
    lines=[[bcn("⊕ ", GREEN_D), it("s")]], size=9, shape=MSO_SHAPE.OVAL)
curve((ETA_X+0.35, R1+LANE), (S_X-0.29, R1+0.27), color=GRAYLN)
seg(ETA_X+0.35, R1, S_X-node_d/2-0.03, R1, arrow=True)
curve((ETA_X+0.35, R1-LANE), (S_X-0.29, R1-0.27), color=GRAYLN)
text(S_X, R1-1.02, [[cn("三路全程独立，仅在此聚合距离")]], size=7.5, color=GREEN_D, w=3.4)
verdict(R1, "✓ 本文 · 稳健最优", GREEN, XG_F, GREEN_D)

# ---------- 行2: 特征级 F1 ----------
lane_letters(R2)
enc_container(3.6, R2, 3.0)
attn_w = 1.3
for v, off in LANES.items():
    y = R2+off
    chip(F_X, y, 0.72, [[it("f"), sub(v)]])
    seg(0.55, y, F_X-0.36, y)
    seg(F_X+0.36, y, 7.25-attn_w/2, y)             # f -> 注意力(全带宽)
enc_label(3.6, R2, "共享双路径编码器")
box(7.25, R2, attn_w, 1.4, line=RED, fill=XR_F, lw=1.4,
    lines=[[bcn("跨视角", RED_D)], [bcn("注意力", RED_D)]], size=8.5)
chip(ETA_X, R2, 0.70, [[it("η")]])
seg(7.25+attn_w/2, R2, ETA_X-0.35, R2, color=REDLN, width=1.8)   # 耦合后的宽带流
snode(S_X, R2)
seg(ETA_X+0.35, R2, S_X-0.31, R2, color=REDLN, width=1.8, arrow=True)
text(7.25, R2-1.05, [[cn("三路特征全带宽耦合")]], size=7.5, color=RED_D, w=3.0)
verdict(R2, "✗ 随机切分有害", RED, XR_F, RED_D)

# ---------- 行3: 瓶颈中层 F2 ----------
lane_letters(R3)
enc_container(2.70, R3, 1.15)
enc_container(4.50, R3, 1.15)
cap_w = 0.45
box(3.60, R3, cap_w, 1.2, line=ORANGE, fill=XO_F, lw=1.3, radius=0.5)
text(3.60, R3-1.15, [[cn("瓶颈 token：窄带交换")]], size=7.5, color=OR_D, w=3.0)
for v, off in LANES.items():
    y = R3+off
    seg(0.55, y, 3.60-cap_w/2, y)                              # 灰: 交换前
    seg(3.60+cap_w/2, y, F_X-0.38, y, color=ORLN, width=1.0)   # 橙: 交换后(窄带)
    chip(F_X, y, 0.76, [[it(FT), sub(v)]], color=OR_D)
    seg(F_X+0.38, y, ETA_X-0.35, y, color=ORLN, width=1.0)
    chip(ETA_X, y, 0.70, [[it("η"), sup("("+v+")")]])
enc_label(2.70, R3, "前段", w=1.0)
enc_label(4.50, R3, "后段", w=1.0)
snode(S_X, R3)
curve((ETA_X+0.35, R3+LANE), (S_X-0.24, R3+0.19), color=ORLN)
seg(ETA_X+0.35, R3, S_X-0.31, R3, color=ORLN, arrow=True)
curve((ETA_X+0.35, R3-LANE), (S_X-0.24, R3-0.19), color=ORLN)
verdict(R3, "≈ 打平 · LOSO 变差", ORANGE, XO_F, OR_D)

# ---- 5. 图例 ----
ly = -2.28
seg(0.0, ly, 0.75, ly, color=GRAYLN, width=1.0)
text(1.85, ly, [[cn("视角内独立数据流")]], size=7.5, color=SUBT, w=2.4)
seg(3.65, ly, 4.40, ly, color=REDLN, width=1.8)
text(5.65, ly, [[cn("跨视角耦合后的数据流")]], size=7.5, color=SUBT, w=2.9)
d = 0.30
box(7.65, ly, d, d, line=GREEN, fill=XG_F, lw=1.1, shape=MSO_SHAPE.OVAL)
text(9.35, ly, [[cn("跨视角交换位置（色 = 策略）")]], size=7.5, color=SUBT, w=3.5)

prs.save("fig6_fusion_v3.pptx")
print("saved fig6_fusion_v3.pptx  %.2fx%.2f in" % (SLIDE_W, SLIDE_H))
