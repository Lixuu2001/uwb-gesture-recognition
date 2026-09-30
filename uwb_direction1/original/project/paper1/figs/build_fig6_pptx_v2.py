# -*- coding: utf-8 -*-
"""fig6 三种跨视角融合策略对比 -> 期刊风 PPTX v2。与 fig1 v2 同一设计语言:
浅灰面板 + 白底细灰框; 每面板唯一的彩色高亮块 = 跨视角交换发生的位置。"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN, MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

FT = "f̃"

def RGB(h): return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
PANEL   = RGB("F5F6F8")
STROKEN = RGB("9AA3B2")
TXT     = RGB("1A1A1A"); SUBT = RGB("667085"); HDR = RGB("3A404A")
BLUE    = RGB("3C5488"); RED = RGB("E64B35"); GREEN = RGB("00A087"); ORANGE = RGB("E18727")
GREEN_D = RGB("00795F"); RED_D = RGB("B03A2A"); OR_D = RGB("A9671A")
XG_F    = RGB("EAF6F2"); XR_F = RGB("FBE9E5"); XO_F = RGB("FCF3E3")
ARROW   = "55606E"
EA = "Microsoft YaHei"; LA = "Times New Roman"

X_MIN, X_MAX = -0.60, 11.0
Y_MIN, Y_MAX = -1.60, 7.85
MARGIN = 0.35; SLIDE_W = 10.5
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

def panel_band(cx, cy, w, h, radius=0.05):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(X(cx-w/2)), Inches(Y(cy+h/2)), Inches(w*S), Inches(h*S))
    try: shp.adjustments[0] = radius
    except Exception: pass
    shp.fill.solid(); shp.fill.fore_color.rgb = PANEL
    shp.line.fill.background(); shp.shadow.inherit = False
    return shp

def box(cx, cy, w, h, line=STROKEN, fill=None, lw=0.9, radius=0.10,
        lines=None, size=9, bold=False, color=TXT):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(X(cx-w/2)), Inches(Y(cy+h/2)), Inches(w*S), Inches(h*S))
    try: shp.adjustments[0] = radius
    except Exception: pass
    shp.fill.solid(); shp.fill.fore_color.rgb = fill if fill is not None else RGB("FFFFFF")
    shp.line.color.rgb = line; shp.line.width = Pt(lw); shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(lines or []):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        emit(p, parts(ln), size, bold=bold, color=color)
    return shp

def text(cx, cy, lines, size=8.5, color=TXT, bold=False, w=3.0, left=False):
    left_in = X(cx) if left else X(cx - w/2)
    tb = slide.shapes.add_textbox(Inches(left_in), Inches(Y(cy)-0.28), Inches(w*S), Inches(0.56))
    tf = tb.text_frame; tf.word_wrap = not left
    if left: tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT if left else PP_ALIGN.CENTER
        emit(p, parts(ln), size, bold=bold, color=color)
    return tb

def seg(x1, y1, x2, y2, color=ARROW, arrow=True, width=0.9, head='med'):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Inches(X(x1)), Inches(Y(y1)), Inches(X(x2)), Inches(Y(y2)))
    c.shadow.inherit = False; style_line(c, color, width, arrow=arrow, head=head); return c

# ==========================================================================
DY = 0.62
LAB = {"L": DY, "T": 0.0, "R": -DY}

def panel(base, title_parts, drawer, note_xy, note_lines, note_color, chip=None):
    panel_band(5.125, base-0.275, 11.35, 2.45)
    text(-0.55, base+1.22, [title_parts], size=11, color=HDR, bold=True, left=True, w=11.0)
    if chip is not None:
        cx, w, txt_, col_line, col_fill, col_txt = chip
        cb = box(cx, base+1.22, w, 0.42, line=col_line, fill=col_fill, lw=1.0,
                 radius=0.5, lines=[[bcn(txt_, col_txt)]], size=8)
    for v, off in LAB.items():
        text(0.10, base+off, [[cn("视角 "), it(v)]], size=8.5, color=SUBT, w=1.1)
    drawer(base)
    nx, ny = note_xy
    text(nx, base+ny, note_lines, size=8, color=note_color, w=3.4)

# ---------------- (a) 决策级聚合 ----------------
def draw_a(b):
    box(2.0, b, 1.4, 1.5, lines=["共享", "编码器"])
    for v, off in LAB.items():
        box(4.2, b+off, 1.0, 0.56, lines=[[it("f"), sub(v)]], size=9.5)
        box(6.5, b+off, 1.5, 0.56, lines=[[cn("检查器 "), it("η"), sup("("+v+")")]], size=8.5)
    box(8.9, b, 1.15, 1.45, line=GREEN, fill=XG_F, lw=1.4,
        lines=[[bcn("⊕", GREEN_D)], [cn("异常分数 "), it("s")]], size=9)
    for v, off in LAB.items():
        y = b+off
        seg(0.62, y, 1.30, y); seg(2.70, y, 3.70, y)
        seg(4.70, y, 5.75, y); seg(7.25, y, 8.325, y)

# ---------------- (b) 特征级融合 F1 ----------------
def draw_b(b):
    box(1.8, b, 1.2, 1.5, lines=["共享", "编码器"])
    box(4.3, b, 1.6, 1.5, line=RED, fill=XR_F, lw=1.4,
        lines=[[bcn("跨视角", RED_D)], [bcn("注意力", RED_D)]], size=9)
    box(6.6, b, 1.25, 1.0, lines=["融合特征", [it(FT), sub("v")]], size=8.5)
    box(8.9, b, 1.2, 0.9, lines=["检查器"])
    for v, off in LAB.items():
        y = b+off
        seg(0.62, y, 1.20, y); seg(2.40, y, 3.50, y)
    seg(5.10, b, 5.975, b); seg(7.225, b, 8.30, b)

# ---------------- (c) 瓶颈中层融合 F2 ----------------
def draw_c(b):
    box(1.7, b, 1.15, 1.5, lines=["编码器", "前段"])
    box(3.9, b, 1.3, 1.5, line=ORANGE, fill=XO_F, lw=1.4,
        lines=[[bcn("瓶颈", OR_D)], [bcn("token", OR_D)]], size=9)
    box(6.0, b, 1.15, 1.5, lines=["编码器", "后段"])
    box(8.0, b, 0.95, 0.66, lines=[[it(FT), sub("v")]], size=9.5)
    box(10.0, b, 1.15, 0.9, lines=["检查器"])
    for v, off in LAB.items():
        y = b+off
        seg(0.62, y, 1.125, y); seg(2.275, y, 3.25, y); seg(4.55, y, 5.425, y)
    seg(6.575, b, 7.525, b); seg(8.475, b, 9.425, b)

# ==========================================================================
panel(6.35, [bcn("(a) 决策级聚合：无跨视角特征交换", HDR)], draw_a,
      (8.9, -1.16), ["三路全程独立，仅在此处聚合"], GREEN_D,
      chip=(3.45, 1.15, "本文采用", GREEN, XG_F, GREEN_D))
panel(3.15, [bcn("(b) 特征级融合（F1）：编码后注意力交换", HDR)], draw_b,
      (4.3, -1.16), ["三路特征在此耦合（随机切分有害）"], RED_D)
panel(-0.05, [bcn("(c) 瓶颈中层融合（F2，MBT 式）：编码器中层经少量瓶颈 token 交换", HDR)], draw_c,
      (3.9, -1.16), ["窄带交换（正则）：随机切分打平、LOSO 变差"], OR_D)

prs.save("fig6_fusion_v2.pptx")
print("saved fig6_fusion_v2.pptx  %.2fx%.2f in" % (SLIDE_W, SLIDE_H))
