# -*- coding: utf-8 -*-
"""图6 三种跨视角融合策略对比 -> 原生可编辑 PPTX。风格与 fig1 统一。"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN, MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

FT = "f̃"   # f 带 ~

C_BACK  = RGBColor(0x3C,0x54,0x88); C_BACKF = RGBColor(0xEE,0xF1,0xF7)
C_ENCF  = RGBColor(0xE6,0xEC,0xF5)
C_RED   = RGBColor(0xE6,0x4B,0x35); C_REDF = RGBColor(0xFD,0xEE,0xEB)
C_ACC   = RGBColor(0x00,0xA0,0x87); C_ACCF = RGBColor(0xE6,0xF4,0xF1)
C_ORANGE= RGBColor(0xE1,0x87,0x27)
XB_RF   = RGBColor(0xF7,0xCD,0xC4); XB_RT = RGBColor(0xB0,0x3A,0x2A)   # (b) 注意力 强红
XB_OF   = RGBColor(0xF8,0xDB,0xB0); XB_OT = RGBColor(0xA9,0x67,0x1A)   # (c) 瓶颈 强橙
N_ACC   = RGBColor(0x17,0x74,0x63); N_RED = RGBColor(0xC0,0x39,0x2B); N_OR = RGBColor(0xB9,0x70,0x1C)
TITLE   = RGBColor(0x33,0x33,0x33); TXT = RGBColor(0x22,0x22,0x22)
ARROW   = "7A7A7A"
EA = "Microsoft YaHei"; LA = "Times New Roman"

X_MIN, X_MAX = -0.4, 11.0
Y_MIN, Y_MAX = -1.55, 7.85
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
def sub(t, italic=True):  return dict(t=t, it=italic, base='sub', cn=False)
def sup(t, italic=False): return dict(t=t, it=italic, base='sup', cn=False)

def emit(para, ps, size, bold=False, color=TXT):
    for p in ps:
        r = para.add_run(); r.text = p['t']; sz = size; base = None
        if p['base'] == 'sub':  sz = size*0.72; base = -25000
        elif p['base'] == 'sup': sz = size*0.72; base = 30000
        style_run(r, sz, bold=bold, italic=p['it'], color=color,
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

def box(cx, cy, w, h, fill, line, lines, size=9, bold=False, color=TXT, lw=1.1, radius=0.10):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(X(cx-w/2)), Inches(Y(cy+h/2)), Inches(w*S), Inches(h*S))
    try: shp.adjustments[0] = radius
    except Exception: pass
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line; shp.line.width = Pt(lw); shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        emit(p, parts(ln), size, bold=bold, color=color)
    return shp

def text(cx, cy, lines, size=8.5, color=TXT, bold=False, align=PP_ALIGN.CENTER,
         w=3.0, left=False):
    left_in = X(cx) if left else X(cx - w/2)
    tb = slide.shapes.add_textbox(Inches(left_in), Inches(Y(cy)-0.28), Inches(w*S), Inches(0.56))
    tf = tb.text_frame; tf.word_wrap = not left
    if left:
        tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT if left else align
        emit(p, parts(ln), size, bold=bold, color=color)
    return tb

def seg(x1, y1, x2, y2, color=ARROW, arrow=True, width=0.9, head='med'):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Inches(X(x1)), Inches(Y(y1)), Inches(X(x2)), Inches(Y(y2)))
    c.shadow.inherit = False; style_line(c, color, width, arrow=arrow, head=head); return c

# ==========================================================================
DY = 0.62
LAB = {"L": DY, "T": 0.0, "R": -DY}

def panel(base, title, drawer, note_xy, note_lines, note_color):
    text(-0.30, base+1.30, [title], size=12, color=TITLE, bold=True, left=True, w=11.0)
    for v, off in LAB.items():
        text(0.10, base+off, [[cn("视角 "), it(v)]], size=9, color=C_BACK, w=1.1)
    drawer(base)
    nx, ny = note_xy
    text(nx, base+ny, note_lines, size=8.5, color=note_color, w=3.4)

# ---------------- (a) 决策级聚合 ----------------
def draw_a(b):
    box(2.0, b, 1.4, 1.5, C_ENCF, C_BACK, ["共享", "编码器"], size=9)
    for v, off in LAB.items():
        box(4.2, b+off, 1.0, 0.58, C_BACKF, C_BACK, [[it("f"), sub(v)]], size=9.5)
        box(6.5, b+off, 1.5, 0.58, C_REDF, C_RED,
            [[cn("检查器 "), it("η"), sup("("+v+")")]], size=8.5)
    box(8.9, b, 1.15, 1.45, C_ACCF, C_ACC, ["⊕", [cn("异常分数 "), it("s")]], size=9)
    for v, off in LAB.items():
        y = b+off
        seg(0.62, y, 1.30, y); seg(2.70, y, 3.70, y)
        seg(4.70, y, 5.75, y); seg(7.25, y, 8.325, y)

# ---------------- (b) 特征级融合 F1 ----------------
def draw_b(b):
    box(1.8, b, 1.2, 1.5, C_ENCF, C_BACK, ["共享", "编码器"], size=9)
    box(4.3, b, 1.6, 1.5, XB_RF, C_RED, ["跨视角", "注意力"], size=9, bold=True, color=XB_RT, lw=1.5)
    box(6.6, b, 1.25, 1.0, C_BACKF, C_BACK, ["融合特征", [it(FT), sub("v")]], size=8.8)
    box(8.9, b, 1.2, 0.9, C_REDF, C_RED, ["检查器"], size=9)
    for v, off in LAB.items():
        y = b+off
        seg(0.62, y, 1.20, y); seg(2.40, y, 3.50, y)
    seg(5.10, b, 5.975, b); seg(7.225, b, 8.30, b)

# ---------------- (c) 瓶颈中层融合 F2 ----------------
def draw_c(b):
    box(1.7, b, 1.15, 1.5, C_ENCF, C_BACK, ["编码器", "前段"], size=9)
    box(3.9, b, 1.3, 1.5, XB_OF, C_ORANGE, ["瓶颈", "token"], size=9, bold=True, color=XB_OT, lw=1.5)
    box(6.0, b, 1.15, 1.5, C_ENCF, C_BACK, ["编码器", "后段"], size=9)
    box(8.0, b, 0.95, 0.7, C_BACKF, C_BACK, [[it(FT), sub("v")]], size=9.5)
    box(10.0, b, 1.15, 0.9, C_REDF, C_RED, ["检查器"], size=9)
    for v, off in LAB.items():
        y = b+off
        seg(0.62, y, 1.125, y); seg(2.275, y, 3.25, y); seg(4.55, y, 5.425, y)
    seg(6.575, b, 7.525, b); seg(8.475, b, 9.425, b)

# ==========================================================================
panel(6.4, "(a) 决策级聚合（无跨视角特征交换 —— 本文采用）", draw_a,
      (8.9, -1.15), ["三路全程独立，", "仅在此处聚合"], N_ACC)
panel(3.2, "(b) 特征级融合（F1）：编码后注意力交换", draw_b,
      (4.3, -1.15), ["三路特征在此耦合", "（随机切分有害）"], N_RED)
panel(0.0, "(c) 瓶颈中层融合（F2，MBT 式）：编码器中层经少量瓶颈 token 交换", draw_c,
      (3.9, -1.15), ["窄带交换（正则）", "随机切分打平、LOSO 变差"], N_OR)

prs.save("fig6_fusion.pptx")
print("saved fig6_fusion.pptx  %.2fx%.2f in  scale=%.4f" % (SLIDE_W, SLIDE_H, S))
