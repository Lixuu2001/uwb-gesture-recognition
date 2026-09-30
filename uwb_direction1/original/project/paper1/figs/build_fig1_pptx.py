# -*- coding: utf-8 -*-
"""MV-OSR 架构图 (fig1 v2) -> 原生可编辑 PPTX。坐标从 TikZ(cm) 端口, y 轴翻转。"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

# 组合字符用显式码点
KH   = "k̂"        # k̂
DBAR = "d̄"        # d̄
SBAR = "σ̄"   # σ̄

C_BACK   = RGBColor(0x3C,0x54,0x88); C_BACKF = RGBColor(0xEE,0xF1,0xF7)
C_ENCF   = RGBColor(0xE6,0xEC,0xF5)
C_ORANGE = RGBColor(0xE1,0x87,0x27); C_ORANGEF = RGBColor(0xFD,0xF3,0xE7)
C_RED    = RGBColor(0xE6,0x4B,0x35); C_REDF = RGBColor(0xFD,0xEE,0xEB)
C_ACC    = RGBColor(0x00,0xA0,0x87); C_ACCF = RGBColor(0xE6,0xF4,0xF1)
C_ARROW  = "5F5F5F"; C_GRAY = "6E6E6E"
TXT      = RGBColor(0x22,0x22,0x22); GRY = RGBColor(0x55,0x55,0x55)
WHITE    = RGBColor(0xFF,0xFF,0xFF)
EA = "Microsoft YaHei"; LA = "Times New Roman"

X_MIN, X_MAX = -0.75, 21.35
Y_MIN, Y_MAX = -1.40, 6.55
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
def sub(t, italic=True):  return dict(t=t, it=italic, base='sub', cn=False)
def sup(t, italic=False): return dict(t=t, it=italic, base='sup', cn=False)

def emit(para, parts, size, bold=False, color=TXT):
    for p in parts:
        r = para.add_run(); r.text = p['t']; sz = size; base = None
        if p['base'] == 'sub':  sz = size*0.72; base = -25000
        elif p['base'] == 'sup': sz = size*0.72; base = 30000
        style_run(r, sz, bold=bold, italic=p['it'], color=color,
                  latin=(EA if p['cn'] else LA), baseline=base)

def parts(c): return [cn(c)] if isinstance(c, str) else c

_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
def style_line(shape, color_hex, width_pt, dashed=False, arrow=False):
    # 用 get_or_add_ln() 把 a:ln 放到 spPr 内的合法位置(在 effectLst 之前),
    # 否则 PowerPoint 会丢弃箭头端 (tailEnd)。
    ln = shape.line._get_or_add_ln()
    for ch in list(ln): ln.remove(ch)
    ln.set('w', str(int(width_pt*12700))); ln.set('cap', 'rnd')
    ln.append(parse_xml('<a:solidFill %s><a:srgbClr val="%s"/></a:solidFill>' % (_NS, color_hex)))
    if dashed: ln.append(parse_xml('<a:prstDash %s val="dash"/>' % _NS))
    ln.append(parse_xml('<a:round %s/>' % _NS))
    if arrow: ln.append(parse_xml('<a:tailEnd %s type="triangle" w="lg" len="lg"/>' % _NS))

def add_box(cx, cy, w, h, fill, line, line_w=1.25, radius=0.09,
            title=None, sub_=None, title_size=11, sub_size=8,
            title_color=TXT, title_bold=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(X(cx-w/2)), Inches(Y(cy+h/2)), Inches(w*S), Inches(h*S))
    try: shp.adjustments[0] = radius
    except Exception: pass
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line; shp.line.width = Pt(line_w); shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    first = True
    if title is not None:
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        emit(p, parts(title), title_size, bold=title_bold, color=title_color); first = False
    if sub_ is not None:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        emit(p, parts(sub_), sub_size, bold=False, color=TXT)
    return shp

def add_textbox(cx, cy, w, lines, size=8, color=TXT, bold=False,
                align=PP_ALIGN.CENTER, fill_white=False, h=0.5):
    tb = slide.shapes.add_textbox(Inches(X(cx-w/2)), Inches(Y(cy+h/2)),
                                  Inches(w*S), Inches(h*S))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill_white:
        tb.fill.solid(); tb.fill.fore_color.rgb = WHITE; tb.line.fill.background()
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        emit(p, parts(ln), size, bold=bold, color=color)
    return tb

def add_group(cx, cy, w, h, color_hex, width_pt=1.0, radius=0.05):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(X(cx-w/2)), Inches(Y(cy+h/2)), Inches(w*S), Inches(h*S))
    try: shp.adjustments[0] = radius
    except Exception: pass
    shp.fill.background(); shp.shadow.inherit = False
    style_line(shp, color_hex, width_pt, dashed=True); return shp

def seg(x1, y1, x2, y2, color=C_ARROW, dashed=False, arrow=False, width=1.3):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Inches(X(x1)), Inches(Y(y1)), Inches(X(x2)), Inches(Y(y2)))
    c.shadow.inherit = False; style_line(c, color, width, dashed=dashed, arrow=arrow); return c

def path(pts, color=C_ARROW, dashed=False, width=1.3):
    for i in range(len(pts)-1):
        (x1,y1),(x2,y2) = pts[i], pts[i+1]
        seg(x1,y1,x2,y2, color=color, dashed=dashed, arrow=(i==len(pts)-2), width=width)

def dot(x, y, r=0.055):
    d = slide.shapes.add_shape(MSO_SHAPE.OVAL,
            Inches(X(x)-r/2), Inches(Y(y)-r/2), Inches(r), Inches(r))
    d.fill.solid(); d.fill.fore_color.rgb = RGBColor(0x5F,0x5F,0x5F)
    d.line.fill.background(); d.shadow.inherit = False; return d

# ==========================================================================
yL, yT, yR, yS = 3.20, 1.75, 0.30, 5.50
xbus = 10.95
VMAP = {yL:"L", yT:"T", yR:"R"}

# 1. 分组框
add_group(0.85, 1.75, 2.402, 4.24, C_GRAY, 0.9)
add_group(15.09, 5.50, 7.127, 1.372, "3C5488", 1.0)
add_group(13.10, 1.75, 2.892, 4.26, "E64B35", 1.0)

# 2. (箭头改到方框之后绘制, 使箭头三角浮在方框之上, 见文件末尾)

# 3. 方框
add_box(0.85, yL, 2.05, 0.92, C_BACKF, C_BACK, title=[cn("左雷达 "), it("L")], sub_="CIR 90×189")
add_box(0.85, yT, 2.05, 0.92, C_BACKF, C_BACK, title=[cn("上雷达 "), it("T")], sub_="CIR 90×189")
add_box(0.85, yR, 2.05, 0.92, C_BACKF, C_BACK, title=[cn("右雷达 "), it("R")], sub_="CIR 90×189")

# 归一化: 空框 + 上部标题 + 下部副文
add_box(3.30, yT, 2.0, 4.05, C_BACKF, C_BACK)
add_textbox(3.30, 2.85, 1.95, ["振幅归一化"], size=11.5, bold=True)
add_textbox(3.30, 1.55, 1.95, [[up("|"), it("x"), up("|"), cn(" 后逐样本、")],
            "逐视角除以标准差", "(abs_std)"], size=8)

# 编码器
add_box(6.55, yT, 3.55, 4.05, C_ENCF, C_BACK, radius=0.05)
add_textbox(6.55, 3.52, 3.3, ["共享双路径编码器"], size=12.5, color=C_BACK, bold=True)
add_box(6.55, 2.48, 3.15, 0.88, WHITE, C_BACK, title="时间路径 Transformer",
        sub_="90 个时间 token × 4 层", title_size=10.5)
add_box(6.55, 1.18, 3.15, 0.88, WHITE, C_BACK, title="距离路径 Transformer",
        sub_="189 个距离 token × 4 层", title_size=10.5)
add_textbox(6.55, 0.12, 3.4, ["三视角权重共享 + 视角嵌入", "两路 token 展平拼接"], size=8, color=GRY)

# 特征码
for y in (yL, yT, yR):
    add_box(9.80, y, 1.58, 0.76, C_BACKF, C_BACK,
            title=[it("f"), sub(VMAP[y]), cn("  (8928 维)")], title_size=10, title_bold=False)

# 阶段一
add_box(12.80, yS, 2.05, 0.88, C_BACKF, C_BACK, title="特征聚合")
add_box(15.25, yS, 2.05, 0.88, C_BACKF, C_BACK, title="闭集分类器", sub_="交叉熵训练")
add_box(17.60, yS, 1.62, 0.88, C_ORANGEF, C_ORANGE, title=[cn("预测类 "), it(KH)])

# 阶段二
for y in (yL, yT, yR):
    v = VMAP[y]
    add_box(13.10, y, 2.40, 0.94, C_REDF, C_RED,
            title=[cn("检查器 "), it("η"), sub(KH), sup("("+v+")")],
            sub_=[it("d"), sub(v), up(" = ‖"), it("l"), sub(v), up(" − "),
                  it("φ"), sub(KH), sup("("+v+")"), up("‖")],
            title_size=11, sub_size=8.5)
add_box(16.60, yT, 3.1, 1.24, C_REDF, C_RED, title="多视角聚合（异常分数）",
        sub_=[it("s"), up(" = "), it(DBAR), up("/"), it(SBAR), up(" + "), it("β"),
              up("·var("), it("d"), sub("v"), up(")/"), it(SBAR), sup("2")],
        title_size=10.5, sub_size=9)

# 输出
add_box(20.10, 2.40, 2.02, 0.92, C_ACCF, C_ACC,
        title=[it("s"), up(" ≤ "), it("τ"), cn("：接受")],
        sub_=[cn("输出已知类 "), it(KH)], title_size=10.5)
add_box(20.10, 0.78, 2.02, 0.92, C_REDF, C_RED,
        title=[it("s"), up(" > "), it("τ"), cn("：拒识")], sub_="未知手势", title_size=10.5)

# 3b. 箭头 (浮在方框之上, 箭头三角可见)
for y in (yL, yT, yR):
    path([(1.875,y),(2.30,y)]); path([(4.30,y),(4.775,y)])
    path([(8.325,y),(9.01,y)]); path([(10.59,y),(11.90,y)])
seg(xbus, yR, xbus, yS)
path([(xbus,yS),(11.775,yS)]); path([(13.825,yS),(14.225,yS)]); path([(16.275,yS),(16.79,yS)])
path([(17.60,5.06),(17.60,4.30),(13.10,4.30),(13.10,3.88)], color="E18727", dashed=True)
path([(14.30,yL),(14.62,yL),(14.62,2.07),(15.05,2.07)])
path([(14.30,yT),(15.05,yT)])
path([(14.30,yR),(14.62,yR),(14.62,1.43),(15.05,1.43)])
seg(18.15, yT, 18.22, yT)
path([(18.22,yT),(18.22,2.40),(19.09,2.40)]); path([(18.22,yT),(18.22,0.78),(19.09,0.78)])
for y in (yL, yT, yR): dot(xbus, y)

# 4. 标签
add_textbox(15.09, 6.206, 3.2, ["阶段一：闭集分类"], size=10.5, color=C_BACK, bold=True, fill_white=True)
add_textbox(13.10, -0.75, 6.5, ["阶段二：逐类 × 逐视角深度检查（开集判决）"], size=10.5, color=C_RED, bold=True)
add_textbox(15.35, 4.46, 3.6, [[cn("按 "), it(KH), cn(" 选对应类的检查器组")]],
            size=8.5, color=C_ORANGE, fill_white=True)
add_textbox(0.90, -0.95, 3.0, ["训练期随机丢视角",
            [it("p"), sub("drop", italic=False), cn("=0.5，保留 1–2 路")]], size=8, color=GRY)

prs.save("fig1_mvosr.pptx")
print("saved fig1_mvosr.pptx  %.2fx%.2f in  scale=%.4f" % (SLIDE_W, SLIDE_H, S))
