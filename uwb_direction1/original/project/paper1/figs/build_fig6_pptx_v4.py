# -*- coding: utf-8 -*-
"""fig6 v4: 融合深度轴 × IEEE 工程图视觉语言。
结构(按 spec): 行序=融合由早到晚(瓶颈→特征级→决策级), 无面板字母/结论徽章/实验结果,
底部"融合深度"轴带三个策略色刻度点。
视觉(按用户参考图): 白底大留白; 分组容器=浅灰填充+深色虚线圆角框+粗体内标题;
模块=白底细深描边; 交换模块=竖排文字药丸; 特征码=轻立体叠片; 细线小箭头;
饱和色仅限交换记号与耦合流; 衬线字体(宋体/黑体+Times)。"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN, MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

FT = "f̃"

def RGB(h): return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
INK     = RGB("2B2F36")            # 主文字/描边: 近黑碳灰
SUBT    = RGB("6A6F78")
CONT_F  = RGB("F1F2F4")            # 分组容器浅灰
CONT_L  = RGB("53575F")            # 容器虚线
SLAB_B  = RGB("DDE0E5")            # 叠片背片
AMBER   = RGB("C8821F"); AMBER_F = RGB("F8EED9")
CORAL   = RGB("C7523F"); CORAL_F = RGB("F9E6E1")
TEAL    = RGB("1E8E79"); TEAL_F  = RGB("E3F1ED")
LN      = "3A3F47"                  # 中性数据流
LN_AM   = "C8821F"; LN_CO = "C7523F"; LN_TE = "1E8E79"
HAIRC   = "C9CDD4"                  # 表头细线/分隔
EA_R = "SimSun"; EA_B = "SimHei"; LA = "Times New Roman"

X_MIN, X_MAX = -2.80, 11.95
Y_MIN, Y_MAX = -3.25, 4.95
MARGIN = 0.35; SLIDE_W = 13.333
S = (SLIDE_W - 2*MARGIN)/(X_MAX - X_MIN)
SLIDE_H = MARGIN*2 + (Y_MAX - Y_MIN)*S
def X(xc): return MARGIN + (xc - X_MIN)*S
def Y(yc): return MARGIN + (Y_MAX - yc)*S

prs = Presentation()
prs.slide_width = Inches(SLIDE_W); prs.slide_height = Inches(SLIDE_H)
slide = prs.slides.add_slide(prs.slide_layouts[6])

def style_run(r, size, bold=False, italic=False, color=INK, latin=LA, baseline=None):
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = color; r.font.name = latin
    ea = EA_B if bold else EA_R
    rPr = r._r.get_or_add_rPr(); latin_el = rPr.find(qn('a:latin'))
    for tag, face in (('a:ea', ea), ('a:cs', latin)):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {}); latin_el.addnext(el)
        el.set('typeface', face)
    if baseline is not None: rPr.set('baseline', str(baseline))

def it(t):  return dict(t=t, it=True,  base='n')
def up(t):  return dict(t=t, it=False, base='n')
def cn(t):  return dict(t=t, it=False, base='n')
def bcn(t, col=None): return dict(t=t, it=False, base='n', b=True, col=col)
def sub(t, italic=True):  return dict(t=t, it=italic, base='sub')
def sup(t, italic=False): return dict(t=t, it=italic, base='sup')

def emit(para, ps, size, bold=False, color=INK):
    for p in ps:
        r = para.add_run(); r.text = p['t']; sz = size; base = None
        if p['base'] == 'sub':  sz = size*0.72; base = -25000
        elif p['base'] == 'sup': sz = size*0.72; base = 30000
        style_run(r, sz, bold=p.get('b', bold), italic=p['it'],
                  color=p.get('col') or color, baseline=base)

def parts(c): return [cn(c)] if isinstance(c, str) else c

_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
def style_line(shape, color_hex, width_pt, dashed=False, arrow=False, head='med', dot=False):
    ln = shape.line._get_or_add_ln()
    for ch in list(ln): ln.remove(ch)
    ln.set('w', str(int(width_pt*12700))); ln.set('cap', 'rnd')
    ln.append(parse_xml('<a:solidFill %s><a:srgbClr val="%s"/></a:solidFill>' % (_NS, color_hex)))
    if dashed: ln.append(parse_xml('<a:prstDash %s val="dash"/>' % _NS))
    elif dot:  ln.append(parse_xml('<a:prstDash %s val="sysDot"/>' % _NS))
    ln.append(parse_xml('<a:round %s/>' % _NS))
    if arrow: ln.append(parse_xml('<a:tailEnd %s type="triangle" w="%s" len="%s"/>' % (_NS, head, head)))

def box(cx, cy, w, h, line=INK, fill=None, lw=0.8, radius=0.12,
        lines=None, size=8.5, bold=False, color=INK, dashed=False,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, vert=False):
    shp = slide.shapes.add_shape(shape,
            Inches(X(cx-w/2)), Inches(Y(cy+h/2)), Inches(w*S), Inches(h*S))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try: shp.adjustments[0] = radius
        except Exception: pass
    shp.fill.solid(); shp.fill.fore_color.rgb = fill if fill is not None else RGB("FFFFFF")
    shp.shadow.inherit = False
    style_line(shp, "%02X%02X%02X" % (line[0], line[1], line[2]) if isinstance(line, RGBColor) else line, lw, dashed=dashed)
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.01)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    if vert:
        tf._txBody.bodyPr.set('vert', 'eaVert')
    for i, ln_ in enumerate(lines or []):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        emit(p, parts(ln_), size, bold=bold, color=color)
    return shp

def text(cx, cy, lines, size=8, color=INK, bold=False, w=3.0, left=False, fill_white=False):
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

def seg(x1, y1, x2, y2, color=LN, arrow=False, width=0.8, dashed=False, dot=False, head='med'):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Inches(X(x1)), Inches(Y(y1)), Inches(X(x2)), Inches(Y(y2)))
    c.shadow.inherit = False
    style_line(c, color, width, dashed=dashed, arrow=arrow, head=head, dot=dot); return c

def curve(p0, p1, color=LN, width=0.8, arrow=True, samples=28):
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

def slab(cx, cy, w, h, lines, size=8.5, color=INK):
    """轻立体叠片: 灰背片 + 白面片"""
    box(cx+0.07, cy+0.055, w, h, line=SLAB_B, fill=SLAB_B, lw=0.5, radius=0.15)
    return box(cx, cy, w, h, lw=0.8, radius=0.15, lines=lines, size=size, color=color)

def container(cx, cy, w, h, label=None, label_w=2.2):
    shp = box(cx, cy, w, h, line=CONT_L, fill=CONT_F, lw=0.8, radius=0.06, dashed=True)
    if label:
        text(cx + w/2 - label_w/2 - 0.12, cy + h/2 - 0.22, [[bcn(label)]],
             size=8, w=label_w)
    return shp

# ==========================================================================
R1, R2, R3 = 3.10, 1.02, -1.06          # 早→晚: 瓶颈 / 特征级 / 决策级
LANE = 0.40
LANES = {"L": LANE, "T": 0.0, "R": -LANE}
F_X, ETA_X, S_X = 6.15, 8.90, 10.70
CAP_X, ATT_X = 3.60, 7.35

# ---- 1. 表头: 阶段名 + 维度 + 细线; 列界虚点分隔 ----
heads = [
    (0.80,  "输入",   [cn("三视角 CIR，"), it("x"), sub("v"), up(" ∈ ℝ"), sup("90×189")]),
    (3.60,  "编码器", [cn("权重共享双路径 Transformer")]),
    (6.60,  "特征码", [it("f"), sub("v"), up(" ∈ ℝ"), sup("8928")]),
    (8.90,  "检查器", [cn("逐视角，"), it("η"), sub("k", italic=False), sup("(v)")]),
    (10.80, "分数",   [cn("异常分数 "), it("s")]),
]
for hx, name, dim in heads:
    text(hx, 4.52, [[bcn(name)]], size=9.5, w=3.4)
    text(hx, 4.16, [dim], size=7, color=SUBT, w=3.6)
seg(-0.1, 3.94, 11.75, 3.94, color=HAIRC, width=0.75)
for bx in (1.75, 5.45, 7.90, 9.85):
    seg(bx, 3.94, bx, -1.70, color="DFE2E7", width=0.5, dot=True)

# ---- 2. 行标签(参考图 Original image 式: 彩色粗体衬线) ----
def row_label(y, color, name, sub_):
    text(-2.70, y+0.17, [[bcn(name, color)]], size=10, left=True, w=2.7)
    text(-2.70, y-0.24, [[cn(sub_)]], size=7.5, color=SUBT, left=True, w=2.7)
row_label(R1, AMBER, "瓶颈中层融合", "MBT 式")
row_label(R2, CORAL, "特征级融合", "跨视角注意力")
row_label(R3, TEAL,  "决策级聚合", "本文")

def lane_letters(ry):
    for v, off in LANES.items():
        text(0.32, ry+off, [[it(v)]], size=7.5, color=SUBT, w=0.5)

def eta_chip(cx, y, v=None):
    lab = [it("η")] if v is None else [it("η"), sup("("+v+")")]
    return box(cx, y, 0.66, 0.42, lw=0.8, radius=0.3, lines=[lab], size=8.5)

def s_node(cx, ry, d=0.54):
    return box(cx, ry, d, d, lw=0.9, lines=[[it("s")]], size=8.5, shape=MSO_SHAPE.OVAL)

# ---------- 行1: 瓶颈中层融合(最早交换) ----------
lane_letters(R1)
container(2.70, R1, 1.30, 1.55, "前段", label_w=0.75)
container(4.55, R1, 1.30, 1.55, "后段", label_w=0.75)
box(CAP_X, R1, 0.46, 1.30, line=AMBER, fill=AMBER_F, lw=1.2, radius=0.5,
    lines=[[bcn("瓶颈", AMBER)]], size=7.5, vert=True)
for v, off in LANES.items():
    y = R1+off
    seg(0.55, y, CAP_X-0.23, y)                                  # 灰: 交换前
    seg(CAP_X+0.23, y, F_X-0.44, y, color=LN_AM, width=0.8)      # 琥珀: 窄带交换后
    seg(F_X+0.44, y, ETA_X-0.33, y, color=LN_AM, width=0.8)
    slab(F_X, y, 0.80, 0.46, [[it(FT), sub(v)]], color=AMBER)
    eta_chip(ETA_X, y, v)
s_node(S_X, R1)
curve((ETA_X+0.33, R1+LANE), (S_X-0.22, R1+0.18), color=LN_AM)
seg(ETA_X+0.33, R1, S_X-0.29, R1, color=LN_AM, arrow=True)
curve((ETA_X+0.33, R1-LANE), (S_X-0.22, R1-0.18), color=LN_AM)
text(CAP_X, R1-1.12, [[cn("少量共享 token 窄带交换")]], size=7.5, color=AMBER, w=3.0)

# ---------- 行2: 特征级融合 ----------
lane_letters(R2)
container(3.60, R2, 3.10, 1.55, "共享编码器", label_w=1.75)
for v, off in LANES.items():
    y = R2+off
    seg(0.55, y, F_X-0.44, y)
    slab(F_X, y, 0.80, 0.46, [[it("f"), sub(v)]])
    seg(F_X+0.44, y, ATT_X-0.31, y)                              # f -> 注意力(全带宽)
box(ATT_X, R2, 0.62, 1.50, line=CORAL, fill=CORAL_F, lw=1.2, radius=0.30,
    lines=[[bcn("跨视角注意力", CORAL)]], size=7.5, vert=True)
eta_chip(ETA_X, R2)
seg(ATT_X+0.31, R2, ETA_X-0.33, R2, color=LN_CO, width=1.5)      # 耦合后的宽带流
s_node(S_X, R2)
seg(ETA_X+0.33, R2, S_X-0.29, R2, color=LN_CO, width=1.5, arrow=True)
text(ATT_X, R2-1.12, [[cn("三路特征全带宽耦合")]], size=7.5, color=CORAL, w=2.8)

# ---------- 行3: 决策级聚合(最晚, 本文) ----------
lane_letters(R3)
container(3.60, R3, 3.10, 1.55, "共享编码器", label_w=1.75)
for v, off in LANES.items():
    y = R3+off
    seg(0.55, y, F_X-0.44, y)
    slab(F_X, y, 0.80, 0.46, [[it("f"), sub(v)]])
    seg(F_X+0.44, y, ETA_X-0.33, y)
    eta_chip(ETA_X, y, v)
box(S_X, R3, 0.66, 0.66, line=TEAL, fill=TEAL_F, lw=1.2,
    lines=[[bcn("⊕", TEAL), it(" s")]], size=9, shape=MSO_SHAPE.OVAL)
curve((ETA_X+0.33, R3+LANE), (S_X-0.26, R3+0.21), color=LN)
seg(ETA_X+0.33, R3, S_X-0.35, R3, arrow=True)
curve((ETA_X+0.33, R3-LANE), (S_X-0.26, R3-0.21), color=LN)
text(S_X-0.15, R3-1.12, [[cn("三路全程独立，仅在此聚合距离")]], size=7.5, color=TEAL, w=3.2)

# ---- 3. 融合深度轴(底部): 三个策略色刻度点 ----
AX_Y = -2.50
seg(1.20, AX_Y, 11.30, AX_Y, color="6A6F78", width=0.9, arrow=True)
for dx_, col in ((CAP_X, AMBER), (ATT_X, CORAL), (S_X, TEAL)):
    box(dx_, AX_Y, 0.16, 0.16, line=col, fill=col, lw=0.5, shape=MSO_SHAPE.OVAL)
text(0.30, AX_Y, [[bcn("融合深度")]], size=8.5, w=1.6)
text(1.95, AX_Y-0.30, [[cn("早")]], size=7.5, color=SUBT, w=0.6)
text(10.95, AX_Y-0.30, [[cn("晚")]], size=7.5, color=SUBT, w=0.6)

# ---- 4. 图例(参考图式: 图元样本 + 名称) ----
LG_Y = -3.00
seg(0.30, LG_Y, 0.95, LG_Y, color=LN, width=0.8)
text(2.00, LG_Y, [[cn("视角内独立数据流")]], size=7.5, color=SUBT, w=2.3)
seg(3.60, LG_Y, 4.25, LG_Y, color=LN_CO, width=1.5)
text(5.42, LG_Y, [[cn("交换后耦合数据流")]], size=7.5, color=SUBT, w=2.4)
box(7.05, LG_Y, 0.22, 0.22, line=TEAL, fill=TEAL_F, lw=0.9, shape=MSO_SHAPE.OVAL)
text(8.60, LG_Y, [[cn("跨视角交换位置（色＝策略）")]], size=7.5, color=SUBT, w=3.3)

prs.save("fig6_fusion_v4.pptx")
print("saved fig6_fusion_v4.pptx  %.2fx%.2f in" % (SLIDE_W, SLIDE_H))
