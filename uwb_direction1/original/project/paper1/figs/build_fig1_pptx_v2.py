# -*- coding: utf-8 -*-
"""fig1 MV-OSR 架构 -> 期刊风 PPTX v2。
设计语言: 浅灰面板分组 + 白底细灰描边模块 + 语义色仅三处(橙=枢纽/红=开集/绿=接受),
常规字重为主, 加粗只给阶段标题与判决词。"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN, MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

KH = "k̂"

def RGB(h): return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
PANEL   = RGB("F5F6F8")           # 面板浅灰
STROKEN = RGB("9AA3B2")           # 中性描边
ENCF    = RGB("EDEFF4")           # 编码器容器
ENCLN   = RGB("D7DBE3")
TXT     = RGB("1A1A1A"); SUBT = RGB("667085"); HDR = RGB("3A404A")
BLUE    = RGB("3C5488"); RED = RGB("E64B35"); GREEN = RGB("00A087"); ORANGE = RGB("E18727")
GREEN_D = RGB("00795F"); RED_D = RGB("C0392B")
KH_F    = RGB("FEF9F0"); ACC_F = RGB("EFF8F5"); REJ_F = RGB("FDF1EF")
ARROW   = "55606E"
EA = "Microsoft YaHei"; LA = "Times New Roman"

X_MIN, X_MAX = -0.75, 21.35
Y_MIN, Y_MAX = -1.40, 6.75
MARGIN = 0.35; SLIDE_W = 13.333
S = (SLIDE_W - 2*MARGIN)/(X_MAX - X_MIN)
SLIDE_H = MARGIN*2 + (Y_MAX - Y_MIN)*S
def X(xc): return MARGIN + (xc - X_MIN)*S
def Y(yc): return MARGIN + (Y_MAX - yc)*S

prs = Presentation()
prs.slide_width = Inches(SLIDE_W); prs.slide_height = Inches(SLIDE_H)
slide = prs.slides.add_slide(prs.slide_layouts[6])

# ---------- run 级富文本 ----------
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

# ---------- 图元 ----------
def panel(cx, cy, w, h, radius=0.04):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(X(cx-w/2)), Inches(Y(cy+h/2)), Inches(w*S), Inches(h*S))
    try: shp.adjustments[0] = radius
    except Exception: pass
    shp.fill.solid(); shp.fill.fore_color.rgb = PANEL
    shp.line.fill.background(); shp.shadow.inherit = False
    return shp

def stage_header(x, y, color, txt, size=10):
    sq = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
            Inches(X(x)-0.05), Inches(Y(y)-0.05), Inches(0.10), Inches(0.10))
    sq.fill.solid(); sq.fill.fore_color.rgb = color
    sq.line.fill.background(); sq.shadow.inherit = False
    tb = slide.shapes.add_textbox(Inches(X(x)+0.09), Inches(Y(y)-0.15),
                                  Inches(5.0), Inches(0.30))
    tf = tb.text_frame; tf.word_wrap = False; tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(0.0)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    emit(p, [bcn(txt, HDR)], size)
    return tb

def box(cx, cy, w, h, line=STROKEN, fill=None, lw=0.9, radius=0.08,
        title=None, sub_=None, title_size=9.5, sub_size=7.5,
        title_bold=False, sub_color=None):
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
    first = True
    if title is not None:
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        emit(p, parts(title), title_size, bold=title_bold); first = False
    if sub_ is not None:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        emit(p, parts(sub_), sub_size, color=(sub_color or SUBT))
    return shp

def text(cx, cy, lines, size=8, color=TXT, bold=False, w=3.0, left=False):
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

def seg(x1, y1, x2, y2, color=ARROW, dashed=False, arrow=False, width=0.9, head='med'):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Inches(X(x1)), Inches(Y(y1)), Inches(X(x2)), Inches(Y(y2)))
    c.shadow.inherit = False
    style_line(c, color, width, dashed=dashed, arrow=arrow, head=head); return c

def path(pts, color=ARROW, dashed=False, width=0.9):
    for i in range(len(pts)-1):
        (x1,y1),(x2,y2) = pts[i], pts[i+1]
        seg(x1,y1,x2,y2, color=color, dashed=dashed, arrow=(i==len(pts)-2))

def dot(x, y, r=0.05):
    d = slide.shapes.add_shape(MSO_SHAPE.OVAL,
            Inches(X(x)-r/2), Inches(Y(y)-r/2), Inches(r), Inches(r))
    d.fill.solid(); d.fill.fore_color.rgb = RGB(ARROW)
    d.line.fill.background(); d.shadow.inherit = False; return d

# ==========================================================================
yL, yT, yR, yS = 3.20, 1.75, 0.30, 5.42
xbus = 10.95
VMAP = {yL:"L", yT:"T", yR:"R"}

# ---- 1. 面板(最底层) ----
panel( 5.075, 1.56, 11.25, 5.72)     # A 共享特征提取
panel(15.10, 5.69,  7.10, 1.82)      # B 阶段一
panel(15.05, 1.585, 6.70, 4.87)      # C 阶段二
stage_header(-0.18, 4.10, STROKEN, "共享特征提取")
stage_header(11.85, 6.28, BLUE, "阶段一 · 闭集分类")
stage_header(11.95, -0.55, RED, "阶段二 · 开集判决（逐类 × 逐视角深度检查）")

# ---- 2. 训练期丢视角: 浅色虚线组(语义=训练期行为) ----
grp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(X(0.85-1.20)), Inches(Y(1.75+2.12)), Inches(2.40*S), Inches(4.24*S))
try: grp.adjustments[0] = 0.05
except Exception: pass
grp.fill.background(); grp.shadow.inherit = False
style_line(grp, "B7BDC9", 0.8, dashed=True)

# ---- 3. 方框 ----
for y in (yL, yT, yR):
    v = VMAP[y]
    box(0.85, y, 2.05, 0.92, title=[cn("左" if v=="L" else ("上" if v=="T" else "右")),
        cn("雷达 "), it(v)], sub_="CIR 90×189")
# 归一化
box(3.30, yT, 2.0, 4.05)
text(3.30, 2.85, ["振幅归一化"], size=10)
text(3.30, 1.55, [[up("|"), it("x"), up("|"), cn(" 后逐样本、")],
     "逐视角除以标准差", "(abs_std)"], size=7.5, color=SUBT, w=1.95)
# 编码器容器
c = box(6.55, yT, 3.55, 4.05, line=ENCLN, fill=ENCF, lw=0.6, radius=0.05)
text(6.55, 3.50, [[bcn("共享双路径编码器", BLUE)]], size=10, w=3.3)
box(6.55, 2.48, 3.15, 0.88, title="时间路径 Transformer", sub_="90 个时间 token × 4 层")
box(6.55, 1.18, 3.15, 0.88, title="距离路径 Transformer", sub_="189 个距离 token × 4 层")
text(6.55, 0.12, ["三视角权重共享 + 视角嵌入", "两路 token 展平拼接"], size=7.5, color=SUBT, w=3.4)
# 特征码
for y in (yL, yT, yR):
    box(9.80, y, 1.58, 0.72, title=[it("f"), sub(VMAP[y]),
        dict(t="  (8928 维)", it=False, base='n', cn=True, col=SUBT)], title_size=9)
# 阶段一
box(12.80, yS, 2.05, 0.88, title="特征聚合")
box(15.25, yS, 2.05, 0.88, title="闭集分类器", sub_="交叉熵训练")
box(17.60, yS, 1.62, 0.88, line=ORANGE, fill=KH_F, lw=1.1, title=[cn("预测类 "), it(KH)])
# 阶段二 检查器
for y in (yL, yT, yR):
    v = VMAP[y]
    box(13.10, y, 2.40, 0.94, line=RED, lw=1.0,
        title=[cn("检查器 "), it("η"), sub(KH), sup("("+v+")")],
        sub_=[it("d"), sub(v), up(" = ‖"), it("l"), sub(v), up(" − "),
              it("φ"), sub(KH), sup("("+v+")"), up("‖")],
        title_size=9.5, sub_size=8, sub_color=TXT)
box(16.60, yT, 3.1, 1.24, line=RED, lw=1.0,
    title="多视角聚合（异常分数）",
    sub_=[it("s"), up(" = "), it("d̄"), up("/"), it("σ̄"), up(" + "), it("β"),
          up("·var("), it("d"), sub("v"), up(")/"), it("σ̄"), sup("2")],
    title_size=9, sub_size=8.5, sub_color=TXT)
# 判决输出
box(20.10, 2.40, 2.02, 0.92, line=GREEN, fill=ACC_F, lw=1.1,
    title=[it("s"), up(" ≤ "), it("τ"), cn("："), bcn("接受", GREEN_D)],
    sub_=[cn("输出已知类 "), it(KH)], title_size=9.5)
box(20.10, 0.78, 2.02, 0.92, line=RED, fill=REJ_F, lw=1.1,
    title=[it("s"), up(" > "), it("τ"), cn("："), bcn("拒识", RED_D)],
    sub_="未知手势", title_size=9.5)

# ---- 4. 箭头(浮在方框之上) ----
for y in (yL, yT, yR):
    path([(1.875,y),(2.30,y)]); path([(4.30,y),(4.775,y)])
    path([(8.325,y),(9.01,y)]); path([(10.59,y),(11.90,y)])
seg(xbus, yR, xbus, yS)
path([(xbus,yS),(11.775,yS)]); path([(13.825,yS),(14.225,yS)]); path([(16.275,yS),(16.79,yS)])
# 橙色: 预测类 -> 阶段二面板
path([(17.60,4.98),(17.60,4.40),(13.10,4.40),(13.10,4.02)], color="E18727", dashed=True, width=1.0)
text(15.35, 4.62, [[cn("按 "), it(KH), cn(" 选对应类的检查器组")]], size=8, color=ORANGE, w=3.6)
# 阶段二汇聚
path([(14.30,yL),(14.62,yL),(14.62,2.07),(15.05,2.07)])
path([(14.30,yT),(15.05,yT)])
path([(14.30,yR),(14.62,yR),(14.62,1.43),(15.05,1.43)])
# 判决分叉(面板外)
seg(18.15, yT, 18.62, yT)
path([(18.62,yT),(18.62,2.40),(19.09,2.40)]); path([(18.62,yT),(18.62,0.78),(19.09,0.78)])
for y in (yL, yT, yR): dot(xbus, y)

# ---- 5. 注释 ----
text(0.90, -0.95, ["训练期随机丢视角",
     [it("p"), sub("drop", italic=False), cn("=0.5，保留 1–2 路")]], size=7.5, color=SUBT, w=3.0)

prs.save("fig1_mvosr_v2.pptx")
print("saved fig1_mvosr_v2.pptx  %.2fx%.2f in" % (SLIDE_W, SLIDE_H))
