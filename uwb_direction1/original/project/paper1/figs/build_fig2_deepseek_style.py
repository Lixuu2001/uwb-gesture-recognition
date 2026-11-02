# -*- coding: utf-8 -*-
"""按 DeepSeek-V3 Fig.2(MLA + DeepSeekMoE)的视觉语言重绘论文一图2。

参考稿视觉语言(配色由 arXiv:2412.19437 第 7 页逐像素采样得到):
  蓝色虚线圆角容器 + 容器外左上角粗体标题 / 米黄模块框(金色描边) /
  隐状态画成"圆角框内一排小圆" / 多份对象用错位叠层卡片 /
  ⊕ 圆圈运算符 / 细黑箭头 / 右上角图例方块。

三个变体共用同一行栅格,绿色(跨视角耦合)模块出现的最低位置即融合深度。
"""
import sys, io
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'

# ---- 由 DeepSeek-V3 Fig.2 采样得到的配色 ----------------------------------
CREAM   = RGBColor(0xFF, 0xF2, 0xCC)   # 模块框填充
GOLD    = RGBColor(0xFF, 0xD9, 0x66)   # 模块框描边
DBLUE   = RGBColor(0x2E, 0x75, 0xB6)   # 虚线容器 / 引线
LBLUE   = RGBColor(0xDE, 0xEB, 0xF7)   # 分类块 A
LGREEN  = RGBColor(0xC5, 0xE0, 0xB4)   # 分类块 B(本文用作跨视角耦合)
INK     = RGBColor(0x10, 0x10, 0x10)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
GREY    = RGBColor(0x59, 0x59, 0x59)

CN, MATH = 'Microsoft YaHei', 'Times New Roman'
F_TITLE, F_BLK, F_SM, F_TINY = 11.5, 10.0, 8.5, 7.5

CW, CH = 10.20, 5.62
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(CW), Inches(CH)
sld = prs.slides.add_slide(prs.slide_layouts[6])
S = sld.shapes

ML, PANW, GAP = 0.16, 3.16, 0.18
CT, CB = 0.70, 5.32                     # 容器上下缘
INX, INW = 0.14, 2.88                   # 容器内缩进与内宽
SW, SG = 0.88, 0.12                     # 单视角列宽与列距

R_S     = (0.86, 0.26)
R_PLUS  = (1.23, 0.22)
R_D     = (1.58, 0.24)
R_INSP  = (1.94, 0.58)
R_SWAP  = (2.64, 0.38)
R_F     = (3.30, 0.26)
R_ENC   = (3.68, 1.04)
R_IN    = (4.84, 0.28)


def _rad(sh, r=0.05):
    mn = min(sh.width, sh.height) / 914400.0
    try:
        sh.adjustments[0] = max(0.0, min(0.5, r / mn))
    except Exception:
        pass


def box(x, y, w, h, fill, line=INK, lw=0.75, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        r=0.05, dash=None):
    sh = S.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        _rad(sh, r)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(lw)
        if dash:
            ln = sh.line._get_or_add_ln()
            ln.append(parse_xml('<a:prstDash xmlns:a="%s" val="%s"/>' % (A, dash)))
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(9144)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def txt(host, lines, size=F_BLK, color=INK, align=PP_ALIGN.CENTER, bold=False):
    tf = host.text_frame if hasattr(host, 'text_frame') else host
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if isinstance(ln, str):
            ln = [ln]
        ln = [(x, size, color, CN, bold) if isinstance(x, str) else x for x in ln]
        for run in ln:
            t, sz, col, fnt, bd = run[:5]
            base = run[5] if len(run) > 5 else 0
            r = p.add_run(); r.text = t
            r.font.size = Pt(sz); r.font.bold = bd
            r.font.color.rgb = col; r.font.name = fnt
            rPr = r._r.get_or_add_rPr()
            ea = rPr.find(qn('a:ea'))
            if ea is None:
                ea = parse_xml('<a:ea xmlns:a="%s"/>' % A); rPr.append(ea)
            ea.set('typeface', CN)
            if base:
                rPr.set('baseline', str(int(base * 1000)))


def mt(t, sz=F_BLK, bd=False, col=INK):
    return (t, sz, col, MATH, bd, 0)


def sb(t, sz=F_BLK, bd=False, col=INK):
    return (t, sz * 0.80, col, MATH, bd, -25)


def zh(t, sz=F_BLK, bd=False, col=INK):
    return (t, sz, col, CN, bd, 0)


def label(x, y, w, h, anchor=MSO_ANCHOR.MIDDLE):
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tb


def module(x, y, w, h, fill=CREAM, line=GOLD, lw=1.0, layers=1, dx=0.042, dy=0.042):
    """米黄模块框;layers>1 时画成错位叠层卡片(对应 ×N)"""
    for k in range(layers - 1, 0, -1):
        box(x + k * dx, y + k * dy, w, h, fill, line, lw)
    return box(x, y, w, h, fill, line, lw)


def hidden(x, y, w, h, n=6, d=0.072):
    """隐状态:圆角白框内一排小圆"""
    b = box(x, y, w, h, WHITE, INK, 0.75)
    tot = n * d + (n - 1) * d * 0.42
    for i in range(n):
        o = S.add_shape(MSO_SHAPE.OVAL,
                        Inches(x + w / 2 - tot / 2 + i * d * 1.42),
                        Inches(y + h / 2 - d / 2), Inches(d), Inches(d))
        o.fill.background()
        o.line.color.rgb = INK; o.line.width = Pt(0.6)
        o.shadow.inherit = False
    return b


def oplus(cx, cy, d=0.22):
    o = S.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2),
                    Inches(d), Inches(d))
    o.fill.solid(); o.fill.fore_color.rgb = WHITE
    o.line.color.rgb = INK; o.line.width = Pt(0.9); o.shadow.inherit = False
    tf = o.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    txt(o, [[mt('+', 11.0, True)]])
    return o


def up(x, y_from, y_to, color=INK, lw=0.9, dash=None):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y_from),
                        Inches(x), Inches(y_to))
    c.line.color.rgb = color; c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    if dash:
        ln.append(parse_xml('<a:prstDash xmlns:a="%s" val="%s"/>' % (A, dash)))
    ln.append(parse_xml('<a:tailEnd xmlns:a="%s" type="triangle" w="sm" len="sm"/>' % A))
    return c


def diag(x1, y1, x2, y2, color=INK, lw=0.9):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                        Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    ln.append(parse_xml('<a:tailEnd xmlns:a="%s" type="triangle" w="sm" len="sm"/>' % A))


# ===========================================================================
def panel(px, title, mode):
    """mode: 'dec' | 'feat' | 'bn'"""
    txt(label(px + 0.02, CT - 0.30, PANW, 0.24),
        [title], F_TITLE, INK, align=PP_ALIGN.LEFT, bold=True)
    box(px, CT, PANW, CB - CT, None, DBLUE, 1.25, r=0.10, dash='dash')

    ix = px + INX
    sx = [ix + i * (SW + SG) for i in range(3)]
    sc = [x + SW / 2 for x in sx]
    V = ['L', 'T', 'R']

    # 输入
    y, h = R_IN
    for i in range(3):
        hidden(sx[i], y, SW, h, n=6)
        txt(label(sx[i], y + h + 0.02, SW, 0.18),
            [[zh('视角 ', F_TINY), mt(V[i], F_TINY, True)]], F_TINY)

    # 共享编码器(×4 叠层)
    ey, eh = R_ENC
    for i in range(3):
        up(sc[i], y, ey + eh - 0.09)      # 叠层框的主框底边
    module(ix, ey, INW - 0.09, eh - 0.09, CREAM, GOLD, 1.0, layers=3)
    txt(label(ix + 0.06, ey + 0.05, INW - 0.20, 0.18, MSO_ANCHOR.TOP),
        [[zh('共享双路径编码器', F_SM, True), mt('  ×4', F_SM, True)]],
        F_SM, align=PP_ALIGN.LEFT)
    bx, bw = ix + 0.09, INW - 0.27
    if mode == 'bn':
        bh, bg, by = 0.23, 0.045, ey + 0.26
        items = [('自注意力', CREAM), ('瓶颈 token 交换（跨视角）', LGREEN), ('前馈', CREAM)]
    else:
        bh, bg, by = 0.27, 0.07, ey + 0.30
        items = [('自注意力', CREAM), ('前馈', CREAM)]
    for j, (lab, col) in enumerate(items):
        b = box(bx, by + j * (bh + bg), bw, bh, col, GOLD if col is CREAM else INK, 0.75)
        txt(b, [lab], F_SM)

    # 特征码
    fy, fh = R_F
    for i in range(3):
        up(sc[i], ey, fy + fh)
        hidden(sx[i], fy, SW, fh, n=6)
        # 标签靠左放,让居中的箭头从旁边穿过,避免压字
        txt(label(sx[i] + 0.02, fy - 0.21, 0.38, 0.18),
            [[mt('f', F_TINY, True), sb(V[i], F_TINY, True)]],
            F_TINY, align=PP_ALIGN.LEFT)

    # 交换带:仅特征级在此耦合
    wy, wh = R_SWAP
    iy, ih = R_INSP
    if mode == 'feat':
        for i in range(3):
            up(sc[i], fy, wy + wh)
        b = box(ix, wy, INW, wh, LGREEN, INK, 0.75)
        txt(b, [['跨视角注意力（Q/K/V 互为键值）']], F_SM)
        for i in range(3):
            up(sc[i], wy, iy + ih)
    else:
        for i in range(3):
            up(sc[i], fy, iy + ih)

    # 逐视角检查器(逐类叠层)
    box(ix, iy, INW, ih, None, DBLUE, 0.9, r=0.06, dash='sysDash')
    txt(label(ix + 0.06, iy + 0.03, INW - 0.12, 0.17, MSO_ANCHOR.TOP),
        [['逐视角深度检查器（逐类 × 逐视角）']], F_TINY, DBLUE,
        align=PP_ALIGN.LEFT, bold=True)
    for i in range(3):
        b = module(sx[i] + 0.03, iy + 0.24, SW - 0.13, 0.22, CREAM, GOLD, 0.75,
                   layers=3, dx=0.030, dy=0.030)
        txt(b, [[mt('η', F_TINY), sb(V[i], F_TINY)]], F_TINY)

    # 逐视角距离
    dy_, dh = R_D
    for i in range(3):
        up(sc[i], iy, dy_ + dh)
        b = box(sx[i] + 0.22, dy_, SW - 0.44, dh, WHITE, INK, 0.75, r=0.06)
        txt(b, [[mt('d', F_SM, True), sb(V[i], F_SM, True)]], F_SM)

    # ⊕ 决策级聚合
    pcy = R_PLUS[0] + R_PLUS[1] / 2
    for i in range(3):
        if i == 1:
            up(sc[i], dy_, R_PLUS[0] + R_PLUS[1])
        else:
            diag(sc[i], dy_, sc[1] + (0.10 if i == 2 else -0.10), pcy + 0.06)
    oplus(sc[1], pcy)
    txt(label(sc[1] + 0.16, pcy - 0.11, 1.60, 0.22),
        [[zh('决策级聚合', F_TINY, True, DBLUE)]], F_TINY, DBLUE, align=PP_ALIGN.LEFT)

    # 异常分数
    sy, sh_ = R_S
    up(sc[1], R_PLUS[0], sy + sh_)
    b = box(sc[1] - 0.62, sy, 1.24, sh_, LBLUE, INK, 0.75)
    txt(b, [[zh('异常分数 ', F_SM, True), mt('s', F_SM, True)]], F_SM)


panel(ML,                        '(a) 决策级聚合（本文）', 'dec')
panel(ML + PANW + GAP,           '(b) 特征级融合', 'feat')
panel(ML + 2 * (PANW + GAP),     '(c) 瓶颈中层融合（MBT 式）', 'bn')

# ---- 右上角图例 -----------------------------------------------------------
lx, ly = CW - 3.30, 0.10
for i, (col, lab, line) in enumerate([(CREAM, '视角内独立计算', GOLD),
                                      (LGREEN, '跨视角耦合模块', INK)]):
    box(lx + i * 1.66, ly, 0.24, 0.17, col, line, 0.75, r=0.03)
    txt(label(lx + i * 1.66 + 0.30, ly - 0.02, 1.34, 0.20),
        [lab], F_TINY, INK, align=PP_ALIGN.LEFT)

OUT = r'D:\uwb_gestures\paper1\figs\fig2_fusion_deepseek_style.pptx'
prs.save(OUT)
print('saved', OUT, '  canvas %.2f x %.2f in' % (CW, CH))
