# -*- coding: utf-8 -*-
"""按 CoRe-Sleep (IEEE JBHI) Fig.1(b) 的视觉语言重绘论文一图2:3 种跨视角融合策略对比。

参考稿视觉语言(配色由原图逐区域采样得到):
  彩色圆角大容器 + 左缘竖排标签 / 内嵌白色圆角框(左上角 ×N) /
  功能子块分色(自注意力 #E8FFC9、前馈 #D6D6F5、跨视角耦合 #FCD2D0) /
  token 画成小圆点堆 / 细黑箭头 / 面板下方 (a)(b)(c) 小标题。

三个面板共用同一套行栅格,因此"耦合发生在第几行"可以横向对读:
瓶颈中层在编码器内部、特征级在编码器之后、决策级只在最顶端,
红色块出现的最低位置即融合深度。
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

# ---- 由 CoRe-Sleep Fig.1(b) 采样得到的配色 --------------------------------
BOX_TAN    = RGBColor(0xF0, 0xE9, 0xCF)
BOX_PURPLE = RGBColor(0xDB, 0xD5, 0xF2)
BOX_BLUE   = RGBColor(0xDC, 0xE7, 0xF2)
SELFATT    = RGBColor(0xE8, 0xFF, 0xC9)   # 自注意力
FFN        = RGBColor(0xD6, 0xD6, 0xF5)   # 前馈
COUPLE     = RGBColor(0xFC, 0xD2, 0xD0)   # 跨视角耦合(红)
TOK_L      = RGBColor(0xE6, 0xF5, 0x82)
TOK_T      = RGBColor(0xDA, 0xE8, 0xFC)
TOK_R      = RGBColor(0xFF, 0xDC, 0xC2)
TOK_BN     = RGBColor(0x9B, 0x8E, 0xC4)   # 瓶颈 token
INK        = RGBColor(0x1A, 0x1A, 0x1A)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
GREY       = RGBColor(0x8C, 0x8C, 0x8C)

CN, MATH = 'Microsoft YaHei', 'Times New Roman'
F_CAP, F_TITLE, F_BLK, F_SM = 12.0, 10.5, 10.0, 8.5

CW, CH = 10.20, 5.60
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(CW), Inches(CH)
sld = prs.slides.add_slide(prs.slide_layouts[6])
S = sld.shapes

# ---- 行栅格(三面板共用,自上而下;数据流自下而上) --------------------------
PAN_T, PAN_B = 0.10, 4.95
R_AGG   = (0.28, 0.40)
R_DIST  = (0.82, 0.26)
R_INSP  = (1.22, 0.56)
R_SWAP  = (1.94, 0.40)
R_FEAT  = (2.50, 0.28)
R_ENC   = (2.94, 1.30)
R_IN    = (4.40, 0.42)
CAPY    = 5.05

PANW, GAP, ML = 3.16, 0.18, 0.16
LBLW = 0.30
INX, INW = 0.34, 2.70
SW, SG = 0.82, 0.12


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


def mt(t, sz=F_BLK, bd=False):
    """正体数学符号"""
    return (t, sz, INK, MATH, bd, 0)


def sb(t, sz=F_BLK, bd=False):
    """下标(用 run baseline 近似)"""
    return (t, sz * 0.82, INK, MATH, bd, -25)


def zh(t, sz=F_BLK, bd=False):
    return (t, sz, INK, CN, bd, 0)


def label(x, y, w, h, anchor=MSO_ANCHOR.MIDDLE):
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tb


def vlabel(x, y, w, h, text, color=INK):
    """左缘竖排标签(自下而上阅读)"""
    tb = label(x, y, w, h)
    tb.text_frame._txBody.find(qn('a:bodyPr')).set('vert', 'vert270')
    txt(tb, [text], F_TITLE, color, bold=True)
    return tb


def up(x, y_from, y_to, color=INK, lw=0.9, dash=None):
    """自下而上的细箭头"""
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y_from),
                        Inches(x), Inches(y_to))
    c.line.color.rgb = color; c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    if dash:
        ln.append(parse_xml('<a:prstDash xmlns:a="%s" val="%s"/>' % (A, dash)))
    ln.append(parse_xml('<a:tailEnd xmlns:a="%s" type="triangle" w="sm" len="sm"/>' % A))
    return c


def stack_box(x, y, w, h, fill, line=INK, lw=0.75, layers=3, dx=0.045, dy=0.045,
              r=0.05):
    """错位叠层的立体框:先画后面的副本,再画最前面的主框(pptx 按插入顺序叠放)"""
    for k in range(layers - 1, 0, -1):
        box(x + k * dx, y + k * dy, w, h, fill, line, lw, r=r)
    return box(x, y, w, h, fill, line, lw, r=r)


def token_pile(cx, cy, fill, n=5, rows=3, ew=0.135, eh=0.115, ov=0.40,
               rdx=-0.030, rdy=0.032):
    """错位堆叠的 token 椭圆堆:自后向前逐排画,排内椭圆相互交叠形成纵深"""
    step = ew * (1 - ov)
    tot = ew + (n - 1) * step
    for rr in range(rows - 1, -1, -1):
        ox, oy = rr * rdx, rr * rdy
        for i in range(n - 1, -1, -1):
            o = S.add_shape(MSO_SHAPE.OVAL,
                            Inches(cx - tot / 2 + i * step + ox),
                            Inches(cy - eh / 2 + oy), Inches(ew), Inches(eh))
            o.fill.solid(); o.fill.fore_color.rgb = fill
            o.line.color.rgb = INK; o.line.width = Pt(0.5)
            o.shadow.inherit = False


def panel(px, cont_fill, name, depth_text, mode):
    """mode: 'dec' 决策级 | 'feat' 特征级 | 'bn' 瓶颈中层"""
    box(px, PAN_T, PANW, PAN_B - PAN_T, cont_fill, INK, 0.75, r=0.09)
    vlabel(px + 0.03, PAN_T + 0.10, LBLW, PAN_B - PAN_T - 0.20, depth_text)

    ix = px + INX
    sx = [ix + i * (SW + SG) for i in range(3)]          # 三路视角的左端
    sc = [x + SW / 2 for x in sx]                        # 三路视角的中心
    views = [('L', TOK_L), ('T', TOK_T), ('R', TOK_R)]

    # --- 输入 token ---
    y, h = R_IN
    for i, (v, col) in enumerate(views):
        token_pile(sc[i] + 0.02, y + 0.13, col)
        tb = label(sx[i], y + 0.26, SW, 0.18)
        txt(tb, [[zh('视角 ', F_SM), mt(v, F_SM, True)]], F_SM)

    # --- 编码器 ---
    ey, eh = R_ENC
    for i in range(3):
        up(sc[i], y, ey + eh - 0.10)      # 叠层框的主框底边
    stack_box(ix, ey, INW - 0.10, eh - 0.10, WHITE, INK, 0.75, layers=3)
    tb = label(ix + 0.05, ey + 0.04, INW - 0.20, 0.20, MSO_ANCHOR.TOP)
    txt(tb, [[('×4', F_SM, INK, MATH, True),
              ('   共享双路径编码器（权重共享）', F_SM, INK, CN, True)]],
        F_SM, align=PP_ALIGN.LEFT)
    bw = INW - 0.30
    if mode == 'bn':
        bh, bg, by = 0.26, 0.05, ey + 0.28
        for j, (lab, col) in enumerate([('自注意力', SELFATT),
                                        ('瓶颈 token 交换（4 个共享 token）', COUPLE),
                                        ('前馈', FFN)]):
            b = box(ix + 0.10, by + j * (bh + bg), bw, bh, col, INK, 0.6)
            txt(b, [lab], F_BLK)
    else:
        bh, bg, by = 0.32, 0.09, ey + 0.36
        for j, (lab, col) in enumerate([('自注意力', SELFATT), ('前馈', FFN)]):
            b = box(ix + 0.10, by + j * (bh + bg), bw, bh, col, INK, 0.6)
            txt(b, [lab], F_BLK)

    # --- 特征码 ---
    fy, fh = R_FEAT
    for i in range(3):
        up(sc[i], ey, fy + fh)
        b = box(sx[i], fy, SW, fh, WHITE, INK, 0.6)
        txt(b, [[mt('f', F_BLK, True), sb(views[i][0], F_BLK, True),
                 zh('  8928 维', F_SM)]], F_SM)

    # --- 交换带:仅特征级在此耦合,其余三路直穿 ---
    wy, wh = R_SWAP
    iy, ih = R_INSP
    if mode == 'feat':
        for i in range(3):
            up(sc[i], fy, wy + wh)
        b = box(ix, wy, INW, wh, COUPLE, INK, 0.75)
        txt(b, [['跨视角注意力（Q/K/V 互为键值）']], F_BLK)
        for i in range(3):
            up(sc[i], wy, iy + ih)
    else:
        for i in range(3):
            up(sc[i], fy, iy + ih)

    # --- 逐视角检查器 ---
    box(ix, iy, INW, ih, WHITE, INK, 0.75)
    tb = label(ix + 0.05, iy + 0.04, INW - 0.10, 0.18, MSO_ANCHOR.TOP)
    txt(tb, [['逐视角深度检查器（逐类 × 逐视角）']], F_SM, align=PP_ALIGN.LEFT, bold=True)
    for i in range(3):
        b = stack_box(sx[i] + 0.02, iy + 0.24, SW - 0.10, 0.22, WHITE, GREY, 0.6,
                      layers=3, dx=0.030, dy=0.030)
        txt(b, [[mt('η', F_SM), sb(views[i][0], F_SM),
                 zh('  8928→128', F_SM)]], F_SM)

    # --- 逐视角距离 ---
    dy, dh = R_DIST
    for i in range(3):
        up(sc[i], iy, dy + dh)
        b = box(sx[i] + 0.16, dy, SW - 0.32, dh, WHITE, INK, 0.6, r=0.06)
        txt(b, [[mt('d', F_BLK, True), sb(views[i][0], F_BLK, True)]], F_BLK)

    # --- 决策级聚合(三者都有,故均为耦合红) ---
    ay, ah = R_AGG
    for i in range(3):
        up(sc[i], dy, ay + ah)
    b = box(ix + 0.22, ay, INW - 0.44, ah, COUPLE, INK, 0.75)
    txt(b, [[zh('决策级聚合', F_BLK, True)],
            [mt('s', F_SM, True), mt(' = mean', F_SM), sb('v', F_SM),
             mt('(d', F_SM), sb('v', F_SM), mt(')/σ̄ + β·var', F_SM), sb('v', F_SM),
             mt('(d', F_SM), sb('v', F_SM), mt(')/σ̄', F_SM),
             ('2', F_SM * 0.82, INK, MATH, False, 33)]], F_SM)

    txt(label(px, CAPY, PANW, 0.30), [name], F_CAP, INK, bold=True)


panel(ML,                    BOX_BLUE,   '(a) 决策级聚合（本文）',
      '耦合深度：决策级', 'dec')
panel(ML + PANW + GAP,       BOX_TAN,    '(b) 特征级融合',
      '耦合深度：编码器之后', 'feat')
panel(ML + 2 * (PANW + GAP), BOX_PURPLE, '(c) 瓶颈中层融合（MBT 式）',
      '耦合深度：编码器中层', 'bn')

OUT = r'D:\uwb_gestures\paper1\figs\fig2_fusion_coresleep_style.pptx'
prs.save(OUT)
print('saved', OUT, '  canvas %.2f x %.2f in' % (CW, CH))
