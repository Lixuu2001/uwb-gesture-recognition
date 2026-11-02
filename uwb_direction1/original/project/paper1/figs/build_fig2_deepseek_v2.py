# -*- coding: utf-8 -*-
r"""论文一 图2:按 DeepSeek-V2/V3 Fig.2 的视觉语言重绘"跨视角融合层级对比"。

参考稿(arXiv:2405.04434 / 2412.19437 Fig.2)的视觉语法:
  · 左侧一列"主干块",米黄圆角框(金色描边)+ 细黑箭头,自下而上;
  · 蓝色虚线圆角容器把主干里的一个块"拉出去"展开成右侧大图(两条蓝虚线连角);
  · 隐状态画成白色圆角框内一排小圆(带省略号);多份对象用错位叠层卡片;
  · ⊕ 圆圈算子;顶部一行方块图例;标注文字用灰色小字。

本图的信息骨架(与正文 \S融合策略比较 自洽):
  主干 = 三视角共享管线(编码 → 视角嵌入 → 特征码 → 逐视角检查器 → 原型距离
         → 跨视角聚合 → 异常分数 → 阈值判决);
  三个展开面板 = 唯一自变量"跨视角交换插在哪一层":
    (a) 决策级(本文)、(b) 编码器之后、(c) 编码器中层(MBT 式瓶颈);
  绿色 = 发生跨视角耦合的模块,米黄 = 视角内独立计算。
  展开面板在竖直方向的高低,即交换层级在管线中的深浅。

输出: fig2_fusion_deepseek.pptx
"""
import io
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'

# ---- 配色(自参考稿采样) ----------------------------------------------------
CREAM = RGBColor(0xFF, 0xF2, 0xCC)   # 米黄模块:视角内独立计算
GOLD = RGBColor(0xE8, 0xB1, 0x2C)    # 米黄模块描边
GREEN = RGBColor(0xC5, 0xE0, 0xB4)   # 浅绿:跨视角耦合
GREENL = RGBColor(0x6E, 0xA6, 0x4B)  # 绿描边 / 绿箭头
LBLUE = RGBColor(0xDE, 0xEB, 0xF7)   # 浅蓝:输出量
DBLUE = RGBColor(0x2E, 0x75, 0xB6)   # 蓝虚线容器 / 引出线
INK = RGBColor(0x0D, 0x0D, 0x0D)
GREY = RGBColor(0x75, 0x75, 0x75)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SONG, HEI, MATH = 'SimSun', 'SimHei', 'Times New Roman'
F_TIT, F_BLK, F_SM, F_TINY = 11.0, 9.0, 8.0, 7.0

CW, CH = 11.40, 7.15
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(CW), Inches(CH)
sld = prs.slides.add_slide(prs.slide_layouts[6])
S = sld.shapes


# ---- 基础图元 --------------------------------------------------------------
def _rad(sh, r):
    mn = min(sh.width, sh.height) / 914400.0
    try:
        sh.adjustments[0] = max(0.0, min(0.5, r / mn))
    except Exception:
        pass


def box(x, y, w, h, fill, line=INK, lw=0.75, r=0.055, dash=None,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = S.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        _rad(sh, r)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
        if dash:
            sh.line._get_or_add_ln().append(
                parse_xml('<a:prstDash xmlns:a="%s" val="%s"/>' % (A, dash)))
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(18288)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def label(x, y, w, h, anchor=MSO_ANCHOR.MIDDLE):
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tb


def txt(host, lines, size=F_BLK, color=INK, align=PP_ALIGN.CENTER):
    """lines: [str | [run, ...]];run = (文本, 字号, 颜色, 字体, 斜体, 上下标)"""
    tf = host.text_frame if hasattr(host, 'text_frame') else host
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 0.95
        if isinstance(ln, str):
            ln = [(ln, size, color, SONG, False, 0)]
        for run in ln:
            t, sz, col, fnt, it, base = run
            r = p.add_run()
            r.text = t
            r.font.size = Pt(sz)
            r.font.italic = it
            r.font.color.rgb = col
            r.font.name = fnt
            rPr = r._r.get_or_add_rPr()
            ea = rPr.find(qn('a:ea'))
            if ea is None:
                ea = parse_xml('<a:ea xmlns:a="%s"/>' % A)
                rPr.append(ea)
            ea.set('typeface', fnt)
            if base:
                rPr.set('baseline', str(int(base * 1000)))


def zh(t, sz=F_BLK, col=INK):
    return (t, sz, col, SONG, False, 0)


def bz(t, sz=F_BLK, col=INK):          # 黑体(粗)中文
    return (t, sz, col, HEI, False, 0)


def mt(t, sz=F_BLK, col=INK, it=True):  # 数学 / 拉丁
    return (t, sz, col, MATH, it, 0)


def sub(t, sz=F_BLK, col=INK):
    return (t, sz * 0.85, col, MATH, False, -25)


def sup(t, sz=F_BLK, col=INK):
    return (t, sz * 0.85, col, MATH, False, 30)


def gy(t, sz=F_TINY, col=GREY):
    return (t, sz, col, SONG, False, 0)


def arrow(x1, y1, x2, y2, color=INK, lw=0.85, head=True, dash=None):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                        Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    if dash:
        ln.append(parse_xml('<a:prstDash xmlns:a="%s" val="%s"/>' % (A, dash)))
    if head:
        ln.append(parse_xml(
            '<a:tailEnd xmlns:a="%s" type="triangle" w="sm" len="sm"/>' % A))
    return c


def stack(x, y, w, h, fill=CREAM, line=GOLD, lw=0.9, layers=3, d=0.038):
    """错位叠层卡片,表示同一模块的 N 份。"""
    for k in range(layers - 1, 0, -1):
        box(x + k * d, y - k * d, w, h, fill, line, lw)
    return box(x, y, w, h, fill, line, lw)


def hidden(x, y, w, h, nl=3, nr=2, dots=True):
    """隐状态:白色圆角框内一排小圆,中间留省略号"""
    b = box(x, y, w, h, WHITE, INK, 0.7, r=0.05)
    d = min(0.092, h * 0.34)
    pitch = d * 1.62
    gap = d * 1.75 if nr else 0.0
    tot = (nl + nr - 1) * pitch + gap + d
    x0 = x + w / 2 - tot / 2
    cy = y + h / 2
    for i in range(nl + nr):
        cx = x0 + i * pitch + (gap if i >= nl else 0)
        o = S.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(cy - d / 2),
                        Inches(d), Inches(d))
        o.fill.background()
        o.line.color.rgb = INK
        o.line.width = Pt(0.6)
        o.shadow.inherit = False
    if dots:
        e = label(x0 + (nl - 1) * pitch + d, cy - 0.09, gap, 0.18)
        txt(e, [[mt('…', 8.5, INK, False)]], align=PP_ALIGN.CENTER)
    return b


def oplus(cx, cy, d=0.26, fill=GREEN, line=GREENL):
    o = S.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2),
                    Inches(d), Inches(d))
    o.fill.solid()
    o.fill.fore_color.rgb = fill
    o.line.color.rgb = line
    o.line.width = Pt(1.0)
    o.shadow.inherit = False
    tf = o.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    txt(o, [[mt('+', 10.5, INK, False)]])
    return o


# ===========================================================================
# 一、左侧主干:三视角共享管线(自下而上)
# ===========================================================================
BX, BW = 0.20, 1.74           # 虚线容器
BIX, BIW = 0.34, 1.46         # 容器内可用区
BCX = BIX + BIW / 2
BT, BB = 0.60, 6.48

txt(label(BX + 0.02, BT - 0.28, 2.0, 0.24),
    [[bz('逐视角管线 ', F_SM), mt('×3', F_SM, INK, False)]],
    F_SM, align=PP_ALIGN.LEFT)
box(BX, BT, BW, BB - BT, None, DBLUE, 1.3, r=0.09, dash='dash')

# (顶 → 底) 名称, y, h, 填充
CHAIN = [
    ('judge', 0.80, 0.36, CREAM),
    ('score', 1.44, 0.34, LBLUE),
    ('agg', 2.06, 0.54, GREEN),
    ('dist', 2.88, 0.30, WHITE),
    ('insp', 3.46, 0.60, CREAM),
    ('feat', 4.34, 0.30, WHITE),
    ('emb', 4.92, 0.30, CREAM),
    ('enc', 5.50, 0.76, CREAM),
]
Y = {k: (y, h) for k, y, h, _ in CHAIN}

# 输入(容器外)
hidden(BIX + 0.10, 6.62, BIW - 0.20, 0.28, 3, 2)
txt(label(BX, 6.93, BW, 0.20, MSO_ANCHOR.TOP),
    [[zh('三视角 ', F_TINY), mt('CIR', F_TINY, INK, False), zh('  ', F_TINY),
      mt('X', F_TINY), sub('L,T,R', F_TINY)]], F_TINY)

b = stack(BIX, Y['enc'][0], BIW - 0.10, Y['enc'][1], CREAM, GOLD, 0.9, 3)
txt(b, [[bz('共享双路径', F_SM)], [bz('Transformer 编码器', F_SM)],
        [mt('L = 4', F_TINY, INK, False), zh('  层', F_TINY)]], F_SM)

b = box(BIX + 0.06, Y['emb'][0], BIW - 0.12, Y['emb'][1], CREAM, GOLD, 0.9)
txt(b, [[zh('+ 视角嵌入 ', F_TINY), mt('e', F_TINY), sub('v', F_TINY)]], F_TINY)

hidden(BIX + 0.10, Y['feat'][0], BIW - 0.20, Y['feat'][1], 3, 2)
txt(label(BCX + 0.06, Y['feat'][0] - 0.20, 0.80, 0.18),
    [[gy('特征码 '), mt('f', F_TINY, GREY), sub('v', F_TINY, GREY)]],
    F_TINY, GREY, align=PP_ALIGN.LEFT)

b = stack(BIX, Y['insp'][0], BIW - 0.10, Y['insp'][1], CREAM, GOLD, 0.9, 3)
txt(b, [[bz('逐类 × 逐视角', F_SM)], [bz('深度检查器 ', F_SM), mt('η', F_SM)]], F_SM)

b = box(BIX + 0.28, Y['dist'][0], BIW - 0.56, Y['dist'][1], WHITE, INK, 0.75)
txt(b, [[mt('d', F_SM), sub('v', F_SM)]], F_SM)
txt(label(BCX + 0.06, Y['dist'][0] - 0.19, 0.80, 0.17),
    [[gy('原型距离')]], F_TINY, GREY, align=PP_ALIGN.LEFT)

b = box(BIX + 0.04, Y['agg'][0], BIW - 0.08, Y['agg'][1], GREEN, GREENL, 0.9)
txt(b, [[bz('跨视角距离聚合', F_SM)], [zh('式 (2)', F_TINY)]], F_SM)

b = box(BIX + 0.14, Y['score'][0], BIW - 0.28, Y['score'][1], LBLUE, INK, 0.75)
txt(b, [[zh('异常分数 ', F_SM), mt('s', F_SM)]], F_SM)

b = box(BIX + 0.02, Y['judge'][0], BIW - 0.04, Y['judge'][1], CREAM, GOLD, 0.9)
txt(b, [[zh('阈值判决:接受 / 拒识', F_TINY)]], F_TINY)

# 主干竖直箭头(自下而上)
LINK = [(6.62, 'enc'), ('enc', 'emb'), ('emb', 'feat'), ('feat', 'insp'),
        ('insp', 'dist'), ('dist', 'agg'), ('agg', 'score'), ('score', 'judge')]
for a_, b_ in LINK:
    y1 = a_ if isinstance(a_, float) else Y[a_][0]
    y2 = Y[b_][0] + Y[b_][1]
    arrow(BCX, y1, BCX, y2)

# ===========================================================================
# 二、右侧三个展开面板
# ===========================================================================
PX, PW = 2.35, 8.85
PH = 1.72
PANEL = {                       # 面板顶边
    'a': 0.60,
    'b': 2.68,
    'c': 4.76,
}
ATTACH = {'a': 'agg', 'b': 'feat', 'c': 'enc'}


def panel_frame(key, title_runs, tag):
    y0 = PANEL[key]
    ay, ah = Y[ATTACH[key]]
    # 蓝虚线引出线(两条,连主干块的上下缘到面板左侧上下角)
    arrow(BX + BW, ay, PX, y0, DBLUE, 0.9, head=False, dash='dash')
    arrow(BX + BW, ay + ah, PX, y0 + PH, DBLUE, 0.9, head=False, dash='dash')
    box(PX, y0, PW, PH, None, DBLUE, 1.3, r=0.055, dash='dash')
    txt(label(PX + 0.02, y0 - 0.26, 5.0, 0.22), [[title_runs]], F_TIT,
        align=PP_ALIGN.LEFT)
    t = label(PX + PW - 4.2, y0 - 0.25, 4.2, 0.20)
    txt(t, [[gy('跨视角交换层级:', F_SM), (tag, F_SM, GREENL, HEI, False, 0)]],
        F_SM, align=PP_ALIGN.RIGHT)
    return y0


def rows(y0):
    """面板内三行(视角 L/T/R)的行顶 y 与行高"""
    return [y0 + 0.34, y0 + 0.74, y0 + 1.14], 0.28


VIEW = ['L', 'T', 'R']

# ---------------------------------------------------------------- (a) 决策级
y0 = panel_frame('a', bz('(a) 决策级聚合(本文方法)', F_TIT), '决策级 · 最末端')
RY, RH = rows(y0)
cy_mid = y0 + PH / 2 - 0.06

txt(label(PX + 0.12, y0 + 0.08, 1.5, 0.18),
    [[gy('来自主干特征码')]], F_TINY, GREY, align=PP_ALIGN.LEFT)
txt(label(4.00, y0 + 0.08, 2.4, 0.18),
    [[gy('冻结特征上逐类训练,视角间不共享')]], F_TINY, GREY, align=PP_ALIGN.LEFT)
txt(label(7.86, y0 + 0.08, 2.6, 0.18),
    [[gy('唯一的跨视角汇合点')]], F_TINY, GREENL, align=PP_ALIGN.LEFT)

for i, v in enumerate(VIEW):
    y = RY[i]
    txt(label(2.44, y, 0.34, RH), [[mt('f', F_SM), sub(v, F_SM)]], F_SM)
    hidden(2.82, y, 0.94, RH, 3, 2)
    arrow(3.76, y + RH / 2, 4.00, y + RH / 2)
    b = stack(4.00, y, 2.40, RH, CREAM, GOLD, 0.85, 3, 0.030)
    txt(b, [[mt('η', F_TINY), sup('(' + v + ')', F_TINY),
             sub('k', F_TINY), zh('  ', F_TINY),
             mt('8928 → 1024 → 128', F_TINY, INK, False)]], F_TINY)
    arrow(6.48, y + RH / 2, 6.64, y + RH / 2)
    b = box(6.64, y, 0.52, RH, WHITE, INK, 0.75)
    txt(b, [[mt('d', F_SM), sub(v, F_SM)]], F_SM)
    arrow(7.16, y + RH / 2, 7.46, cy_mid + (y + RH / 2 - cy_mid) * 0.10,
          GREENL, 0.85)

oplus(7.58, cy_mid)
arrow(7.71, cy_mid, 7.90, cy_mid, GREENL)
b = box(7.90, cy_mid - 0.35, 2.46, 0.70, GREEN, GREENL, 0.9)
txt(b, [[mt('s', F_SM), mt(' = mean', F_SM, INK, False), sub('v', F_SM),
         mt('(d', F_SM), sub('v', F_SM), mt(') / σ̄', F_SM)],
        [mt('+ β · var', F_SM, INK, False), sub('v', F_SM), mt('(d', F_SM),
         sub('v', F_SM), mt(') / σ̄', F_SM), sup('2', F_SM)]], F_SM)
arrow(10.36, cy_mid, 10.56, cy_mid, GREENL)
o = box(10.56, cy_mid - 0.19, 0.42, 0.38, LBLUE, INK, 0.8, r=0.19,
        shape=MSO_SHAPE.OVAL)
txt(o, [[mt('s', F_SM)]], F_SM)

txt(label(4.00, y0 + PH - 0.26, 3.4, 0.20),
    [[gy('三路特征码全程独立,只有距离在此汇合')]], F_TINY, GREY,
    align=PP_ALIGN.LEFT)

# ---------------------------------------------------------------- (b) 特征级
y0 = panel_frame('b', bz('(b) 特征级融合', F_TIT), '编码器之后')
RY, RH = rows(y0)
cy_mid = y0 + PH / 2 - 0.06

txt(label(4.00, y0 + 0.08, 3.9, 0.18),
    [[gy('在特征码之上叠一层跨视角注意力,再送入检查器')]], F_TINY, GREENL,
    align=PP_ALIGN.LEFT)

for i, v in enumerate(VIEW):
    y = RY[i]
    txt(label(2.44, y, 0.34, RH), [[mt('f', F_SM), sub(v, F_SM)]], F_SM)
    hidden(2.82, y, 0.94, RH, 3, 2)
    arrow(3.76, y + RH / 2, 4.00, y + RH / 2, GREENL)
    b = stack(4.00, y, 1.00, RH, GREEN, GREENL, 0.85, 3, 0.028)
    txt(b, [[mt('Q, K, V ', F_TINY, INK, False), zh('投影', F_TINY)]], F_TINY)
    arrow(5.06, y + RH / 2, 5.24, cy_mid + (y + RH / 2 - cy_mid) * 0.35, GREENL)

b = box(5.24, cy_mid - 0.50, 1.42, 1.00, GREEN, GREENL, 0.9)
txt(b, [[bz('多头', F_SM)], [bz('跨视角注意力', F_SM)]], F_SM)
arrow(6.66, cy_mid, 6.90, cy_mid, GREENL)
b = box(6.90, cy_mid - 0.50, 1.00, 1.00, GREEN, GREENL, 0.9)
txt(b, [[bz('相加与', F_SM)], [bz('归一化', F_SM)]], F_SM)

# 残差:特征码旁路直接汇入"相加与归一化"
ry = y0 + PH - 0.20
arrow(3.29, RY[2] + RH, 3.29, ry, GREENL, 0.8, head=False)
arrow(3.29, ry, 7.40, ry, GREENL, 0.8, head=False)
arrow(7.40, ry, 7.40, cy_mid + 0.50, GREENL, 0.8)
txt(label(4.60, ry - 0.20, 1.4, 0.18), [[gy('残差旁路')]], F_TINY, GREY)

for i, v in enumerate(VIEW):
    y = RY[i]
    arrow(7.90, cy_mid + (y + RH / 2 - cy_mid) * 0.35, 8.02, y + RH / 2, GREENL)
    txt(label(8.02, y, 0.36, RH),
        [[mt('f′', F_SM), sub(v, F_SM)]], F_SM)
    hidden(8.42, y, 0.94, RH, 3, 2)
    arrow(9.36, y + RH / 2, 9.60, y + RH / 2)

b = box(9.60, cy_mid - 0.50, 1.44, 1.00, CREAM, GOLD, 0.9)
txt(b, [[bz('逐视角检查器', F_SM)], [mt('→ d', F_SM), sub('v', F_SM),
        zh(' → ', F_SM), zh('式 (2)', F_SM)], [gy('(其后同 (a))')]], F_SM)

# ---------------------------------------------------------------- (c) 瓶颈中层
y0 = panel_frame('c', bz('(c) 瓶颈中层融合(MBT 式)', F_TIT), '编码器中层')
RY, RH = rows(y0)
cy_mid = y0 + PH / 2 - 0.06

txt(label(2.44, y0 + 0.08, 2.0, 0.18),
    [[gy('前 4−K 层:视角内独立')]], F_TINY, GREY, align=PP_ALIGN.LEFT)
txt(label(4.30, y0 + 0.08, 1.9, 0.18),
    [[gy('唯一跨视角通道')]], F_TINY, GREENL, align=PP_ALIGN.CENTER)
txt(label(6.20, y0 + 0.08, 1.8, 0.18),
    [[gy('后 K 层:经瓶颈互读写')]], F_TINY, GREENL, align=PP_ALIGN.LEFT)

for i, v in enumerate(VIEW):
    y = RY[i]
    txt(label(2.44, y, 0.34, RH), [[mt('T', F_SM), sub(v, F_SM)]], F_SM)
    b = stack(2.82, y, 1.24, RH, CREAM, GOLD, 0.85, 3, 0.028)
    txt(b, [[zh('独立编码层 ', F_TINY), mt('×(4−K)', F_TINY, INK, False)]], F_TINY)
    arrow(4.12, y + RH / 2, 4.58, cy_mid + (y + RH / 2 - cy_mid) * 0.16, GREENL)
    arrow(5.72, cy_mid + (y + RH / 2 - cy_mid) * 0.16, 6.20, y + RH / 2, GREENL)
    b = box(6.20, y, 1.55, RH, GREEN, GREENL, 0.85)
    txt(b, [[zh('瓶颈融合层 ', F_TINY), mt('×K', F_TINY, INK, False)]], F_TINY)
    arrow(7.75, y + RH / 2, 8.02, y + RH / 2, GREENL)
    txt(label(8.02, y, 0.36, RH), [[mt('f′', F_SM), sub(v, F_SM)]], F_SM)
    hidden(8.42, y, 0.94, RH, 3, 2)
    arrow(9.36, y + RH / 2, 9.60, y + RH / 2)

for j in range(4):
    bx = 4.66 + j * 0.27
    b = box(bx, cy_mid - 0.16, 0.20, 0.32, GREEN, GREENL, 0.9, r=0.04)
    if j == 0:
        txt(b, [[mt('B', F_TINY)]], F_TINY)
txt(label(4.30, cy_mid + 0.22, 1.9, 0.18),
    [[zh('共享瓶颈 ', F_TINY), mt('token ×4', F_TINY, INK, False)]], F_TINY)
txt(label(4.30, cy_mid + 0.40, 1.9, 0.18),
    [[gy('视角间取平均')]], F_TINY, GREY)

b = box(9.60, cy_mid - 0.50, 1.44, 1.00, CREAM, GOLD, 0.9)
txt(b, [[bz('逐视角检查器', F_SM)], [mt('→ d', F_SM), sub('v', F_SM),
        zh(' → ', F_SM), zh('式 (2)', F_SM)], [gy('(其后同 (a))')]], F_SM)

txt(label(2.44, y0 + PH - 0.24, 4.2, 0.18),
    [[gy('总深度保持 4 层;时间路径与距离路径同时经瓶颈交换')]], F_TINY, GREY,
    align=PP_ALIGN.LEFT)

# ===========================================================================
# 三、顶部图例
# ===========================================================================
LEG = [(CREAM, GOLD, '视角内独立计算'), (GREEN, GREENL, '跨视角交换'),
       (WHITE, INK, '隐状态 / 特征码'), (LBLUE, INK, '开集判决量')]
SWW, TGAP, IGAP, LGH = 0.24, 0.06, 0.20, 0.18


def _tw(t):
    return sum(0.100 if ord(ch) > 0x2000 else 0.046 for ch in t) + 0.03


ly = 0.13
lx = 11.20 - sum(SWW + TGAP + _tw(n) for _, _, n in LEG) - IGAP * (len(LEG) - 1)
for fill, line, name in LEG:
    if fill is WHITE:                      # 隐状态图例:框内画三个小圆
        hidden(lx, ly, SWW, LGH, 3, 0, dots=False)
    else:
        box(lx, ly + 0.01, SWW, LGH - 0.02, fill, line, 0.8, r=0.03)
    w = _tw(name)
    txt(label(lx + SWW + TGAP, ly - 0.02, w, LGH + 0.04), [[gy(name)]],
        F_TINY, INK, align=PP_ALIGN.LEFT)
    lx += SWW + TGAP + w + IGAP

OUT = r'D:\uwb_gestures\paper1\figs\fig2_fusion_deepseek.pptx'
prs.save(OUT)
print('saved', OUT, '  canvas %.2f x %.2f in' % (CW, CH))
