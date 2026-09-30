# -*- coding: utf-8 -*-
"""按 VTS-Agent_三图统一风格-5.pptx 的视觉语言重绘论文一图1(MV-OSR 总体架构)。

风格取自参考稿:蓝底白字横幅(圆角矩形 + 方角矩形补平底) / 白底蓝边大块 /
黑色虚线分组(深蓝加粗组标题) / 马卡龙叶子块(0.75pt 灰边) / 块间红色粗下箭头,
全部微软雅黑加粗。画布 10.5x7.5in,置于 \textwidth 时正文字号约 6.7pt。
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

# ---- 参考稿抠出的样式常量 -------------------------------------------------
BLUE      = RGBColor(0x2F, 0x6F, 0xB6)   # 横幅底 / 大块描边
DARKBLUE  = RGBColor(0x23, 0x4F, 0x87)   # 组标题、强调块文字
LIGHTBLUE = RGBColor(0xEA, 0xF2, 0xFB)   # 强调块底
PEACH     = RGBColor(0xF6, 0xE1, 0xCF)
CREAM     = RGBColor(0xF5, 0xF0, 0xD6)
GREEN     = RGBColor(0xDD, 0xEC, 0xCF)
LEAFLINE  = RGBColor(0x55, 0x55, 0x55)   # 叶子块描边
DASHLINE  = RGBColor(0x33, 0x33, 0x33)   # 虚线分组描边
ORANGE    = RGBColor(0xED, 0x7D, 0x31)
RED       = RGBColor(0xD7, 0x26, 0x1E)   # 块间箭头
INK       = RGBColor(0x33, 0x33, 0x33)   # 正文
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

CN, MATH = 'Microsoft YaHei', 'Times New Roman'
F_BANNER, F_GROUP, F_LEAF, F_NOTE = 15.0, 11.5, 10.5, 9.0

CW, CH = 10.5, 7.50
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(CW), Inches(CH)
sld = prs.slides.add_slide(prs.slide_layouts[6])
S = sld.shapes


def _rad(sh, r=0.055):
    mn = min(sh.width, sh.height) / 914400.0
    try:
        sh.adjustments[0] = max(0.0, min(0.5, r / mn))
    except Exception:
        pass


def box(x, y, w, h, fill=None, line=LEAFLINE, lw=0.75, dash=None,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, r=0.055):
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
            for old in ln.findall(qn('a:prstDash')):
                ln.remove(old)
            ln.append(parse_xml('<a:prstDash xmlns:a="http://schemas.openxmlformats.org/'
                                'drawingml/2006/main" val="%s"/>' % dash))
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(18288)
    tf.margin_top = tf.margin_bottom = Emu(9144)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def label(x, y, w, h, anchor=MSO_ANCHOR.MIDDLE):
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tb


def put(host, lines, size=F_LEAF, color=INK, align=PP_ALIGN.CENTER):
    """lines: [[(text, size, color, font, baseline), ...], ...] 或 [str, ...]"""
    tf = host.text_frame if hasattr(host, 'text_frame') else host
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if isinstance(ln, str):
            ln = [ln]
        ln = [(x, size, color, CN, 0) if isinstance(x, str) else x for x in ln]
        for t, sz, col, fnt, base in ln:
            r = p.add_run(); r.text = t
            r.font.size = Pt(sz); r.font.bold = True
            r.font.color.rgb = col; r.font.name = fnt
            rPr = r._r.get_or_add_rPr()
            rPr.set('altLang', 'en-US')
            ea = rPr.find(qn('a:ea'))
            if ea is None:
                ea = parse_xml('<a:ea xmlns:a="http://schemas.openxmlformats.org/'
                               'drawingml/2006/main"/>'); rPr.append(ea)
            ea.set('typeface', CN)
            if base:
                rPr.set('baseline', str(int(base * 1000)))


def sub(t, sz=None, col=INK, color=None):
    return (t, (sz or F_LEAF) * 0.86, color or col, MATH, -25)


def sup(t, sz=None, col=INK, color=None):
    return (t, (sz or F_LEAF) * 0.86, color or col, MATH, 33)


def mt(t, sz=None, col=INK, color=None):
    return (t, sz or F_LEAF, color or col, MATH, 0)


def cn(t, sz=None, col=INK, color=None):
    return (t, sz or F_LEAF, color or col, CN, 0)


def arrow(x1, y1, x2, y2, color=BLUE, lw=1.1, dash=None):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                        Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    if dash:
        ln.append(parse_xml('<a:prstDash xmlns:a="http://schemas.openxmlformats.org/'
                            'drawingml/2006/main" val="%s"/>' % dash))
    # tailEnd 必须落在 ln 末尾(prstDash 之后),否则 PowerPoint 丢箭头
    ln.append(parse_xml('<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/'
                        'drawingml/2006/main" type="triangle" w="med" len="med"/>'))
    return c


def block(y, h, title):
    """白底蓝边大块 + 顶部蓝色横幅(圆角矩形 + 方角矩形补平底)"""
    outer = box(ML, y, BW, h, WHITE, BLUE, 1.4, r=0.075)
    bh = 0.34
    box(ML, y, BW, bh, BLUE, BLUE, 1.0, r=0.075)
    box(ML, y + bh / 2, BW, bh / 2, BLUE, BLUE, 1.0, shape=MSO_SHAPE.RECTANGLE)
    put(label(ML, y, BW, bh), [title], F_BANNER, WHITE)
    return outer


def group(x, y, w, h, title):
    """黑色虚线分组 + 深蓝加粗组标题"""
    box(x, y, w, h, None, DASHLINE, 0.9, dash='dash', r=0.07)
    put(label(x, y + 0.05, w, 0.26), [title], F_GROUP, DARKBLUE)


ML, MR = 0.22, 0.22
BW = CW - ML - MR

# ===========================================================================
# 块一:共享特征提取
# ===========================================================================
block(0.20, 3.02, '共享特征提取（三视角权重共享 + 视角嵌入）')

group(0.42, 0.66, 2.13, 2.02, '三视角输入')
for i, (nm, sym) in enumerate([('左雷达', 'L'), ('上雷达', 'T'), ('右雷达', 'R')]):
    b = box(0.52, 0.98 + i * 0.56, 1.93, 0.46, PEACH)
    put(b, [[cn(nm + ' '), mt(sym)], [cn('CIR  90×189', F_NOTE)]])

# 三路视角并行贯穿共享模块:每一路各出一根箭头,共享模块画成跨三行的高框
ROWY = [1.21, 1.77, 2.33]
for yy in ROWY:
    arrow(2.57, yy, 2.85, yy)
b = box(2.87, 0.98, 1.16, 1.58, CREAM)
put(b, [['振幅归一化'],
        [cn('逐样本、逐视角', F_NOTE)], [cn('标准化 abs_std', F_NOTE)]])

for yy in ROWY:
    arrow(4.05, yy, 4.33, yy)
group(4.35, 0.66, 3.20, 2.02, '共享双路径编码器')
b = box(4.50, 1.02, 2.90, 0.54, LIGHTBLUE, BLUE)
put(b, [[cn('时间路径 Transformer', color=DARKBLUE)],
        [cn('90 个时间 token × 4 层', F_NOTE, DARKBLUE)]])
b = box(4.50, 1.66, 2.90, 0.54, LIGHTBLUE, BLUE)
put(b, [[cn('距离路径 Transformer', color=DARKBLUE)],
        [cn('189 个距离 token × 4 层', F_NOTE, DARKBLUE)]])
put(label(4.50, 2.26, 2.90, 0.30), [[cn('两路 token 展平拼接', F_NOTE)]], F_NOTE)

for yy in ROWY:
    arrow(7.57, yy, 7.85, yy)
for i, v in enumerate(['L', 'T', 'R']):
    b = box(7.87, 0.98 + i * 0.56, 2.21, 0.46, CREAM)
    put(b, [[mt('f'), sub(v), cn('　特征码（8928 维）')]])

b = box(0.42, 2.76, 3.61, 0.32, None, ORANGE, 0.9, dash='dash', r=0.05)
put(b, [[cn('训练期随机丢视角：', F_NOTE, ORANGE), mt('p', F_NOTE, ORANGE),
         sub('drop', F_NOTE, ORANGE), cn(' = 0.5，随机保留 1–2 路', F_NOTE, ORANGE)]])

# ===========================================================================
sh = S.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(5.02), Inches(3.30),
                 Inches(0.46), Inches(0.40))
sh.fill.solid(); sh.fill.fore_color.rgb = RED
sh.line.color.rgb = RED; sh.line.width = Pt(1.0); sh.shadow.inherit = False

# ===========================================================================
# 块二:阶段一 闭集分类
# ===========================================================================
block(3.80, 1.12, '阶段一 · 闭集分类')
b = box(1.30, 4.26, 2.30, 0.52, CREAM)
put(b, [['跨视角特征聚合']])
arrow(3.62, 4.52, 4.08, 4.52)
b = box(4.10, 4.26, 2.30, 0.52, LIGHTBLUE, BLUE)
put(b, [[cn('闭集分类器', color=DARKBLUE), cn('（交叉熵训练）', F_NOTE, DARKBLUE)]])
arrow(6.42, 4.52, 6.88, 4.52)
b = box(6.90, 4.26, 2.30, 0.52, PEACH)
put(b, [[cn('预测类 '), mt('k̂')]])

# ===========================================================================
sh = S.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(5.02), Inches(5.00),
                 Inches(0.46), Inches(0.40))
sh.fill.solid(); sh.fill.fore_color.rgb = RED
sh.line.color.rgb = RED; sh.line.width = Pt(1.0); sh.shadow.inherit = False
put(label(5.60, 5.04, 3.20, 0.32),
    [[cn('按 ', F_NOTE, ORANGE), mt('k̂', F_NOTE, ORANGE),
      cn(' 选定对应类的检查器组', F_NOTE, ORANGE)]], F_NOTE, align=PP_ALIGN.LEFT)

# ===========================================================================
# 块三:阶段二 开集判决
# ===========================================================================
block(5.50, 1.76, '阶段二 · 开集判决（逐类 × 逐视角深度检查）')

group(0.42, 5.96, 4.18, 1.18, '逐视角深度检查器')
for i, v in enumerate(['L', 'T', 'R']):
    b = box(0.52 + i * 1.36, 6.32, 1.26, 0.60, LIGHTBLUE, BLUE)
    put(b, [[cn('检查器 ', color=DARKBLUE), mt('η', color=DARKBLUE),
             sub('k̂', col=DARKBLUE), sup('(%s)' % v, col=DARKBLUE)],
            [mt('d', F_NOTE, DARKBLUE), sub(v, F_NOTE, DARKBLUE),
             mt(' = ‖l', F_NOTE, DARKBLUE), sub(v, F_NOTE, DARKBLUE),
             mt(' − φ', F_NOTE, DARKBLUE), sub('k̂', F_NOTE, DARKBLUE),
             sup('(%s)' % v, F_NOTE, DARKBLUE), mt('‖', F_NOTE, DARKBLUE)]])

arrow(4.62, 6.62, 4.95, 6.62)
b = box(4.97, 6.28, 2.42, 0.68, LIGHTBLUE, BLUE, 1.2)
put(b, [[cn('多视角聚合（异常分数）', color=DARKBLUE)],
        [mt('s', F_NOTE, DARKBLUE), mt(' = mean', F_NOTE, DARKBLUE),
         sub('v', F_NOTE, DARKBLUE), mt('(d', F_NOTE, DARKBLUE),
         sub('v', F_NOTE, DARKBLUE), mt(')/σ̄ + β·var', F_NOTE, DARKBLUE),
         sub('v', F_NOTE, DARKBLUE), mt('(d', F_NOTE, DARKBLUE),
         sub('v', F_NOTE, DARKBLUE), mt(')/σ̄', F_NOTE, DARKBLUE),
         sup('2', F_NOTE, DARKBLUE)]])

arrow(7.41, 6.62, 7.75, 6.62)
b = box(7.77, 6.11, 2.31, 0.48, GREEN)
put(b, [[mt('s ≤ τ'), cn('：接受，输出已知类 '), mt('k̂')]])
b = box(7.77, 6.65, 2.31, 0.48, PEACH)
put(b, [[mt('s > τ'), cn('：拒识为未知手势')]])

OUT = r'D:\uwb_gestures\paper1\figs\fig1_mvosr_vtsstyle.pptx'
prs.save(OUT)
print('saved', OUT, '  canvas %.2f x %.2f in' % (CW, CH))
