# -*- coding: utf-8 -*-
"""
图2（3.4节）跨视角融合策略对比 —— 独立重绘版。

设计主线：把"跨视角信息交换所处的层级"作为唯一自变量，三列共用同一视觉语法
（输入CIR -> 共享编码器 -> 逐类检查器 -> 距离聚合），仅橙色交换点的位置变化。
顶部层级轴给出"交换更早 = 耦合更强"的方向，底部给出随机切分不可分、LOSO 才分化的结论。

输出：paper1/figs/fig2_fusion_variants_claude.pptx
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

# ---------------------------------------------------------------- 画布与配色
W, H = 10.0, 5.40

C_TEXT      = RGBColor(0x22, 0x30, 0x3C)
C_MUTED     = RGBColor(0x6B, 0x7A, 0x88)
C_CARD      = RGBColor(0xF7, 0xF9, 0xFB)
C_CARD_LN   = RGBColor(0xDD, 0xE4, 0xEA)
C_DATA      = RGBColor(0xDC, 0xE9, 0xF7)
C_DATA_BK   = RGBColor(0xEC, 0xF3, 0xFB)
C_DATA_LN   = RGBColor(0x7F, 0xA6, 0xD0)
C_ENC       = RGBColor(0xE3, 0xEA, 0xF2)
C_ENC_BK    = RGBColor(0xEF, 0xF3, 0xF8)
C_ENC_LN    = RGBColor(0x9B, 0xA9, 0xB8)
C_INSP      = RGBColor(0xD3, 0xDD, 0xE7)
C_INSP_LN   = RGBColor(0x8B, 0x9B, 0xAB)
C_ACC       = RGBColor(0xE0, 0x7B, 0x39)     # 强调色：交换点
C_ACC_FILL  = RGBColor(0xFA, 0xE4, 0xD1)
C_BLUE      = RGBColor(0x1F, 0x6F, 0xB2)     # 本文方案
C_LINE      = RGBColor(0x8D, 0x9A, 0xA6)
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

FS_BIG, FS_SML = 13.5, 9.5                    # 全图仅两种字号
EA, LATIN = "Microsoft YaHei", "Arial"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
sld = prs.slides.add_slide(prs.slide_layouts[6])
SH = sld.shapes


# ---------------------------------------------------------------- 基础工具
def _font(run, size, bold, color):
    f = run.font
    f.size, f.bold, f.name = Pt(size), bold, LATIN
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    rPr.append(rPr.makeelement(qn('a:ea'), {'typeface': EA}))


def _no_shadow(shp):
    shp.shadow.inherit = False


def label(shp, s, size=FS_SML, bold=False, color=C_TEXT, align=PP_ALIGN.CENTER):
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, m, 0)
    p = tf.paragraphs[0]
    p.alignment = align
    for i, line in enumerate(s.split('\n')):
        para = p if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = 0.92
        _font(para.add_run(), size, bold, color)
        para.runs[0].text = line
    return shp


def box(x, y, w, h, fill, line, rad=None, lw=0.75, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = SH.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    _no_shadow(s)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    if rad is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = rad
    s.text_frame.text = ''
    return s


def txt(x, y, w, h, s, size=FS_SML, bold=False, color=C_TEXT, align=PP_ALIGN.LEFT):
    tb = SH.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    return label(tb, s, size, bold, color, align)


def arrow(x1, y1, x2, y2, color=C_LINE, lw=1.0, both=False, dash=False):
    c = SH.add_connector(MSO_CONNECTOR.STRAIGHT,
                         Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    if dash:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    if both:
        ln.append(ln.makeelement(qn('a:headEnd'),
                                 {'type': 'triangle', 'w': 'sm', 'len': 'sm'}))
    ln.append(ln.makeelement(qn('a:tailEnd'),
                             {'type': 'triangle', 'w': 'sm', 'len': 'sm'}))
    return c


def stack(x, y, w, h, n, fill, back, line, text=None, dx=0.030, dy=0.030,
          size=FS_SML, bold=False):
    """后层向右上偏移的层叠块，制造网络堆叠的立体感；最前层承载文字。"""
    for i in range(n - 1, 0, -1):
        box(x + dx * i, y - dy * i, w, h, back, line, rad=0.14, lw=0.5)
    front = box(x, y, w, h, fill, line, rad=0.14)
    if text:
        label(front, text, size, bold)
    return front


def grid(x, y, w, h, nc, nr, fill, line):
    """token 网格：用于瓶颈 token 与 CIR 采样的具象化。"""
    cw, ch = w / nc, h / nr
    for i in range(nc):
        for j in range(nr):
            box(x + i * cw, y + j * ch, cw * 0.78, ch * 0.78, fill, line,
                lw=0.4, shape=MSO_SHAPE.RECTANGLE)


# ---------------------------------------------------------------- 顶部层级轴
MG, GAP = 0.22, 0.12
PW = (W - 2 * MG - 2 * GAP) / 3.0
PX = [MG + i * (PW + GAP) for i in range(3)]
PY, PH = 0.60, 4.00

txt(MG, 0.10, 4.2, 0.24, "唯一自变量：跨视角信息交换所处的层级",
    FS_SML, True, C_TEXT)
arrow(PX[0] + PW / 2, 0.46, PX[2] + PW / 2, 0.46, C_MUTED, 1.0)
txt(PX[2] + PW / 2 - 1.55, 0.24, 1.5, 0.22, "交换更早，3路耦合更强",
    FS_SML, False, C_MUTED, PP_ALIGN.RIGHT)
for px in PX:                                   # 轴上刻度
    t = box(px + PW / 2 - 0.006, 0.425, 0.012, 0.05, C_MUTED, None,
            shape=MSO_SHAPE.RECTANGLE)
    _no_shadow(t)

TITLES = [("(a) 决策级聚合（本文）", "交换层级：决策级",
           "随机切分 95.99   ·   LOSO 85.22±5.55"),
          ("(b) 特征级融合", "交换层级：编码器之后",
           "随机切分 95.30   ·   接入检查器 −0.69"),
          ("(c) 瓶颈中层融合（MBT式）", "交换层级：编码器中层",
           "随机切分 95.96   ·   LOSO 83.43±8.45")]
VIEWS = ["左 L", "上 T", "右 R"]

CY = [1.66, 2.54, 3.42]          # 3条视角泳道中心（全图统一）
MID = CY[1]
DW, DH = 0.26, 0.30              # 输入数据块
EH, IW, IH = 0.34, 0.50, 0.26    # 编码器高 / 检查器宽高
CD, BW = 0.44, 0.22              # 聚合节点直径 / 交换条宽
MX = 2.52                        # 聚合节点左边界（3面板对齐，便于横向比较）
NOTE_Y = 3.72                    # 面板底部提示行

NOTES = ["3路全程独立，仅在此处汇合",
         "3路特征在进入检查器前已耦合",
         "瓶颈在视角间取平均，通道狭窄"]

for k in range(3):
    px = PX[k]
    hl = (k == 0)
    box(px, PY, PW, PH, C_WHITE if hl else C_CARD,
        C_BLUE if hl else C_CARD_LN, rad=0.035, lw=1.75 if hl else 0.75)
    txt(px + 0.10, PY + 0.06, PW - 0.20, 0.26, TITLES[k][0], FS_BIG, True,
        C_BLUE if hl else C_TEXT, PP_ALIGN.LEFT)
    txt(px + 0.10, PY + 0.33, PW - 0.20, 0.20, TITLES[k][1], FS_SML, False,
        C_MUTED, PP_ALIGN.LEFT)

    def lane_head(dx0, ex0, ew, y, i, nlayer=4, txt_=None):
        d = stack(dx0, y - DH / 2, DW, DH, 3, C_DATA, C_DATA_BK, C_DATA_LN)
        label(d, VIEWS[i], FS_SML, True)
        arrow(dx0 + DW + 0.08, y, ex0 - 0.02, y)
        stack(ex0, y - EH / 2, ew, EH, nlayer, C_ENC, C_ENC_BK, C_ENC_LN, txt_)

    def lane_tail(ix0, mx0, y, acc_in=False):
        b = box(ix0, y - IH / 2, IW, IH, C_INSP, C_INSP_LN, rad=0.18)
        label(b, "检查器", FS_SML)
        arrow(ix0 + IW + 0.02, y, mx0 - 0.02, MID + (y - MID) * 0.16)

    def merge(mx0, accent):
        m = box(mx0, MID - CD / 2, CD, CD,
                C_ACC_FILL if accent else C_CARD,
                C_ACC if accent else C_INSP_LN,
                lw=1.75 if accent else 0.75, shape=MSO_SHAPE.OVAL)
        label(m, "距离\n聚合", FS_SML, accent, C_ACC if accent else C_TEXT)

    def xbar(bx0, tokens=False):
        bar = box(bx0, CY[0] - 0.22, BW, CY[2] - CY[0] + 0.44,
                  C_ACC_FILL, C_ACC, rad=0.10, lw=1.75)
        if tokens:
            grid(bx0 + 0.050, MID - 0.30, 0.13, 0.60, 2, 6, C_ACC, C_ACC)
        return bar

    # ---------------- (a) 全程独立，仅在距离聚合处汇合
    if k == 0:
        dx0, ex0, ew, ix0, mx0 = px + 0.09, px + 0.45, 0.86, px + 1.55, px + MX
        for i, y in enumerate(CY):
            lane_head(dx0, ex0, ew, y, i, 4, "编码器")
            arrow(ex0 + ew + 0.10, y, ix0 - 0.02, y)
            lane_tail(ix0, mx0, y)
        merge(mx0, True)

    # ---------------- (b) 编码器之后插入跨视角注意力
    elif k == 1:
        dx0, ex0, ew = px + 0.06, px + 0.40, 0.86
        bx0, ix0, mx0 = px + 1.36, px + 1.75, px + MX
        for i, y in enumerate(CY):
            lane_head(dx0, ex0, ew, y, i, 4, "编码器")
            arrow(ex0 + ew + 0.10, y, bx0 - 0.02, y, C_ACC, 1.25)
            arrow(bx0 + BW + 0.02, y, ix0 - 0.02, y, C_ACC, 1.25)
            lane_tail(ix0, mx0, y)
        label(xbar(bx0), "跨\n视\n角\n注\n意\n力", FS_SML, True, C_ACC)
        merge(mx0, False)

    # ---------------- (c) 编码器中层经共享瓶颈 token 交换
    else:
        dx0, e1, e2w = px + 0.02, px + 0.39, 0.44
        bx0, e2, ix0, mx0 = px + 1.00, px + 1.38, px + 1.92, px + MX
        for i, y in enumerate(CY):
            lane_head(dx0, e1, e2w, y, i, 2, "1–2层")
            arrow(e1 + e2w + 0.04, y, bx0 - 0.02, y, C_ACC, 1.25, both=True)
            stack(e2, y - EH / 2, e2w, EH, 2, C_ENC, C_ENC_BK, C_ENC_LN, "3–4层")
            arrow(bx0 + BW + 0.04, y, e2 - 0.02, y, C_ACC, 1.25, both=True)
            arrow(e2 + e2w + 0.04, y, ix0 - 0.02, y)
            lane_tail(ix0, mx0, y)
        xbar(bx0, tokens=True)
        txt(bx0 - 0.44, CY[0] - 0.50, BW + 0.88, 0.20,
            "共享瓶颈 token", FS_SML, True, C_ACC, PP_ALIGN.CENTER)
        merge(mx0, False)

    txt(px + 0.10, NOTE_Y, PW - 0.20, 0.20, NOTES[k], FS_SML, True,
        C_ACC, PP_ALIGN.CENTER)

    # 面板结果条
    sep = SH.add_connector(MSO_CONNECTOR.STRAIGHT,
                           Inches(px + 0.12), Inches(PY + 3.40),
                           Inches(px + PW - 0.12), Inches(PY + 3.40))
    sep.line.color.rgb = C_CARD_LN
    sep.line.width = Pt(0.75)
    txt(px + 0.10, PY + 3.48, PW - 0.20, 0.26, TITLES[k][2], FS_SML,
        k == 0, C_BLUE if k == 0 else C_MUTED, PP_ALIGN.CENTER)

# ---------------------------------------------------------------- 底部结论条
strip = box(MG, 4.76, W - 2 * MG, 0.50, C_CARD, C_CARD_LN, rad=0.10)
txt(MG + 0.16, 4.82, W - 2 * MG - 0.32, 0.20,
    "随机切分（划分1）三者统计不可分   ·   LOSO 跨受试者协议下才分化：无融合 85.22±5.55 ＞ 瓶颈融合 83.43±8.45（配对差 −1.79）",
    FS_SML, False, C_MUTED, PP_ALIGN.CENTER)
txt(MG + 0.16, 5.02, W - 2 * MG - 0.32, 0.22,
    "开集增益来自 3 路近似独立的距离信号集成，任何特征层耦合都会削弱这一独立性",
    FS_BIG, True, C_ACC, PP_ALIGN.CENTER)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs',
                   'fig2_fusion_variants_claude.pptx')
out = os.path.normpath(out)
prs.save(out)
print("saved:", out)
