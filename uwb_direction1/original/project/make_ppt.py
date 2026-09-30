# -*- coding: utf-8 -*-
"""Build a concise PPT for the MV-OSR UWB gesture project.
Method (Eom et al.) -> my architecture improvements -> combined results.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = r"D:/uwb_gestures"
FLOW = ROOT + "/figures/mv_osr_three_view_flow.png"
F1 = ROOT + "/figures/nature/fig1_per_case_oscr.png"
F2 = ROOT + "/figures/nature/fig2_ablation_heatmap.png"

# palette
TEAL = RGBColor(0x1F, 0x6F, 0x78)
DARK = RGBColor(0x22, 0x2B, 0x2E)
GREY = RGBColor(0x5A, 0x63, 0x66)
LIGHT = RGBColor(0xEE, 0xF3, 0xF3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACC = RGBColor(0xD9, 0x53, 0x4F)
FONT = "Microsoft YaHei"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def rect(s, x, y, w, h, color, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp


def txt(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
        space_after=6, line_spacing=1.05):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after); p.space_before = Pt(0)
        p.line_spacing = line_spacing
        if isinstance(para, str):
            para = [(para, {})]
        for t, st in para:
            r = p.add_run(); r.text = t
            f = r.font
            f.name = st.get("sz_name", FONT)
            f.size = Pt(st.get("sz", 18))
            f.bold = st.get("b", False)
            f.italic = st.get("i", False)
            f.color.rgb = st.get("c", DARK)
        if "level" in (para[0][1] if para else {}):
            p.level = para[0][1]["level"]
    return tb


def header(s, kicker, title):
    rect(s, 0, 0, SW, Inches(1.18), WHITE)
    rect(s, 0, Inches(1.18), SW, Pt(2.5), TEAL)
    rect(s, Inches(0.0), 0, Inches(0.18), Inches(1.18), TEAL)
    txt(s, Inches(0.55), Inches(0.12), Inches(12), Inches(0.32),
        [[(kicker, {"sz": 12, "b": True, "c": TEAL})]])
    txt(s, Inches(0.5), Inches(0.4), Inches(12.3), Inches(0.72),
        [[(title, {"sz": 26, "b": True, "c": DARK})]])


def bullets(s, x, y, w, h, items, gap=8, base=16):
    runs = []
    for it in items:
        lvl = it[0]
        text = it[1]
        st = it[2] if len(it) > 2 else {}
        mark = "  • " if lvl == 0 else "      – "
        sz = st.get("sz", base if lvl == 0 else base - 2)
        col = st.get("c", DARK if lvl == 0 else GREY)
        runs.append([(mark, {"sz": sz, "c": TEAL if lvl == 0 else GREY, "b": lvl == 0}),
                     (text, {"sz": sz, "c": col, "b": st.get("b", False)})])
    txt(s, x, y, w, h, runs, space_after=gap, line_spacing=1.06)


def add_image_fit(s, path, x, y, w, h):
    from PIL import Image
    iw, ih = Image.open(path).size
    ar = iw / ih
    bw, bh = w, h
    if bw / bh > ar:
        nw = int(bh * ar); nh = bh
    else:
        nw = bw; nh = int(bw / ar)
    nx = x + (bw - nw) // 2; ny = y + (bh - nh) // 2
    s.shapes.add_picture(path, nx, ny, nw, nh)


# ---------------------------------------------------------------- Slide 1 title
s = add_slide()
rect(s, 0, 0, SW, SH, DARK)
rect(s, 0, Inches(4.55), SW, Pt(3), TEAL)
rect(s, Inches(0.0), Inches(2.0), Inches(0.22), Inches(2.4), TEAL)
txt(s, Inches(0.9), Inches(2.0), Inches(11.6), Inches(0.5),
    [[("UWB 冲激雷达 · 开集手势识别 (OSR)", {"sz": 18, "b": True, "c": RGBColor(0x8E, 0xCB, 0xCF)})]])
txt(s, Inches(0.9), Inches(2.55), Inches(11.8), Inches(1.7),
    [[("从单视角 Transformer 到", {"sz": 38, "b": True, "c": WHITE})],
     [("多视角深度检测的复现与改进", {"sz": 38, "b": True, "c": WHITE})]],
    line_spacing=1.02)
txt(s, Inches(0.9), Inches(4.8), Inches(11.8), Inches(1.6),
    [[("基线论文：Eom, Jeon & Jeong, ", {"sz": 16, "c": RGBColor(0xC9,0xD2,0xD4)}),
      ("“UWB Impulse Radar-Based Open-Set Gesture Recognition Using Transformer and One-Versus-Rest Classifier”", {"sz": 16, "i": True, "c": RGBColor(0xC9,0xD2,0xD4)})],
     [("IEEE Internet of Things Journal, 2025  ·  数据集：UWB-Gestures (Figshare 12652592)", {"sz": 15, "c": GREY if False else RGBColor(0x9A,0xA6,0xA9)})]],
    space_after=8)

# ---------------------------------------------------------------- Slide 2 background
s = add_slide()
header(s, "研究背景 / BACKGROUND", "任务定义：开集手势识别 (OSR)")
bullets(s, Inches(0.55), Inches(1.5), Inches(7.3), Inches(5.5), [
    (0, "开集问题：现实中用户会做未定义手势，系统不仅要", {}),
    (1, "准确分类 12 类已知手势（闭集）", {}),
    (1, "并正确拒识 未知手势（误判会引发系统误动作）", {}),
    (0, "为什么用 UWB 冲激雷达", {}),
    (1, "不受光照影响、隐私友好；单设备易同步", {}),
    (1, "带宽 / 中心频率 / 调制简单，可与通信、测距集成", {}),
    (0, "数据：UWB-Gestures（公开集）", {}),
    (1, "12 类手势 × 8 受试者 × 每类 100 样本", {}),
    (1, "CIR 矩阵 X：90 帧(时间 I) × 189 采样(空间 M)", {}),
    (1, "同步采集 3 个雷达视角：Left / Top / Right", {}),
], gap=9, base=16)
rect(s, Inches(8.1), Inches(1.6), Inches(4.7), Inches(4.9), LIGHT)
txt(s, Inches(8.35), Inches(1.8), Inches(4.3), Inches(0.5),
    [[("论文核心思想", {"sz": 17, "b": True, "c": TEAL})]])
bullets(s, Inches(8.2), Inches(2.4), Inches(4.5), Inches(4.0), [
    (0, "把 OSR 分解为两步（divide & conquer）", {"sz": 15}),
    (1, "Stage 1：闭集分类，预测一个已知类 k", {"sz": 14}),
    (1, "Stage 2：深度检测，判 “k” 是真已知还是未知", {"sz": 14}),
    (0, "原论文只用 Left 单雷达视角", {"sz": 15, "c": ACC, "b": True}),
], gap=10)

# ---------------------------------------------------------------- Slide 3 paper method overview
s = add_slide()
header(s, "论文方法 / METHOD", "两阶段架构总览 (Eom et al.)")
# pipeline boxes
def pbox(x, y, w, h, title, sub, color, tcol=WHITE):
    b = rect(s, x, y, w, h, color)
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = title; r.font.size = Pt(15); r.font.bold = True
    r.font.name = FONT; r.font.color.rgb = tcol
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = sub; r2.font.size = Pt(11); r2.font.name = FONT
    r2.font.color.rgb = tcol
    return b

y1 = Inches(2.0)
pbox(Inches(0.55), y1, Inches(2.0), Inches(1.2), "CIR 矩阵 X", "90×189 单视角", GREY)
pbox(Inches(2.85), y1, Inches(2.6), Inches(1.2), "双路径 Transformer\n特征提取器", "时间路 + 空间路", TEAL)
pbox(Inches(5.75), y1, Inches(2.3), Inches(1.2), "闭集分类器", "DNN 2048→256→K", TEAL)
pbox(Inches(8.35), y1, Inches(2.1), Inches(1.2), "预测已知类 k", "Stage 1 输出", RGBColor(0xE7,0xC8,0x6B), DARK)
# arrows row1 (simple connectors via small rects)
for ax in [Inches(2.6), Inches(5.5), Inches(8.1), Inches(10.5)]:
    txt(s, ax, y1+Inches(0.35), Inches(0.4), Inches(0.5), [[("→", {"sz": 22, "b": True, "c": TEAL})]], align=PP_ALIGN.CENTER)
pbox(Inches(10.7), y1, Inches(2.05), Inches(1.2), "激活 inspector-k", "Stage 2 入口", RGBColor(0xE7,0xC8,0x6B), DARK)

y2 = Inches(3.9)
txt(s, Inches(0.55), Inches(3.5), Inches(12), Inches(0.4),
    [[("Stage 2 — 深度检测：每个已知类一个二分类 deep inspector（one-vs-rest）", {"sz": 15, "b": True, "c": TEAL})]])
bullets(s, Inches(0.55), y2, Inches(12.3), Inches(3.2), [
    (0, "原型学习：profile φ_k = 类内 latent 向量均值；距离概率 p = exp(−d(l,φ_k))", {"sz": 15}),
    (0, "profile-distance 损失把正样本拉近 φ_k、负样本推远；数据增强：manifold mixup 造负样本 + 高斯扰动造正样本", {"sz": 15}),
    (0, "判决：测试时距离 d(l,φ_k) < 阈值 η·σ_k → 判已知 k；否则 → 未知（η=3 启发式）", {"sz": 15}),
    (0, "距离阈值优于 softmax/sigmoid 概率阈值（论文 Fig.6：概率聚集在 0/1，难设阈值）", {"sz": 15, "c": GREY}),
], gap=11)

# ---------------------------------------------------------------- Slide 4 dual-path encoder
s = add_slide()
header(s, "论文方法 / METHOD", "双路径 Transformer 特征提取器")
bullets(s, Inches(0.55), Inches(1.5), Inches(12.4), Inches(5.5), [
    (0, "输入 CIR 矩阵 X：I=90 帧（时间）× M=189 采样（空间 / 距离）", {}),
    (0, "时间路 (Temporal path)：切成 I 个 patch，每个 M 维 → Transformer 聚焦时间特征", {}),
    (0, "空间路 (Spatial path)：切成 M 个 patch，每个 I 维 → Transformer 聚焦空间特征", {}),
    (1, "每个 patch 线性投影到 D 维 + 可学习位置编码", {}),
    (1, "ViT 式编码器：L 层，每层 MSA(多头自注意力) + MLP，LayerNorm + 残差", {}),
    (0, "两路编码 concat → 综合特征码 c ∈ R^{D(I+M)}（flatten=8928 维）", {"b": True}),
    (1, "同时喂给闭集分类器与 deep inspector", {}),
    (0, "超参：D=32, D_key=32, L=4, 头数 N_h=4；lr=1e-4, batch=16, 150 epoch", {"c": GREY}),
], gap=11, base=17)
rect(s, Inches(9.3), Inches(1.6), Inches(3.5), Inches(2.0), LIGHT)
txt(s, Inches(9.5), Inches(1.75), Inches(3.2), Inches(1.8),
    [[("关键直觉", {"sz": 15, "b": True, "c": TEAL})],
     [("手势同时含时间动态与空间反射模式 → 用两条独立路径分别捕获，再拼接。", {"sz": 13, "c": DARK})]],
    space_after=6)

# ---------------------------------------------------------------- Slide 5 my architecture
s = add_slide()
header(s, "我的改进 / OUR ARCHITECTURE", "多视角融合 + 每视角深度检测")
add_image_fit(s, FLOW, Inches(6.5), Inches(1.35), Inches(6.7), Inches(5.9))
bullets(s, Inches(0.45), Inches(1.45), Inches(6.0), Inches(5.7), [
    (0, "三个同步雷达视角 L/T/R（论文只用 Left）", {"sz": 15, "b": True}),
    (0, "共享权重 Siamese 双路径编码器", {"sz": 15, "b": True}),
    (1, "三视角共用同一套 Transformer 权重", {"sz": 13}),
    (0, "跨视角注意力融合 (cross-view fusion)", {"sz": 15, "b": True}),
    (1, "沿慢时间轴对 V 个视角做注意力；可选物理 bias(ρ/Δt/方向)", {"sz": 13}),
    (0, "每视角 deep inspector + 多视角打分", {"sz": 15, "b": True}),
    (1, "s = mean_v(d_v)/σ̄ + β·var_v(d_v)/σ̄²", {"sz": 13}),
    (1, "mean 项=距离均值(降噪+几何互补)，var 项=视角分歧", {"sz": 13}),
    (0, "|x|+std (abs_std) 归一化", {"sz": 15, "b": True, "c": ACC}),
    (1, "对齐论文 “amplitude” 表述，复现关键", {"sz": 13}),
], gap=6)

# ---------------------------------------------------------------- Slide 6 main results table
s = add_slide()
header(s, "主要结果 / RESULTS", "主对照表：论文 vs 我的结果（5-case 均值±std，%）")
rows = [
    ["配置 Configuration", "Accuracy", "F1", "AUROC", "OSCR"],
    ["Eom et al. 单视角（论文）", "83.64", "83.91", "89.28", "88.42"],
    ["Ours 单视角 (Left) · MSP", "72.37", "77.63", "78.49", "77.02"],
    ["Ours 单视角 (Left) · inspector", "78.03", "80.84", "84.92", "82.90"],
    ["Ours 三视角 (L+T+R) · MSP", "78.41", "82.62", "83.34", "81.85"],
    ["Ours 三视角 (L+T+R) · inspector ★", "92.29", "92.54", "97.12", "95.14"],
]
nrow, ncol = len(rows), len(rows[0])
tbl_w = Inches(12.3); tbl_h = Inches(3.5)
gt = s.shapes.add_table(nrow, ncol, Inches(0.5), Inches(1.55), tbl_w, tbl_h).table
gt.columns[0].width = Inches(5.1)
for c in range(1, ncol):
    gt.columns[c].width = Inches(1.8)
for r in range(nrow):
    for c in range(ncol):
        cell = gt.cell(r, c)
        cell.margin_top = Pt(3); cell.margin_bottom = Pt(3)
        cell.margin_left = Pt(7); cell.margin_right = Pt(5)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        run = p.add_run(); run.text = rows[r][c]
        f = run.font; f.name = FONT; f.size = Pt(14)
        if r == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = TEAL
            f.color.rgb = WHITE; f.bold = True
        elif r == nrow - 1:
            cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xDB,0xEC,0xEC)
            f.color.rgb = DARK; f.bold = True
        elif r == 1:
            cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xF6,0xE2,0xE1)
            f.color.rgb = DARK; f.bold = (c == 0)
        else:
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
            f.color.rgb = DARK
bullets(s, Inches(0.5), Inches(5.3), Inches(12.4), Inches(1.9), [
    (0, "三视角 inspector：OSCR 95.14 ± 0.81，超论文单视角 88.42 共 +6.72 分，且标准差更小", {"b": True}),
    (0, "单视角复现：OSCR 82.90，与论文差 5.5 分（约 1.2σ），基本复现到位", {"c": GREY}),
    (0, "重叠指标统一采用我的结果；论文行仅保留其报告的单视角数值作对比", {"c": GREY, "sz": 14}),
], gap=7, base=16)

# ---------------------------------------------------------------- Slide 7 per-case fig
s = add_slide()
header(s, "主要结果 / RESULTS", "逐 case OSCR：三视角全面超越论文基线")
add_image_fit(s, F1, Inches(0.5), Inches(1.4), Inches(8.6), Inches(5.8))
bullets(s, Inches(9.25), Inches(1.7), Inches(3.9), Inches(5.3), [
    (0, "5 个 case 三视角全部超过论文 88.42 线", {"sz": 15}),
    (0, "多视角增益均值 +12.24 OSCR", {"sz": 15, "b": True}),
    (0, "Case 4 单视角最难(73.5)，三视角增益最大 +20.5", {"sz": 15, "c": ACC}),
    (0, "三视角标准差 0.81，远小于单视角 4.83 → 更稳", {"sz": 15}),
], gap=12)

# ---------------------------------------------------------------- Slide 8 ablation heatmap
s = add_slide()
header(s, "关键消融 / ABLATION", "池化 × 归一化：abs_std 是首要因素")
add_image_fit(s, F2, Inches(0.5), Inches(1.4), Inches(8.9), Inches(5.8))
bullets(s, Inches(9.55), Inches(1.7), Inches(3.65), Inches(5.3), [
    (0, "归一化 per_view_std → abs_std", {"sz": 15, "b": True}),
    (1, "单视角 +10 分，三视角 +4.5 分", {"sz": 14}),
    (0, "池化 mean → flatten 是次要因素(+5)", {"sz": 15}),
    (0, "最优组合 ★", {"sz": 15, "b": True, "c": TEAL}),
    (1, "三视角 + flatten + abs_std", {"sz": 14}),
    (1, "OSCR 95.14 ± 0.81（主结果）", {"sz": 14, "b": True}),
], gap=11)

# ---------------------------------------------------------------- Slide 9 other experiments
s = add_slide()
header(s, "其它实验 / FURTHER EXPERIMENTS", "鲁棒性与诚实负结果")
exp = [
    ["实验", "设置", "结果 / 结论"],
    ["缺视角鲁棒性", "训练用 3 视角，测试丢 1–2 路", "L+R 双视角 OSCR 86.22（省 Top，硬件 −33%）；inspector 全程碾压 MSP"],
    ["score 分量", "mean / var / combined", "mean-only(β=0) 95.90 最高；var 分量信息冗余（负结果）"],
    ["物理 bias", "加 ρ/Δt/方向 注意力先验", "无增益 95.10 vs 95.14；注意力已从数据学到几何"],
    ["Raw vs ClutterRemoved", "换原始 ADC 输入", "打平 95.53 vs 95.14 → Transformer 端到端自学杂波抑制"],
    ["加宽 D_key 8→32", "对齐论文模型宽度", "单视角掉 4.8 分（过拟合）→ 我们的 D_key=8 更优"],
]
nrow, ncol = len(exp), 3
gt = s.shapes.add_table(nrow, ncol, Inches(0.5), Inches(1.55), Inches(12.3), Inches(5.2)).table
gt.columns[0].width = Inches(2.5); gt.columns[1].width = Inches(3.4); gt.columns[2].width = Inches(6.4)
for r in range(nrow):
    for c in range(ncol):
        cell = gt.cell(r, c)
        cell.margin_top = Pt(4); cell.margin_bottom = Pt(4)
        cell.margin_left = Pt(8); cell.margin_right = Pt(6)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
        run = p.add_run(); run.text = exp[r][c]
        f = run.font; f.name = FONT; f.size = Pt(13.5)
        if r == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = TEAL
            f.color.rgb = WHITE; f.bold = True
        else:
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
            f.color.rgb = DARK; f.bold = (c == 0)

# ---------------------------------------------------------------- Slide 10 conclusion
s = add_slide()
rect(s, 0, 0, SW, SH, DARK)
rect(s, Inches(0.0), Inches(0.9), Inches(0.22), Inches(1.0), TEAL)
txt(s, Inches(0.9), Inches(0.95), Inches(11.5), Inches(0.9),
    [[("总结 / CONCLUSION", {"sz": 28, "b": True, "c": WHITE})]])
items = [
    ("复现", "单视角方案 OSCR 82.9，在 1.2σ 内基本复现论文 Eom et al."),
    ("主结果", "三视角 + abs_std + per-view inspector → OSCR 95.14 ± 0.81，超论文 +6.72 分且更稳"),
    ("机理", "多视角增益来自几何互补 + 距离均值降噪，而非“视角分歧”（var 项冗余，诚实负结果）"),
    ("工程", "L+R 双视角即可用（省 Top 雷达）；可省杂波对消预处理，直接喂原始数据"),
]
y = Inches(2.2)
for tag, body in items:
    rect(s, Inches(0.9), y, Inches(1.7), Inches(0.78), TEAL)
    txt(s, Inches(0.9), y+Inches(0.12), Inches(1.7), Inches(0.6),
        [[(tag, {"sz": 16, "b": True, "c": WHITE})]], align=PP_ALIGN.CENTER)
    txt(s, Inches(2.8), y+Inches(0.02), Inches(9.8), Inches(0.85),
        [[(body, {"sz": 17, "c": RGBColor(0xE6,0xEC,0xED)})]], anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(1.02)

OUT = ROOT + "/MV-OSR_汇报.pptx"
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
