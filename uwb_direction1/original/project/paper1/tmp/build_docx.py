# -*- coding: utf-8 -*-
"""把 pandoc 产出的 docx 套上《电子与信息学报》版式,并插入批注-原文对照框。

版式依据 2025 版投稿细则:
  单栏,5 号字,两端对齐;正文中文宋体、单倍行距,英文 Euclid、行距 16 磅;
  表格一律三线表。
"""
import io
import re
import sys

import docx
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, "tmp")
from annot_data import ANNOTS

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC, DST = "tmp/_raw.docx", "mvosr_paper_cn_投稿版.docx"
CJK = re.compile(r"[一-鿿]")
SIZE = Pt(10.5)          # 5 号字
EN_FONT = "Euclid"       # 细则指定;若本机未装,Word 会回退,不影响编辑部排版
ZH_FONT = "宋体"

doc = docx.Document(SRC)


def set_run_font(run, zh):
    run.font.size = SIZE
    rPr = run._element.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = rPr.makeelement(qn("w:rFonts"), {})
        rPr.insert(0, rf)
    rf.set(qn("w:ascii"), EN_FONT)
    rf.set(qn("w:hAnsi"), EN_FONT)
    rf.set(qn("w:eastAsia"), ZH_FONT)


def style_par(p, force_zh=None):
    zh = bool(CJK.search(p.text)) if force_zh is None else force_zh
    pf = p.paragraph_format
    # 段内有图或公式时,p.text 是空的(w:drawing / m:oMath 都不算 w:t),
    # 若按"无中文=英文段"套 16 磅固定行距,Word 会把图和公式裁成 16 磅高的一条。
    has_obj = bool(p._p.findall(".//{*}drawing")) or bool(p._p.findall(".//{*}oMath"))
    if p.style.name.startswith("Heading") or p.style.name in ("Title", "Author"):
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER if p.style.name in ("Title", "Author") \
            else WD_ALIGN_PARAGRAPH.LEFT
    elif has_obj and not p.text.strip():
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER      # 独立的图/公式居中
    else:
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if zh or has_obj:
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    else:
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing = Pt(16)
    for r in p.runs:
        set_run_font(r, zh)


# ---- 1. 页面:A4 单栏 --------------------------------------------------------
for s in doc.sections:
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(2.4)
    s.top_margin = s.bottom_margin = Cm(2.5)

# ---- 2. 正文段落 ------------------------------------------------------------
for p in doc.paragraphs:
    style_par(p)

# ---- 2.5 图题/表题:补编号、居中;标题与交叉引用改黑 --------------------------
n_tab = n_fig = 0
for p in doc.paragraphs:
    st = p.style.name
    if st == "Table Caption":
        n_tab += 1
        lbl, num = "表", n_tab
    elif st == "Image Caption":
        n_fig += 1
        lbl, num = "图", n_fig
    else:
        continue
    tag = "%s%d　" % (lbl, num)
    if p.runs:
        p.runs[0].text = tag + p.runs[0].text
    else:
        p.add_run(tag)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
print("已补编号:图 %d 个,表 %d 个" % (n_fig, n_tab))

# 标题与交叉引用链接一律黑色(pandoc 默认蓝色)
for p in doc.paragraphs:
    if p.style.name.startswith("Heading"):
        for r in p.runs:
            r.font.color.rgb = RGBColor(0, 0, 0)
            r.bold = True
for h in doc.element.body.iter(qn("w:hyperlink")):
    for rPr in h.iter(qn("w:rPr")):
        for tag in ("w:color", "w:u"):
            for e in rPr.findall(qn(tag)):
                rPr.remove(e)
        c = rPr.makeelement(qn("w:color"), {})
        c.set(qn("w:val"), "000000")
        rPr.append(c)

# ---- 3. 三线表 --------------------------------------------------------------
def set_borders(row, top=None, bottom=None):
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        old = tcPr.find(qn("w:tcBorders"))
        if old is not None:
            tcPr.remove(old)
        b = tcPr.makeelement(qn("w:tcBorders"), {})
        for side in ("top", "left", "bottom", "right"):
            e = b.makeelement(qn("w:" + side), {})
            sz = {"top": top, "bottom": bottom}.get(side)
            e.set(qn("w:val"), "single" if sz else "nil")
            if sz:
                e.set(qn("w:sz"), str(sz))
                e.set(qn("w:color"), "000000")
            b.append(e)
        tcPr.append(b)


for ti, t in enumerate(doc.tables, 1):
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER  # 边框全部显式设置,不依赖内置表格样式
    # 表头行数:pandoc 把 \multirow 展成重复文字,首列重复即为两行表头
    nhead = 2 if (len(t.rows) > 1 and t.rows[0].cells[0].text.strip()
                  == t.rows[1].cells[0].text.strip() and t.rows[0].cells[0].text.strip()) else 1
    for ri, row in enumerate(t.rows):
        top = 12 if ri == 0 else None                      # 顶线(粗)
        bottom = 6 if ri == nhead - 1 else None             # 表头下细线
        if ri == len(t.rows) - 1:
            bottom = 12                                     # 底线(粗)
        set_borders(row, top, bottom)
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                style_par(p)
    # 表头里横向相邻的重复单元格合并(还原 \multicolumn)
    for ri in range(nhead):
        j = 0
        cells = t.rows[ri].cells
        while j < len(cells) - 1:
            k = j
            while (k + 1 < len(cells) and cells[k + 1].text.strip()
                   and cells[k + 1].text.strip() == cells[j].text.strip()):
                k += 1
            if k > j:
                cells[j].merge(cells[k])
            j = k + 1
    print("表%d: %d行,表头%d行,已设三线" % (ti, len(t.rows), nhead))

# ---- 4. Word 原生批注(显示在右侧批注栏,不进正文) ----------------------------
def runs_covering(p, anchor):
    """返回覆盖 anchor 那几个 run;定位不到就退回整段的 run。

    注意 p.runs 不含 OMML 公式,所以这里按 run 文本自行拼接定位,不能用 p.text。
    """
    runs = p.runs
    if not runs:
        return None
    spans, pos = [], 0
    for r in runs:
        spans.append((pos, pos + len(r.text), r))
        pos += len(r.text)
    full = "".join(r.text for r in runs)
    i = full.find(anchor)
    if i < 0:
        return runs
    j = i + len(anchor)
    sel = [r for (a, b, r) in spans if b > i and a < j]
    return sel or runs


placed, missed = [], []
for ann in ANNOTS:
    tgt = None
    for p in doc.paragraphs:
        if ann["anchor"] in p.text:
            tgt = p
            break
    sel = runs_covering(tgt, ann["anchor"]) if tgt is not None else None
    if not sel:
        missed.append(ann["n"])
        continue
    c = doc.add_comment(sel, text=ann["comment"], author="导师批注", initials="导")
    c.add_paragraph("【原文】" + ann["old"])
    placed.append(ann["n"])

print("批注框已插入:", placed)
if missed:
    print("!! 未定位到锚点:", missed)

# ---- 5. 脚注(通信作者/基金)不在 document.paragraphs 里,单独设字体 ------------
from lxml import etree

for part in doc.part.package.parts:
    if str(part.partname).endswith("footnotes.xml"):
        # 这是个通用 Part,没有 _element,只能解析并回写字节流
        root = etree.fromstring(part.blob)
        for r in root.iter(qn("w:r")):
            rPr = r.find(qn("w:rPr"))
            if rPr is None:
                rPr = r.makeelement(qn("w:rPr"), {})
                r.insert(0, rPr)
            for tag in ("w:rFonts", "w:sz", "w:szCs"):
                for e in rPr.findall(qn(tag)):
                    rPr.remove(e)
            rf = rPr.makeelement(qn("w:rFonts"), {})
            rf.set(qn("w:ascii"), EN_FONT)
            rf.set(qn("w:hAnsi"), EN_FONT)
            rf.set(qn("w:eastAsia"), ZH_FONT)
            rPr.insert(0, rf)
            for tag, val in (("w:sz", "18"), ("w:szCs", "18")):  # 9pt = 18 半磅
                e = rPr.makeelement(qn(tag), {})
                e.set(qn("w:val"), val)
                rPr.append(e)
        part._blob = etree.tostring(root, xml_declaration=True,
                                    encoding="UTF-8", standalone=True)
        print("脚注字体已统一")

try:
    doc.save(DST)
    print("输出 ->", DST)
except PermissionError:
    alt = DST.replace(".docx", "_新.docx")
    doc.save(alt)
    print("!! %s 正被 Word 占用,已改存 -> %s" % (DST, alt))
