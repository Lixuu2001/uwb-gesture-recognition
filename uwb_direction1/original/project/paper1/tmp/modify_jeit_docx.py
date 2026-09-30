from __future__ import annotations

import re
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt


ROOT = Path(r"D:\uwb_gestures\paper1")
SRC = ROOT / "mvosr_paper_cn_投稿版.docx"
DST = ROOT / "mvosr_paper_cn_投稿版_按模板修改.docx"


def remove_paragraph(paragraph) -> None:
    p = paragraph._element
    p.getparent().remove(p)


def replace_across_text_nodes(paragraph, old: str, new: str, count: int = -1) -> int:
    """Replace visible text while preserving existing runs, hyperlinks, comments and math."""
    nodes = paragraph._p.xpath(".//w:t")
    if not nodes:
        return 0
    texts = [n.text or "" for n in nodes]
    joined = "".join(texts)
    starts = []
    start = 0
    while True:
        pos = joined.find(old, start)
        if pos < 0 or (count >= 0 and len(starts) >= count):
            break
        starts.append(pos)
        start = pos + len(old)
    for pos in reversed(starts):
        end = pos + len(old)
        offsets = []
        cursor = 0
        for t in texts:
            offsets.append((cursor, cursor + len(t)))
            cursor += len(t)
        first = next(i for i, (a, b) in enumerate(offsets) if a <= pos < b or (a == b == pos))
        last = next(i for i, (a, b) in enumerate(offsets) if a < end <= b)
        a0, _ = offsets[first]
        al, _ = offsets[last]
        prefix = texts[first][: pos - a0]
        suffix = texts[last][end - al :]
        texts[first] = prefix + new + (suffix if first == last else "")
        for i in range(first + 1, last):
            texts[i] = ""
        if last != first:
            texts[last] = suffix
        joined = "".join(texts)
    for node, value in zip(nodes, texts):
        node.text = value
        if value.startswith(" ") or value.endswith(" "):
            node.set(qn("xml:space"), "preserve")
    return len(starts)


def set_fonts(run, size: float, east_asia: str = "宋体", latin: str = "Euclid", bold=None) -> None:
    run.font.name = latin
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:ascii"), latin)
    fonts.set(qn("w:hAnsi"), latin)
    fonts.set(qn("w:eastAsia"), east_asia)


def format_para(
    p,
    *,
    size=10.5,
    east_asia="宋体",
    latin="Euclid",
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    exact=16,
    before=0,
    after=0,
    first=21,
    left=0,
    right=0,
    bold=None,
):
    pf = p.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.left_indent = Pt(left)
    pf.right_indent = Pt(right)
    pf.first_line_indent = Pt(first) if first is not None else None
    if exact:
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing = Pt(exact)
    for r in p.runs:
        set_fonts(r, size, east_asia, latin, bold)


def set_keep(p, keep_next=False, keep_lines=True, page_break_before=False):
    p.paragraph_format.keep_with_next = keep_next
    p.paragraph_format.keep_together = keep_lines
    p.paragraph_format.page_break_before = page_break_before


def add_prefix_run(p, text: str, *, size=8, bold=False):
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), "Euclid")
    fonts.set(qn("w:hAnsi"), "Euclid")
    fonts.set(qn("w:eastAsia"), "宋体")
    rpr.append(fonts)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size * 2)))
    rpr.append(sz)
    if bold:
        rpr.append(OxmlElement("w:b"))
    r.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    ppr = p._p.pPr
    idx = 1 if ppr is not None else 0
    p._p.insert(idx, r)


def add_equation_number(p, number: int, usable_twips: int):
    ppr = p._p.get_or_add_pPr()
    tabs = ppr.find(qn("w:tabs"))
    if tabs is not None:
        ppr.remove(tabs)
    tabs = OxmlElement("w:tabs")
    for val, pos in (("center", usable_twips // 2), ("right", usable_twips)):
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), val)
        tab.set(qn("w:pos"), str(pos))
        tabs.append(tab)
    ppr.append(tabs)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Pt(0)
    p.paragraph_format.right_indent = Pt(0)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    pre = OxmlElement("w:r")
    pre.append(OxmlElement("w:tab"))
    insert_idx = 1 if p._p.pPr is not None else 0
    p._p.insert(insert_idx, pre)

    post = OxmlElement("w:r")
    post.append(OxmlElement("w:tab"))
    t = OxmlElement("w:t")
    t.text = f"({number})"
    post.append(t)
    p._p.append(post)
    for r in p.runs:
        set_fonts(r, 10.5)


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.clear()
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    result = OxmlElement("w:t")
    result.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for el in (fld_begin, instr, fld_sep, result, fld_end):
        run._r.append(el)
    set_fonts(run, 9)


def paragraph_by_text(doc, text):
    return next(p for p in doc.paragraphs if p.text == text)


def count_comment_markers(path: Path):
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        xml = z.read("word/document.xml").decode("utf-8")
        comments = z.read("word/comments.xml").decode("utf-8") if "word/comments.xml" in names else ""
        return {
            "comments_part": int(bool(comments)),
            "comments": comments.count("<w:comment "),
            "starts": xml.count("<w:commentRangeStart"),
            "ends": xml.count("<w:commentRangeEnd"),
            "refs": xml.count("<w:commentReference"),
        }


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if DST.exists():
        DST.unlink()

    source_comments = count_comment_markers(SRC)
    doc = Document(str(SRC))

    # A4 and the journal template's page geometry.
    for sec in doc.sections:
        sec.page_width = Cm(21.0)
        sec.page_height = Cm(29.7)
        sec.left_margin = Inches(0.748)
        sec.right_margin = Inches(0.748)
        sec.top_margin = Inches(1.221)
        sec.bottom_margin = Inches(0.945)
        sec.header_distance = Inches(0.748)
        sec.footer_distance = Inches(0.689)
        add_page_number(sec)

    # Record structural blocks before editing.
    original = list(doc.paragraphs)
    english = original[5:12]
    refs = original[90:142]
    author_front = original[1]
    stray = original[89]
    author_back = original[142:145]

    # Double-blind submission: remove author/unit and author biography placeholders.
    remove_paragraph(author_front)
    remove_paragraph(stray)
    for p in author_back:
        remove_paragraph(p)

    # Move the required 800–1200-word English structured abstract to the end.
    for p in english:
        p._p.getparent().remove(p._p)
    body = doc._element.body
    sect_pr = body.sectPr
    for _ in range(2):
        blank = OxmlElement("w:p")
        body.insert(body.index(sect_pr), blank)
    for p in english:
        body.insert(body.index(sect_pr), p._p)

    # Insert the missing reference heading before reference 1.
    ref_heading = refs[0].insert_paragraph_before("参考文献")

    # Front matter and terminology fixes.
    title = doc.paragraphs[0]
    replace_across_text_nodes(title, "基于多视角超宽带脉冲雷达的开集手势识别方法", "多视角超宽带脉冲雷达开集手势识别方法")
    zh_abs = next(p for p in doc.paragraphs if p.text.startswith("摘要："))
    replace_across_text_nodes(zh_abs, "摘要：", "摘  要：", 1)
    replace_across_text_nodes(zh_abs, "超宽带雷达手势识别", "超宽带（Ultra-WideBand，UWB）雷达手势识别", 1)
    kw = next(p for p in doc.paragraphs if p.text.startswith("关键词："))
    replace_across_text_nodes(kw, "关键词：", "关键词：", 1)
    cls = next(p for p in doc.paragraphs if p.text.startswith("中图分类号"))
    replace_across_text_nodes(cls, cls.text, "中图分类号：TN951    文献标识码：A    文章编号：")
    doi = cls.insert_paragraph_before("DOI：")
    cls._p.addnext(doi._p)  # DOI follows the classification line.

    en_title = next(p for p in doc.paragraphs if p.text.startswith("Open-Set Gesture Recognition Based"))
    replace_across_text_nodes(en_title, en_title.text, "Open-Set Gesture Recognition with Multi-View UWB Impulse Radar")

    # Abbreviations at their first occurrence in the Chinese main text.
    p17 = next(p for p in doc.paragraphs if "如MBT[36]" in p.text)
    replace_across_text_nodes(p17, "如MBT", "如多模态瓶颈Transformer（Multimodal Bottleneck Transformer，MBT）", 1)
    p24 = next(p for p in doc.paragraphs if "同步采集的CIR距离" in p.text)
    replace_across_text_nodes(p24, "CIR距离-慢时间矩阵", "信道冲激响应（Channel Impulse Response，CIR）距离-慢时间矩阵", 1)
    p29 = next(p for p in doc.paragraphs if "每个视角的样本为信道冲激响应" in p.text)
    replace_across_text_nodes(p29, "信道冲激响应（CIR）距离-慢时间矩阵", "CIR距离-慢时间矩阵", 1)
    p30 = next(p for p in doc.paragraphs if "每层由4头自注意力" in p.text)
    replace_across_text_nodes(p30, "两层MLP", "两层多层感知机（Multi-Layer Perceptron，MLP）", 1)
    p58 = next(p for p in doc.paragraphs if "RTX 4060 Laptop" in p.text)
    replace_across_text_nodes(p58, "GPU", "图形处理器（Graphics Processing Unit，GPU）", 1)
    p60 = next(p for p in doc.paragraphs if "OSCR与AUROC" in p.text)
    replace_across_text_nodes(p60, "AUROC", "接收者操作特征曲线下面积（Area Under the Receiver Operating Characteristic Curve，AUROC）", 1)

    # Wording, list markers and unresolved equation cross-references.
    eom = next(p for p in doc.paragraphs if "Eom等[33]" in p.text)
    replace_across_text_nodes(eom, "Eom等", "Eom等人", 1)
    for p in doc.paragraphs:
        if "[eq:score]" in p.text:
            replace_across_text_nodes(p, "[eq:score]", "2")
    contrib = next(p for p in doc.paragraphs if "主要工作如下：1）" in p.text)
    for n in range(1, 4):
        replace_across_text_nodes(contrib, f"{n}）", f"({n}) ", 1)
    robust = next(p for p in doc.paragraphs if "随机丢视角训练则可以修复" in p.text)
    for n in range(1, 5):
        replace_across_text_nodes(robust, f"{n}）", f"({n}) ", 1)

    # Visible chapter numbering, preserving the journal's simple Arabic hierarchy.
    h1_num = 0
    h2_num = 0
    for p in doc.paragraphs:
        if p is ref_heading:
            continue
        if p.style.name == "Heading 1":
            h1_num += 1
            h2_num = 0
            if not re.match(r"^\d+\s", p.text):
                add_prefix_run(p, f"{h1_num}  ", size=14, bold=True)
        elif p.style.name == "Heading 2":
            h2_num += 1
            if not re.match(r"^\d+\.\d+\s", p.text):
                add_prefix_run(p, f"{h1_num}.{h2_num}  ", size=12, bold=True)

    # References numbered in citation order. Their text remains unchanged.
    for i, p in enumerate(refs, 1):
        add_prefix_run(p, f"[{i}] ", size=8)

    # Number the six standalone displayed equations without converting them to tables/images.
    usable_twips = int((Cm(21.0) - Inches(0.748) - Inches(0.748)) / 635)
    equation_paras = []
    for p in doc.paragraphs:
        has_math = bool(p._p.xpath("./m:oMath | ./m:oMathPara | .//m:oMath"))
        visible = "".join(p._p.xpath(".//w:t/text()"))
        if has_math and not visible.strip():
            equation_paras.append(p)
    if len(equation_paras) != 6:
        raise RuntimeError(f"Expected 6 standalone equations, found {len(equation_paras)}")
    for i, p in enumerate(equation_paras, 1):
        add_equation_number(p, i, usable_twips)

    # Global paragraph formatting, then special roles.
    special_elements = {title._p, zh_abs._p, kw._p, cls._p, doi._p, ref_heading._p, en_title._p}
    special_elements.update(p._p for p in english)
    special_elements.update(p._p for p in refs)
    special_elements.update(p._p for p in equation_paras)
    for p in doc.paragraphs:
        if p._p in special_elements:
            continue
        if p.style.name == "Heading 1":
            format_para(p, size=14, east_asia="黑体", align=WD_ALIGN_PARAGRAPH.CENTER, exact=20, before=12, after=6, first=0, bold=True)
            set_keep(p, keep_next=True)
        elif p.style.name == "Heading 2":
            format_para(p, size=12, east_asia="黑体", align=WD_ALIGN_PARAGRAPH.LEFT, exact=18, before=8, after=4, first=0, bold=True)
            set_keep(p, keep_next=True)
        elif p.style.name in {"Image Caption", "Table Caption"}:
            format_para(p, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, exact=14, before=3, after=3, first=0)
            for r in p.runs:
                r.italic = False
            set_keep(p, keep_next=p.style.name == "Table Caption")
        elif p.style.name == "Captioned Figure":
            format_para(p, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, exact=None, first=0)
            p.paragraph_format.keep_together = True
        elif not p.text.strip() and not p._p.xpath(".//w:drawing"):
            format_para(p, size=10.5, align=WD_ALIGN_PARAGRAPH.LEFT, exact=8, first=0)
        else:
            # The supplied journal template uses single spacing for ordinary
            # 5-size body text; fixed 16 pt is reserved for abstract blocks.
            format_para(p, exact=None)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.line_spacing = 1
            set_keep(p)

    format_para(title, size=15, east_asia="黑体", align=WD_ALIGN_PARAGRAPH.CENTER, exact=20, before=0, after=5, first=0, bold=True)
    set_keep(title, keep_next=True)
    format_para(zh_abs, size=9, align=WD_ALIGN_PARAGRAPH.JUSTIFY, exact=16, before=2, after=0, first=-21, left=21)
    format_para(kw, size=9, align=WD_ALIGN_PARAGRAPH.JUSTIFY, exact=16, before=0, after=0, first=-21, left=21)
    for p in (cls, doi):
        format_para(p, size=9, align=WD_ALIGN_PARAGRAPH.LEFT, exact=14, before=0, after=0, first=0)

    # Classification metadata: journal labels in bold, values in regular type.
    cls.clear()
    for value, is_bold in (
        ("中图分类号：", True), ("TN951    ", False),
        ("文献标识码：", True), ("A    ", False),
        ("文章编号：", True), ("", False),
    ):
        r = cls.add_run(value)
        set_fonts(r, 9, bold=is_bold)
    doi.clear()
    r = doi.add_run("DOI：")
    set_fonts(r, 9, bold=True)

    # Bold only the front labels.
    for p, label in ((zh_abs, "摘  要："), (kw, "关键词：")):
        for r in p.runs:
            if label in r.text:
                r.bold = True

    format_para(ref_heading, size=14, east_asia="黑体", align=WD_ALIGN_PARAGRAPH.CENTER, exact=20, before=12, after=6, first=0, bold=True)
    set_keep(ref_heading, keep_next=True)
    for p in refs:
        format_para(p, size=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY, exact=12, before=0, after=0, first=-18, left=18)
        set_keep(p, keep_lines=True)

    format_para(en_title, size=14, east_asia="Times New Roman", latin="Times New Roman", align=WD_ALIGN_PARAGRAPH.CENTER, exact=18, before=0, after=6, first=0, bold=True)
    set_keep(en_title, keep_next=True)
    for p in english[1:]:
        format_para(p, size=9, east_asia="Times New Roman", latin="Euclid", align=WD_ALIGN_PARAGRAPH.JUSTIFY, exact=16, before=0, after=0, first=0)
        # The long structured abstract is intentionally allowed to split across pages;
        # otherwise each long section is pushed wholesale to the next page.
        set_keep(p, keep_lines=False)
        if p.text in {"Abstract:"}:
            set_keep(p, keep_next=True, keep_lines=True)
            for r in p.runs:
                r.bold = True
        elif ":" in p.text:
            label = p.text.split(":", 1)[0] + ":"
            remaining = len(label)
            for r in p.runs:
                if remaining <= 0:
                    break
                if r.text:
                    r.bold = True
                    remaining -= len(r.text)

    # Compact, consistent three-line tables without changing their borders or values.
    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for row in table.rows:
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p in cell.paragraphs:
                    format_para(p, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER, exact=12, first=0)

    # Set style defaults to prevent hidden 9 pt before/after spacing from reappearing.
    for name in ("Normal", "Body Text", "First Paragraph"):
        if name in doc.styles:
            style = doc.styles[name]
            style.paragraph_format.space_before = Pt(0)
            style.paragraph_format.space_after = Pt(0)
            style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            style.paragraph_format.line_spacing = 1
            style.font.size = Pt(10.5)
            style.font.name = "Euclid"
            style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")

    # Keep figures, tables and captions from splitting awkwardly.
    for p in doc.paragraphs:
        if p._p.xpath(".//w:drawing"):
            p.paragraph_format.keep_together = True
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.line_spacing = 1
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(0)

    doc.core_properties.title = "多视角超宽带脉冲雷达开集手势识别方法"
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    settings = doc.settings.element
    update_fields = settings.find(qn("w:updateFields"))
    if update_fields is None:
        update_fields = OxmlElement("w:updateFields")
        settings.append(update_fields)
    update_fields.set(qn("w:val"), "true")
    doc.save(str(DST))

    output_comments = count_comment_markers(DST)
    if source_comments != output_comments:
        raise RuntimeError(f"Comment structure changed: {source_comments} -> {output_comments}")
    print(f"Saved: {DST}")
    print(f"Comments preserved: {output_comments}")


if __name__ == "__main__":
    main()
