from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


SOURCE = Path(
    r"D:\uwb_gestures\paper1\mvosr_paper_cn_投稿版_12页终稿_表格修正版.docx"
)
OUTPUT = Path(
    r"D:\uwb_gestures\paper1\mvosr_paper_cn_投稿版_12页终稿_表格统一修正版.docx"
)


def replace_child(parent, tag):
    old = parent.find(qn(tag))
    if old is not None:
        parent.remove(old)
    node = OxmlElement(tag)
    parent.append(node)
    return node


def set_cell_bottom_nil(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    old = borders.find(qn("w:bottom"))
    if old is not None:
        borders.remove(old)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "nil")
    borders.append(bottom)


def add_trimmed_group_rule(cell):
    paragraph = cell.paragraphs[0]
    p_pr = paragraph._p.get_or_add_pPr()

    p_borders = p_pr.find(qn("w:pBdr"))
    if p_borders is None:
        p_borders = OxmlElement("w:pBdr")
        p_pr.append(p_borders)
    old = p_borders.find(qn("w:bottom"))
    if old is not None:
        p_borders.remove(old)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "000000")
    p_borders.append(bottom)

    ind = p_pr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        p_pr.append(ind)
    ind.set(qn("w:left"), "80")
    ind.set(qn("w:right"), "80")


def lock_table_geometry(table):
    grid_widths = [int(col.get(qn("w:w"))) for col in table._tbl.tblGrid]
    total_width = sum(grid_widths)
    tbl_pr = table._tbl.tblPr

    table.autofit = False

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.insert(0, tbl_w)
    tbl_w.set(qn("w:type"), "dxa")
    tbl_w.set(qn("w:w"), str(total_width))

    cell_mar = tbl_pr.find(qn("w:tblCellMar"))
    if cell_mar is None:
        cell_mar = OxmlElement("w:tblCellMar")
        tbl_pr.append(cell_mar)
    for edge, width in (("top", 0), ("left", 108), ("bottom", 0), ("right", 108)):
        node = cell_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            cell_mar.append(node)
        node.set(qn("w:type"), "dxa")
        node.set(qn("w:w"), str(width))

    for row in table._tbl.tr_lst:
        grid_index = 0
        for tc in row.tc_lst:
            tc_pr = tc.get_or_add_tcPr()
            span_node = tc_pr.find(qn("w:gridSpan"))
            span = int(span_node.get(qn("w:val"))) if span_node is not None else 1
            width = sum(grid_widths[grid_index : grid_index + span])
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.insert(0, tc_w)
            tc_w.set(qn("w:type"), "dxa")
            tc_w.set(qn("w:w"), str(width))
            grid_index += span


doc = Document(SOURCE)
if len(doc.tables) != 4:
    raise RuntimeError(f"Expected 4 tables, found {len(doc.tables)}")

for table in doc.tables:
    lock_table_geometry(table)

for table, columns in ((doc.tables[0], (1, 3)), (doc.tables[2], (2, 4))):
    for column in columns:
        cell = table.cell(0, column)
        set_cell_bottom_nil(cell)
        add_trimmed_group_rule(cell)

doc.save(OUTPUT)
print(OUTPUT)
