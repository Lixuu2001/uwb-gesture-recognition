from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


SOURCE = Path(r"D:\uwb_gestures\paper1\mvosr_paper_cn_投稿版_12页终稿.docx")
OUTPUT = Path(
    r"D:\uwb_gestures\paper1\mvosr_paper_cn_投稿版_12页终稿_表格修正版.docx"
)

OUTER_RULE = 12  # eighths of a point: 1.5 pt
INNER_RULE = 6   # eighths of a point: 0.75 pt


def _set_border(parent_pr, edge, val, size=None):
    """Replace one table/cell border edge deterministically."""
    borders_tag = "w:tblBorders" if parent_pr.tag == qn("w:tblPr") else "w:tcBorders"
    borders = parent_pr.find(qn(borders_tag))
    if borders is None:
        borders = OxmlElement(borders_tag)
        parent_pr.append(borders)

    old = borders.find(qn(f"w:{edge}"))
    if old is not None:
        borders.remove(old)

    node = OxmlElement(f"w:{edge}")
    node.set(qn("w:val"), val)
    if size is not None:
        node.set(qn("w:sz"), str(size))
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), "000000")
    borders.append(node)


def _unique_cells(row):
    seen = set()
    result = []
    for cell in row.cells:
        key = id(cell._tc)
        if key not in seen:
            seen.add(key)
            result.append(cell)
    return result


def _clear_table_rules(table):
    for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
        _set_border(table._tbl.tblPr, edge, "nil")

    seen = set()
    for row in table.rows:
        for cell in row.cells:
            key = id(cell._tc)
            if key in seen:
                continue
            seen.add(key)
            for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
                _set_border(cell._tc.get_or_add_tcPr(), edge, "nil")


def _rule_on_row(table, row_index, edge, size):
    for cell in _unique_cells(table.rows[row_index]):
        _set_border(cell._tc.get_or_add_tcPr(), edge, "single", size)


def _group_rule(table, row_index, physical_column_indexes):
    seen = set()
    for column_index in physical_column_indexes:
        cell = table.cell(row_index, column_index)
        key = id(cell._tc)
        if key in seen:
            continue
        seen.add(key)
        _set_border(cell._tc.get_or_add_tcPr(), "bottom", "single", INNER_RULE)


def _format_grouped_header_table(table, group_start_columns):
    _clear_table_rules(table)
    _rule_on_row(table, 0, "top", OUTER_RULE)
    _group_rule(table, 0, group_start_columns)
    _rule_on_row(table, 1, "bottom", INNER_RULE)
    _rule_on_row(table, -1, "bottom", OUTER_RULE)


def _format_single_header_table(table):
    _clear_table_rules(table)
    _rule_on_row(table, 0, "top", OUTER_RULE)
    _rule_on_row(table, 0, "bottom", INNER_RULE)
    _rule_on_row(table, -1, "bottom", OUTER_RULE)


doc = Document(SOURCE)
if len(doc.tables) != 4:
    raise RuntimeError(f"Expected 4 tables, found {len(doc.tables)}")

# Table 1: two grouped column pairs; first and last cells span the two header rows.
_format_grouped_header_table(doc.tables[0], group_start_columns=(1, 3))

# Table 2: one header row.
_format_single_header_table(doc.tables[1])

# Table 3: two grouped column pairs; first two cells span the two header rows.
_format_grouped_header_table(doc.tables[2], group_start_columns=(2, 4))

# Table 4: one header row; body has vertically merged metric labels.
_format_single_header_table(doc.tables[3])

doc.save(OUTPUT)
print(OUTPUT)
