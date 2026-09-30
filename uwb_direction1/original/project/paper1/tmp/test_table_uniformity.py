from pathlib import Path
import sys

from docx import Document
from docx.oxml.ns import qn


DOCX = Path(sys.argv[1])
doc = Document(DOCX)


def edge(parent, borders_name, edge_name):
    borders = parent.find(qn(borders_name))
    return borders.find(qn(edge_name)) if borders is not None else None


def attr(node, name):
    return node.get(qn(name)) if node is not None else None


assert len(doc.tables) == 4

for index, table in enumerate(doc.tables, 1):
    layout = table._tbl.tblPr.find(qn("w:tblLayout"))
    width = table._tbl.tblPr.find(qn("w:tblW"))
    assert attr(layout, "w:type") == "fixed", f"table {index}: layout is not fixed"
    assert attr(width, "w:type") == "dxa", f"table {index}: width is not explicit DXA"
    assert int(attr(width, "w:w")) > 0, f"table {index}: explicit width is missing"

# Adjacent grouped headers must use trimmed paragraph rules, not touching cell borders.
group_cells = (
    (doc.tables[0], (1, 3)),
    (doc.tables[2], (2, 4)),
)
for table, columns in group_cells:
    for column in columns:
        cell = table.cell(0, column)
        cell_bottom = edge(cell._tc.tcPr, "w:tcBorders", "w:bottom")
        assert attr(cell_bottom, "w:val") == "nil", "group cell bottom rule still touches its neighbor"

        paragraph = cell.paragraphs[0]
        p_pr = paragraph._p.get_or_add_pPr()
        p_bottom = edge(p_pr, "w:pBdr", "w:bottom")
        assert attr(p_bottom, "w:val") == "single", "trimmed group paragraph rule missing"
        assert attr(p_bottom, "w:sz") == "6", "group rule has the wrong weight"

        ind = p_pr.find(qn("w:ind"))
        left = int(attr(ind, "w:left") or 0)
        right = int(attr(ind, "w:right") or 0)
        assert left >= 50 and right >= 50, "group rule is not visibly trimmed"

print("PASS: table geometry is fixed and grouped header rules are visibly separated")
