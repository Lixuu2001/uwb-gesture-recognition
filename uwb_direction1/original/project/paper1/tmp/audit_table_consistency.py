from pathlib import Path
from collections import Counter

from docx import Document
from docx.oxml.ns import qn


DOCX = Path(r"D:\uwb_gestures\paper1\mvosr_paper_cn_投稿版_12页终稿_表格修正版.docx")


def attr(node, name):
    return node.get(qn(name)) if node is not None else None


def border(tc, edge):
    borders = tc.tcPr.find(qn("w:tcBorders"))
    node = borders.find(qn(f"w:{edge}")) if borders is not None else None
    return (attr(node, "w:val"), attr(node, "w:sz")) if node is not None else None


def effective_run_format(run):
    rpr = run._r.rPr
    rfonts = rpr.find(qn("w:rFonts")) if rpr is not None else None
    return (
        run.font.size.pt if run.font.size else None,
        bool(run.bold),
        attr(rfonts, "w:eastAsia"),
        attr(rfonts, "w:ascii"),
    )


doc = Document(DOCX)
for ti, table in enumerate(doc.tables, 1):
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    grid = [int(attr(col, "w:w")) for col in table._tbl.tblGrid]
    table_cell_mar = tbl_pr.find(qn("w:tblCellMar"))
    margins = {}
    if table_cell_mar is not None:
        for edge in ("top", "left", "bottom", "right"):
            node = table_cell_mar.find(qn(f"w:{edge}"))
            margins[edge] = attr(node, "w:w") if node is not None else None

    print(f"TABLE {ti}")
    print(
        f"  alignment={table.alignment} autofit={table.autofit} "
        f"tblW=({attr(tbl_w, 'w:type')},{attr(tbl_w, 'w:w')}) "
        f"tblInd=({attr(tbl_ind, 'w:type')},{attr(tbl_ind, 'w:w')}) "
        f"grid={grid} total={sum(grid)} margins={margins}"
    )

    row_meta = []
    for row in table.rows:
        tr_pr = row._tr.trPr
        tr_h = tr_pr.find(qn("w:trHeight")) if tr_pr is not None else None
        row_meta.append((attr(tr_h, "w:val"), attr(tr_h, "w:hRule")))
    print(f"  row_heights={row_meta}")

    paragraph_formats = Counter()
    run_formats = Counter()
    vertical_alignments = Counter()
    cell_margins = Counter()
    for row in table.rows:
        seen = set()
        for cell in row.cells:
            key = id(cell._tc)
            if key in seen:
                continue
            seen.add(key)
            vertical_alignments[str(cell.vertical_alignment)] += 1
            tc_mar = cell._tc.tcPr.find(qn("w:tcMar"))
            cm = []
            for edge in ("top", "left", "bottom", "right"):
                node = tc_mar.find(qn(f"w:{edge}")) if tc_mar is not None else None
                cm.append(attr(node, "w:w") if node is not None else None)
            cell_margins[tuple(cm)] += 1
            for p in cell.paragraphs:
                pf = p.paragraph_format
                paragraph_formats[(
                    str(p.alignment),
                    pf.space_before.pt if pf.space_before else None,
                    pf.space_after.pt if pf.space_after else None,
                    pf.line_spacing,
                )] += 1
                for run in p.runs:
                    if run.text:
                        run_formats[effective_run_format(run)] += 1

    print(f"  vertical_alignments={dict(vertical_alignments)}")
    print(f"  cell_margins={dict(cell_margins)}")
    print(f"  paragraph_formats={dict(paragraph_formats)}")
    print(f"  run_formats={dict(run_formats)}")
    print("  top=" + str([border(cell._tc, "top") for cell in table.rows[0].cells]))
    print("  header_bottom=" + str([border(cell._tc, "bottom") for cell in table.rows[0].cells]))
    print("  last_bottom=" + str([border(cell._tc, "bottom") for cell in table.rows[-1].cells]))
    print()
