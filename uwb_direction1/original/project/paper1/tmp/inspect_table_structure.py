from pathlib import Path
import sys

from docx import Document
from docx.oxml.ns import qn


SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\uwb_gestures\paper1\mvosr_paper_cn_投稿版_12页终稿.docx"
)


def border_summary(tc):
    tc_pr = tc.tcPr
    tc_borders = tc_pr.find(qn("w:tcBorders"))
    if tc_borders is None:
        return "-"
    parts = []
    for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
        node = tc_borders.find(qn(f"w:{edge}"))
        if node is not None:
            parts.append(
                f"{edge}={node.get(qn('w:val'))}:{node.get(qn('w:sz'))}"
            )
    return ",".join(parts) or "-"


doc = Document(SOURCE)
for ti, table in enumerate(doc.tables, 1):
    print(f"TABLE {ti}: {len(table.rows)} rows x {len(table.columns)} cols")
    for ri, row in enumerate(table.rows):
        seen = set()
        cells = []
        for ci, cell in enumerate(row.cells):
            tc = cell._tc
            key = id(tc)
            if key in seen:
                continue
            seen.add(key)
            tc_pr = tc.tcPr
            grid_span = tc_pr.find(qn("w:gridSpan"))
            vmerge = tc_pr.find(qn("w:vMerge"))
            span = grid_span.get(qn("w:val")) if grid_span is not None else "1"
            vm = vmerge.get(qn("w:val")) if vmerge is not None else "-"
            text = " ".join(cell.text.split())
            cells.append(
                f"c{ci}[span={span},vm={vm},text={text!r},bdr={border_summary(tc)}]"
            )
        print(f"  r{ri}: " + " | ".join(cells))
    print()
