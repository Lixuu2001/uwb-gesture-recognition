# -*- coding: utf-8 -*-
"""把审阅版 PDF 加宽出右侧批注栏,让画在页面外的批注气泡完整显示。

审阅软件(Acrobat/福昕等)习惯把 popup 放在纸面右侧之外,而多数阅读器
(pdf.js / VS Code 内置预览等)按页面边界裁剪,导致气泡只露一条。
这里只把 MediaBox 的右边界外推,正文在页面坐标系里位置不变,
因此原有的 .synctex.gz 仍然可用(脚本会自动复制一份同名索引)。

用法: python tmp/widen_for_annots.py [annotated.pdf] [--pad 24]
"""
import shutil
import sys
from pathlib import Path

import fitz

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PDF = Path(args[0] if args else "mvosr_paper_cn-0803.pdf")
PAD = 24.0
if "--pad" in sys.argv:
    PAD = float(sys.argv[sys.argv.index("--pad") + 1])

OUT = PDF.with_name(PDF.stem + "_批注可读.pdf")

doc = fitz.open(PDF)

# 1) 量出所有气泡需要多宽的右侧栏
need_right = 0.0
for page in doc:
    for a in page.annots() or []:
        need_right = max(need_right, a.popup_rect.x1)
if need_right <= doc[0].rect.x1:
    print("所有气泡都在页内,无需加宽")
    sys.exit(0)

new_w = need_right + PAD
print(f"页宽 {doc[0].rect.x1:.1f} -> {new_w:.1f} pt (气泡最右 {need_right:.1f})")

moved = 0
for pno, page in enumerate(doc, 1):
    h = page.rect.y1
    # 2) 只向右外推 MediaBox:PDF 原点在左下,左边界和上边界都不动,
    #    所以正文相对"页面左上角"的坐标不变,synctex 映射保持有效
    page.set_mediabox(fitz.Rect(0, 0, new_w, h))

    # 3) 个别气泡竖直方向也出界,夹回页内
    for a in page.annots() or []:
        r = a.popup_rect
        dy = 0.0
        if r.y1 > h:
            dy = h - r.y1
        elif r.y0 < 0:
            dy = -r.y0
        if dy:
            a.set_popup(fitz.Rect(r.x0, r.y0 + dy, r.x1, r.y1 + dy))
            a.update()
            moved += 1

print(f"竖直方向夹回 {moved} 个气泡")
doc.save(OUT, garbage=3, deflate=True)
doc.close()

# 4) 让加宽版也能双击跳转:synctex 按 PDF 主名找索引
src_syn = PDF.with_suffix(".synctex.gz")
if src_syn.exists():
    dst_syn = OUT.with_suffix("").with_suffix(".synctex.gz")
    shutil.copyfile(src_syn, dst_syn)
    print(f"已复制 synctex 索引 -> {dst_syn.name}")

print(f"输出: {OUT.name}")
