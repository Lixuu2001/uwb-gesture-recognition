# -*- coding: utf-8 -*-
"""按正文首次引用顺序重排 thebibliography 中的 \bibitem(GB/T 7714 顺序编码制)。

同时报告:未被引用的条目、引用了但没有条目的 key。
用法: python tmp/reorder_bib.py mvosr_paper_cn.tex [--write]
"""
import re
import sys
from pathlib import Path

TEX = Path(sys.argv[1] if len(sys.argv) > 1 else "mvosr_paper_cn.tex")
WRITE = "--write" in sys.argv

src = TEX.read_text(encoding="utf-8")
head, sep, rest = src.partition(r"\begin{thebibliography}")
if not sep:
    sys.exit("找不到 thebibliography")
body, sep2, tail = rest.partition(r"\end{thebibliography}")

# 1) 正文里的引用顺序(\cite 与 \upcite 都算,支持逗号分隔多 key)
order, seen = [], set()
for m in re.finditer(r"\\(?:up)?cite\{([^}]*)\}", head):
    for k in (x.strip() for x in m.group(1).split(",")):
        if k.startswith("#"):  # \newcommand{\upcite}[1]{...\cite{#1}} 里的占位符
            continue
        if k and k not in seen:
            seen.add(k)
            order.append(k)

# 2) 拆出每个 \bibitem 块(含其后正文,直到下一个 \bibitem)
opening, _, items_src = body.partition(r"\bibitem")
items_src = r"\bibitem" + items_src
blocks = {}
for m in re.finditer(r"\\bibitem\{([^}]*)\}(.*?)(?=\\bibitem\{|\Z)", items_src, re.S):
    blocks[m.group(1)] = m.group(2).strip("\n")

missing = [k for k in order if k not in blocks]
uncited = [k for k in blocks if k not in seen]
print(f"正文引用 {len(order)} 个 key,条目 {len(blocks)} 条")
if missing:
    print("!! 引用了但没有 bibitem:", missing)
if uncited:
    print("!! 有 bibitem 但正文未引用:", uncited)

# 3) 按引用顺序重排;未被引用的保持原相对次序附在末尾(便于人工处置)
new_order = [k for k in order if k in blocks] + uncited
out = [opening.rstrip("\n")]
for k in new_order:
    out.append(f"\\bibitem{{{k}}}\n{blocks[k]}")
new_body = "\n\n".join(out) + "\n\n"

changed = [i for i, (a, b) in enumerate(zip(list(blocks), new_order)) if a != b]
print(f"重排后条目数 {len(new_order)};顺序{'有变化' if changed else '无变化'}")

if WRITE:
    TEX.write_text(head + sep + new_body + sep2 + tail, encoding="utf-8")
    print("已写回", TEX.name)
else:
    print("(试运行,加 --write 才写回)")
