# -*- coding: utf-8 -*-
"""把审阅版 PDF 的批注导出为 Markdown,并把每条批注定位到 .tex 的行号。

用法: python tmp/dump_annots.py [annotated.pdf] [source.tex]
"""
import re
import subprocess
import sys
from pathlib import Path

import fitz

PDF = Path(sys.argv[1] if len(sys.argv) > 1 else "mvosr_paper_cn-0803.pdf")
TEX = Path(sys.argv[2] if len(sys.argv) > 2 else "mvosr_paper_cn.tex")
OUT = PDF.with_suffix("").with_name(PDF.stem + "_批注.md")

tex_lines = TEX.read_text(encoding="utf-8").splitlines()


def squeeze(s):
    """去掉空白与 LaTeX 命令,只留可比对的裸字符,供模糊定位用。"""
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", "", s)
    return re.sub(r"[\s\u3000{}$~\\]+", "", s)


tex_squeezed = [squeeze(l) for l in tex_lines]


def locate(quote, min_len=4):
    """用高亮原文在 tex 里找行号。

    PDF 抽取会把上标引用打散、把行末词序打乱,所以不整段比对,而是切成连续
    汉字块后拿最长的块去找,失败再退到更短的块。
    """
    chunks = [squeeze(c) for c in re.findall(r"[一-鿿，。；：、（）]{%d,}" % min_len, quote)]
    chunks = sorted({c for c in chunks if len(c) >= min_len}, key=len, reverse=True)
    for probe in chunks:
        hits = [i + 1 for i, l in enumerate(tex_squeezed) if probe in l]
        if hits:
            return hits[0], probe
    # 退路:整段前缀匹配(纯数字/英文的高亮走这里)
    q = squeeze(quote)
    for n in range(len(q), min_len - 1, -1):
        hits = [i + 1 for i, l in enumerate(tex_squeezed) if q[:n] in l]
        if hits:
            return hits[0], q[:n]
    return None, ""


def locate_synctex(pno, rect):
    """首选:用 synctex 按批注坐标反查行号(需 <pdf名>.synctex.gz 与 PDF 同目录)。"""
    x, y = rect.x0 + 2, (rect.y0 + rect.y1) / 2
    try:
        out = subprocess.run(["synctex", "edit", "-o", f"{pno}:{x}:{y}:{PDF.name}"],
                             capture_output=True, text=True, cwd=PDF.parent).stdout
    except OSError:
        return None
    hits = [int(l.split(":", 1)[1]) for l in out.splitlines() if l.startswith("Line:")]
    return min(hits) if hits else None


rows = []
doc = fitz.open(PDF)
for pno in range(doc.page_count):
    page = doc[pno]
    for a in page.annots() or []:
        kind = a.type[1]
        # 高亮/删除线覆盖的正文原文
        quote = ""
        if a.vertices:
            quads = [fitz.Quad(a.vertices[i:i + 4]) for i in range(0, len(a.vertices), 4)]
            quote = " ".join(page.get_textbox(q.rect).replace("\n", " ") for q in quads)
        quote = re.sub(r"\s+", " ", quote).strip()
        line = locate_synctex(pno + 1, a.rect)
        if line is None and quote:
            line, _ = locate(quote)
        rows.append(dict(page=pno + 1, kind=kind, quote=quote,
                         comment=(a.info.get("content") or "").strip(), line=line))

KIND_CN = {"Highlight": "高亮", "StrikeOut": "删除线", "Underline": "下划线",
           "Text": "便签", "Squiggly": "波浪线"}

md = [f"# {PDF.name} 批注汇总", "",
      f"共 {len(rows)} 条。行号指向 `{TEX.name}`（源码自 {TEX.name} 未变，行号可直接用）。", ""]
for i, r in enumerate(rows, 1):
    loc = f"{TEX.name}:{r['line']}" if r["line"] else "（未定位）"
    md.append(f"## {i}. p{r['page']} · {KIND_CN.get(r['kind'], r['kind'])} · {loc}")
    if r["quote"]:
        md.append(f"> 原文：{r['quote']}")
    md.append("")
    md.append(f"**批注**：{r['comment'] or '（无文字，仅标记）'}")
    md.append("")

OUT.write_text("\n".join(md), encoding="utf-8")
print(f"wrote {OUT}  ({len(rows)} annots, {sum(r['line'] is None for r in rows)} unlocated)")
