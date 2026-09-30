# -*- coding: utf-8 -*-
"""把 mvosr_paper_cn.tex 预处理成便于 pandoc 转 Word 的中间稿。

做三件事:
  1. 图 PDF -> 600 dpi PNG(Word 里显示用;矢量原图按细则另行打包上传)
  2. \\upcite{k,...} / \\cite{k} 按 .aux 里的编号就地展开成 [n] 上标或方括号
  3. 去掉只对 LaTeX 排版有意义、pandoc 会译坏的命令
"""
import io
import re
import shutil
import sys
from pathlib import Path

import fitz

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("mvosr_paper_cn.tex")
OUT = Path("tmp/_docx_src.tex")
IMGDIR = Path("tmp/_docx_img")
IMGDIR.mkdir(parents=True, exist_ok=True)

tex = SRC.read_text(encoding="utf-8")

# ---- 1. 图片转 PNG ----------------------------------------------------------
for m in set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)):
    src = Path("figs") / m
    if not src.exists():
        print("!! 缺图:", src)
        continue
    png = IMGDIR / (src.stem + ".png")
    d = fitz.open(src)
    d[0].get_pixmap(dpi=600).save(png)
    d.close()
    tex = tex.replace("{" + m + "}", "{" + png.as_posix() + "}")
    print("图 -> %s (600 dpi, %s)" % (png.name, src.name))

# ---- 2. 引用编号就地展开 -----------------------------------------------------
aux = Path("mvosr_paper_cn.aux").read_text(encoding="utf-8")
num = {k: int(v) for k, v in re.findall(r"\\bibcite\{([^}]*)\}\{(\d+)\}", aux)}


def nums(keys):
    out = []
    for k in (x.strip() for x in keys.split(",")):
        out.append(str(num.get(k, "?")))
        if k not in num:
            print("!! 未知引用 key:", k)
    return ",".join(out)


tex = re.sub(r"\\upcite\{([^}]*)\}", lambda m: r"\textsuperscript{[%s]}" % nums(m.group(1)), tex)
tex = re.sub(r"(?<!up)\\cite\{([^}]*)\}", lambda m: "[%s]" % nums(m.group(1)), tex)
# 上面的 \upcite 定义行已无用,删掉以免 pandoc 误解析
tex = re.sub(r"\\newcommand\{\\upcite\}.*\n", "", tex)

# ---- 2.5 简单数值退回纯文本 --------------------------------------------------
# 细则:"一般的简单参量,直接使用 Word 常规输入即可,复杂的公式、参量请使用 MathType"。
# 表格里的 $86.74\pm1.29$ 之类若留作公式,在 Word 里既难编辑又不合体例。
# 注意:范围号必须用 U+223C“∼”,不能用 ASCII 的 ~ —— 后者在 LaTeX 里是不断行空格,
# 会被 pandoc 当成空格吃掉,数字区间就断成了两个数。
UNI = {r"\pm": "±", r"\to": "→", r"\sim": "∼", r"\times": "×",
       r"\ge": "≥", r"\le": "≤", "-": "−"}
SIMPLE = re.compile(r"\$(\\mathbf\{)?([0-9.,+\-\s]|\\pm|\\to|\\sim|\\times|\\ge|\\le)+\}?\$")


def to_plain(m):
    s = m.group(0).strip("$")
    bold = s.startswith(r"\mathbf{")
    if bold:
        s = s[len(r"\mathbf{"):].rstrip("}")
    for k, v in UNI.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", "", s)
    return r"\textbf{%s}" % s if bold else s


n_simple = len(SIMPLE.findall(tex))
tex = SIMPLE.sub(to_plain, tex)
print("简单数值公式退回纯文本: %d 处" % n_simple)

# 表格里的 \cmidrule 会被 pandoc 当成单元格文字漏出来
tex = re.sub(r"\\cmidrule(\([a-z]*\))?\{[\d-]+\}", "", tex)

# ---- 3. 清掉纯排版命令 -------------------------------------------------------
tex = re.sub(r"\\(clearpage|raggedbottom)\b", "", tex)
tex = re.sub(r"\\vspace\{[^}]*\}", "", tex)
tex = re.sub(r"\\mbox\{([^}]*)\}", r"\1", tex)
tex = re.sub(r"\\captionsetup(\[[^\]]*\])?\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "", tex)

OUT.write_text(tex, encoding="utf-8")
print("中间稿 ->", OUT)
