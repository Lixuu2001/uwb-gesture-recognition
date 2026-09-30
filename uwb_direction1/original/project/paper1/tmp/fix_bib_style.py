# -*- coding: utf-8 -*-
"""把参考文献表里的页码范围连字符改成 en dash(该刊排印体例)。

只动 thebibliography 内、且紧跟在 ": " 之后的 数字-数字,
因此不会碰到 DOI 里的连字符(如 j.tbit1001-0645)或 Ultra-WideBand 这类词。
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
P = "mvosr_paper_cn.tex"
s = open(P, encoding="utf-8").read()

head, sep, rest = s.partition(r"\begin{thebibliography}")
body, sep2, tail = rest.partition(r"\end{thebibliography}")

pat = re.compile(r"(:\s*\d+)-(\d+)")
n = len(pat.findall(body))
body = pat.sub(r"\1--\2", body)

# 跨行断开的页码范围,如 "395-\n411"
pat2 = re.compile(r"(:\s*\d+)-(\s*\n\s*)(\d+)")
n2 = len(pat2.findall(body))
body = pat2.sub(r"\1--\2\3", body)

open(P, "w", encoding="utf-8").write(head + sep + body + sep2 + tail)
print("页码范围改为 en dash: %d 处(含跨行 %d 处)" % (n + n2, n2))
