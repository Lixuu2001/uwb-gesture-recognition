# -*- coding: utf-8 -*-
"""校验参考文献编号是否严格等于正文首次引用顺序(GB/T 7714 顺序编码制)。"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

tex = open("mvosr_paper_cn.tex", encoding="utf-8").read()
head = tex.split(r"\begin{thebibliography}")[0]

order, seen = [], set()
for m in re.finditer(r"\\(up)?cite\{([^}]*)\}", head):
    for k in (x.strip() for x in m.group(2).split(",")):
        if k and not k.startswith("#") and k not in seen:
            seen.add(k)
            order.append(k)

aux = open("mvosr_paper_cn.aux", encoding="utf-8").read()
num = {m.group(1): int(m.group(2)) for m in re.finditer(r"\\bibcite\{([^}]*)\}\{(\d+)\}", aux)}

bad = [(k, num.get(k), i + 1) for i, k in enumerate(order) if num.get(k) != i + 1]
print("正文首次引用 %d 个 key" % len(order))
print("编号与首次引用顺序不符:", bad if bad else "无 ✓")
print("参考文献总数:", len(num))

cn_keys = ["wang2019multiparam", "han2022jeit", "liyao2021",
           "jin2022uwb", "wang2018jr", "dong2023bit", "gaofei2022"]
cn = sorted((num[k], k) for k in cn_keys if k in num)
print("中文文献 %d 篇,占比 %.0f%%:" % (len(cn), 100 * len(cn) / len(num)))
for n, k in cn:
    print("   [%d] %s" % (n, k))
