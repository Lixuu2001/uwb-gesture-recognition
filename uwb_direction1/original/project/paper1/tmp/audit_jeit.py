# -*- coding: utf-8 -*-
"""按《电子与信息学报》投稿细则(2023)体检参考文献表。

检查项:
  1. 近两年会议/期刊文献占比 >= 30%
  2. 条数 >= 15
  3. 中文文献是否给了对应英文著录
  4. 页码范围是否用短横(该刊排印用 en dash)
  5. 是否缺 doi
  6. 会议文献著录是否为该刊的 "[C]. 会议名, 城市, 国家, 年: 页码" 式
"""
import io
import re
import sys
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CURRENT_YEAR = 2026  # 投稿年份;"近两年"按 >= CURRENT_YEAR-2 计

tex = open("mvosr_paper_cn.tex", encoding="utf-8").read()
body = tex.split("\\begin{thebibliography}")[1].split("\\end{thebibliography}")[0]
items = re.findall(r"\\bibitem\{([^}]*)\}(.*?)(?=\\bibitem\{|\Z)", body, re.S)
print("参考文献 %d 条 (细则要求 >= 15)  %s\n" % (len(items), "✓" if len(items) >= 15 else "✗"))

years, cn, no_doi, conf_old, hyphen = {}, [], [], [], []
for k, t in items:
    t = " ".join(t.split())
    ys = [int(y) for y in re.findall(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)", t)]
    years[k] = max(ys) if ys else None
    if re.search(r"[\u4e00-\u9fff]", t):
        cn.append((k, t))
    if "doi" not in t.lower():
        no_doi.append(k)
    if "[C]//" in t or "[C] //" in t:
        conf_old.append(k)
    if re.search(r":\s*\d+-\d+", t):
        hyphen.append(k)

recent = [k for k, v in years.items() if v and v >= CURRENT_YEAR - 2]
pct = 100 * len(recent) / len(items)
print("【近两年文献占比】%d/%d = %.1f%%  (细则要求 >= 30%%)  %s"
      % (len(recent), len(items), pct, "✓" if pct >= 30 else "✗ 不达标"))
print("   %d 年及以后: %s" % (CURRENT_YEAR - 2, sorted(recent)))
print("   年份分布:", dict(sorted(Counter(years.values()).items(),
                                 key=lambda x: (x[0] is None, x[0]))))

print("\n【中文文献需附英文著录】共 %d 条中文文献:" % len(cn))
for k, t in cn:
    has_en = bool(re.search(r"[A-Z]{2,}\s+[A-Z][a-z]+", t))
    print("   %-22s %s" % (k, "已附英文 ✓" if has_en else "缺英文著录 ✗"))

print("\n【其他体例差异】")
print("   缺 doi 的条目: %d/%d" % (len(no_doi), len(items)))
print("   页码用半角连字符(该刊排印用 en dash): %d 条" % len(hyphen))
print("   会议著录用 [C]// 式(该刊用 '[C]. 会议名, 城市, 国家, 年: 页'): %d 条" % len(conf_old))
if conf_old:
    print("      ", conf_old)
