# -*- coding: utf-8 -*-
"""按《电子与信息学报》投稿细则(2025)体检正文侧的体例项。"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t = open("mvosr_paper_cn.tex", encoding="utf-8").read()


def grab(pat):
    m = re.search(pat, t, re.S)
    return m.group(1) if m else ""


zh = grab(r"摘\\quad 要：\}(.*?)\n")
# 长摘要为多段带标签结构,从 Abstract: 一直取到 Key words 之前
en = grab(r"\\textbf\{Abstract:\}(.*?)\\noindent\\textbf\{Key words:\}")
en = re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", en)
kw = grab(r"关键词：\}(.*?)\n")
title = grab(r"\\title\{[^{}]*\{?([^{}\\]*)\}")

print("标题:", title.strip())
print("   细则:标题尽量避免“基于……”句式  ->", "✗ 以“基于”开头" if title.strip().startswith("基于") else "✓")
print()
print("中文摘要 %d 字" % len(re.sub(r"[\\{}a-zA-Z0-9\s%.±~]", "", zh)))
print("   出现“本文” %d 处  -> %s" % (zh.count("本文"), "✗ 细则要求改用“该文”" if zh.count("本文") else "✓"))
print("英文摘要 %d 词  -> %s" % (len(en.split()), "✗ 细则要求 800~1200 词的长摘要" if len(en.split()) < 800 else "✓"))
print("   含 Objective/Methods/Results and Discussions/Conclusions 分节标签 ->",
      "✓" if "Objective" in en else "✗ 缺")
print()
print("关键词 %d 个 (细则 3~5)  %s" % (len(kw.split("；")), "✓" if 3 <= len(kw.split("；")) <= 5 else "✗"))
print("末章节名:", "结束语 ✓" if "\\section{结束语}" in t else ("结论 (细则用“结束语”)" if "\\section{结论}" in t else "?"))
print("作者简介:", "有 ✓" if "作者简介" in t else "✗ 缺(细则要求列于参考文献后,须写明硕士生/博士生)")
print("基金标注:", "有" if re.search(r"基金|thanks|footnote", t) else "✗ 缺(如有资助须首页页脚标注,并给英文)")
print("通信作者标注:", "有" if "通信作者" in t else "✗ 缺(首页页脚)")
print()
print("全文“本文”共 %d 处(摘要内必须改,正文可保留)" % t.count("本文"))
