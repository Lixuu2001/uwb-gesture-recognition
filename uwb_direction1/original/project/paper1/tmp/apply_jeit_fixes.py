# -*- coding: utf-8 -*-
"""落实《电子与信息学报》细则中剩余的正文侧体例项。"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

P = "mvosr_paper_cn.tex"
ABS = sys.argv[1]
s = open(P, encoding="utf-8").read()
log = []

# 1) 用 800~1200 词的四段式长英文摘要替换原短摘要
new_abs = open(ABS, encoding="utf-8").read().strip()
s2 = re.sub(r"\\noindent\\textbf\{Abstract:\}.*?(?=\n\n)",
            lambda m: new_abs, s, count=1, flags=re.S)  # lambda:避免替换串里的反斜杠被当转义
log.append("英文长摘要已替换" if s2 != s else "!! 英文摘要未替换")
s = s2

# 2) 中文摘要中的"本文"改"该文"(细则:摘要以第三人称,不能出现"本文")
head, sep, rest = s.partition("\\vspace{4pt}\n\\noindent\\textbf{关键词：}")
n = head.count("本文")
head = head.replace("本文", "该文")
s = head + sep + rest
log.append("中文摘要 本文->该文: %d 处" % n)

# 3) 末章"结论"改"结束语"
if "\\section{结论}" in s:
    s = s.replace("\\section{结论}", "\\section{结束语}")
    log.append("末章 结论->结束语")

# 4) 通信作者与基金页脚
if "通信作者" not in s:
    s = s.replace(
        "\\author{（作者姓名待填）\\\\[2pt]\n{\\small （作者单位待填，城市\\quad 邮编）}}",
        "\\author{（作者姓名待填）\\\\[2pt]\n{\\small （作者单位待填，城市\\quad 邮编）}\n"
        "\\thanks{*通信作者：（姓名待填）\\quad （Email待填）\\\\\n"
        "基金项目：（基金名称与编号待填）\\\\\n"
        "Foundation Item: （英文基金信息待填）}}")
    log.append("已加通信作者/基金页脚占位")

# 5) 作者简介(细则:列于参考文献之后,硕士生/博士生须写明)
if "作者简介" not in s:
    s = s.replace(
        "\\end{thebibliography}",
        "\\end{thebibliography}\n\n"
        "% ===========================================================================\n"
        "% 作者简介(细则要求列于参考文献后,全部作者;硕士、硕士生、博士、博士生须写清楚)\n"
        "% ===========================================================================\n"
        "\\vspace{6pt}\n"
        "\\noindent\\textbf{作者简介}\n\n"
        "\\noindent （第1作者姓名）：（性别），（职称/学位,如\"硕士生\"），研究方向为（待填）.\n\n"
        "\\noindent （第2作者姓名）：（性别），（职称/学位），研究方向为（待填）.\n")
    log.append("已加作者简介占位")

open(P, "w", encoding="utf-8").write(s)
for line in log:
    print(" ", line)
