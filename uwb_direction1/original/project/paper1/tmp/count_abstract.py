# -*- coding: utf-8 -*-
"""统计英文长摘要词数(细则:800~1200 词),并分段报告。"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

path = sys.argv[1]
t = open(path, encoding="utf-8").read()


def words(s):
    s = re.sub(r"\\[a-zA-Z]+", " ", s)      # LaTeX 命令
    s = re.sub(r"[{}$\\]", " ", s)          # 括号与转义
    return [w for w in s.split() if re.search(r"[A-Za-z0-9]", w)]


secs = ["Objective", "Methods", "Results and Discussions", "Conclusions"]
total = 0
parts = []
for i, sec in enumerate(secs):
    nxt = secs[i + 1] if i + 1 < len(secs) else None
    pat = re.escape(sec) + r":\}(.*?)" + (re.escape(nxt) + r":" if nxt else r"\Z")
    m = re.search(pat, t, re.S)
    n = len(words(m.group(1))) if m else 0
    parts.append((sec, n))
    total += n

print("英文长摘要 %d 词  (细则要求 800~1200)  %s" % (total, "✓" if 800 <= total <= 1200 else "✗"))
for sec, n in parts:
    print("   %-26s %4d 词" % (sec, n))
