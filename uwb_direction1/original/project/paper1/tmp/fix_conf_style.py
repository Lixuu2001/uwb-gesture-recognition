# -*- coding: utf-8 -*-
"""把会议文献由 GB/T 7714 的 [C]//论文集 式改成《电子与信息学报》的
   "[C]. 会议名, 城市, 国家, 年: 页码." 式。

细则:"会议论文...要给出举办国家和城市,线上会议用[C/OL]表示,可以不给出举办地"。
下表按会议届次逐条填写举办地;因疫情线上举办的按 [C/OL] 处理、不填地点。
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# key -> (会议名, 举办地或 None 表示线上)
VENUE = {
    "bendale2016":     ("The IEEE Conference on Computer Vision and Pattern Recognition", "Las Vegas, USA"),
    "hendrycks2017":   ("The 5th International Conference on Learning Representations", "Toulon, France"),
    "vaze2022":        ("The 10th International Conference on Learning Representations", None),
    "nagrani2021":     ("The 35th Conference on Neural Information Processing Systems", None),
    "mm25":            ("The 33rd ACM International Conference on Multimedia", "Dublin, Ireland"),
    "lopezpaz2016":    ("The 4th International Conference on Learning Representations", "San Juan, Puerto Rico"),
    "vaswani2017":     ("The 31st Conference on Neural Information Processing Systems", "Long Beach, USA"),
    "dosovitskiy2021": ("The 9th International Conference on Learning Representations", None),
    "li2021simple":    ("The IEEE/CVF International Conference on Computer Vision", None),
    "verma2019":       ("The 36th International Conference on Machine Learning", "Long Beach, USA"),
    "zhang2018mixup":  ("The 6th International Conference on Learning Representations", "Vancouver, Canada"),
    "han2021":         ("The 9th International Conference on Learning Representations", None),
    "dhamija2018":     ("The 32nd Conference on Neural Information Processing Systems", "Montreal, Canada"),
    "he2022mae":       ("The IEEE/CVF Conference on Computer Vision and Pattern Recognition", "New Orleans, USA"),
}

P = "mvosr_paper_cn.tex"
s = open(P, encoding="utf-8").read()
head, sep, rest = s.partition(r"\begin{thebibliography}")
body, sep2, tail = rest.partition(r"\end{thebibliography}")

# 每条 [C]//<论文集名>. <年>[: <页码>].  ->  [C]. <会议名>, <地点>, <年>[: <页码>].
tailpat = re.compile(r"\[C\]//.*?\.\s*(\d{4})(\s*:\s*[\d-]+(?:--[\d]+)?)?\s*\.", re.S)
done, skipped = [], []

for key, (conf, place) in VENUE.items():
    m = re.search(r"(\\bibitem\{" + re.escape(key) + r"\})(.*?)(?=\\bibitem\{|\Z)", body, re.S)
    if not m:
        skipped.append(key)
        continue
    entry = m.group(2)
    tm = tailpat.search(entry)
    if not tm:
        skipped.append(key)
        continue
    year, pages = tm.group(1), (tm.group(2) or "").strip()
    marker = "[C/OL]" if place is None else "[C]"
    loc = "" if place is None else ", " + place
    new_tail = "%s. %s%s, %s%s." % (marker, conf, loc, year, pages)
    body = body[:m.start(2)] + entry[:tm.start()] + new_tail + entry[tm.end():] + body[m.end(2):]
    done.append(key)

open(P, "w", encoding="utf-8").write(head + sep + body + sep2 + tail)
print("已转换 %d 条会议文献" % len(done))
print("  线上会议按 [C/OL] 处理:", [k for k in done if VENUE[k][1] is None])
if skipped:
    print("!! 未处理:", skipped)
