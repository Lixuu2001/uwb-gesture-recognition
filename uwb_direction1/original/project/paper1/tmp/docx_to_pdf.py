# -*- coding: utf-8 -*-
"""用 Word COM 把 docx 导出成 PDF,便于检查排版效果。"""
import io
import os
import sys

import win32com.client

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

src = os.path.abspath(sys.argv[1])
dst = os.path.abspath(sys.argv[2])
markup = "--markup" in sys.argv          # 连右侧批注栏一起导出

# DispatchEx 而非 Dispatch:必须起独立实例,否则末尾的 Quit() 会关掉用户自己开着的 Word
word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    d = word.Documents.Open(src, ReadOnly=True)
    print("页数:", d.ComputeStatistics(2))     # wdStatisticPages
    print("字数:", d.ComputeStatistics(0))     # wdStatisticWords
    print("批注数:", d.Comments.Count)
    # Item=7 是 wdExportDocumentWithMarkup,把批注画到右侧栏
    d.ExportAsFixedFormat(dst, 17, False, 0, 0, 1, 1, 7 if markup else 0)
    d.Close(False)
finally:
    word.Quit()
print("PDF ->", dst)
