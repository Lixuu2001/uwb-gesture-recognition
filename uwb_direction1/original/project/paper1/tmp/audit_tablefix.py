from pathlib import Path
import hashlib
import re
import sys
import zipfile

from docx import Document
from lxml import etree


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\uwb_gestures\paper1\mvosr_paper_cn_投稿版_12页终稿.docx"
)
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    r"D:\uwb_gestures\paper1\mvosr_paper_cn_投稿版_12页终稿_表格修正版.docx"
)


def zip_xml(path, member):
    with zipfile.ZipFile(path) as zf:
        return etree.fromstring(zf.read(member))


def text_nodes(path):
    root = zip_xml(path, "word/document.xml")
    return root.xpath("//w:t/text() | //w:instrText/text()", namespaces=NS)


def count_comment_nodes(path):
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        comments = 0
        if "word/comments.xml" in names:
            root = etree.fromstring(zf.read("word/comments.xml"))
            comments = len(root.xpath("//w:comment", namespaces=NS))
        doc_root = etree.fromstring(zf.read("word/document.xml"))
        starts = len(doc_root.xpath("//w:commentRangeStart", namespaces=NS))
        ends = len(doc_root.xpath("//w:commentRangeEnd", namespaces=NS))
        refs = len(doc_root.xpath("//w:commentReference", namespaces=NS))
    return comments, starts, ends, refs


def media_hashes(path):
    result = {}
    with zipfile.ZipFile(path) as zf:
        for name in sorted(n for n in zf.namelist() if n.startswith("word/media/")):
            result[name] = hashlib.sha256(zf.read(name)).hexdigest()
    return result


src_doc = Document(SOURCE)
out_doc = Document(OUTPUT)
src_text = text_nodes(SOURCE)
out_text = text_nodes(OUTPUT)

paragraphs = [p.text for p in out_doc.paragraphs]
reference_numbers = []
for text in paragraphs:
    m = re.match(r"^\[(\d+)\]", text.strip())
    if m:
        reference_numbers.append(int(m.group(1)))

captions = [
    text for text in paragraphs
    if re.match(r"^[图表]\s*\d+", text.strip())
]

print(f"text_nodes_identical={src_text == out_text} count={len(out_text)}")
print(f"tables={len(out_doc.tables)} inline_shapes={len(out_doc.inline_shapes)}")
print(f"comments={count_comment_nodes(OUTPUT)}")
print(f"comments_same={count_comment_nodes(SOURCE) == count_comment_nodes(OUTPUT)}")
print(f"media_same={media_hashes(SOURCE) == media_hashes(OUTPUT)} media={len(media_hashes(OUTPUT))}")
print(
    "references="
    f"{len(reference_numbers)} first={reference_numbers[:1]} last={reference_numbers[-1:]} "
    f"continuous={reference_numbers == list(range(1, 53))}"
)
for caption in captions:
    print(f"caption={caption!r} has_ideographic_space={chr(0x3000) in caption}")

if not src_text == out_text:
    raise SystemExit("Document text changed unexpectedly")
if len(out_doc.tables) != 4 or len(out_doc.inline_shapes) != 2:
    raise SystemExit("Table/image count changed unexpectedly")
if count_comment_nodes(OUTPUT) != (13, 13, 13, 13):
    raise SystemExit("Comment structure changed unexpectedly")
if media_hashes(SOURCE) != media_hashes(OUTPUT):
    raise SystemExit("Embedded media changed unexpectedly")
if reference_numbers != list(range(1, 53)):
    raise SystemExit("Reference numbering changed unexpectedly")
