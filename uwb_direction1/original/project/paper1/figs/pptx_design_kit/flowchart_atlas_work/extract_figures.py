import json
import re
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw, ImageFont


WORK = Path(__file__).resolve().parent
SOURCE_DIR = WORK / "sources"
OUT_DIR = WORK / "figure_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def caption_regex(label: str) -> re.Pattern:
    number = re.search(r"(\d+)", label).group(1)
    return re.compile(rf"\b(?:Figure|Fig\.?)[ \t]*{number}[\.:]", re.IGNORECASE)


def render_clip(page, clip, output: Path, dpi: int = 190) -> None:
    page.crop(clip).to_image(resolution=dpi, antialias=True).original.convert("RGB").save(output)


def find_caption_block(page, pattern: re.Pattern):
    hits = page.search(pattern, regex=True, return_chars=False)
    return hits[0] if hits else None


def crop_for_caption(page, block):
    width, height = page.width, page.height
    x0, y0, x1, y1 = block["x0"], block["top"], block["x1"], block["bottom"]
    block_width = x1 - x0
    if block_width > width * 0.58:
        left, right = 20, width - 20
        top = max(18, y0 - height * 0.46)
    elif x0 < width / 2:
        left, right = 20, width / 2 - 8
        top = max(18, y0 - height * 0.38)
    else:
        left, right = width / 2 + 8, width - 20
        top = max(18, y0 - height * 0.38)
    bottom = min(height - 18, y1 + 12)
    return (left, top, right, bottom)


def make_contact_sheet(records):
    thumbs = []
    for record in records:
        if not record.get("output"):
            continue
        path = Path(record["output"])
        if not path.is_file():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((520, 330))
        canvas = Image.new("RGB", (560, 390), "white")
        canvas.paste(img, ((560 - img.width) // 2, 18))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 350), f'{record["id"]} · {record["figure"]} · p.{record["page"]}', fill="#172B4D")
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 560, rows * 390), "#E9EEF4")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 560, (index // cols) * 390))
    sheet.save(OUT_DIR / "contact_sheet.png")


def main():
    manifest = json.loads((WORK / "sources_manifest.json").read_text(encoding="utf-8"))
    records = []
    validation = []
    for item in manifest:
        pdf_path = SOURCE_DIR / f'{item["id"]}.pdf'
        if not pdf_path.exists():
            validation.append({"id": item["id"], "status": "missing"})
            continue
        try:
            doc = pdfplumber.open(pdf_path)
            page_count = len(doc.pages)
        except Exception as exc:
            validation.append({"id": item["id"], "status": "invalid", "error": str(exc)})
            continue
        validation.append({"id": item["id"], "status": "ok", "pages": page_count, "bytes": pdf_path.stat().st_size})
        for figure in item["figures"]:
            pattern = caption_regex(figure)
            found = False
            for page_number in range(page_count):
                page = doc.pages[page_number]
                block = find_caption_block(page, pattern)
                if block is None:
                    continue
                output = OUT_DIR / f'{item["id"]}_{figure.lower().replace(" ", "_")}_p{page_number + 1}.png'
                render_clip(page, crop_for_caption(page, block), output)
                records.append({
                    "id": item["id"],
                    "paper": item["paper"],
                    "venue": item["venue"],
                    "figure": figure,
                    "page": page_number + 1,
                    "output": str(output),
                    "source": item["source"],
                })
                found = True
                break
            if not found:
                records.append({
                    "id": item["id"], "paper": item["paper"], "venue": item["venue"],
                    "figure": figure, "page": None, "output": "", "source": item["source"], "status": "caption_not_found"
                })
        doc.close()
    (OUT_DIR / "extraction_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "pdf_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(records)
    print(json.dumps({"validation": validation, "extracted": sum(bool(r.get("output")) for r in records), "requested": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
