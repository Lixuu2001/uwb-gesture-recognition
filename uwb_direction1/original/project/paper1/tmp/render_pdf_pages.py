from pathlib import Path
import sys

import pypdfium2 as pdfium


pdf_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

pdf = pdfium.PdfDocument(pdf_path)
for i, page in enumerate(pdf):
    bitmap = page.render(scale=2.0)
    image = bitmap.to_pil()
    image.save(out_dir / f"page-{i + 1:02d}.png")

print(f"pages={len(pdf)} out={out_dir}")
