from pathlib import Path

from PIL import Image


ROOT = Path(r"D:\uwb_gestures\paper1\tmp\mvosr_tablefix_pages")
OUT = Path(r"D:\uwb_gestures\paper1\tmp\mvosr_tablefix_crops")
OUT.mkdir(parents=True, exist_ok=True)

# Coordinates are from the latest 2x Word render and intentionally include captions.
regions = {
    "table1.png": (ROOT / "page-07.png", (245, 470, 950, 760)),
    "table2.png": (ROOT / "page-07.png", (330, 1070, 875, 1285)),
    "table3.png": (ROOT / "page-08.png", (245, 340, 965, 660)),
    "table4.png": (ROOT / "page-08.png", (260, 1030, 940, 1325)),
}

for name, (source, box) in regions.items():
    with Image.open(source) as im:
        im.crop(box).save(OUT / name)
    print(OUT / name)
