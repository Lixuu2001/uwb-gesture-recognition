from pathlib import Path
from shutil import copy2
from PIL import Image

work = Path(__file__).resolve().parent
figure_dir = work / "figure_candidates"
backup_dir = work / "raw_auto_crops"
backup_dir.mkdir(exist_ok=True)

# Fractions are relative to the first-pass scholarly figure crop.
crops = {
    "densenet_figure_1_p1.png": (0.00, 0.32, 1.00, 1.00),
    "unet_figure_1_p2.png": (0.27, 0.00, 0.88, 1.00),
    "vit_figure_1_p3.png": (0.00, 0.08, 1.00, 1.00),
    "mae_figure_1_p1.png": (0.00, 0.49, 1.00, 1.00),
    "swin_figure_1_p1.png": (0.00, 0.42, 1.00, 1.00),
    "maskrcnn_figure_1_p1.png": (0.00, 0.51, 1.00, 1.00),
    "dinov2_figure_1_p2.png": (0.00, 0.08, 1.00, 1.00),
    "dinov2_figure_2_p3.png": (0.00, 0.08, 1.00, 1.00),
}

for name, ratios in crops.items():
    source = figure_dir / name
    backup = backup_dir / name
    if not backup.exists():
        copy2(source, backup)
    image = Image.open(backup).convert("RGB")
    w, h = image.size
    left, top, right, bottom = ratios
    box = (round(left * w), round(top * h), round(right * w), round(bottom * h))
    image.crop(box).save(source, quality=95)
    print(name, image.size, "->", (box[2] - box[0], box[3] - box[1]))
