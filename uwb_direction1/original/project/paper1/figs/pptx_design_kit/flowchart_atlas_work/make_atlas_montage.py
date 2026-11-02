from pathlib import Path
from PIL import Image, ImageDraw

work = Path(__file__).resolve().parent
render_dir = work / "atlas_render"
files = sorted(render_dir.glob("slide-*.png"))
thumb_w, thumb_h = 480, 270
gap, pad = 18, 18
cols = 3
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (pad * 2 + cols * thumb_w + (cols - 1) * gap, pad * 2 + rows * thumb_h + (rows - 1) * gap), "#E9EEF2")
for i, file in enumerate(files):
    image = Image.open(file).convert("RGB")
    image.thumbnail((thumb_w, thumb_h))
    x = pad + (i % cols) * (thumb_w + gap)
    y = pad + (i // cols) * (thumb_h + gap)
    canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
    canvas.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
    sheet.paste(canvas, (x, y))
sheet.save(Path("C:/Users/19252/Desktop/顶刊深度学习方法图设计组件库_含真实图鉴_预览.webp"), quality=88, method=6)
print(f"montage: {len(files)} slides")
