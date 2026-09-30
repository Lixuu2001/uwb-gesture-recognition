"""Draw a dependency-free SVG figure for the MAE generative spine.

The figure intentionally avoids matplotlib so it can run in minimal Python
environments. It outputs a paper-editable SVG plus a small HTML preview wrapper.
"""
from html import escape
from pathlib import Path


W, H = 1600, 900
OUT = Path(r"D:\uwb_gestures\figures")

COL = {
    "ink": "#263238",
    "muted": "#687780",
    "line": "#3F4B52",
    "panel": "#F8FAFB",
    "panel_edge": "#D6DEE4",
    "input": "#EAF1F8",
    "input_edge": "#4F7FA8",
    "mask": "#FBE9E7",
    "mask_edge": "#D95F4D",
    "encoder": "#E3EEF9",
    "encoder_edge": "#4E83B5",
    "token": "#F0F3F7",
    "token_edge": "#6B7D91",
    "decoder": "#F6ECFA",
    "decoder_edge": "#8E62A9",
    "loss": "#FFF2CC",
    "loss_edge": "#D5A93D",
    "spine": "#E5F5F1",
    "spine_edge": "#079A87",
    "ood": "#FDEEEB",
    "ood_edge": "#D94A3A",
    "synth": "#EAF4EA",
    "synth_edge": "#5E9B68",
    "future": "#F1F2F8",
    "future_edge": "#8491B4",
}


parts = []


def add(s):
    parts.append(s)


def text(x, y, body, size=18, color=None, weight="400", anchor="middle",
         line_gap=1.18, italic=False):
    color = color or COL["ink"]
    lines = str(body).split("\n")
    style = "italic" if italic else "normal"
    if len(lines) == 1:
        add(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{weight}" font-style="{style}" '
            f'fill="{color}">{escape(lines[0])}</text>'
        )
        return
    total = (len(lines) - 1) * size * line_gap
    y0 = y - total / 2
    add(
        f'<text x="{x}" y="{y0}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" font-style="{style}" '
        f'fill="{color}">'
    )
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * line_gap
        add(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    add("</text>")


def rect(x, y, w, h, fill, stroke, rx=16, lw=2, dash=None, opacity=1.0):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" '
        f'stroke-width="{lw}"{dash_attr}/>'
    )


def box(x, y, w, h, body, fill, stroke, size=18, weight="400",
        rx=18, lw=2, dash=None):
    rect(x, y, w, h, fill, stroke, rx=rx, lw=lw, dash=dash)
    text(x + w / 2, y + h / 2 + size * 0.34, body, size=size,
         weight=weight, anchor="middle")


def arrow(x1, y1, x2, y2, color=None, lw=2.2, dash=None, marker=True):
    color = color or COL["line"]
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    add(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{lw}" fill="none"{dash_attr}{marker_attr}/>'
    )


def path(d, color=None, lw=2.2, dash=None, marker=True):
    color = color or COL["line"]
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    add(
        f'<path d="{d}" stroke="{color}" stroke-width="{lw}" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}{marker_attr}/>'
    )


def panel(x, y, w, h, title, subtitle):
    rect(x, y, w, h, COL["panel"], COL["panel_edge"], rx=24, lw=1.5)
    text(x + 28, y + 44, title, size=25, weight="700", anchor="start")
    text(x + 28, y + 78, subtitle, size=17, color=COL["muted"], anchor="start")


def mini_cir(x, y, w, h, label, masked=False, slope=1.0):
    rect(x, y, w, h, "#175E83", COL["mask_edge"] if masked else COL["input_edge"],
         rx=0, lw=2)
    # Static clutter stripe and moving ridge, stylised but recognisably CIR-like.
    add(f'<line x1="{x + 22}" y1="{y + 2}" x2="{x + 22}" y2="{y + h - 2}" '
        f'stroke="#39C5C8" stroke-width="7" opacity="0.65"/>')
    add(f'<line x1="{x + 46}" y1="{y + h - 4}" x2="{x + w - 16}" y2="{y + 7 * slope}" '
        f'stroke="#9FDB3A" stroke-width="13" opacity="0.95"/>')
    add(f'<line x1="{x + 46}" y1="{y + h - 4}" x2="{x + w - 16}" y2="{y + 7 * slope}" '
        f'stroke="#E8F966" stroke-width="5" opacity="0.95"/>')
    if masked:
        rect(x, y, w, h, "#FFFFFF", "none", rx=0, lw=0, opacity=0.78)
        add(f'<line x1="{x + 10}" y1="{y + 8}" x2="{x + w - 10}" y2="{y + h - 8}" '
            f'stroke="{COL["mask_edge"]}" stroke-width="5"/>')
        add(f'<line x1="{x + 10}" y1="{y + h - 8}" x2="{x + w - 10}" y2="{y + 8}" '
            f'stroke="{COL["mask_edge"]}" stroke-width="5"/>')
    text(x + w / 2, y - 10, label, size=17, weight="700")


def token_stack(x, y, w, h, n, label, fill, stroke):
    for i in range(n):
        off = i * 7
        rect(x + off, y + off, w, h, fill, stroke, rx=0, lw=1.2)
    text(x + w / 2 + (n - 1) * 3.5, y + h + 42, label,
         size=15, color=COL["muted"])


def draw_encoder(x, y, w, h):
    box(x, y, w, h, "", COL["encoder"], COL["encoder_edge"], rx=22, lw=2.5)
    text(x + w / 2, y + 34, "Shared dual-path encoder  enc0",
         size=22, weight="700")
    text(x + w / 2, y + 62, "same weights for visible radar views",
         size=15, color=COL["muted"])
    box(x + 32, y + 95, 154, 58, "Temporal\nTransformer",
        "#F8FBFD", COL["encoder_edge"], size=16, rx=12)
    box(x + 32, y + 172, 154, 58, "Range\nTransformer",
        "#F8FBFD", COL["encoder_edge"], size=16, rx=12)
    box(x + 232, y + 132, 154, 70, "visible tokens\nM = [B, 2T, D]",
        COL["token"], COL["token_edge"], size=15, rx=12)
    arrow(x + 188, y + 124, x + 230, y + 158, lw=1.8)
    arrow(x + 188, y + 201, x + 230, y + 175, lw=1.8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    add(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{COL["line"]}"/>
  </marker>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; dominant-baseline: middle; }}
  </style>
</defs>
<rect width="100%" height="100%" fill="white"/>
''')

    text(52, 52, "Cross-view MAE pretraining and the generative spine",
         size=31, weight="700", anchor="start")
    text(52, 92,
         "Mask one radar view, reconstruct it from the other views, then keep enc0+dec0 as a deployed generator.",
         size=18, color=COL["muted"], anchor="start")

    # Panel A
    panel(
        46, 135, 1508, 378,
        "A. Stage-0: cross-view masked reconstruction",
        "The hidden radar view supplies the target. No human gesture label is used in this pretraining objective."
    )

    mini_cir(86, 255, 136, 58, "x_L", masked=False, slope=1.0)
    mini_cir(86, 332, 136, 58, "x_T masked", masked=True, slope=1.5)
    mini_cir(86, 409, 136, 58, "x_R", masked=False, slope=0.7)
    text(154, 448, "x: [B, 3, 90, 189]", size=16, color=COL["muted"])

    box(278, 311, 160, 86, "randomly mask\none view v*\nexample: Top",
        COL["mask"], COL["mask_edge"], size=17, weight="700")
    arrow(224, 360, 276, 354)

    box(490, 242, 172, 76, "visible views\nx_-v*\n[B, 2, 90, 189]",
        COL["input"], COL["input_edge"], size=16)
    box(490, 371, 172, 76, "held-out target\nx_v*\n[B, 90, 189]",
        COL["mask"], COL["mask_edge"], size=16)
    path("M 440 344 C 464 326, 466 286, 488 280")
    path(f"M 440 364 C 464 382, 466 408, 488 409", color=COL["mask_edge"])

    draw_encoder(715, 232, 452, 226)
    arrow(664, 280, 712, 318)

    box(1208, 222, 180, 54, "90 learned queries Q\n[B, 90, D]",
        "#FFFFFF", COL["decoder_edge"], size=15)
    box(1200, 300, 220, 82, "masked-view decoder  dec0\nTransformer decoder\ncross-attention to M",
        COL["decoder"], COL["decoder_edge"], size=16, weight="700")
    path("M 1168 345 C 1182 338, 1186 328, 1198 326")
    arrow(1298, 278, 1298, 298)

    box(1200, 415, 148, 62, "reconstruction\nx_hat_v*\n[B, 90, 189]",
        COL["input"], COL["input_edge"], size=15)
    arrow(1300, 384, 1285, 413)
    box(1378, 415, 92, 62, "L_MAE\nMSE", COL["loss"], COL["loss_edge"],
        size=18, weight="700")
    arrow(1350, 446, 1376, 446)
    path(f"M 662 410 C 880 470, 1120 506, 1376 462", color=COL["mask_edge"],
         dash="8 6", lw=2.2)
    text(973, 494, "L_MAE = ||x_hat_v* - x_v*||^2   updates both enc0 and dec0",
         size=18)

    # Panel B
    panel(
        46, 535, 1508, 325,
        "B. After Stage-0: freeze enc0+dec0 as the generative spine",
        "Standard MAE often discards the decoder. Here the decoder remains useful for open-set evidence and virtual-view synthesis."
    )

    box(640, 645, 330, 96,
        "Frozen generative spine\nenc0 + dec0\nmask-one-view generator",
        COL["spine"], COL["spine_edge"], size=20, weight="700", rx=24, lw=2.6)
    text(805, 763, "weights frozen for downstream readouts",
         size=16, color=COL["muted"])

    box(116, 650, 220, 66, "test sample\nmask each view\nv = L, T, R",
        COL["input"], COL["input_edge"], size=16)
    arrow(338, 683, 638, 693)
    box(112, 755, 390, 62,
        "reconstruction-error signal\n"
        "e_recon = mean_v MSE(x_hat_v, x_v)",
        COL["ood"], COL["ood_edge"], size=15, weight="700")
    path("M 640 716 C 550 738, 535 780, 504 786", lw=2.2)
    box(126, 814, 360, 42,
        "low error: self-consistent known\nhigh error: likely unknown",
        "#FFF7F5", COL["ood_edge"], size=14)

    box(1206, 650, 235, 66,
        "sensor failure\nsurviving views\n{x_L, x_R}",
        COL["input"], COL["input_edge"], size=16)
    arrow(972, 693, 1204, 683)
    box(1176, 755, 302, 62,
        "virtual-view synthesis\nx_tilde_T = dec0(enc0(x_L, x_R))",
        COL["synth"], COL["synth_edge"], size=16, weight="700")
    arrow(1324, 718, 1324, 753)
    box(1210, 814, 235, 42,
        "complete view set again\n[x_L, x_tilde_T, x_R]",
        "#F7FBF7", COL["synth_edge"], size=14)

    box(632, 788, 348, 58,
        "optional future readout\nstructured latent motion z(t)\n(JEPA/EMA only for latent targets)",
        COL["future"], COL["future_edge"], size=15, dash="8 6")
    path(f"M 805 743 L 805 786", color=COL["future_edge"], dash="8 6", lw=2.0)

    text(805, 884,
         "Paper-1 spine readouts: e_recon for open-set evidence + decoder synthesis for missing-view robustness.",
         size=18, color=COL["muted"])

    add("</svg>\n")
    svg = "".join(parts)
    svg_path = OUT / "mae_generative_spine.svg"
    svg_path.write_text(svg, encoding="utf-8")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>MAE Generative Spine</title>"
        "<style>body{margin:0;background:#fff;display:flex;justify-content:center;}"
        "img{max-width:100vw;height:auto;}</style></head>"
        f"<body><img src='{svg_path.name}'></body></html>"
    )
    (OUT / "mae_generative_spine_preview.html").write_text(html, encoding="utf-8")
    print(svg_path)


if __name__ == "__main__":
    main()
