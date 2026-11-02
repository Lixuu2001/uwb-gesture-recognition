# -*- coding: utf-8 -*-
"""Regenerate MV-OSR manuscript figures without external plotting packages.

The local Python used by Codex may not have matplotlib.  This script therefore
uses ReportLab for vector PDF output and Pillow for PNG previews.  The numerical
values are fixed from RESULTS.md and the manuscript tables; only the visual
design is changed.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figs")
os.makedirs(OUT, exist_ok=True)


# Palette shared by all redrawn figures.
C_BASE = "#8290B0"
C_BASE_LIGHT = "#B7C2D8"
C_METHOD = "#079A87"
C_METHOD_DARK = "#006C61"
C_NAVY = "#2F557F"
C_RED = "#D94A3A"
C_RED_LIGHT = "#FCEDEA"
C_AMBER = "#C97922"
C_AMBER_LIGHT = "#FFF2DE"
C_GRAY = "#9AA3AF"
C_GRID = "#E6E8EC"
C_BAND = "#EDF0F5"
C_TEXT = "#222222"
C_MUTED = "#6F7782"
C_PANEL = "#D9DDE5"


FONT_DIR = r"C:\Windows\Fonts"
FONT_REG = os.path.join(FONT_DIR, "msyh.ttc")
FONT_BOLD = os.path.join(FONT_DIR, "msyhbd.ttc")
FONT_PDF = "SimHei"

try:
    pdfmetrics.registerFont(TTFont(FONT_PDF, os.path.join(FONT_DIR, "simhei.ttf")))
except Exception:
    FONT_PDF = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_PDF))


def pt(inch: float) -> float:
    return inch * 72.0


def rgb(hex_color: str) -> tuple[int, int, int]:
    s = hex_color.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


class Figure:
    """Small dual backend: ReportLab PDF plus antialiased Pillow PNG."""

    scale = 3

    def __init__(self, name: str, width_in: float, height_in: float):
        self.name = name
        self.w = pt(width_in)
        self.h = pt(height_in)
        self.pdf_path = os.path.join(OUT, f"{name}.pdf")
        self.png_path = os.path.join(OUT, f"{name}.png")
        self.c = canvas.Canvas(self.pdf_path, pagesize=(self.w, self.h))
        self.img = Image.new("RGB", (round(self.w * self.scale), round(self.h * self.scale)), "white")
        self.d = ImageDraw.Draw(self.img)
        self._font_cache: dict[tuple[float, bool], ImageFont.FreeTypeFont] = {}

    def _py(self, y: float) -> float:
        return self.h - y

    def _font(self, size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key not in self._font_cache:
            path = FONT_BOLD if bold and os.path.exists(FONT_BOLD) else FONT_REG
            if not os.path.exists(path):
                path = os.path.join(FONT_DIR, "simhei.ttf")
            self._font_cache[key] = ImageFont.truetype(path, round(size * self.scale))
        return self._font_cache[key]

    def _xy(self, x: float, y: float) -> tuple[int, int]:
        return round(x * self.scale), round(y * self.scale)

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = C_TEXT, width: float = 0.7, dash: Sequence[float] | None = None) -> None:
        self.c.setStrokeColor(HexColor(color))
        self.c.setLineWidth(width)
        if dash:
            self.c.setDash(list(dash))
        self.c.line(x1, self._py(y1), x2, self._py(y2))
        if dash:
            self.c.setDash()
        if dash:
            self._dash_line_pil(x1, y1, x2, y2, color, width, dash)
        else:
            self.d.line((*self._xy(x1, y1), *self._xy(x2, y2)), fill=rgb(color), width=max(1, round(width * self.scale)))

    def _dash_line_pil(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float, dash: Sequence[float]) -> None:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0:
            return
        ux, uy = dx / length, dy / length
        pos = 0.0
        draw = True
        idx = 0
        while pos < length:
            step = dash[idx % len(dash)]
            end = min(length, pos + step)
            if draw:
                xa, ya = x1 + ux * pos, y1 + uy * pos
                xb, yb = x1 + ux * end, y1 + uy * end
                self.d.line((*self._xy(xa, ya), *self._xy(xb, yb)), fill=rgb(color), width=max(1, round(width * self.scale)))
            pos = end
            draw = not draw
            idx += 1

    def rect(self, x: float, y: float, w: float, h: float, fill: str, outline: str | None = None, width: float = 0.7, radius: float = 0.0, dash: Sequence[float] | None = None) -> None:
        self.c.setFillColor(HexColor(fill))
        self.c.setStrokeColor(HexColor(outline or fill))
        self.c.setLineWidth(width)
        if dash:
            self.c.setDash(list(dash))
        if radius:
            self.c.roundRect(x, self._py(y + h), w, h, radius, stroke=1 if outline else 0, fill=1)
        else:
            self.c.rect(x, self._py(y + h), w, h, stroke=1 if outline else 0, fill=1)
        if dash:
            self.c.setDash()
        xy = (*self._xy(x, y), *self._xy(x + w, y + h))
        if radius:
            self.d.rounded_rectangle(xy, radius=round(radius * self.scale), fill=rgb(fill), outline=rgb(outline or fill), width=max(1, round(width * self.scale)))
        else:
            self.d.rectangle(xy, fill=rgb(fill), outline=rgb(outline or fill), width=max(1, round(width * self.scale)))

    def circle(self, x: float, y: float, r: float, fill: str, outline: str = "#FFFFFF", width: float = 0.7) -> None:
        self.c.setFillColor(HexColor(fill))
        self.c.setStrokeColor(HexColor(outline))
        self.c.setLineWidth(width)
        self.c.circle(x, self._py(y), r, stroke=1, fill=1)
        xy = (*self._xy(x - r, y - r), *self._xy(x + r, y + r))
        self.d.ellipse(xy, fill=rgb(fill), outline=rgb(outline), width=max(1, round(width * self.scale)))

    def text(self, x: float, y: float, s: str, size: float = 7.0, color: str = C_TEXT, anchor: str = "mm", bold: bool = False, leading: float = 1.18) -> None:
        lines = str(s).split("\n")
        line_h = size * leading
        if "t" in anchor:
            ys = [y + i * line_h for i in range(len(lines))]
        elif "b" in anchor:
            start = y - (len(lines) - 1) * line_h
            ys = [start + i * line_h for i in range(len(lines))]
        else:
            start = y - (len(lines) - 1) * line_h / 2
            ys = [start + i * line_h for i in range(len(lines))]

        self.c.setFillColor(HexColor(color))
        self.c.setFont(FONT_PDF, size)
        for line, yy in zip(lines, ys):
            baseline = self._py(yy) - size * 0.35
            if anchor[0] == "c" or anchor[0] == "m":
                self.c.drawCentredString(x, baseline, line)
            elif anchor[0] == "r":
                self.c.drawRightString(x, baseline, line)
            else:
                self.c.drawString(x, baseline, line)

        pil_anchor = {"mm": "mm", "lm": "lm", "rm": "rm", "lt": "lt", "ct": "mt", "rt": "rt", "lb": "lb", "cb": "mb", "rb": "rb"}.get(anchor, "mm")
        font = self._font(size, bold)
        for line, yy in zip(lines, ys):
            self.d.text(self._xy(x, yy), line, fill=rgb(color), font=font, anchor=pil_anchor)

    def rotated_text(self, x: float, y: float, s: str, angle: float, size: float = 7.0, color: str = C_TEXT, bold: bool = False) -> None:
        self.c.saveState()
        self.c.translate(x, self._py(y))
        self.c.rotate(-angle)
        self.c.setFillColor(HexColor(color))
        self.c.setFont(FONT_PDF, size)
        self.c.drawCentredString(0, -size * 0.35, s)
        self.c.restoreState()

        font = self._font(size, bold)
        bbox = self.d.textbbox((0, 0), s, font=font)
        tw, th = bbox[2] - bbox[0] + 8 * self.scale, bbox[3] - bbox[1] + 8 * self.scale
        layer = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
        ld = ImageDraw.Draw(layer)
        ld.text((tw // 2, th // 2), s, fill=rgb(color) + (255,), font=font, anchor="mm")
        rot = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        px, py = self._xy(x, y)
        self.img.paste(rot, (px - rot.width // 2, py - rot.height // 2), rot)

    def arrow(self, x1: float, y1: float, x2: float, y2: float, color: str = "#69707A", width: float = 0.7) -> None:
        self.line(x1, y1, x2, y2, color, width)
        ang = math.atan2(y2 - y1, x2 - x1)
        head = 4.2
        a1 = ang + math.pi * 0.82
        a2 = ang - math.pi * 0.82
        pts = [
            (x2, y2),
            (x2 + head * math.cos(a1), y2 + head * math.sin(a1)),
            (x2 + head * math.cos(a2), y2 + head * math.sin(a2)),
        ]
        self.c.setFillColor(HexColor(color))
        p = self.c.beginPath()
        p.moveTo(pts[0][0], self._py(pts[0][1]))
        p.lineTo(pts[1][0], self._py(pts[1][1]))
        p.lineTo(pts[2][0], self._py(pts[2][1]))
        p.close()
        self.c.drawPath(p, fill=1, stroke=0)
        self.d.polygon([self._xy(px, py) for px, py in pts], fill=rgb(color))

    def hatch(self, x: float, y: float, w: float, h: float, color: str = "#FFFFFF", step: float = 7.0, width: float = 1.1) -> None:
        start = x - h
        while start < x + w:
            x1 = max(x, start)
            y1 = y + max(0, x - start)
            x2 = min(x + w, start + h)
            y2 = y + h - max(0, start + h - (x + w))
            self.line(x1, y1, x2, y2, color, width)
            start += step

    def save(self) -> None:
        self.c.showPage()
        self.c.save()
        self.img.save(self.png_path)
        print(f"  saved {self.name}.pdf/png")


@dataclass
class Chart:
    f: Figure
    left: float
    top: float
    width: float
    height: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def x(self, v: float) -> float:
        return self.left + (v - self.xmin) / (self.xmax - self.xmin) * self.width

    def y(self, v: float) -> float:
        return self.top + self.height - (v - self.ymin) / (self.ymax - self.ymin) * self.height

    def axes(self, xticks: Iterable[float] = (), yticks: Iterable[float] = (), grid: str = "y") -> None:
        f = self.f
        if grid in ("x", "both"):
            for xv in xticks:
                x = self.x(xv)
                f.line(x, self.top, x, self.top + self.height, C_GRID, 0.45)
        if grid in ("y", "both"):
            for yv in yticks:
                y = self.y(yv)
                f.line(self.left, y, self.left + self.width, y, C_GRID, 0.45)
        f.line(self.left, self.top + self.height, self.left + self.width, self.top + self.height, C_TEXT, 0.8)
        f.line(self.left, self.top, self.left, self.top + self.height, C_TEXT, 0.8)


def draw_bar(ch: Chart, x: float, value: float, base: float, width: float, color: str, hatch: bool = False) -> None:
    px = ch.x(x) - width / 2
    py = ch.y(value)
    by = ch.y(base)
    ch.f.rect(px, py, width, by - py, color)
    if hatch:
        ch.f.hatch(px, py, width, by - py, "#FFFFFF", step=7.0, width=1.0)


def draw_error_y(ch: Chart, x: float, y: float, err: float, color: str = "#4A4A4A") -> None:
    px = ch.x(x)
    y1, y2 = ch.y(y - err), ch.y(y + err)
    ch.f.line(px, y1, px, y2, color, 0.85)
    ch.f.line(px - 3.2, y1, px + 3.2, y1, color, 0.85)
    ch.f.line(px - 3.2, y2, px + 3.2, y2, color, 0.85)


def draw_error_x(ch: Chart, x: float, y: float, err: float, color: str) -> None:
    x1, x2, py = ch.x(x - err), ch.x(x + err), y
    ch.f.line(x1, py, x2, py, color, 0.8)
    ch.f.line(x1, py - 3, x1, py + 3, color, 0.8)
    ch.f.line(x2, py - 3, x2, py + 3, color, 0.8)


def fig2_main_per_case() -> None:
    f = Figure("fig2_main_per_case", 6.8, 3.08)
    ch = Chart(f, 38, 35, 425, 145, -0.6, 7.08, 60, 101)
    single = [87.81, 84.53, 87.39, 87.18, 86.81]
    three = [96.19, 94.80, 94.72, 94.04, 95.95]
    eom = 88.42
    s_mean, s_std = 86.74, 1.29
    t_mean, t_std = 95.14, 0.81
    xm = 6.02
    bw = 19

    ch.axes(yticks=[60, 70, 80, 90, 100], grid="y")
    for yv in [60, 70, 80, 90, 100]:
        f.text(ch.left - 7, ch.y(yv), str(yv), 6.8, C_TEXT, "rm")
    f.text(ch.left, ch.top - 14, "OSCR (%)", 7.1, C_TEXT, "lm")

    f.line(ch.x(-0.58), ch.y(eom), ch.x(7.08), ch.y(eom), C_RED, 0.9, dash=(5, 3))
    f.text(ch.x(6.94), ch.y(eom) - 10, "Eom 2025 单视角", 6.2, C_RED, "rm")
    f.text(ch.x(6.94), ch.y(eom) + 10, f"{eom:.2f}", 6.8, C_RED, "rm", True)

    for i, (s, t) in enumerate(zip(single, three)):
        draw_bar(ch, i - 0.17, s, 60, bw, C_BASE_LIGHT)
        draw_bar(ch, i + 0.17, t, 60, bw, C_NAVY)
        f.text(ch.x(i + 0.17), ch.y(t) - 9, f"+{t - s:.1f}", 6.9, C_METHOD_DARK, "cb", True)

    draw_bar(ch, xm - 0.17, s_mean, 60, bw, C_BASE_LIGHT)
    draw_bar(ch, xm + 0.17, t_mean, 60, bw, C_NAVY)
    draw_error_y(ch, xm - 0.17, s_mean, s_std)
    draw_error_y(ch, xm + 0.17, t_mean, t_std)
    f.text(ch.x(xm - 0.17), ch.y(s_mean) + 12, f"{s_mean:.2f}", 6.8, "#FFFFFF", "mm", True)
    f.text(ch.x(xm + 0.17), ch.y(t_mean) + 10, f"{t_mean:.2f}", 6.8, "#FFFFFF", "mm", True)

    f.line(ch.x(5.05), ch.top, ch.x(5.05), ch.top + ch.height, "#D8DADF", 0.75)
    for i, lab in enumerate(["case 1", "case 2", "case 3", "case 4", "case 5"]):
        f.text(ch.x(i), ch.top + ch.height + 16, lab, 7.0, C_TEXT, "ct")
    f.text(ch.x(xm), ch.top + ch.height + 16, "5-case\n均值", 7.0, C_TEXT, "ct")

    # Legend.
    lx = ch.left + 165
    f.rect(lx, 16, 10, 6, C_BASE_LIGHT)
    f.text(lx + 14, 19, "单视角(Left)", 7.0, C_TEXT, "lm")
    f.rect(lx + 92, 16, 10, 6, C_NAVY)
    f.text(lx + 106, 19, "三视角(L+T+R)", 7.0, C_TEXT, "lm")
    f.save()


def fig3_robustness() -> None:
    f = Figure("fig3_robustness", 7.10, 3.36)
    rows = [
        ("全视角 L+T+R", 95.33, 0.79, 94.88, 0.94),
        ("上失效(L+R)", 86.22, 2.63, 90.81, 2.28),
        ("右失效(L+T)", 80.21, 2.62, 91.97, 0.93),
        ("左失效(T+R)", 71.40, 9.75, 92.35, 1.07),
        ("仅剩左 L", 54.71, 6.46, 83.61, 3.21),
        ("仅剩上 T", 53.37, 7.11, 72.12, 3.94),
        ("仅剩右 R", 43.35, 7.51, 81.30, 1.36),
    ]
    left, top, width, height = 104, 39, 330, 153
    delta_x = left + width + 29
    ch = Chart(f, left, top, width, height, 33, 103, 0, 1)
    ticks = [40, 50, 60, 70, 80, 90, 100]
    for tv in ticks:
        x = ch.x(tv)
        f.line(x, top, x, top + height, C_GRID, 0.45)
        f.text(x, top + height + 14, str(tv), 6.8, C_TEXT, "ct")
    f.line(left, top + height, left + width, top + height, C_TEXT, 0.8)
    f.line(left, top, left, top + height, C_TEXT, 0.8)
    f.text(left + width / 2, top + height + 30, "OSCR (%, 5-case 均值; 误差棒为标准差)", 7.2, C_TEXT, "ct")

    sv_ref = 86.74
    refx = ch.x(sv_ref)
    f.line(refx, top, refx, top + height, C_RED, 0.9, dash=(5, 3))
    f.text(refx + 4, top - 8, "单视角直训 86.74", 6.4, C_RED, "lb")
    f.line(left, top + 1.5 * height / 7, left + width, top + 1.5 * height / 7, "#DDE0E6", 0.72)
    f.line(left, top + 4.5 * height / 7, left + width, top + 4.5 * height / 7, "#DDE0E6", 0.72)

    for idx, (lab, b, bs, v, vs) in enumerate(rows):
        y = top + (idx + 0.5) * height / len(rows)
        f.text(left - 7, y, lab, 7.0, C_TEXT, "rm")
        f.line(ch.x(b), y, ch.x(v), y, "#C9CDD4", 1.45)
        draw_error_x(ch, b, y, bs, C_BASE)
        draw_error_x(ch, v, y, vs, C_METHOD)
        f.circle(ch.x(b), y, 3.3, C_BASE)
        f.circle(ch.x(v), y, 3.4, C_METHOD)
        delta = v - b
        label = f"{delta:+.2f} (噪声内)" if abs(delta) < 1 else f"+{delta:.1f}"
        f.text(delta_x, y, label, 6.5, C_METHOD_DARK if delta > 2 else C_MUTED, "ct", delta > 2)

    f.circle(left + 96, 19, 3.4, C_BASE)
    f.text(left + 104, 19, "朴素三视角训练(基线)", 6.6, C_TEXT, "lm")
    f.circle(left + 230, 19, 3.4, C_METHOD)
    f.text(left + 238, 19, "随机丢视角训练(p=0.5)", 6.6, C_TEXT, "lm")
    f.save()


def fig4_ablation() -> None:
    f = Figure("fig4_ablation", 7.25, 3.55)
    # Left panel.
    ch1 = Chart(f, 42, 42, 205, 148, -0.5, 3.5, 60, 103)
    ch1.axes(yticks=[60, 70, 80, 90, 100], grid="y")
    for yv in [60, 70, 80, 90, 100]:
        f.text(ch1.left - 7, ch1.y(yv), str(yv), 6.7, C_TEXT, "rm")
    f.text(ch1.left, 10, "(a) 归一化是首因, 池化是次因", 7.4, C_TEXT, "lt", True)
    f.text(ch1.left, ch1.top - 3, "OSCR (%)", 7.0, C_TEXT, "lm")
    cfgs = ["mean\nper-view\nstd†", "flatten\nper-view\nstd", "mean\nabs-std", "flatten\nabs-std★"]
    sv = [69.71, 72.42, 83.34, 82.90]
    sve = [None, 4.73, 3.34, 4.83]
    tv = [92.56, 90.63, 93.63, 95.14]
    tve = [None, 2.26, 2.05, 0.81]
    bw = 14
    for i in range(4):
        draw_bar(ch1, i - 0.16, sv[i], 60, bw, C_BASE, hatch=(i == 0))
        draw_bar(ch1, i + 0.16, tv[i], 60, bw, C_METHOD, hatch=(i == 0))
        if sve[i] is not None:
            draw_error_y(ch1, i - 0.16, sv[i], sve[i])
            draw_error_y(ch1, i + 0.16, tv[i], tve[i])
        f.text(ch1.x(i - 0.16), ch1.y(sv[i] + (sve[i] or 0)) - 7, f"{sv[i]:.1f}", 5.9, "#56698F", "cb")
        f.text(ch1.x(i + 0.16), ch1.y(tv[i] + (tve[i] or 0)) - 7, f"{tv[i]:.1f}", 5.9, C_METHOD_DARK, "cb")
        f.text(ch1.x(i), ch1.top + ch1.height + 12, cfgs[i], 6.0, C_TEXT, "ct")
    f.rect(ch1.left + 8, ch1.top + 5, 8, 6, C_BASE)
    f.text(ch1.left + 20, ch1.top + 8, "单视角", 6.2, C_TEXT, "lm")
    f.rect(ch1.left + 58, ch1.top + 5, 8, 6, C_METHOD)
    f.text(ch1.left + 70, ch1.top + 8, "三视角", 6.2, C_TEXT, "lm")
    f.text(ch1.left, ch1.top + ch1.height + 50, "† 仅 case-0; 其余为 5-case 均值 ± std", 5.7, C_MUTED, "lt")

    # Right panel.
    ch2 = Chart(f, 320, 42, 150, 148, -2.2, 1.0, 0, 6)
    f.text(ch2.left, 10, "(b) 架构级融合: 零增益或有害", 7.4, C_TEXT, "lt", True)
    f.rect(ch2.x(-0.81), ch2.top, ch2.x(0.81) - ch2.x(-0.81), ch2.height, C_BAND)
    f.text(ch2.left + ch2.width / 2, ch2.top - 6, "灰带: 主结果 ±1 std", 5.8, C_MUTED, "ct")
    for xv in [-2, -1, 0, 1]:
        x = ch2.x(xv)
        f.line(x, ch2.top, x, ch2.top + ch2.height, C_GRID, 0.45)
        f.text(x, ch2.top + ch2.height + 14, str(xv), 6.7, C_TEXT, "ct")
    f.line(ch2.x(0), ch2.top, ch2.x(0), ch2.top + ch2.height, "#3D3D3D", 0.8)
    f.line(ch2.left, ch2.top + ch2.height, ch2.left + ch2.width, ch2.top + ch2.height, C_TEXT, 0.8)
    f.line(ch2.left, ch2.top, ch2.left, ch2.top + ch2.height, C_TEXT, 0.8)
    f.text(ch2.left + ch2.width / 2, ch2.top + ch2.height + 31, "ΔOSCR (相对无融合基线, %)", 6.8, C_TEXT, "ct")
    items = [
        ("F0 移除跨视角注意力", +0.05, "随机·5-case"),
        ("F2 瓶颈中层融合(最优)", -0.03, "随机·case-0"),
        ("E5 物理先验偏置", -0.04, "随机·5-case"),
        ("E4 分歧分量 β=1", -0.57, "随机·5-case"),
        ("F1 融合特征喂检察器", -0.69, "随机·case-0"),
        ("F6 瓶颈融合 @ LOSO", -1.79, "LOSO·case-0"),
    ]
    row_h = ch2.height / 6
    for idx, (lab, val, proto) in enumerate(items):
        y = ch2.top + (idx + 0.5) * row_h
        f.text(ch2.left - 7, y, lab, 6.0, C_TEXT, "rm")
        x0 = ch2.x(0)
        x1 = ch2.x(val)
        color = C_GRAY if abs(val) < 0.1 else C_RED
        f.rect(min(x0, x1), y - 6.8, abs(x1 - x0), 13.6, color)
        f.text((x1 - 6) if val < 0 else (x1 + 6), y, f"{val:+.2f}", 6.0, C_TEXT, "rm" if val < 0 else "lm")
        f.text(ch2.left + ch2.width + 10, y, proto, 5.6, C_MUTED, "lm")
    f.save()


def fig5_repro_waterfall() -> None:
    f = Figure("fig5_repro_waterfall", 6.25, 2.82)
    ch = Chart(f, 48, 20, 385, 150, -0.55, 5.55, 70, 91.5)
    levels = [73.18, 75.38, 75.39, 77.84, 88.35]
    paper = 87.98
    names = ["基线\n(杂波对消)", "+ 式(12)\nmixup 负样本", "+ 检察器\nbatch=16", "+ 忠实\nD_key=32", "+ 原始回波\n(Raw)输入", "复现结果"]
    ch.axes(yticks=[70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5, 90], grid="y")
    for yv in [70, 75, 80, 85, 90]:
        f.text(ch.left - 7, ch.y(yv), f"{yv:.0f}", 6.7, C_TEXT, "rm")
    f.text(ch.left, ch.top - 9, "case-4 OSCR (%)", 7.0, C_TEXT, "lm")
    f.line(ch.left, ch.y(paper), ch.left + ch.width, ch.y(paper), C_RED, 0.9, dash=(5, 3))
    f.text(ch.left + 10, ch.y(paper) - 10, f"论文 case-4: {paper:.2f}", 6.4, C_RED, "lb")
    bw = 28
    draw_bar(ch, 0, levels[0], 70, bw, C_BASE)
    f.text(ch.x(0), ch.y(levels[0]) - 8, f"{levels[0]:.2f}", 6.6, C_TEXT, "cb")
    for i in range(4):
        low, high = levels[i], levels[i + 1]
        color = C_GRAY if high - low < 0.1 else C_METHOD
        f.rect(ch.x(i + 1) - bw / 2, ch.y(high), bw, ch.y(low) - ch.y(high), color)
        f.line(ch.x(i) + bw / 2, ch.y(low), ch.x(i + 1) - bw / 2, ch.y(low), "#AEB4BE", 0.72, dash=(1.5, 2.0))
        f.text(ch.x(i + 1), ch.y(high) - 7, f"+{high - low:.2f}", 6.6, C_METHOD_DARK if color == C_METHOD else C_MUTED, "cb", color == C_METHOD)
    f.line(ch.x(4) + bw / 2, ch.y(levels[-1]), ch.x(5) - bw / 2, ch.y(levels[-1]), "#AEB4BE", 0.72, dash=(1.5, 2.0))
    draw_bar(ch, 5, levels[-1], 70, bw, C_NAVY)
    f.text(ch.x(5), ch.y(levels[-1]) - 8, f"{levels[-1]:.2f}", 6.8, C_NAVY, "cb", True)
    f.text(ch.x(2), ch.y(73.0), "伪因素", 5.8, C_MUTED, "mm")
    f.text(ch.x(4), ch.y(90.3), "主因", 5.8, C_METHOD_DARK, "mm", True)
    for i, lab in enumerate(names):
        f.text(ch.x(i), ch.top + ch.height + 15, lab, 6.2, C_TEXT, "ct")
    f.save()


@dataclass(frozen=True)
class BoxStyle:
    edge: str
    face: str
    text: str = C_TEXT
    lw: float = 0.75
    dash: bool = False


def ubox(f: Figure, x: float, y: float, w: float, h: float, label: str, st: BoxStyle, fs: float = 6.0, bold: bool = False) -> None:
    f.rect(x, y, w, h, st.face, st.edge, st.lw, radius=4, dash=(3, 2) if st.dash else None)
    if label:
        f.text(x + w / 2, y + h / 2, label, fs, st.text, "mm", bold)


def fig6_fusion_variants() -> None:
    f = Figure("fig6_fusion_variants", 6.9, 4.3)
    sx = f.w / 10.0
    sy = f.h / 6.0

    def X(v: float) -> float:
        return v * sx

    def Y(v: float) -> float:
        return v * sy

    def box(x, y, w, h, label, st, fs=6.0, bold=False):
        ubox(f, X(x), Y(y), X(w), Y(h), label, st, fs, bold)

    def arr(x1, y1, x2, y2, color="#69707A"):
        f.arrow(X(x1), Y(y1), X(x2), Y(y2), color, 0.7)

    input_st = BoxStyle("#5C6F94", "#F4F6FA")
    enc_st = BoxStyle("#4D9CB8", "#EEF8FB")
    feat_st = BoxStyle("#6D7892", "#F4F5F8")
    none_st = BoxStyle("#B7BEC9", "#FFFFFF", C_MUTED, dash=True)
    score_st = BoxStyle(C_METHOD, "#EAF7F4", C_METHOD_DARK, 0.9)
    bad_st = BoxStyle(C_RED, C_RED_LIGHT, "#A52E25", 0.9)
    bot_st = BoxStyle(C_AMBER, C_AMBER_LIGHT, "#8C5419", 0.9)
    out_st = BoxStyle("#7D8796", "#F8F9FB")

    f.text(X(0.20), Y(0.25), "跨视角融合策略对比", 7.6, C_TEXT, "lm", True)
    f.text(X(9.78), Y(0.25), "相同输入与检察器; 仅交换层级不同", 5.8, C_MUTED, "rm")
    panels = [
        (0.68, "(a) 决策级聚合: 本文采用", "三路全程独立; OSCR 95.14±0.81", C_METHOD_DARK, "decision"),
        (2.45, "(b) 特征级融合 F1: 编码后交换", "case-0: ΔOSCR -0.69", C_RED, "feature"),
        (4.22, "(c) 瓶颈中层融合 F2/MBT: 编码中交换", "随机近零; LOSO: ΔOSCR -1.79", C_AMBER, "bottleneck"),
    ]
    for y0, title, result, accent, mode in panels:
        box(0.08, y0, 9.82, 1.62, "", BoxStyle(C_PANEL, "#FFFFFF"), 1)
        f.text(X(0.25), Y(y0 + 0.20), title, 6.9, C_TEXT, "lt", True)
        f.text(X(9.70), Y(y0 + 0.20), result, 5.8, accent, "rt", True)
        lanes = [y0 + 0.63, y0 + 0.88, y0 + 1.13]
        for lab, yy in zip(["L", "T", "R"], lanes):
            box(0.32, yy - 0.10, 0.38, 0.20, lab, input_st, 5.7, True)
        f.text(X(0.51), Y(y0 + 1.42), "三视角输入", 5.3, C_MUTED, "mm")
        mid = y0 + 0.88

        if mode == "bottleneck":
            box(1.15, mid - 0.27, 1.10, 0.54, "编码器\n前段", enc_st, 5.7)
            box(2.55, mid - 0.31, 0.92, 0.62, "瓶颈\ntoken", bot_st, 5.8, True)
            box(3.78, mid - 0.27, 1.10, 0.54, "编码器\n后段", enc_st, 5.7)
            feat_x = 5.24
            for yy in lanes:
                arr(0.70, yy, 1.15, yy)
                arr(2.25, yy, 2.55, yy)
                arr(3.47, yy, 3.78, yy)
                arr(4.88, yy, feat_x, yy)
        else:
            box(1.15, mid - 0.35, 1.35, 0.70, "共享\n编码器", enc_st, 5.8)
            feat_x = 2.92
            for yy in lanes:
                arr(0.70, yy, 1.15, yy)
                arr(2.50, yy, feat_x, yy)

        for lab, yy in zip(["f_L", "f_T", "f_R"], lanes):
            box(feat_x, yy - 0.095, 0.54, 0.19, lab, feat_st, 5.6)

        if mode == "feature":
            box(3.88, mid - 0.31, 1.12, 0.62, "跨视角\n注意力", bad_st, 5.8, True)
            for yy in lanes:
                arr(feat_x + 0.54, yy, 3.88, yy)
            box(5.42, mid - 0.23, 0.78, 0.46, "融合特征\nf_v~", feat_st, 5.2)
            arr(5.00, mid, 5.42, mid)
            box(6.62, mid - 0.26, 0.95, 0.52, "检察器", bad_st, 5.8)
            arr(6.20, mid, 6.62, mid)
            score_x = 8.05
            arr(7.57, mid, score_x, mid)
        else:
            no_x = 3.88 if mode == "decision" else 5.98
            box(no_x, mid - 0.23, 0.78, 0.46, "无特征\n交换", none_st, 5.3)
            for yy in lanes:
                arr(feat_x + 0.54, yy, no_x, yy)
            insp_x = 5.20 if mode == "decision" else 7.10
            if mode == "decision":
                for yy in lanes:
                    arr(no_x + 0.78, yy, insp_x, yy)
                    box(insp_x, yy - 0.095, 0.88, 0.19, "η^(v)", feat_st, 5.2)
                    arr(insp_x + 0.88, yy, 7.35, yy)
                score_x = 7.60
            else:
                for yy in lanes:
                    arr(no_x + 0.78, yy, insp_x, yy)
                box(insp_x, mid - 0.26, 0.95, 0.52, "检察器", bot_st, 5.8)
                score_x = 8.05
                arr(insp_x + 0.95, mid, score_x, mid)

        box(score_x, mid - 0.29, 1.10, 0.58, "⊕\n异常分数 s", score_st if mode == "decision" else BoxStyle("#AEB5C0", "#F7F8FA", C_MUTED), 5.7, mode == "decision")
        arr(score_x + 1.10, mid, 9.35, mid)
        box(9.35, mid - 0.21, 0.50, 0.42, "拒识\n判决", out_st, 5.0)
        note = {
            "decision": ("跨视角信息只在分数层汇合", C_METHOD_DARK),
            "feature": ("交换提前到特征层, 破坏逐视角独立性", C_RED),
            "bottleneck": ("少量共享 token 作窄带交换", "#8C5419"),
        }[mode]
        f.text(X(5.15), Y(y0 + 1.45), note[0], 5.5, note[1], "mm")
    f.save()


if __name__ == "__main__":
    print("rendering figures ->", OUT)
    fig5_repro_waterfall()
    fig2_main_per_case()
    fig6_fusion_variants()
    fig3_robustness()
    fig4_ablation()
    print("done.")
