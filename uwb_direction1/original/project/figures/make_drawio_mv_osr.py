from __future__ import annotations

import datetime as _dt
import os
import uuid
import xml.etree.ElementTree as ET


OUT = r"D:\uwb_gestures\figures\mv_osr_three_view_flow_nature.drawio"

PALETTE = {
    "ink": "#23313A",
    "muted": "#66757F",
    "line": "#4A5963",
    "blue_fill": "#EAF2FA",
    "blue_stroke": "#6A91B8",
    "green_fill": "#EEF7EF",
    "green_stroke": "#6FAE74",
    "violet_fill": "#F0EDF7",
    "violet_stroke": "#8C7BB2",
    "rose_fill": "#F8EDF1",
    "rose_stroke": "#B47A93",
    "amber_fill": "#FFF7DE",
    "amber_stroke": "#D8A63A",
    "grey_fill": "#F7F8F9",
    "grey_stroke": "#CCD4D9",
    "map_fill": "#E8F1F4",
    "map_stroke": "#7EAEC4",
    "signal": "#2B8AA0",
    "signal2": "#D6A93C",
}


class Drawio:
    def __init__(self):
        mxfile = ET.Element(
            "mxfile",
            {
                "host": "app.diagrams.net",
                "modified": _dt.datetime.now().isoformat(timespec="seconds"),
                "agent": "Codex",
                "version": "24.7.17",
                "type": "device",
            },
        )
        diagram = ET.SubElement(mxfile, "diagram", {"name": "MV-OSR Nature flow", "id": uuid.uuid4().hex[:12]})
        self.mxfile = mxfile
        self.model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1500",
                "dy": "900",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": "1480",
                "pageHeight": "900",
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.n = 2

    def _id(self) -> str:
        self.n += 1
        return f"n{self.n}"

    @staticmethod
    def base_text(font=15, color=None, bold=False, align="center"):
        c = color or PALETTE["ink"]
        return (
            f"fontFamily=Helvetica;fontSize={font};fontColor={c};"
            f"align={align};verticalAlign=middle;whiteSpace=wrap;html=1;"
            f"spacing=0;spacingTop=0;spacingBottom=0;"
            + ("fontStyle=1;" if bold else "")
        )

    def rect(self, x, y, w, h, value="", fill=None, stroke=None, font=15, bold=False,
             rounded=True, stroke_width=1.4, align="center", extra=""):
        style = (
            f"rounded={1 if rounded else 0};arcSize=8;whiteSpace=wrap;html=1;"
            f"fillColor={fill or '#FFFFFF'};strokeColor={stroke or PALETTE['line']};"
            f"strokeWidth={stroke_width};"
            + self.base_text(font=font, bold=bold, align=align)
            + extra
        )
        cid = self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": cid, "value": value, "style": style, "vertex": "1", "parent": "1"},
        )
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        return cid

    def text(self, x, y, w, h, value, font=15, bold=False, color=None, align="left"):
        style = (
            "text;strokeColor=none;fillColor=none;rounded=0;"
            + self.base_text(font=font, color=color, bold=bold, align=align)
        )
        cid = self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": cid, "value": value, "style": style, "vertex": "1", "parent": "1"},
        )
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        return cid

    def edge(self, source, target, dashed=False, label="", stroke=None, width=1.5, points=None):
        style = (
            f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
            f"jettySize=auto;html=1;endArrow=block;endFill=1;"
            f"strokeColor={stroke or PALETTE['line']};strokeWidth={width};"
            + ("dashed=1;dashPattern=4 3;" if dashed else "")
            + self.base_text(font=12, color=PALETTE["muted"])
        )
        cid = self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": cid, "value": label, "style": style, "edge": "1", "parent": "1", "source": source, "target": target},
        )
        geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            arr = ET.SubElement(geo, "Array", {"as": "points"})
            for px, py in points:
                ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
        return cid

    def line(self, x1, y1, x2, y2, stroke=None, width=1.2, dashed=False):
        style = (
            f"endArrow=none;html=1;rounded=0;strokeColor={stroke or PALETTE['line']};"
            f"strokeWidth={width};"
            + ("dashed=1;dashPattern=4 3;" if dashed else "")
        )
        cid = self._id()
        cell = ET.SubElement(self.root, "mxCell", {"id": cid, "value": "", "style": style, "edge": "1", "parent": "1"})
        geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geo, "mxPoint", {"x": str(x1), "y": str(y1), "as": "sourcePoint"})
        ET.SubElement(geo, "mxPoint", {"x": str(x2), "y": str(y2), "as": "targetPoint"})
        return cid

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ET.indent(self.mxfile, space="  ")
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=True)


def add_cir_map(d: Drawio, x, y, label, seed=0):
    outer = d.rect(x, y, 155, 66, "", PALETTE["map_fill"], PALETTE["map_stroke"], rounded=False, stroke_width=1.2)
    d.text(x, y - 28, 155, 22, label, font=14, bold=True, align="center")
    # Sparse amplitude traces; these remain editable draw.io line objects.
    for k in range(5):
        d.line(x + 25 + k * 13, y + 58 - k * 9, x + 105 + k * 10, y + 8 + k * 8,
               stroke=PALETTE["signal"], width=1.2 + 0.2 * k)
    d.line(x + 37, y + 6, x + 37, y + 61, stroke="#9FC5D5", width=2.0)
    d.line(x + 41, y + 6, x + 41, y + 61, stroke="#C0DCE7", width=1.0)
    return outer


def add_token_stack(d: Drawio, x, y, label):
    for i in range(6):
        d.rect(x + i * 5, y - i * 3, 52, 28, "", "#FFFFFF", PALETTE["blue_stroke"], rounded=False, stroke_width=0.8)
    d.text(x - 18, y + 30, 120, 20, label, font=12, color=PALETTE["muted"], align="center")


def build():
    d = Drawio()

    # Title.
    d.text(50, 35, 1100, 30, "<b>Multi-view UWB open-set gesture recognition</b>", font=23, bold=True)
    d.text(
        50, 70, 1180, 24,
        "Three synchronized CIR views are encoded with a shared dual-path Transformer and inspected using class-specific per-view prototypes.",
        font=13, color=PALETTE["muted"],
    )

    # Section separators.
    d.line(300, 125, 300, 820, stroke="#E1E7EA", width=1.1)
    d.line(760, 125, 760, 820, stroke="#E1E7EA", width=1.1)

    # Inputs.
    d.text(60, 135, 190, 25, "<b>Synchronized radar views</b>", font=16, bold=True)
    left = add_cir_map(d, 70, 185, "Left", 1)
    top = add_cir_map(d, 70, 315, "Top", 2)
    right = add_cir_map(d, 70, 445, "Right", 3)
    d.text(82, 545, 125, 22, "CIR amplitude maps", font=12, color=PALETTE["muted"], align="center")

    amp = d.rect(
        350, 295, 170, 110,
        "<b>Amplitude preprocessing</b><br>|x| + per-view std<br><span style=\"font-size:12px\">abs_std</span>",
        PALETTE["green_fill"], PALETTE["green_stroke"], font=15, bold=False, stroke_width=1.5,
    )
    for src in [left, top, right]:
        d.edge(src, amp, stroke=PALETTE["line"], width=1.4)

    # Encoder block.
    enc = d.rect(575, 150, 635, 270, "", PALETTE["blue_fill"], PALETTE["blue_stroke"], rounded=True, stroke_width=1.6)
    d.text(700, 174, 380, 24, "<b>Shared dual-path Transformer encoder</b>", font=17, bold=True, align="center")
    d.text(715, 201, 350, 18, "same weights for Left, Top and Right views", font=12, color=PALETTE["muted"], align="center")
    d.edge(amp, enc, stroke=PALETTE["line"], width=1.5)

    add_token_stack(d, 620, 255, "90 temporal tokens")
    add_token_stack(d, 620, 340, "189 spatial tokens")
    emb_t = d.rect(780, 238, 145, 58, "Patch + position<br>embedding", "#FFF5E8", "#D59B52", font=13, stroke_width=1.2)
    emb_s = d.rect(780, 323, 145, 58, "Patch + position<br>embedding", "#FFF5E8", "#D59B52", font=13, stroke_width=1.2)
    tr_t = d.rect(970, 238, 150, 58, "Temporal<br>Transformer", "#D9E7F6", PALETTE["blue_stroke"], font=14, stroke_width=1.2)
    tr_s = d.rect(970, 323, 150, 58, "Spatial<br>Transformer", "#D9E7F6", PALETTE["blue_stroke"], font=14, stroke_width=1.2)
    feat = d.rect(1140, 273, 90, 70, "<b>feature<br>code f<sub>v</sub></b><br><span style=\"font-size:12px\">8928</span>", "#FFFFFF", PALETTE["blue_stroke"], font=13, stroke_width=1.2)
    d.edge(emb_t, tr_t, width=1.2)
    d.edge(emb_s, tr_s, width=1.2)
    d.edge(tr_t, feat, width=1.2)
    d.edge(tr_s, feat, width=1.2)
    d.text(1124, 350, 100, 18, "concat", font=11, color=PALETTE["muted"], align="center")

    # Per-view features and buses.
    d.text(590, 455, 220, 22, "per-view feature codes", font=13, color=PALETTE["muted"], align="center")
    f_left = d.rect(600, 485, 165, 50, "f<sub>Left</sub>", "#E5F2F7", PALETTE["map_stroke"], font=16, stroke_width=1.2)
    f_top = d.rect(600, 560, 165, 50, "f<sub>Top</sub>", "#E5F2F7", PALETTE["map_stroke"], font=16, stroke_width=1.2)
    f_right = d.rect(600, 635, 165, 50, "f<sub>Right</sub>", "#E5F2F7", PALETTE["map_stroke"], font=16, stroke_width=1.2)
    # Light dotted connection from encoder to view-specific features.
    for dst in [f_left, f_top, f_right]:
        d.edge(feat, dst, dashed=True, stroke="#8AA7B8", width=1.0)

    bus = d.rect(795, 505, 8, 160, "", "#FFFFFF", PALETTE["line"], rounded=False, stroke_width=1.0)
    for f in [f_left, f_top, f_right]:
        d.edge(f, bus, stroke=PALETTE["line"], width=1.2)

    # Stage 1.
    d.text(900, 476, 300, 28, "<b>Stage 1: closed-set prediction</b>", font=16, bold=True, align="center")
    fusion = d.rect(835, 520, 210, 68, "<b>Cross-view attention fusion</b>", PALETTE["violet_fill"], PALETTE["violet_stroke"], font=14, stroke_width=1.4)
    cls = d.rect(1080, 520, 175, 68, "Known-gesture<br>classifier", PALETTE["violet_fill"], PALETTE["violet_stroke"], font=14, stroke_width=1.4)
    pred = d.rect(1290, 520, 120, 68, "<b>predicted<br>class k&#770;</b>", PALETTE["amber_fill"], PALETTE["amber_stroke"], font=14, stroke_width=1.4)
    d.edge(bus, fusion, stroke=PALETTE["line"], width=1.5)
    d.edge(fusion, cls, stroke=PALETTE["line"], width=1.5)
    d.edge(cls, pred, stroke=PALETTE["line"], width=1.5)

    # Stage 2.
    d.text(860, 650, 360, 28, "<b>Stage 2: multi-view deep inspection</b>", font=16, bold=True, align="center")
    insp = d.rect(
        835, 700, 220, 80,
        "<b>Per-class, per-view inspectors</b><br>&pi;<sup>v</sup><sub>I,k&#770;</sub>(f<sub>v</sub>)<br><span style=\"font-size:12px\">6 classes &times; 3 views = 18 heads</span>",
        PALETTE["rose_fill"], PALETTE["rose_stroke"], font=13, stroke_width=1.4,
    )
    dist = d.rect(1095, 700, 185, 80, "Prototype distances<br>d<sub>v</sub> / &sigma;<sub>v</sub>", PALETTE["rose_fill"], PALETTE["rose_stroke"], font=14, stroke_width=1.4)
    score = d.rect(1320, 700, 120, 80, "<b>anomaly<br>score s</b>", PALETTE["amber_fill"], PALETTE["amber_stroke"], font=14, stroke_width=1.4)
    d.edge(bus, insp, stroke=PALETTE["line"], width=1.5, points=[(805, 705)])
    d.edge(pred, insp, dashed=True, stroke=PALETTE["muted"], width=1.2, label="select inspectors for k&#770;", points=[(1350, 650), (1050, 650)])
    d.edge(insp, dist, stroke=PALETTE["line"], width=1.5)
    d.edge(dist, score, stroke=PALETTE["line"], width=1.5)

    d.text(
        870, 815, 500, 30,
        "s = mean<sub>v</sub>(d<sub>v</sub>)/&sigma;&#772; + &beta; var<sub>v</sub>(d<sub>v</sub>)/&sigma;&#772;<sup>2</sup>",
        font=17, color=PALETTE["ink"], align="center",
    )
    d.text(
        870, 844, 500, 22,
        "mean distance: common mismatch &nbsp;&nbsp; | &nbsp;&nbsp; variance: cross-view disagreement",
        font=12, color=PALETTE["muted"], align="center",
    )

    accept = d.rect(1270, 810, 80, 48, "accept<br>known", "#ECF8ED", PALETTE["green_stroke"], font=13, stroke_width=1.2)
    reject = d.rect(1365, 810, 85, 48, "reject<br>unknown", "#FAEDEB", "#C98A83", font=13, stroke_width=1.2)
    d.edge(score, accept, stroke=PALETTE["line"], width=1.2)
    d.edge(score, reject, stroke=PALETTE["line"], width=1.2)

    # Relation to single-view paper.
    d.rect(
        60, 705, 440, 76,
        "<b>Relation to Eom et al.</b><br>"
        "With V = 1, the cross-view variance term vanishes and the pipeline reduces to the original single-view inspector path.",
        PALETTE["grey_fill"], PALETTE["grey_stroke"], font=13, stroke_width=1.2,
    )

    d.save(OUT)


if __name__ == "__main__":
    build()
    print(OUT)
