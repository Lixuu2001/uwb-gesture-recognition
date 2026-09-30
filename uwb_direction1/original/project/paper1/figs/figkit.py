# -*- coding: utf-8 -*-
"""论文配图工具：一套几何图元 → SVG（网页效果图）与 PPTX（可编辑投稿图）。

用法::

    from figkit import Canvas
    c = Canvas(1280, 820)
    c.box(10, 10, 100, 40, "标签")
    c.write_all(os.path.dirname(__file__), "fig2_exchange", caption="图 2 …")

坐标系约定：1 unit = 0.01 inch。论文 \\textwidth = 16.2 cm = 6.378 in，
故 1280 units 宽的画布打印缩放约 0.498，即**幻灯片字号 × 0.5 ≈ 打印磅值**，
图内最小字号取 12pt（打印 6pt）。

文本 runs 的片段类型：
    zh 中文正文 / en 拉丁与数字 / it 数学斜体 / itsub 斜体下标 / sub 下标 / sup 上标

PPTX 陷阱（踩过的坑，勿回退）：
  * 1 unit = 9144 EMU（0.01 in），不是 91440；
  * 箭头 headEnd/tailEnd 必须 append 到 <a:ln> 末尾，否则 PowerPoint 静默丢箭头；
  * 中文要同时设 a:latin 与 a:ea，只设 font.name 会退回主题字体；
  * 文本框 margin 置 0 后锚点即文本边界，不要再加补偿偏移；
  * 只用直线段，不用贝塞尔自由曲线（传英寸浮点会被截成怪折线）。
"""
import base64
import io
import os

PT = 100 / 72.0                      # 1 pt -> unit
ZH_F = "'Microsoft YaHei','PingFang SC',sans-serif"
EN_F = "'Arial','Helvetica Neue',sans-serif"
IT_F = "'Times New Roman','Cambria',serif"
_FONTS = {"zh": ("Microsoft YaHei", False), "en": ("Arial", False),
          "it": ("Times New Roman", True), "itsub": ("Times New Roman", True),
          "sub": ("Arial", False), "sup": ("Arial", False)}


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Canvas:
    def __init__(self, w, h, ink="1A2430", stroke="AEBBC7", lane="8FA0AF"):
        self.w, self.h = w, h
        self.ink, self.stroke, self.lane = ink, stroke, lane
        self.P = []

    # -------------------------------------------------------- primitives
    def rect(self, x, y, w, h, fill=None, stroke=None, sw=1.1, rx=4,
             dash=None, z=0):
        self.P.append(dict(k="rect", x=x, y=y, w=w, h=h, fill=fill,
                           stroke=stroke, sw=sw, rx=rx, dash=dash, z=z))

    def circ(self, cx, cy, r, fill=None, stroke=None, sw=1.1, dash=None, z=0):
        self.P.append(dict(k="circ", cx=cx, cy=cy, r=r, fill=fill,
                           stroke=stroke, sw=sw, dash=dash, z=z))

    def line(self, x1, y1, x2, y2, color=None, sw=1.1, arrow=False, dash=None,
             z=0, both=False):
        self.P.append(dict(k="line", x1=x1, y1=y1, x2=x2, y2=y2,
                           color=color or self.lane, sw=sw, arrow=arrow,
                           dash=dash, z=z, both=both))

    def arrow(self, x1, y1, x2, y2, color=None, sw=1.25, dash=None, both=False):
        self.line(x1, y1, x2, y2, color=color, sw=sw, arrow=True, dash=dash,
                  both=both)

    def text(self, cx, cy, runs, size=12, color=None, align="center",
             bold=False, z=5):
        if isinstance(runs, str):
            runs = [(runs, "zh")]
        self.P.append(dict(k="text", cx=cx, cy=cy, runs=runs, size=size,
                           color=color or self.ink, align=align, bold=bold,
                           z=z))

    def image(self, x, y, w, h, path, z=0):
        self.P.append(dict(k="image", x=x, y=y, w=w, h=h, path=path, z=z))

    # -------------------------------------------------------- composites
    def lines(self, cx, cy, rows, size=12, color=None, bold=False):
        """多行文本，绕 (cx, cy) 垂直居中。rows 的元素可以是 str 或 runs 列表。"""
        lh = size * 1.45
        y0 = cy - (len(rows) - 1) * lh / 2
        for i, r in enumerate(rows):
            self.text(cx, y0 + i * lh, r, size=size, color=color, bold=bold)

    def box(self, x, y, w, h, rows=None, fill="FFFFFF", stroke=None, sw=1.15,
            rx=4, size=12, color=None, bold=False, dash=None):
        self.rect(x, y, w, h, fill=fill, stroke=stroke or self.stroke, sw=sw,
                  rx=rx, dash=dash)
        if rows is not None:
            self.lines(x + w / 2, y + h / 2,
                       rows if isinstance(rows, list) else [rows],
                       size=size, color=color, bold=bold)

    def stack_box(self, x, y, w, h, rows=None, fill="FFFFFF", stroke=None,
                  n=2, size=12, color=None):
        """带叠层阴影的模块，暗示"若干层"。"""
        for i in range(n, 0, -1):
            self.rect(x + i * 2.6, y - i * 2.6, w, h,
                      fill="F1F4F7" if i > 1 else "F8FAFC", stroke="D3DCE4",
                      sw=0.85, z=-3 - i)
        self.box(x, y, w, h, rows, fill=fill, stroke=stroke, size=size,
                 color=color)

    def chip(self, x, y, w, h, s, size=13, fill="EDF1F5", color=None,
             bold=True, stroke=None, rx=4):
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.1, rx=rx)
        self.text(x + w / 2, y + h / 2, s, size=size, color=color, bold=bold)

    def token_stack(self, cx, y0, w, n, cell_h, gap, fill, stroke,
                    ellipsis_at=None):
        """竖向 token 堆，中间可插省略号。返回底端 y。"""
        y = y0
        for i in range(n):
            if ellipsis_at is not None and i == ellipsis_at:
                self.text(cx, y + cell_h / 2, [("⋮", "en")], size=12,
                          color=stroke)
                y += cell_h + gap
                continue
            self.rect(cx - w / 2, y, w, cell_h, fill=fill, stroke=stroke,
                      sw=0.9, rx=2)
            y += cell_h + gap
        return y - gap

    # -------------------------------------------------------- SVG
    @staticmethod
    def _run_svg(r, size):
        s, kind = r
        px = size * PT
        if kind in ("zh", "en", "it"):
            fam = {"zh": ZH_F, "en": EN_F, "it": IT_F}[kind]
            it = ' font-style="italic"' if kind == "it" else ""
            return f'<tspan font-family="{fam}"{it}>{_esc(s)}</tspan>'
        fam = IT_F if kind == "itsub" else EN_F
        it = ' font-style="italic"' if kind == "itsub" else ""
        dy = -px * 0.34 if kind == "sup" else px * 0.22
        return (f'<tspan font-family="{fam}"{it} font-size="{px*0.72:.2f}" '
                f'dy="{dy:.2f}">{_esc(s)}</tspan>'
                f'<tspan dy="{-dy:.2f}" font-size="0"> </tspan>')

    def to_svg(self):
        o = io.StringIO()
        o.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
                f'font-family="{ZH_F}">\n<defs>\n')
        for col in {p["color"] for p in self.P if p["k"] == "line"}:
            o.write(f'<marker id="a{col}" viewBox="0 0 10 10" refX="9" '
                    f'refY="5" markerWidth="6" markerHeight="6" '
                    f'orient="auto-start-reverse">'
                    f'<path d="M0,1.6 L10,5 L0,8.4 z" fill="#{col}"/>'
                    f'</marker>\n')
        o.write(f'</defs>\n<rect width="{self.w}" height="{self.h}" '
                f'fill="#fff"/>\n')
        for p in sorted(self.P, key=lambda d: d["z"]):
            k = p["k"]
            if k == "rect":
                f = f'#{p["fill"]}' if p["fill"] else "none"
                s = f'#{p["stroke"]}' if p["stroke"] else "none"
                d = f' stroke-dasharray="{p["dash"]}"' if p["dash"] else ""
                o.write(f'<rect x="{p["x"]:.2f}" y="{p["y"]:.2f}" '
                        f'width="{p["w"]:.2f}" height="{p["h"]:.2f}" '
                        f'rx="{p["rx"]}" fill="{f}" stroke="{s}" '
                        f'stroke-width="{p["sw"]*PT:.2f}"{d}/>\n')
            elif k == "circ":
                f = f'#{p["fill"]}' if p["fill"] else "none"
                s = f'#{p["stroke"]}' if p["stroke"] else "none"
                d = f' stroke-dasharray="{p["dash"]}"' if p["dash"] else ""
                o.write(f'<circle cx="{p["cx"]:.2f}" cy="{p["cy"]:.2f}" '
                        f'r="{p["r"]:.2f}" fill="{f}" stroke="{s}" '
                        f'stroke-width="{p["sw"]*PT:.2f}"{d}/>\n')
            elif k == "line":
                m = "a" + p["color"]
                a = f' marker-end="url(#{m})"' if p["arrow"] else ""
                if p["both"]:
                    a += f' marker-start="url(#{m})"'
                d = f' stroke-dasharray="{p["dash"]}"' if p["dash"] else ""
                o.write(f'<line x1="{p["x1"]:.2f}" y1="{p["y1"]:.2f}" '
                        f'x2="{p["x2"]:.2f}" y2="{p["y2"]:.2f}" '
                        f'stroke="#{p["color"]}" '
                        f'stroke-width="{p["sw"]*PT:.2f}" '
                        f'stroke-linecap="round"{a}{d}/>\n')
            elif k == "image":
                b64 = base64.b64encode(open(p["path"], "rb").read()).decode()
                o.write(f'<image x="{p["x"]:.2f}" y="{p["y"]:.2f}" '
                        f'width="{p["w"]:.2f}" height="{p["h"]:.2f}" '
                        f'preserveAspectRatio="none" '
                        f'xlink:href="data:image/png;base64,{b64}"/>\n')
            else:
                anc = {"center": "middle", "left": "start",
                       "right": "end"}[p["align"]]
                wt = ' font-weight="600"' if p["bold"] else ""
                body = "".join(self._run_svg(r, p["size"]) for r in p["runs"])
                o.write(f'<text xml:space="preserve" x="{p["cx"]:.2f}" '
                        f'y="{p["cy"]:.2f}" text-anchor="{anc}" '
                        f'dominant-baseline="central" '
                        f'font-size="{p["size"]*PT:.2f}" fill="#{p["color"]}"'
                        f'{wt}>{body}</text>\n')
        o.write("</svg>\n")
        return o.getvalue()

    # -------------------------------------------------------- PPTX
    @staticmethod
    def _est_w(runs, size):
        n = 0.0
        for s, kind in runs:
            f = 0.72 if kind in ("sub", "sup", "itsub") else 1.0
            for ch in s:
                n += (1.0 if ord(ch) > 0x2E7F else 0.53) * f
        return n * size * PT

    def to_pptx(self, path):
        from pptx import Presentation
        from pptx.util import Emu, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
        from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
        from pptx.oxml.ns import qn

        U = 9144
        prs = Presentation()
        prs.slide_width, prs.slide_height = Emu(self.w * U), Emu(self.h * U)
        shapes = prs.slides.add_slide(prs.slide_layouts[6]).shapes

        def rgb(h):
            return RGBColor.from_string(h)

        def style(sh, fill, stroke, sw, dash=None):
            if fill:
                sh.fill.solid()
                sh.fill.fore_color.rgb = rgb(fill)
            else:
                sh.fill.background()
            if stroke:
                sh.line.color.rgb = rgb(stroke)
                sh.line.width = Pt(sw)
                if dash:
                    ln = sh.line._get_or_add_ln()
                    ln.append(ln.makeelement(qn("a:prstDash"),
                                             {"val": "dash"}))
            else:
                sh.line.fill.background()
            sh.shadow.inherit = False

        for p in sorted(self.P, key=lambda d: d["z"]):
            k = p["k"]
            if k == "rect":
                shp = (MSO_SHAPE.ROUNDED_RECTANGLE if p["rx"]
                       else MSO_SHAPE.RECTANGLE)
                sh = shapes.add_shape(shp, Emu(int(p["x"] * U)),
                                      Emu(int(p["y"] * U)),
                                      Emu(int(p["w"] * U)),
                                      Emu(int(p["h"] * U)))
                if p["rx"]:
                    sh.adjustments[0] = min(0.5,
                                            p["rx"] / min(p["w"], p["h"]))
                style(sh, p["fill"], p["stroke"], p["sw"], p["dash"])
            elif k == "circ":
                r = p["r"]
                sh = shapes.add_shape(MSO_SHAPE.OVAL,
                                      Emu(int((p["cx"] - r) * U)),
                                      Emu(int((p["cy"] - r) * U)),
                                      Emu(int(2 * r * U)), Emu(int(2 * r * U)))
                style(sh, p["fill"], p["stroke"], p["sw"], p["dash"])
            elif k == "line":
                sh = shapes.add_connector(
                    MSO_CONNECTOR.STRAIGHT, Emu(int(p["x1"] * U)),
                    Emu(int(p["y1"] * U)), Emu(int(p["x2"] * U)),
                    Emu(int(p["y2"] * U)))
                ln = sh.line._get_or_add_ln()
                sh.line.color.rgb = rgb(p["color"])
                sh.line.width = Pt(p["sw"])
                if p["dash"]:
                    ln.append(ln.makeelement(qn("a:prstDash"),
                                             {"val": "dash"}))
                # headEnd / tailEnd 必须在 <a:ln> 末尾，否则 PowerPoint 丢箭头
                if p["both"]:
                    ln.append(ln.makeelement(qn("a:headEnd"), {
                        "type": "triangle", "w": "med", "len": "med"}))
                if p["arrow"]:
                    ln.append(ln.makeelement(qn("a:tailEnd"), {
                        "type": "triangle", "w": "med", "len": "med"}))
            elif k == "image":
                shapes.add_picture(p["path"], Emu(int(p["x"] * U)),
                                   Emu(int(p["y"] * U)), Emu(int(p["w"] * U)),
                                   Emu(int(p["h"] * U)))
            else:
                wpx = max(34.0, self._est_w(p["runs"], p["size"]) * 1.22 + 18)
                hpx = p["size"] * PT * 2.0
                x = {"center": p["cx"] - wpx / 2, "left": p["cx"],
                     "right": p["cx"] - wpx}[p["align"]]
                tb = shapes.add_textbox(Emu(int(x * U)),
                                        Emu(int((p["cy"] - hpx / 2) * U)),
                                        Emu(int(wpx * U)), Emu(int(hpx * U)))
                tf = tb.text_frame
                tf.word_wrap = False
                tf.margin_left = tf.margin_right = 0
                tf.margin_top = tf.margin_bottom = 0
                tf.vertical_anchor = MSO_ANCHOR.MIDDLE
                para = tf.paragraphs[0]
                para.alignment = {"center": PP_ALIGN.CENTER,
                                  "left": PP_ALIGN.LEFT,
                                  "right": PP_ALIGN.RIGHT}[p["align"]]
                for s, kind in p["runs"]:
                    run = para.add_run()
                    run.text = s
                    name, italic = _FONTS[kind]
                    sub = kind in ("sub", "sup", "itsub")
                    run.font.size = Pt(p["size"] * (0.72 if sub else 1.0))
                    run.font.name = name
                    run.font.bold = p["bold"]
                    run.font.italic = italic
                    run.font.color.rgb = rgb(p["color"])
                    rPr = run.font._rPr
                    rPr.append(rPr.makeelement(qn("a:ea"), {"typeface": name}))
                    if kind in ("sub", "itsub"):
                        rPr.set("baseline", "-25000")
                    elif kind == "sup":
                        rPr.set("baseline", "30000")
        prs.save(path)
        return path

    # -------------------------------------------------------- 一次性输出
    def write_all(self, outdir, stem, caption=""):
        svg = self.to_svg()
        with open(os.path.join(outdir, stem + ".svg"), "w",
                  encoding="utf-8") as f:
            f.write(svg)
        html = ('<meta charset="utf-8"><title>' + stem + '</title><style>'
                'body{margin:0;background:#eceff2;font-family:' + ZH_F + '}'
                '.wrap{padding:24px}.card{background:#fff;border-radius:10px;'
                'box-shadow:0 2px 20px rgba(20,35,50,.12);padding:16px;'
                'max-width:1360px;margin:0 auto}.cap{max-width:1360px;'
                'margin:14px auto 0;color:#3d4a58;font-size:14px;'
                'line-height:1.7}svg{width:100%;height:auto;display:block}'
                '</style><div class="wrap"><div class="card">' + svg +
                '</div><div class="cap">' + caption + '</div></div>')
        with open(os.path.join(outdir, stem + ".html"), "w",
                  encoding="utf-8") as f:
            f.write(html)
        self.to_pptx(os.path.join(outdir, stem + ".pptx"))
        return len(self.P)
