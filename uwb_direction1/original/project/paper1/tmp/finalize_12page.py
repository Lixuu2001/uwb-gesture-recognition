from __future__ import annotations

import re
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(r"D:\uwb_gestures\paper1")
SRC = ROOT / "mvosr_paper_cn_投稿版_按模板修改.docx"
DST = ROOT / "mvosr_paper_cn_投稿版_12页终稿.docx"


OBJECTIVE = (
    "Ultra-WideBand (UWB) impulse radar enables contactless gesture sensing without forming images and "
    "remains insensitive to illumination and light occlusion. Practical deployment nevertheless introduces "
    "two coupled problems. First, motions outside the trained vocabulary continuously enter the field of view, "
    "whereas a closed-set classifier must assign every input to a known class and may therefore trigger falsely. "
    "Open-set recognition must preserve known-class classification while rejecting gestures never observed during "
    "training. Second, sensing hardware may be blocked or fail. Existing UWB open-set gesture recognition relies "
    "on a single radar and is limited by observation geometry: the echo is the radial projection of a "
    "three-dimensional hand trajectory onto the radar line of sight, so gestures with similar projections are "
    "difficult to separate and the prototype-distance geometry used for rejection degrades. A single sensor also "
    "provides no redundancy. Multiple radars supply complementary projection directions and fault tolerance, but "
    "premature cross-view interaction can couple their representations and make a model fragile when a view "
    "disappears. This study therefore determines where cross-view information should be combined for open-set "
    "decision making and develops one model that remains usable under arbitrary sensor-availability states."
)

METHODS = (
    "A three-view open-set gesture recognition framework is proposed in which radar echoes are encoded and "
    "inspected independently and are combined only at the decision layer. Channel Impulse Response (CIR) "
    "range-slow-time matrices from the left, top and right radars are normalized per sample and per view by taking "
    "absolute values and then standardizing them. This removes session-level scale differences while preserving "
    "the geometric structure of the echo. A single weight-shared dual-path Transformer encodes every view. Its "
    "temporal path treats the matrix as a sequence of range profiles, whereas its range path treats the transpose "
    "as a sequence of slow-time curves. Learnable view embeddings identify the radar and can be sliced with the "
    "input, allowing any subset of views to use the same weights. The closed-set head classifies the cross-view "
    "mean of the feature codes, and no cross-view connection is placed inside the encoder. For open-set decision "
    "making, a deep inspector is trained for every pair of known class and view. Each inspector owns a prototype "
    "and a root-mean-square distance scale in a 128-dimensional latent space. For the predicted class, the "
    "per-view normalized prototype distances are aggregated into an anomaly score containing a cross-view mean "
    "and a cross-view variance. The mean strengthens distance evidence that is consistent across observation "
    "geometries, while the variance captures disagreement among views. To obtain sensor-failure robustness, view "
    "dropout is triggered during training with a fixed probability; when triggered, a view subset is sampled "
    "uniformly from all non-empty proper subsets. Architecture-level controls include feature-level cross-view "
    "attention, bottleneck mid-level fusion, geometry-prior attention and evidential fusion. Experiments use the "
    "public UWB-Gestures dataset under five known/unknown class splits. Both random splitting and Leave-One-Subject-"
    "Out (LOSO) are evaluated with OSCR and AUROC, and sensor-failure tests cover all seven non-empty view subsets "
    "without retraining or recalibration."
)

RESULTS = (
    "Decision-level aggregation increases the mean Open-Set Classification Rate (OSCR) from 86.74% for the "
    "single-view baseline to 95.14%, a gain of 8.40 percentage points that is consistent across all five splits. "
    "The corresponding AUROC is 97.12%, and the best-threshold Accuracy and Macro-F1 are 92.29% and 92.54%, "
    "respectively. On split 1, feature-level fusion and bottleneck fusion obtain OSCR values of 95.30% and 95.96%, "
    "which are statistically indistinguishable from the 95.99% of decision-level aggregation. Routing the fused "
    "features into the inspectors instead lowers OSCR by 0.69 points. Under LOSO, bottleneck fusion falls to "
    "83.43% +/- 8.45% compared with 85.22% +/- 5.55% without fusion, showing both lower accuracy and greater "
    "cross-subject variability. Geometry-prior attention changes OSCR by only -0.04 points, and evidential fusion "
    "reaches 81.2%, below the 92.6% of the corresponding distance-inspector configuration. Naive three-view "
    "training is fragile when sensors fail: OSCR falls to 43.35% when only the right radar remains. View dropout "
    "removes this defect with negligible full-view cost, giving 94.88% instead of 95.33% with all views, retaining "
    "90.81%-92.35% after any single-sensor failure, and raising the worst single-view result from 43.35% to 81.30%. "
    "Under LOSO, three-view recognition achieves 84.34% versus 55.31% for the single-view baseline, so the "
    "multi-view gain expands from 8.40 to 29.03 points; the cross-subject cost is 10.80 points for three views "
    "compared with 31.43 points for one view."
)

CONCLUSIONS = (
    "The open-set benefit of multi-view sensing originates from the ensemble of several nearly independent "
    "per-view distance signals rather than from richer features produced by early cross-view fusion. Because "
    "closed-set accuracy is already saturated, cross-entropy provides little pressure to learn useful inter-view "
    "geometry, and feature coupling can weaken the independence required by distance aggregation. Random view "
    "dropout is an effective and inexpensive remedy for the resulting sensor-failure fragility, allowing one model "
    "to degrade gracefully across all sensor states. A residual cross-subject degradation of about ten points, "
    "together with the input-version difference in the single-view comparison, remains the main limitation. Future "
    "work will address continuous sensor degradation and cross-view masked reconstruction, and will integrate "
    "unknown rejection, missing-view completion and sensor-fault detection within one framework."
)


REF41 = (
    "[41] LI M C, YANG D K, ZHAO X, et al. Correlation-decoupled knowledge distillation for multimodal "
    "sentiment analysis with incomplete modalities[C]. The IEEE/CVF Conference on Computer Vision and Pattern "
    "Recognition, Seattle, USA, 2024: 12458-12468."
)

REF42 = (
    "[42] ZHUANG Y, LIU M H, BAI W, et al. CMAD: correlation-aware and modalities-aware distillation for "
    "multimodal sentiment analysis with missing modalities[C]. The IEEE/CVF International Conference on Computer "
    "Vision, 2025: 4626-4636."
)


def set_fonts(run, size=9, bold=None, superscript=None):
    run.font.name = "Euclid"
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if superscript is not None:
        run.font.superscript = superscript
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:ascii"), "Euclid")
    fonts.set(qn("w:hAnsi"), "Euclid")
    fonts.set(qn("w:eastAsia"), "宋体")


def replace_visible_text_preserve_markup(p, new_text: str):
    nodes = p._p.xpath(".//w:t")
    if not nodes:
        r = p.add_run(new_text)
        set_fonts(r, 10.5)
        return
    nodes[0].text = new_text
    nodes[0].set(qn("xml:space"), "preserve")
    for node in nodes[1:]:
        node.text = ""


def replace_plain(p, text: str, *, size=10.5):
    p.clear()
    r = p.add_run(text)
    set_fonts(r, size)


def replace_with_superscript_citation(p, before: str, citation: str, after: str):
    p.clear()
    r = p.add_run(before)
    set_fonts(r, 10.5)
    r = p.add_run(citation)
    set_fonts(r, 10.5, superscript=True)
    r = p.add_run(after)
    set_fonts(r, 10.5)


def replace_labeled_english(p, label: str, body: str):
    p.clear()
    r = p.add_run(label)
    set_fonts(r, 9, bold=True)
    r = p.add_run(" " + body)
    set_fonts(r, 9, bold=False)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.line_spacing = Pt(16)
    p.paragraph_format.keep_together = False


def remove_paragraph(p):
    p._p.getparent().remove(p._p)


def find_start(doc, prefix):
    return next(p for p in doc.paragraphs if p.text.startswith(prefix))


def comment_counts(path: Path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
        comments = z.read("word/comments.xml").decode("utf-8")
    return (
        comments.count("<w:comment "),
        xml.count("<w:commentRangeStart"),
        xml.count("<w:commentRangeEnd"),
        xml.count("<w:commentReference"),
    )


def word_count(*parts):
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?%?", " ".join(parts)))


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    before_comments = comment_counts(SRC)
    doc = Document(str(SRC))

    # Capture target paragraphs before any deletion shifts document indices.
    overview = find_start(doc, "该架构沿用文献[33]")
    roadmap = find_start(doc, "后文组织与此对应")
    fusion_random = find_start(doc, "3种策略在统计上无法区分")
    fusion_loso = find_start(doc, "随机切分下三者难以区分")
    fusion_extra = find_start(doc, "2.4节给出的两种更显式")

    # Compress repeated Chinese exposition while keeping every decisive result.
    replace_visible_text_preserve_markup(
        overview,
        "该架构沿用文献[33]的两阶段训练范式：第一阶段训练编码器与闭集分类器，第二阶段冻结特征并为每个（已知类，视角）对训练检查器。与单视角基线相比，本文将单一距离判决扩展为3个观测几何下的独立判决，并仅在决策级聚合。",
    )
    remove_paragraph(roadmap)
    replace_with_superscript_citation(
        fusion_random,
        "随机切分下，决策级聚合、特征级融合和瓶颈融合的OSCR分别为95.99、95.30和95.96，差异不显著；将特征级融合结果接入检查器后OSCR下降0.69，说明特征耦合削弱了逐视角距离的独立性。LOSO下，瓶颈融合亦由无融合的85.22±5.55降至83.43±8.45，且折间波动增大。几何先验注意力仅变化−0.04，证据式融合为81.2，均未优于距离式决策级聚合。结果表明，跨视角交互不宜前移到特征层，开集增益主要来自独立距离信号的集成，与文献",
        "[37]",
        "一致。",
    )
    remove_paragraph(fusion_loso)
    remove_paragraph(fusion_extra)

    # Condense the long English abstract while staying above the journal's 800-word floor.
    en_map = {
        "Objective:": OBJECTIVE,
        "Methods:": METHODS,
        "Results and Discussions:": RESULTS,
        "Conclusions:": CONCLUSIONS,
    }
    for label, body in en_map.items():
        p = find_start(doc, label)
        replace_labeled_english(p, label, body)

    wc = word_count(OBJECTIVE, METHODS, RESULTS, CONCLUSIONS)
    if not 800 <= wc <= 1200:
        raise RuntimeError(f"English abstract word count out of range: {wc}")

    # Replace two old generic distillation references with recent, directly relevant work.
    ref41 = find_start(doc, "[41]")
    ref42 = find_start(doc, "[42]")
    replace_plain(ref41, REF41, size=8)
    replace_plain(ref42, REF42, size=8)

    if DST.exists():
        DST.unlink()
    doc.save(str(DST))

    after_comments = comment_counts(DST)
    if after_comments != before_comments:
        raise RuntimeError(f"Comment structure changed: {before_comments} -> {after_comments}")

    check = Document(str(DST))
    refs = [p.text for p in check.paragraphs if re.match(r"^\[\d+\]\s", p.text)]
    recent = sum(("2024" in r or "2025" in r) for r in refs)
    print(f"Saved: {DST}")
    print(f"English abstract words: {wc}")
    print(f"Recent references: {recent}/{len(refs)} = {recent/len(refs):.2%}")
    print(f"Comments preserved: {after_comments}")


if __name__ == "__main__":
    main()
