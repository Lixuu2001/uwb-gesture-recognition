# Master's Proposal Data Survey Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Locally revise only the data-survey section in the current desktop proposal, preserving the existing UWB prose while adding two sourced figures and a formal FMCW dataset description.

**Architecture:** Download two primary-source figures, then use a focused `python-docx` editor to modify the single table cell containing the survey section. Insert all figures inline with adjacent captions/source lines, save back to the same DOCX path, and verify both OOXML structure and every rendered page.

**Tech Stack:** Bundled Python 3.12, python-docx, Pillow, requests, Microsoft Word COM, Poppler, OOXML audits.

---

### Task 1: Acquire and verify primary-source figures

**Files:**
- Create: `C:/Users/19252/Documents/Codex/proposal_data_review_20260830/assets/uwb_dataset_structure.png`
- Create: `C:/Users/19252/Documents/Codex/proposal_data_review_20260830/assets/fmcw_pipeline.png`

- [ ] Download Scientific Data Fig. 5 for UWB-Gestures and Cambridge Core Fig. 1 for the FMCW acquisition/labeling pipeline from their full-size image URLs.
- [ ] Open both local images and confirm that the UWB figure shows the participant/raw/clutter-removed/gesture/view hierarchy and the FMCW figure shows raw radar tensor, range profile, extracted features, and frame labels.
- [ ] Record source page URLs in the edit script for caption source lines.

### Task 2: Build a narrowly scoped DOCX editor

**Files:**
- Create: `C:/Users/19252/Documents/Codex/proposal_data_review_20260830/edit_data_survey.py`
- Modify in place: `C:/Users/19252/Desktop/硕士研究生选题报告书_恢复候选_自动保存于1120.docx`

- [ ] Assert that the document contains exactly one paragraph beginning `UWB-Gestures 数据由` and exactly one paragraph beginning `FMCW 数据包含` before editing.
- [ ] Preserve the UWB paragraph text and formatting; insert the UWB structure image, caption `图2-1  UWB-Gestures数据集整体结构`, and a source line with the Scientific Data article URL immediately afterward.
- [ ] Replace only the existing FMCW paragraph with formal Chinese text covering the exact dataset name, Zenodo DOI, sensor, 49,000-record composition, tensor/label structure, environment imbalance, fixed-window label caveat, and the user1-user8 main protocol.
- [ ] Insert the FMCW pipeline image, caption `图2-2  60 GHz FMCW雷达数据采集、特征提取与逐帧标注流程`, and a source line with the Cambridge Core/Zenodo URLs immediately after the replacement text.
- [ ] Keep images inline, centered, proportional, and within the existing cell width; retain all unrelated tables, sections, comments, and document properties.
- [ ] Before authoring, run the Documents artifact marker once with operation `edit`, output count `1`, and format `docx`; then save directly to the user-named desktop file.

### Task 3: Structural and content verification

**Files:**
- Inspect: `C:/Users/19252/Desktop/硕士研究生选题报告书_恢复候选_自动保存于1120.docx`

- [ ] Reopen the saved DOCX and assert that the UWB paragraph is unchanged, the old FMCW paragraph is absent, the formal dataset name and both DOI/source URLs are present, and exactly two new inline images/captions were added.
- [ ] Run the Documents image audit and inspect DOCX ZIP relationships/content types for both image parts.
- [ ] Confirm the file remains readable by Microsoft Word and that no temporary placeholders or internal citation tokens are present.

### Task 4: Render and inspect the complete document

**Files:**
- Create for QA only: `C:/Users/19252/Documents/Codex/proposal_data_review_20260830/final_render/proposal.pdf`
- Create for QA only: `C:/Users/19252/Documents/Codex/proposal_data_review_20260830/final_render/page-*.png`

- [ ] Export the edited DOCX to PDF with Word COM and rasterize every page with bundled Poppler.
- [ ] Inspect every page image at full-page scale, then inspect the two figure pages at original resolution.
- [ ] If a caption separates from its figure, a figure is blurry/cropped, text overlaps, or a large avoidable blank area appears, adjust sizing/paragraph pagination and repeat the full render.
- [ ] Record final page count, file size, hash, inspected page list, and the pages containing `图2-1` and `图2-2` before delivery.
