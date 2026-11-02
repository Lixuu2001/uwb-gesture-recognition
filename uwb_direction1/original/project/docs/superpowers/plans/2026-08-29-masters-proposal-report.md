# Masters Proposal Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete, source-grounded Wuhan University of Technology master's topic-proposal DOCX using the official template, with a 5000+ Chinese-character literature review, 40+ verified references, one technical-route figure, and full-page visual QA.

**Architecture:** Preserve the original DOC as the format authority, convert it once to a working DOCX, and insert three clearly labeled attachment blocks for literature review, investigation report, and topic proposal while leaving all identity and evaluation fields blank. Build content from a repository evidence ledger and a separately verified bibliography, then generate the DOCX and render it through Microsoft Word for structural and visual checks.

**Tech Stack:** Microsoft Word COM automation, Python 3.13 with `python-docx` 1.2.0, OOXML, SVG/PNG, PowerShell, `pdftoppm`, official publisher/DOI/dataset pages.

---

### Task 1: Freeze inputs and distill the official template

**Files:**
- Read: `C:/Users/19252/Desktop/附件3_硕士研究生选题报告书.doc`
- Read: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/template_reference.docx`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/artifact.md`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/template_inventory.json`

- [ ] **Step 1: Record the immutable source hash and page contract**

Run:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'C:\Users\19252\Desktop\附件3_硕士研究生选题报告书.doc'
pdfinfo 'D:\uwb_gestures\.codex_tmp\masters_proposal_20260829\template_reference.pdf'
```

Expected: SHA-256 `11B7C95DF59323470043A9849D76E85B03608432CCEEBC3F82CD375952A1271D`, 8 A4 pages.

- [ ] **Step 2: Inventory sections, tables, styles, headers, footers, fields and drawings**

Use `python-docx` plus read-only ZIP/XML inspection to write exact measurements to `artifact.md` and machine-readable counts to `template_inventory.json`. Record two sections, seven tables, page margins, header/footer distances, row/column grids, the page-number field, and every fixed form label.

- [ ] **Step 3: Define the editable slots**

In `artifact.md`, define only these changes: title on cover and page 3; topic classification as application research; source as self-selected topic; page-3 literature-review cell as “详见附页一”; page-4 investigation cell as “详见附页二”; page-5 proposal cell as “详见附页三”; insert the three attachment blocks before the evaluation forms. Explicitly preserve identity fields, adviser fields, signature fields, opinions, dates, checkboxes not requested by the user, and all evaluation tables.

- [ ] **Step 4: Verify the source remains unchanged**

Run the SHA-256 command again. Expected: the same source hash.

### Task 2: Build a repository evidence ledger

**Files:**
- Read: `D:/uwb_gestures/paper1/mvosr_paper.tex`
- Read: `D:/uwb_gestures/paper3/fsgr_paper.tex`
- Read: `D:/uwb_gestures/proposal_v2/01_已有工作台账.md`
- Read: `D:/uwb_gestures/proposal_v2/28_块二方向v3_FMCW连续流手势时序分割_核查与冒烟实验.md`
- Read: `D:/uwb_gestures/docs/RESULTS.md`
- Read: `D:/uwb_gestures/tools/fmcw_stream.py`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/evidence_ledger.md`

- [ ] **Step 1: Record the six research points and evidence status**

Create a table with columns `research_point`, `claim`, `status`, `artifact`, `exact_location`, `allowed_wording`. Use only `已完成`, `已建立基线/先导验证`, and `拟开展/待验证` as status values.

- [ ] **Step 2: Independently verify headline numbers**

Cross-check source tables/logs for: multi-view OSCR `95.14±0.81`; single-view OSCR `86.74±1.29`; LOSO multi-view `84.34±2.80`; LOSO single-view `55.31±5.62`; missing-sensor range `90.8–92.4`; few-shot new-class accuracy `87.3%` after routing correction; reconstruction-signal AUROC `95.48`; FMCW user1–8 event F1 `0.799`, onset MAE `1.61` frames, and false alarms `2.44/min` at the recorded threshold.

- [ ] **Step 3: Record mandatory limitations**

Include the fixed-window label construction, two incompatible FMCW annotation conventions, semi-synthetic concatenated streams, unresolved MIRA reproduction, no UWB continuous-stream claim, and unrun timing/open-set closure experiments.

- [ ] **Step 4: Run ledger consistency checks**

Search for unsupported status language:

```powershell
rg -n '已经解决|全面优于|首次|真实长流|真实终点|UWB连续流已验证' 'D:\uwb_gestures\.codex_tmp\masters_proposal_20260829\evidence_ledger.md'
```

Expected: no matches unless inside a clearly marked prohibited-wording column.

### Task 3: Assemble and verify at least 40 references

**Files:**
- Read: `D:/uwb_gestures/paper1/mvosr_paper.tex`
- Read: `D:/uwb_gestures/paper3/fsgr_paper.tex`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/references_verified.json`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/references_gbt7714.md`

- [ ] **Step 1: Build four literature groups**

Select authoritative work for: radar/UWB gesture sensing and datasets; open-set recognition, few-shot registration and conformal calibration; multi-view/missing-modality reconstruction; online action detection, temporal segmentation and open-set temporal localization.

- [ ] **Step 2: Verify every bibliographic field against a primary source**

For each entry store `authors`, `title`, `venue`, `year`, `volume`, `issue`, `pages`, `doi_or_official_url`, `country_group`, and `verified_on`. Prefer the published version over arXiv. Do not retain an entry if title, venue or year cannot be confirmed.

- [ ] **Step 3: Produce GB/T 7714 entries**

Generate at least 42 entries so the final count remains above 40 after deduplication. Ensure at least 14 are foreign publications; target at least 30 foreign publications for margin.

- [ ] **Step 4: Run reference checks**

Expected checks: unique normalized titles; contiguous numbering; `count >= 40`; `foreign_count * 3 >= count`; every item has DOI or official URL; no duplicate DOI; no fabricated page ranges.

### Task 4: Draft the complete Chinese report text

**Files:**
- Read: `D:/uwb_gestures/docs/superpowers/specs/2026-08-29-masters-proposal-report-design.md`
- Read: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/evidence_ledger.md`
- Read: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/references_gbt7714.md`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/report_body.md`

- [ ] **Step 1: Write attachment one — literature review**

Write at least 5000 Chinese characters excluding the reference list. Use the four literature groups, compare methods and limitations, and end by deriving the two-block research question. Keep citations numeric and adjacent to supported claims.

- [ ] **Step 2: Write attachment two — investigation report**

State the investigation period as project-document and public-data investigation conducted in 2026, the location as the local research workspace and public online repositories, the investigated objects, procedures, and principal findings. Do not invent visits to laboratories, companies, or interviews.

- [ ] **Step 3: Write attachment three — topic proposal**

Use these exact top-level sections: `1 课题题目及来源`, `2 研究目的与意义`, `3 国内外研究现状`, `4 研究目标、研究内容与关键问题`, `5 研究方法与技术路线`, `6 预期成果与创新点`, `7 研究进度与论文写作计划`.

- [ ] **Step 4: Separate completed and planned work**

For each of the six research points, include one status sentence. Completed claims must cite the evidence ledger. Planned claims must use future tense and include baselines, split, metrics and a stop/acceptance rule.

- [ ] **Step 5: Run prose and requirement checks**

Check: literature review Chinese-character count `>=5000`; all seven required sections exist; no identity placeholders; no `TBD`, `TODO`, `Claude`, `GPT`, `Codex`, internal memory citations, or tool tokens; no paragraph beginning with formulaic phrases such as “随着……快速发展” more than once; every numeric project claim occurs in the evidence ledger.

### Task 5: Create the technical-route figure

**Files:**
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/technical_route.svg`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/technical_route.png`

- [ ] **Step 1: Draw the two-lane route**

Use a left-to-right layout with a common problem statement, an upper UWB lane and lower FMCW lane, six compact method blocks, two output blocks, and a shared deployment-evaluation block. Use black/grey typography with restrained blue and teal; no gradients or decorative icons.

- [ ] **Step 2: Export a Word-safe raster**

Export at 2400 px minimum width with a white background. Keep all text in Chinese, ensure the smallest label remains readable at 15.5 cm page width, and add the caption `图1 面向开放环境的多视角与连续流雷达手势感知技术路线`.

- [ ] **Step 3: Inspect the image**

Confirm no clipped text, crossing arrows, tiny labels, or scientific mismatch with the report body.

### Task 6: Author the DOCX from the template

**Files:**
- Read: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/template_reference.docx`
- Read: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/artifact.md`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/build_proposal.py`
- Create: `C:/Users/19252/Desktop/硕士研究生选题报告书_多视角与连续流雷达手势感知_初稿.docx`

- [ ] **Step 1: Mark the artifact operation**

Run exactly once before authoring:

```powershell
node 'C:\Users\19252\.codex\plugins\cache\openai-primary-runtime\documents\26.813.12317\skills\documents\container_tools\mark_artifact_operation_started.mjs' --operation-kind create --expected-output-count 1 --output-format docx
```

Expected: exit code 0.

- [ ] **Step 2: Build from a copied template**

The script must copy the converted template, fill the title, add the three “详见附页” pointers, insert attachment pages before the evaluation-form section, reuse the existing page geometry/header/footer, apply Songti 12 pt and exactly 18 pt line spacing to body paragraphs, and insert the route PNG at no more than 15.5 cm width.

- [ ] **Step 3: Preserve empty fields and evaluation pages**

Assert that student name, number, category, major, advisers, unit and dates remain empty; verify the text of sections four to six and the evaluation tables remains unchanged from the reference.

- [ ] **Step 4: Save the desktop deliverable without overwriting the source**

Expected: one new DOCX with a distinct filename; the original `.doc` hash remains unchanged.

### Task 7: Run structural and content QA

**Files:**
- Read: `C:/Users/19252/Desktop/硕士研究生选题报告书_多视角与连续流雷达手势感知_初稿.docx`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/final_qa.json`

- [ ] **Step 1: Check package integrity**

Open with `python-docx`; verify the package is readable, all images resolve, section geometry matches the source, headers/footers exist, and there are no comments, tracked changes, or internal citation tokens.

- [ ] **Step 2: Check requirements programmatically**

Assert title present; seven proposal sections present; literature review count `>=5000`; reference count `>=40`; foreign ratio `>=1/3`; one route figure and caption present; identity/evaluation fields blank; all reference numbers used in text exist in the bibliography.

- [ ] **Step 3: Check source fidelity**

Compare fixed form labels and evaluation-table text against `template_inventory.json`. Fail if any protected label, checkbox group, signature line or scoring table disappears.

### Task 8: Render, inspect every page, and deliver

**Files:**
- Read: `C:/Users/19252/Desktop/硕士研究生选题报告书_多视角与连续流雷达手势感知_初稿.docx`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/final_render/final.pdf`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/final_render/page-*.png`

- [ ] **Step 1: Export through Microsoft Word**

Open the final DOCX with a hidden Word COM instance, update page fields, export PDF, close without altering the source template, and report the resulting page count.

- [ ] **Step 2: Render every page to PNG**

Run `pdftoppm -png -r 144 final.pdf page`. Expected: one PNG per PDF page.

- [ ] **Step 3: Inspect every page at 100%**

Check cover, instructions, page-3 form, all literature-review pages, investigation report, topic report, route figure, references, and all evaluation forms. Reject clipping, overlapping, broken tables, missing glyphs, orphaned headings, large accidental gaps, or blank trailing pages.

- [ ] **Step 4: Rebuild and rerender until clean**

Each layout change requires a fresh Word export and inspection of every regenerated page.

- [ ] **Step 5: Final verification**

Re-run `final_qa.json` checks and SHA-256 the deliverable. Confirm the desktop DOCX exists, opens, and is non-empty before reporting completion.

