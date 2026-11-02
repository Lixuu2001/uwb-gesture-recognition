# Master's Proposal Defense Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an editable 11-slide Chinese PowerPoint for a 10-minute master's proposal defense, with presenter notes and three polished technical diagrams.

**Architecture:** A single JavaScript ES module uses `@oai/artifact-tool` to create the deck from scratch in a custom navy/teal/orange academic visual system. Source text and evidence are locked in a separate text ledger before authoring; the builder exports one PPTX, then bundled rendering and overflow tools provide structural and visual QA.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled presentation runtime, Microsoft PowerPoint/LibreOffice renderer, bundled slide QA helpers.

---

## File Structure

- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/source-notes.txt` — exact claims, evidence boundaries, slide-by-slide sources.
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/talk-track.txt` — visible copy and presenter notes for all 11 slides.
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/build_deck.mjs` — complete deck authoring module.
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/qa-notes.txt` — render, overflow, notes, and content checks.
- Create: `C:/Users/19252/Desktop/硕士开题答辩PPT_多视角与连续流雷达手势感知_10分钟.pptx` — final deliverable.

### Task 1: Lock source evidence and talk track

**Files:**
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/source-notes.txt`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/talk-track.txt`
- Read: `C:/Users/19252/Desktop/硕士研究生选题报告书_多视角与连续流雷达手势感知_初稿.docx`
- Read: `D:/uwb_gestures/.codex_tmp/masters_proposal_20260829/report_body.md`

- [ ] **Step 1: Record the claims that may appear as completed work**

Write only the following numeric claims into the evidence ledger: UWB OSCR 95.14±0.81%, LOSO OSCR 84.34±2.80%, worst single-view 43.35%→81.30%, 5-shot new-class 75.5%→87.3%, FMCW event F1 0.799, onset MAE 1.61 frames ≈49 ms, false alarms 2.44/min, MSP open-set AUROC 0.652, and unknown-event false commitment 63.0%.

- [ ] **Step 2: Write the 11-slide visible copy**

Keep each slide to one audience-facing conclusion and 60–90 Chinese characters of visible copy. Use exactly two research-content labels and never use “研究内容三”.

- [ ] **Step 3: Write presenter notes**

Write 130–220 Chinese characters per content slide, shorter notes for cover and closing, and end every slide's notes with a `[Sources]` block referencing the proposal DOCX and relevant repository evidence.

- [ ] **Step 4: Validate the script duration**

Count Chinese characters in all notes. Target 2,200–2,800 Chinese characters, corresponding to approximately 9.5–10 minutes at 250–280 Chinese characters per minute including pauses.

### Task 2: Build the visual system and slide helpers

**Files:**
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/build_deck.mjs`

- [ ] **Step 1: Initialize the deck**

Create a 16:9 presentation, set document metadata, define navy `#0B1F33`, teal `#1E8A8A`, orange `#E58A2B`, pale backgrounds, and font families Microsoft YaHei/Aptos.

- [ ] **Step 2: Implement repeated layout helpers**

Implement helpers for slide title, footer, rounded evidence callout, metric strip, process node, connector, and speaker notes. Footer is omitted on slide 1 and shows the short topic plus page number on slides 2–11.

- [ ] **Step 3: Implement note injection**

Attach each slide's talk track as native PowerPoint speaker notes, preserving the `[Sources]` block verbatim.

### Task 3: Author slides 1–6

**Files:**
- Modify: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/build_deck.mjs`

- [ ] **Step 1: Create cover and deployment-gap slides**

Slide 1 uses only the proposal title, defense label, and blank identity line. Slide 2 contrasts pre-segmented closed-set samples with continuous open-environment input through one left-to-right comparison.

- [ ] **Step 2: Create the three-uncertainty story diagram**

Slide 3 uses three native-shape branches for spatial, temporal, and semantic uncertainty, with concise labels and a single bottom conclusion that connects them to the two research contents.

- [ ] **Step 3: Create the feasibility-evidence slide**

Slide 4 uses a restrained metric composition: UWB evidence on the left, FMCW evidence on the right, and an orange “尚未解决” callout for the 0.652/63.0% open-set gap.

- [ ] **Step 4: Create research content one slides**

Slide 5 explains view-independent encoding, decision-level open-set aggregation, and random-view-dropout. Slide 6 shows the editable process “mask/reconstruct → reliability evidence → missing-view decision → few-shot registration”.

### Task 4: Author slides 7–11

**Files:**
- Modify: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/build_deck.mjs`

- [ ] **Step 1: Create the FMCW task-definition slide**

Slide 7 shows continuous radar frames entering a causal model and producing event, onset, class, and uncertainty outputs; label the current data as constructed continuous replay, not natural long stream.

- [ ] **Step 2: Create the commitment-state-machine slide**

Slide 8 uses connectors behind nodes for “background → candidate → retractable early prediction → stable commitment → cooldown”, with a rejection/cache branch for stable unknown evidence and a small risk–latency axis.

- [ ] **Step 3: Create the overall technical-route slide**

Slide 9 uses two parallel editable lanes for UWB and FMCW, merging at known output, unknown cache, manual confirmation, and vocabulary update.

- [ ] **Step 4: Create innovation/risk and schedule slides**

Slide 10 presents exactly two innovations with matching evidence gates and three risk controls. Slide 11 uses a four-stage horizontal schedule and ends with a one-sentence research value statement.

### Task 5: Export and perform structural QA

**Files:**
- Modify: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/build_deck.mjs`
- Create: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/qa-notes.txt`
- Create: `C:/Users/19252/Desktop/硕士开题答辩PPT_多视角与连续流雷达手势感知_10分钟.pptx`

- [ ] **Step 1: Run the artifact operation marker exactly once**

Run the presentations-skill marker with `--operation-kind create --expected-output-count 1 --output-format pptx` immediately before the first authoring execution.

- [ ] **Step 2: Export the deck**

Run `build_deck.mjs` with the bundled Node runtime and confirm the final PPTX exists and contains 11 slides.

- [ ] **Step 3: Verify notes and content boundaries**

Inspect the PPTX to confirm 11 speaker-note sections, exactly one “研究内容二：FMCW 雷达连续流手势事件感知” heading, zero “研究内容三”, and all five key FMCW values.

- [ ] **Step 4: Run overflow tests**

Run `slides_test.py` against the final PPTX. Expected result: no elements outside the slide canvas; investigate and fix every reported overlap or clipping issue.

### Task 6: Render and visually verify every slide

**Files:**
- Modify: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/build_deck.mjs`
- Modify: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/qa-notes.txt`

- [ ] **Step 1: Render all slides**

Run the bundled `render_slides.py` and confirm 11 PNG files are produced.

- [ ] **Step 2: Inspect the montage for deck-level rhythm**

Create a montage and verify title hierarchy, color consistency, alternating silhouettes, footer consistency, and balanced information density.

- [ ] **Step 3: Inspect all 11 slides individually**

Check full-size renders for title wrapping, Chinese font fallback, connector crossings, clipped labels, distorted diagrams, excessive empty space, and small unreadable annotations.

- [ ] **Step 4: Correct and rerun all QA**

Revise `build_deck.mjs`, re-export, rerender, and rerun overflow and notes checks until all issues are resolved. Record final slide count, render count, notes count, file hash, and inspected-slide list in `qa-notes.txt`.
