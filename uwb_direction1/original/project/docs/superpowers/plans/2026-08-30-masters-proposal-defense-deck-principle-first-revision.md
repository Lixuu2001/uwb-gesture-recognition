# Masters Proposal Defense Deck Principle-First Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the existing 11-slide proposal defense PPTX so it removes experimental result numbers and explains the research background, sensing principles, and two solution routes clearly within a ten-minute talk.

**Architecture:** Keep the existing Artifact Tool builder and visual system. Rewrite the talk-track source first, then replace the result-heavy slide and evaluation-heavy wording in the builder, export the deck in place, and verify both rendered slides and speaker-note XML.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled presentation render/overflow helpers, PowerShell ZIP/XML inspection.

---

### Task 1: Rewrite the narrative and speaker notes

**Files:**
- Modify: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/talk-track.txt`

- [ ] **Step 1: Replace the result-led sequence**

Set the slide sequence to: background gap, radar sensing principle, three uncertainties, UWB three-view method, masked reconstruction, FMCW causal event model, event state machine, unified route, innovations and feasibility, research plan.

- [ ] **Step 2: Remove experiment-number narration**

Delete references to OSCR, F1, AUROC, exact latency, exact false-alarm rate, and exact error/accuracy percentages. Keep method names only where they help explain a design decision.

- [ ] **Step 3: Preserve direct-speaking notes and sources**

Keep one Chinese spoken paragraph and one `[Sources]` block for each of the 11 slides. The spoken paragraphs must use the pattern “现象—原因—方案—验证方式” and remain suitable for a ten-minute defense.

- [ ] **Step 4: Run narrative checks**

Run:

```powershell
rg -n "95\.14|84\.34|43\.35|81\.30|0\.799|0\.652|63\.0|2\.44|49 ms|AUROC|OSCR" D:\uwb_gestures\.codex_tmp\masters_proposal_ppt_20260830\talk-track.txt
```

Expected: no output.

Run a section/source count and expect `slides=11 sources=11`.

### Task 2: Rebuild slides around principles and solutions

**Files:**
- Modify: `D:/uwb_gestures/.codex_tmp/masters_proposal_ppt_20260830/build_deck.mjs`
- Output: `C:/Users/19252/Desktop/硕士开题答辩PPT_多视角与连续流雷达手势感知_10分钟.pptx`

- [ ] **Step 1: Replace the data slide with a radar sensing principle slide**

Build an editable native-shape flow:

```text
雷达发射 → 手部运动调制 → 回波序列 → 时空特征 → 手势类别 / 未知判断
```

Add two concise branches below the shared flow: `UWB：CIR / 距离结构` and `FMCW：距离-速度 / 微多普勒`.

- [ ] **Step 2: Move the uncertainty framing to slide 4**

Retain the existing spatial, temporal, and category uncertainty diagram, but replace the removed result language with an explanation of how the three uncertainties map to the two research contents.

- [ ] **Step 3: Remove result-oriented wording from the remaining slides**

On the reconstruction slide, replace metric names with qualitative reliability checks: cross-subject stability, sensor-failure degradation, and decision consistency. On the innovation slide, express each innovation as `机制 → 解决的问题 → 验证方式`, and replace “部署指标” language with feasibility resources and protocol boundaries.

- [ ] **Step 4: Export the revised deck**

Immediately before this first edit-authoring run, execute the artifact-operation marker once with `--operation-kind edit --expected-output-count 1 --output-format pptx`. Then run `build_deck.mjs` with the bundled `RUNTIME_NODE`, `RUNTIME_NODE_MODULES`, and `RUNTIME_BIN_DIR` values.

Expected: `slides=11` and the desktop PPTX is overwritten successfully.

### Task 3: Verify structure, notes, and visual quality

**Files:**
- Inspect: `C:/Users/19252/Desktop/硕士开题答辩PPT_多视角与连续流雷达手势感知_10分钟.pptx`

- [ ] **Step 1: Run overflow detection**

Run `slides_test.py` with the final PPTX and expect `Test passed. No overflow detected.`

- [ ] **Step 2: Render every slide**

Run `render_slides.py`, create a montage, inspect the complete montage, and inspect any dense or changed slide individually at full size. Fix all unintended wrapping, clipping, overlap, and connector-direction issues before continuing.

- [ ] **Step 3: Inspect PPTX XML**

Verify:

```text
Slides = 11
NoteSlides = 11
NotesWithSources = 11
ResearchContent3Count = 0
HasExactContent2 = true
```

Also search slide and note text for the deleted experiment-number tokens and expect zero matches.

- [ ] **Step 4: Remove generated desktop scratch files**

Delete only the exact generated `*.pptx.inspect.ndjson` companion file. Preserve the final PPTX and all user-authored files.

- [ ] **Step 5: Record final file evidence**

Report the final file size and SHA-256 hash, then deliver the PPTX with one output citation.
