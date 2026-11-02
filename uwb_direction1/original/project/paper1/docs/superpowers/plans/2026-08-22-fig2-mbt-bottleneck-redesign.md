# Figure 2 MBT Bottleneck Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace panel (c)'s middle MBT mechanism with an editable, implementation-correct, high-level-journal bottleneck diagram while preserving the rest of the one-slide deck.

**Architecture:** Import the current PPTX with `@oai/artifact-tool`, inspect and resolve panel (c) elements by stable ids/names, delete only the approved middle-region objects, then add connector-first native shapes and editable text. Export back to the same user-authorized path after keeping a temporary backup and verify the final slide structurally and visually.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled presentation render and overflow tools.

---

### Task 1: Inspect and map the existing slide

**Files:**
- Create: `.codex_tmp/fig2_mbt_redesign_20260822/inspect_source.mjs`
- Create: `.codex_tmp/fig2_mbt_redesign_20260822/template-audit.txt`
- Create: `.codex_tmp/fig2_mbt_redesign_20260822/template-frame-map.json`
- Create: `.codex_tmp/fig2_mbt_redesign_20260822/deviation-log.txt`

- [ ] Copy the user-authorized PPTX to the temporary directory as `source-backup.pptx`.
- [ ] Import and inspect slide, shape, textbox, image, notes, and layout records.
- [ ] Render the source slide at scale 2 and export layout JSON.
- [ ] Identify every panel (c) object with a bounding box in the editable middle region and classify it as keep, delete, or replace.

### Task 2: Implement the editable MBT mechanism

**Files:**
- Create: `.codex_tmp/fig2_mbt_redesign_20260822/edit_mbt_panel.mjs`
- Modify in place: `C:/Users/19252/Desktop/fig2_three_fusion_立体注意力权重_可编辑.pptx`

- [ ] Run the presentation edit marker exactly once.
- [ ] Import `source-backup.pptx` and delete only mapped panel (c) middle objects.
- [ ] Add twelve colored curves before nodes so the lines remain behind tokens.
- [ ] Add grouped input/output token columns, group hairlines, `90/189` labels, and updated-state labels.
- [ ] Add the two-layer purple bottleneck frame and four 40 x 17 token rectangles.
- [ ] Add `N_B=4`, view-average formula, temporal/range annotation, and separate-readout note.
- [ ] Preserve and extend the existing `[Sources]` speaker-note block.
- [ ] Export to the user-authorized in-place destination.

### Task 3: Verify the edited presentation

**Files:**
- Create: `.codex_tmp/fig2_mbt_redesign_20260822/verify_final.mjs`
- Create: `.codex_tmp/fig2_mbt_redesign_20260822/final-slide-1.png`
- Create: `.codex_tmp/fig2_mbt_redesign_20260822/final-slide-1.layout.json`

- [ ] Re-import the final PPTX and verify one slide, retained panel (b) attention image, retained CIR images, four bottleneck rectangles, and removed old MBT legend objects.
- [ ] Render the final slide and inspect panel (c) at full size for line crossings, label legibility, and visual balance.
- [ ] Run `slides_test.py` and require `Test passed. No overflow detected.`
- [ ] Confirm the final PPTX is non-empty and the temporary source backup remains available.

