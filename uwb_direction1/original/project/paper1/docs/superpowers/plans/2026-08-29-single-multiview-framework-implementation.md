# Single-view and Multi-view Framework Implementation Plan

> **For agentic workers:** Execute inline in the current session; do not dispatch subagents.

**Goal:** Produce a two-slide editable PPTX preview containing the single-view baseline and the three-view proposed framework.

**Architecture:** Duplicate the inspected editable source slide twice, replace its slide-local diagram objects with two layouts that share geometry and typography, export through Artifact Tool, then render and inspect both slides.

**Tech Stack:** PowerPoint OOXML imported through `@oai/artifact-tool`, bundled Node runtime, bundled slide renderer and overflow checker.

---

### Task 1: Prepare duplicated source slides

**Files:**
- Create: `.codex_tmp/fig_framework_preview_20260829/template-frame-map.json`
- Create: `.codex_tmp/fig_framework_preview_20260829/template-starter.pptx`

- [ ] Validate that the source has one slide and no raster media dependency.
- [ ] Duplicate source slide 1 into output slides 1 and 2.
- [ ] Render the starter and confirm both copies preserve the source canvas.

### Task 2: Build the editable diagrams

**Files:**
- Create: `.codex_tmp/fig_framework_preview_20260829/build_framework_preview.mjs`
- Create: `figs/fig1_single_vs_multiview_framework_preview.pptx`

- [ ] Replace slide 1 objects with the single-view data flow defined in the design spec.
- [ ] Replace slide 2 objects with the three-view data flow defined in the design spec.
- [ ] Export per-slide PNG and layout JSON evidence.

### Task 3: Verify the deliverable

**Files:**
- Create: `.codex_tmp/fig_framework_preview_20260829/qa-ledger.txt`

- [ ] Run slide overflow detection and require zero overflow errors.
- [ ] Inspect both rendered slides at full size for hierarchy, connector routing, wrapping, and legibility.
- [ ] Reopen the exported PPTX with Artifact Tool and verify two slides and editable text/shape records.
