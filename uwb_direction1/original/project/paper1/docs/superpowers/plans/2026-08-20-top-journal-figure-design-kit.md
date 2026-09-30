# Top-Journal Deep-Learning Figure Design Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished, fully editable PPTX library of publication-scale deep-learning figure components and complete workflow templates that can be copied directly into the user's UWB paper figures.

**Architecture:** Create a standalone 16:9 PowerPoint deck with native shapes, text, and connectors. Separate publisher requirements from reusable design recommendations, use a consistent Nature-leaning visual grammar, and include source URLs in speaker notes. Export the editable deck plus full-slide renders and a montage for visual QA.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled PowerPoint render and slide-validation utilities.

---

### Task 1: Lock the information and visual system

**Files:**
- Create: `figs/pptx_design_kit/build_top_journal_figure_kit.mjs`

- [ ] Define a 16:9 white-canvas theme using Arial, navy text, neutral grey structure, and a restrained colour-blind-safe accent palette.
- [ ] Define reusable helpers for titles, labels, rounded modules, tokens, containers, arrows, scale bars, and source notes.
- [ ] Encode official Nature and IEEE dimensions separately from the deck's inferred design recommendations.

### Task 2: Author the editable component-library slides

**Files:**
- Modify: `figs/pptx_design_kit/build_top_journal_figure_kit.mjs`

- [ ] Create slides for final-size rules, line and arrow semantics, dashed containers, module grammar, and colour palettes.
- [ ] Label every reusable sample with its intended final-print size, line weight, padding, and semantic role.
- [ ] Add a visible warning wherever a value is a recommended starting point rather than a journal requirement.

### Task 3: Author the complete workflow templates

**Files:**
- Modify: `figs/pptx_design_kit/build_top_journal_figure_kit.mjs`

- [ ] Build an editable one-way encoder template inspired by compact ViT/NeRF method overviews.
- [ ] Build an editable three-strategy fusion comparison template for early, intermediate/bottleneck, and decision fusion.
- [ ] Build an editable dual-stream cross-attention/bottleneck template.
- [ ] Build a copy-ready UWB three-view Figure 2 master using the same visual system.

### Task 4: Export and verify

**Files:**
- Create: `figs/pptx_design_kit/output/top_journal_dl_figure_design_kit.pptx`
- Create: `figs/pptx_design_kit/output/slide-*.png`
- Create: `figs/pptx_design_kit/output/deck-montage.webp`

- [ ] Export every slide as PNG, export a montage, and export the editable PPTX.
- [ ] Run slide-overflow and layout validation.
- [ ] Inspect every slide at full size and revise any overlap, wrapping, weak contrast, or ambiguous arrow routing.
- [ ] Copy the verified PPTX and montage to the Desktop for direct use.

