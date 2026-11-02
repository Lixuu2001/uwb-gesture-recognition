# Figure 2 Typography and Attention Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize the full-slide typography and replace panel (b)'s English-labeled attention bitmap with a text-free scientific visual in the existing editable PowerPoint.

**Architecture:** Use ImageGen to create one text-free raster asset at the current attention-image aspect ratio. Import the current PPTX with `@oai/artifact-tool`, replace only the panel (b) image at its verified frame, normalize text styles by semantic role and position, then export in place and verify the rendered slide and editable structure.

**Tech Stack:** OpenAI ImageGen, JavaScript ES modules, `@oai/artifact-tool`, bundled presentation render and overflow tools.

---

### Task 1: Generate the text-free attention asset

**Files:**
- Reference: `C:\Users\19252\AppData\Local\Temp\codex-clipboard-a06b184c-dc04-4604-b5b6-3a9eaa4ba693.png`
- Create: `D:\uwb_gestures\paper1\.codex_tmp\fig2_typography_attention_20260822\attention_text_free.png`

- [ ] **Step 1: Inspect the supplied reference image**

Use the local image viewer and confirm the four stacked green matrices, three colored interaction paths, shallow 3D depth, and current wide aspect ratio.

- [ ] **Step 2: Generate the replacement**

Use ImageGen in edit/reference mode with an explicit prohibition on words, letters, numbers, equations, legends, and typographic symbols. Preserve a white/transparent-looking background, emerald matrices, muted gold/blue/red paths, and a print-safe scientific illustration style.

- [ ] **Step 3: Inspect the generated image**

Reject any result containing text-like glyphs, garbled labels, neon glow, excessive cartoon styling, or a composition that cannot crop cleanly to the current panel (b) image frame.

### Task 2: Build the typography and image-replacement edit

**Files:**
- Source and output: `C:\Users\19252\Desktop\fig2_three_fusion_立体注意力权重_可编辑.pptx`
- Create: `D:\uwb_gestures\paper1\.codex_tmp\fig2_typography_attention_20260822\source-backup.pptx`
- Create: `D:\uwb_gestures\paper1\.codex_tmp\fig2_typography_attention_20260822\edit_typography_attention.mjs`

- [ ] **Step 1: Preserve the current deck**

Copy the current PPTX to `source-backup.pptx`, then import the backup so every run starts from the same known input.

- [ ] **Step 2: Replace the attention bitmap**

Find the sole panel (b) image at approximately `[675.18, 407, 344.65, 217]`, replace its source with `attention_text_free.png`, retain the frame, and use `contain` or a visually verified crop that does not clip matrix planes or arrows.

- [ ] **Step 3: Normalize text by semantic role**

Apply Microsoft YaHei to Chinese text, Arial to Latin prose/identifiers, and Times New Roman to formulas and mathematical variables. Use the approved sizes: 22 pt panel titles, 14 pt view labels, 12.5–13 pt encoder/core labels, 11.5 pt token-family labels, 10.5–11 pt ranges/variables, 12.5–13 pt equations, 9.5–10 pt secondary annotations, and 19 pt bottom conclusions.

- [ ] **Step 4: Preserve emphasis and layout**

Keep existing bold/italic semantics, avoid changing panel geometry, and ensure all single-line titles remain single line.

- [ ] **Step 5: Record provenance and export**

Update the slide speaker-note `[Sources]` block with the supplied reference image and the generated replacement asset, export one PPTX in place, and save an inspection record and preview under the temporary directory.

### Task 3: Verify the final PowerPoint

**Files:**
- Verify: `C:\Users\19252\Desktop\fig2_three_fusion_立体注意力权重_可编辑.pptx`
- Create: `D:\uwb_gestures\paper1\.codex_tmp\fig2_typography_attention_20260822\verify_final.mjs`
- Create: `D:\uwb_gestures\paper1\.codex_tmp\fig2_typography_attention_20260822\rendered\slide-1.png`

- [ ] **Step 1: Render the exported PPTX**

Render the actual final file, not an unsaved in-memory preview, and inspect the entire slide at full size.

- [ ] **Step 2: Run overflow validation**

Run `slides_test.py` against the exported PPTX. Expected output: `Test passed. No overflow detected.`

- [ ] **Step 3: Run structural validation**

Assert one slide, ten images, the new attention image at the original panel (b) frame, four editable MBT bottleneck tokens, twelve panel (c) curved connectors, and complete source notes.

- [ ] **Step 4: Confirm typography and text-free asset visually**

Confirm same-level labels now have consistent sizes, no panel title wraps, bottom conclusions are balanced, and the replacement bitmap contains no visible English, letters, numbers, equations, or legends.
