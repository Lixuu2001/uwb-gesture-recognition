# Figure 2 GPT Image Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and visually verify one high-fidelity GPT Image preview of the approved three-column Figure 2 fusion-strategy comparison.

**Architecture:** Use the current rendered PPT slide as the composition reference and the three real L/T/R CIR images as scientific-content references. Generate one publication-style raster concept with corrected method logic, then inspect the output at full size for technical structure, hierarchy, color semantics, and text limitations before presenting it to the user.

**Tech Stack:** Built-in GPT Image generation, local PowerPoint render, Codex image viewer, Markdown image preview.

---

### Task 1: Validate visual references

**Files:**
- Read: `D:/uwb_gestures/paper1/.codex_tmp/fig2_review_20260819/fig2_three_fusion_source/slide-1.png`
- Read: `D:/uwb_gestures/paper1/figs/assets/fig2_real_cir_L.png`
- Read: `D:/uwb_gestures/paper1/figs/assets/fig2_real_cir_T.png`
- Read: `D:/uwb_gestures/paper1/figs/assets/fig2_real_cir_R.png`

- [ ] **Step 1: Open all four references at full size**

Confirm that the current slide has three panels in the order decision-level, post-encoding feature-level, and MBT-style mid-level fusion. Confirm that the three CIR images correspond to L, T, and R views and are visually distinct.

- [ ] **Step 2: Record invariant content**

Use the following invariants in the generation prompt:

```text
Keep three equal-width panels in the order (a), (b), (c).
Keep real UWB CIR heatmap thumbnails for L, T, and R.
Keep view-wise shared-parameter encoding and a common downstream inspector/distance-score path.
Show cross-view communication only at the decision end in (a), after complete encoding in (b), and through shared bottleneck tokens inside the encoder in (c).
```

### Task 2: Generate the high-fidelity concept

**Files:**
- Reference: `D:/uwb_gestures/paper1/docs/superpowers/specs/2026-08-19-fig2-fusion-strategies-gpt-image-design.md`
- Create: built-in GPT Image output under the Codex generated-images location

- [ ] **Step 1: Submit one structured GPT Image prompt**

Use the taxonomy `scientific-educational` and request a wide, white-background, publication-ready academic infographic. Require concise Chinese labels, restrained flat colors, no gradients, no shadows, no 3D effects, no decorative icons, and no invented modules or numerical results.

- [ ] **Step 2: Encode the corrected technical flow**

The prompt must explicitly require:

```text
(a) L/T/R CIR -> independent shared-parameter dual-path encoders -> view-wise inspectors/prototypes -> d_L,d_T,d_R -> normalized mean-variance anomaly score; no cross-view connection before the final score.
(b) L/T/R CIR -> independent shared-parameter dual-path encoders -> complete feature codes z_L,z_T,z_R -> cross-view attention -> updated view-wise features -> the same inspectors, distances, and score; both time-path and range-path information are included in z_v.
(c) L/T/R CIR -> encoder first half -> compact time-plus-range feature blocks -> a few shared purple bottleneck tokens -> bottleneck averaging and write-back -> encoder second half -> the same inspectors, distances, and score; no direct view-to-view bypass around the bottleneck.
```

- [ ] **Step 3: Encode the visual system**

Use low-saturation yellow for L, blue for T, coral red for R, green only for cross-view exchange, purple only for bottleneck tokens, and neutral gray for common modules. Use aligned vertical tracks, strong primary arrows, thin internal interaction lines, and ample white space.

### Task 3: Inspect and present the result

**Files:**
- Read: generated GPT Image output

- [ ] **Step 1: Inspect the generated image at full size**

Verify panel order, shared structure, exchange position, lack of forbidden cross-view shortcuts, color semantics, and overall publication-style finish.

- [ ] **Step 2: Classify text and formula fidelity**

Separate layout quality from generated-text accuracy. Record any malformed Chinese labels or formulas as items to rebuild with editable PowerPoint text rather than treating them as accepted technical content.

- [ ] **Step 3: Present the preview inline**

Show the generated image at its absolute local path. Summarize the strongest improvements, identify any remaining content or typography defects, and ask whether to proceed to a fully editable PowerPoint reconstruction or revise the concept first.
