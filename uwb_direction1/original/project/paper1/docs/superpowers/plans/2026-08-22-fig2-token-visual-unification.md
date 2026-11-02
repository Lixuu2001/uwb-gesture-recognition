# Figure 2 Token Visual Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify temporal and range token representations across panels (a), (b), and (c), while adding an editable Chinese annotation beside the approved text-free attention image.

**Architecture:** Import the current one-slide PPTX with `@oai/artifact-tool`, preserve all approved titles, CIR images, encoders, attention bitmap, and panel boundaries, then replace only the existing token-related shapes and affected connectors with a named native-shape system. A structural verifier and full-slide render provide independent acceptance checks.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled presentation renderer, `slides_test.py`, PowerPoint native shapes and text.

---

### Task 1: Freeze source and inspect editable objects

**Files:**
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_token_unification_20260822/source-backup.pptx`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_token_unification_20260822/current-inspect.ndjson`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_token_unification_20260822/source-notes.txt`

- [ ] **Step 1: Copy the current user deck to the operation-specific backup**

Run PowerShell `Copy-Item -LiteralPath 'C:/Users/19252/Desktop/fig2_three_fusion_立体注意力权重_可编辑.pptx' -Destination 'D:/uwb_gestures/paper1/.codex_tmp/fig2_token_unification_20260822/source-backup.pptx'`.

- [ ] **Step 2: Inspect all slide objects**

Import `source-backup.pptx` and call:

```js
const inspected = await deck.inspect({
  kind: "slide,textbox,shape,image,connector,notes",
  include: "id,slide,name,bbox,text,textPreview,geometry,alt",
  maxChars: 400000,
});
await fs.writeFile(INSPECT, inspected.ndjson, "utf8");
```

Expected: one slide, ten images, three panel titles, four bottleneck token shapes, and the approved attention image at approximately `[675, 407, 345, 217]`.

- [ ] **Step 3: Record provenance**

Write a plain-text source note identifying the user deck, approved attention bitmap, CIR assets, implementation-semantic source, and approved design specification.

### Task 2: Create a failing structural acceptance test

**Files:**
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_token_unification_20260822/verify_token_system.mjs`

- [ ] **Step 1: Implement exact required-name checks**

The verifier imports a deck and requires these native editable objects:

```js
const required = [
  ...["decision", "feature-in", "feature-out", "mbt-in", "mbt-out"].flatMap(stage =>
    ["L", "T", "R"].flatMap(view => [
      `${stage}-${view}-time-label`, `${stage}-${view}-range-label`,
      `${stage}-${view}-time-track`, `${stage}-${view}-range-track`,
    ])
  ),
  "feature-attention-note-rule",
  "feature-attention-note-title",
  "feature-attention-note-qk",
  "feature-attention-note-writeback",
];
for (const name of required) {
  if (!records.some(r => r.name === name)) throw new Error(`Missing ${name}`);
}
```

The verifier also asserts one slide, ten images, three 22-pt titles, four bottleneck tokens, twelve panel-(c) mechanism connectors, and the approved attention-image dimensions.

- [ ] **Step 2: Run the verifier against the untouched backup**

Expected: fail with `Missing decision-L-time-track`, proving the new representation is not already present.

### Task 3: Implement the unified editable token system

**Files:**
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_token_unification_20260822/edit_fig2_token_unification.mjs`
- Modify: `C:/Users/19252/Desktop/fig2_three_fusion_立体注意力权重_可编辑.pptx`

- [ ] **Step 1: Mark one PPTX edit operation**

Run `mark_artifact_operation_started.mjs --operation-kind edit --expected-output-count 1 --output-format pptx` exactly once.

- [ ] **Step 2: Define the shared visual tokens**

Use these fixed design tokens:

```js
const VIEW = {
  L: { line: "#D29A1A", time: "#F8D77C", range: "#FFF5D6" },
  T: { line: "#3E9FC4", time: "#A9DDF0", range: "#E4F5FB" },
  R: { line: "#D96B7B", time: "#F1B7C1", range: "#FBE8EC" },
};
const TOKEN = { cellW: 12, cellH: 8, gap: 2, radius: 1.5, linePt: 0.8 };
const FONT = { cn: "Microsoft YaHei", math: "Times New Roman" };
```

- [ ] **Step 3: Implement one reusable track constructor**

```js
function addTokenTrack(slide, { name, x, y, view, kind, label }) {
  const palette = VIEW[view];
  const fill = kind === "time" ? palette.time : palette.range;
  const group = slide.addGroup({ name, left: x, top: y, width: 82, height: 26 });
  group.addText(label, {
    name: `${name}-label`, left: 0, top: 0, width: 82, height: 11,
    fontFamily: FONT.cn, fontSize: 9.5, bold: true, color: "#303945",
    align: "center", valign: "middle", margin: 0,
  });
  [0, 1, 2, 3].forEach(i => group.addShape("roundRect", {
    name: `${name}-cell-${i + 1}`, left: i * 14, top: 14,
    width: TOKEN.cellW, height: TOKEN.cellH,
    fill: { color: fill }, line: { color: palette.line, width: TOKEN.linePt },
    radius: TOKEN.radius,
  }));
  group.addText("⋯", {
    name: `${name}-ellipsis`, left: 57, top: 11, width: 10, height: 12,
    fontFamily: FONT.math, fontSize: 9, color: "#55606D", margin: 0,
  });
  group.addShape("roundRect", {
    name: `${name}-cell-last`, left: 69, top: 14,
    width: TOKEN.cellW, height: TOKEN.cellH,
    fill: { color: fill }, line: { color: palette.line, width: TOKEN.linePt },
    radius: TOKEN.radius,
  });
  return group;
}
```

- [ ] **Step 4: Replace the token regions in all three panels**

Remove only existing token rectangles, token labels, indexes, token ellipses, and connectors terminating in those objects. Add paired tracks using the names required by Task 2, with `时序 token × 90` above `距离 token × 189`. Keep three view columns aligned within each panel.

- [ ] **Step 5: Preserve panel-specific semantics**

In (a), route each paired track to `dL`, `dT`, or `dR`. In (b), route only time tracks into the attention image and route range tracks around it to their matching output tracks. In (c), connect both tracks to the existing bottleneck mechanism while retaining four purple bottleneck tokens and the compact note `×2：时序 / 距离各一组`.

- [ ] **Step 6: Add the editable attention annotation**

Add a 2-pixel-equivalent emerald rule and three textboxes at the right side of the attention image:

```js
const noteCopy = [
  ["feature-attention-note-title", "四头视角自注意力", 9.8, true, "#173D35"],
  ["feature-attention-note-qk", "行＝查询 · 列＝键", 8.3, false, "#45505E"],
  ["feature-attention-note-writeback", "残差连接后写回各视角", 8.3, false, "#45505E"],
];
```

The annotation must have no card, rounded border, shadow, badge, or icon.

- [ ] **Step 7: Update speaker-note sources and export**

Retain the existing `[Sources]` block and add the approved token-unification specification as an editorial design source. Export the modified deck in place to the requested desktop path.

### Task 4: Verify structure and appearance

**Files:**
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_token_unification_20260822/rendered_final/slide-1.png`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_token_unification_20260822/final-inspect.ndjson`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_token_unification_20260822/qa-ledger.txt`

- [ ] **Step 1: Run the structural verifier**

Expected: all required token-track and annotation names exist; one slide; ten images; four bottleneck tokens; twelve panel-(c) mechanism connectors.

- [ ] **Step 2: Render the final slide**

Run the bundled slide renderer against the actual desktop output, not an intermediate deck.

- [ ] **Step 3: Inspect the full-size render**

Confirm the three panels share identical token shape grammar, labels do not wrap, (b) range bypasses are visually separate from attention, (c) input/output tracks align, and the annotation does not cover the attention asset.

- [ ] **Step 4: Run overflow testing**

Run `slides_test.py` against the final desktop PPTX. Expected: `Test passed. No overflow detected.`

- [ ] **Step 5: Record the final QA result**

Write the structural counts, overflow result, image dimensions, rendered-slide inspection result, file size, and SHA-256 hash to `qa-ledger.txt`.

### Task 5: Deliver the revised deck

**Files:**
- Verify: `C:/Users/19252/Desktop/fig2_three_fusion_立体注意力权重_可编辑.pptx`

- [ ] **Step 1: Use verification-before-completion**

Re-run the final verifier, overflow test, and render inspection immediately before claiming completion.

- [ ] **Step 2: Report only the user-facing outcome**

State that the three panels now share one token grammar, describe the (b) time-attention/range-bypass distinction and the added annotation, cite the final PPTX exactly once, and show the actual final slide render.
