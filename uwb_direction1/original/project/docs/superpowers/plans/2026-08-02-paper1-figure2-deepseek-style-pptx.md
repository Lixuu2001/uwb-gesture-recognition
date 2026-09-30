# Paper 1 Figure 2 DeepSeek-Style PPTX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one high-density, fully editable PowerPoint slide that redraws Paper 1 Figure 2 as a common backbone with three DeepSeek-style technical insets.

**Architecture:** Author a single 1280 × 720 slide from native PowerPoint shapes through `@oai/artifact-tool`. A left-side backbone explains the shared dual-path encoder and inspector pipeline; three right-side dashed insets expose decision-, feature-, and bottleneck-level cross-view exchange while preserving the scientific topology frozen in the approved design.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled PowerPoint render/overflow tools, and Node structural validation.

---

## File Structure

- Create `paper1/tmp/fig2_deepseek_style/build_fig2_deepseek_style_v2.mjs`: temporary artifact-tool builder and export logic.
- Create `paper1/tmp/fig2_deepseek_style/validate_fig2_deepseek_style_v2.mjs`: temporary structural validator.
- Generate `paper1/tmp/fig2_deepseek_style/fig2_fusion_deepseek_style_v2.layout.json`: temporary object/layout snapshot.
- Generate `paper1/figs/fig2_fusion_deepseek_style_v2.pptx`: final editable PowerPoint.
- Generate `paper1/figs/fig2_fusion_deepseek_style_v2.png`: final preview.
- Read only `paper1/mvosr_paper_cn.tex` and the user reference image for content/style evidence.
- Do not modify or overwrite any existing figure, manuscript include, or prior builder.

### Task 1: Create the failing structural validator

**Files:**
- Create: `paper1/tmp/fig2_deepseek_style/validate_fig2_deepseek_style_v2.mjs`
- Test target: `paper1/figs/fig2_fusion_deepseek_style_v2.pptx`

- [ ] **Step 1: Create the validation script**

```js
import assert from "node:assert/strict";
import fs from "node:fs";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const pptx = process.argv[2];
assert.ok(fs.existsSync(pptx), `missing PPTX: ${pptx}`);
const deck = await PresentationFile.importPptx(await FileBlob.load(pptx));
const snapshot = await deck.inspect({ kind: "slide,textbox,shape", maxChars: 120000 });
const text = snapshot.ndjson;
for (const required of [
  "公共主干定位", "决策级聚合（本文）", "特征级融合",
  "瓶颈中层融合（MBT 式）", "feature-residual-route",
  "bottleneck-token-1", "bottleneck-token-2",
  "bottleneck-token-3", "bottleneck-token-4",
]) assert.match(text, new RegExp(required.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")));
assert.equal((text.match(/bottleneck-token-/g) ?? []).length, 4);
for (const banned of ["OSCR", "随机丢视角", "闭集分类头", "Accept", "Reject"])
  assert.equal(text.includes(banned), false, banned);
console.log("structure validation passed");
```

- [ ] **Step 2: Run the validator and confirm RED**

Run:

```powershell
node paper1/tmp/fig2_deepseek_style/validate_fig2_deepseek_style_v2.mjs paper1/figs/fig2_fusion_deepseek_style_v2.pptx
```

Expected: failure because the final PPTX does not yet exist.

### Task 2: Initialize the artifact-tool workspace and author the slide shell

**Files:**
- Create: `paper1/tmp/fig2_deepseek_style/build_fig2_deepseek_style_v2.mjs`
- Generate: `paper1/tmp/fig2_deepseek_style/node_modules/`

- [ ] **Step 1: Initialize the temporary workspace**

Run:

```powershell
node "$SKILL_DIR/container_tools/setup_artifact_tool_workspace.mjs" --workspace "D:/uwb_gestures/paper1/tmp/fig2_deepseek_style"
```

Expected: `node_modules/@oai/artifact-tool` exists in the temporary workspace.

- [ ] **Step 2: Add the artifact-tool builder foundation**

The builder must use this output contract and native-shape helpers:

```js
import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "D:/uwb_gestures/paper1";
const OUT_PPTX = path.join(ROOT, "figs/fig2_fusion_deepseek_style_v2.pptx");
const OUT_PNG = path.join(ROOT, "figs/fig2_fusion_deepseek_style_v2.png");
const OUT_LAYOUT = path.join(ROOT, "tmp/fig2_deepseek_style/fig2_fusion_deepseek_style_v2.layout.json");
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slide = deck.slides.add();
slide.background.fill = "#FFFFFF";

function addBox(name, left, top, width, height, text, fill, stroke, fontSize = 13) {
  const shape = slide.shapes.add({
    geometry: "roundRect", name,
    position: { left, top, width, height },
    fill, line: { style: "solid", fill: stroke, width: 1.1 },
    borderRadius: 5, shadow: "shadow-none",
  });
  shape.text = text;
  shape.text.style = {
    fontSize, typeface: "Microsoft YaHei", color: "#17202A",
    alignment: "center", verticalAlignment: "middle",
    autoFit: "shrinkText", insets: { left: 4, right: 4, top: 2, bottom: 2 },
  };
  return shape;
}

function addText(name, left, top, width, height, text, fontSize, bold = false, typeface = "Microsoft YaHei") {
  const shape = slide.shapes.add({
    geometry: "textbox", name, position: { left, top, width, height },
    fill: "none", line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { fontSize, bold, typeface, color: "#17202A", autoFit: "shrinkText", verticalAlignment: "middle" };
  return shape;
}
```

- [ ] **Step 3: Add the title, left dashed backbone, and the three right dashed inset frames**

Use the approved geometry: title at `x=24, y=18`; left backbone at `x=24, y=84, w=292, h=552`; right insets at `x=370, w=886` with y/heights `(84,174)`, `(276,185)`, `(479,175)`. Use a 1.5 px blue dashed line and no fill.

### Task 3: Build the high-density common backbone

**Files:**
- Modify: `paper1/tmp/fig2_deepseek_style/build_fig2_deepseek_style_v2.mjs`

- [ ] **Step 1: Add L/T/R input nodes and the shared encoder group**

Create L/T/R nodes, two encoder paths, and labels:

```js
const VIEW_X = [42, 127, 212];
for (const [index, view] of ["L", "T", "R"].entries())
  addBox(`backbone-input-${view}`, VIEW_X[index], 126, 66, 34, view, "#FFFFFF", "#27323D", 16);

addBox("backbone-time-path", 50, 234, 220, 58,
  "时间路径 Transformer\n90 tokens · D=32 · 4 heads · ×4",
  "#E3EDF7", "#27323D", 11);
addBox("backbone-range-path", 50, 321, 220, 58,
  "距离路径 Transformer\n189 tokens · D=32 · 4 heads · ×4",
  "#E3EDF7", "#27323D", 11);
```

Overlay small editable `LN + MHA`, `Add`, and `LN + MLP` shapes inside both path boxes, then add the feature-code box labeled `展平 + 拼接 + 视角嵌入` and `fL, fT, fR ∈ R^8928`.

- [ ] **Step 2: Add the view-specific inspector and prototype-distance nodes**

Add a 8928 → 1024 → 128 inspector box, prototype token stack, and native distance boxes `dL`, `dT`, `dR`. Connect them to a small green `异常分数 s` node while keeping all ordinary L/T/R feature lanes independent.

### Task 4: Build the three scientific insets

**Files:**
- Modify: `paper1/tmp/fig2_deepseek_style/build_fig2_deepseek_style_v2.mjs`

- [ ] **Step 1: Build inset (a), decision-level aggregation**

Add three independent `fv → ηk(v)` lanes, three distance nodes, and the green score box with:

```text
s(X) = meanv(dv)/σ̄ + β · varv(dv)/σ̄²
```

No connector may cross views before the three distances enter the score node.

- [ ] **Step 2: Build inset (b), feature-level fusion**

Add stacked view features, Q and K/V nodes, a green multi-head cross-view attention block, a green `残差相加 + LayerNorm` block, three fused outputs, and three view-specific inspectors. Name the residual connector `feature-residual-route`.

- [ ] **Step 3: Build inset (c), MBT-style bottleneck fusion**

Add independent-layer token lanes, exactly four shapes named `bottleneck-token-1` through `bottleneck-token-4`, three bottleneck-fusion rows, and three output distances. The four bottleneck tokens are the only cross-view channel; do not create direct L↔T, T↔R, or L↔R ordinary-token connectors.

- [ ] **Step 4: Add callout connectors and the legend**

Use blue dashed callout lines from the relevant backbone depths to the three inset frames. Add legend samples for pale yellow within-view computation, pale blue shared encoder, pale green cross-view exchange, and blue dashed expansion areas.

### Task 5: Export and run structural validation

**Files:**
- Generate: `paper1/figs/fig2_fusion_deepseek_style_v2.pptx`
- Generate: `paper1/figs/fig2_fusion_deepseek_style_v2.png`
- Generate: `paper1/tmp/fig2_deepseek_style/fig2_fusion_deepseek_style_v2.layout.json`

- [ ] **Step 1: Export preview, layout, and PPTX**

Finish the builder with:

```js
const preview = await deck.export({ slide, format: "png", scale: 2 });
await fs.writeFile(OUT_PNG, new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(OUT_LAYOUT, await layout.text());
slide.speakerNotes.textFrame.setText(
  "[Sources]\n- User-provided DeepSeek-V2/V3 architecture image: visual direction only.\n- D:/uwb_gestures/paper1/mvosr_paper_cn.tex: scientific topology and labels."
);
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(OUT_PPTX);
```

- [ ] **Step 2: Run the builder and validator**

Run:

```powershell
node paper1/tmp/fig2_deepseek_style/build_fig2_deepseek_style_v2.mjs
node paper1/tmp/fig2_deepseek_style/validate_fig2_deepseek_style_v2.mjs paper1/figs/fig2_fusion_deepseek_style_v2.pptx
```

Expected: both commands exit 0 and the validator prints `structure validation passed`.

### Task 6: Render, inspect, and fix visual defects

**Files:**
- Verify: `paper1/figs/fig2_fusion_deepseek_style_v2.pptx`
- Verify: `paper1/figs/fig2_fusion_deepseek_style_v2.png`
- Modify if needed: `paper1/tmp/fig2_deepseek_style/build_fig2_deepseek_style_v2.mjs`

- [ ] **Step 1: Render the final slide independently**

```powershell
python "$SKILL_DIR/container_tools/render_slides.py" "D:/uwb_gestures/paper1/figs/fig2_fusion_deepseek_style_v2.pptx"
```

Expected: `paper1/figs/fig2_fusion_deepseek_style_v2/slide-1.png` exists.

- [ ] **Step 2: Inspect the full-resolution slide**

Open both the artifact-tool preview and the independently rendered slide. Confirm no wrong-direction arrow, accidental ordinary-token cross-link, illegible title, hidden label, or apparent fifth bottleneck token.

- [ ] **Step 3: Run overflow QA**

```powershell
python "$SKILL_DIR/container_tools/slides_test.py" "D:/uwb_gestures/paper1/figs/fig2_fusion_deepseek_style_v2.pptx"
```

Expected: `Test passed. No overflow detected.`

- [ ] **Step 4: Regenerate after any correction and repeat Steps 1–3**

Only finish after the final render, structural validator, and overflow test all pass.

### Task 7: Final scope check and delivery

**Files:**
- Verify only; do not modify the manuscript.

- [ ] **Step 1: Verify output scope**

Run `git status --short -- paper1/figs paper1/tmp/fig2_deepseek_style paper1/mvosr_paper.tex paper1/mvosr_paper_cn.tex` and confirm both manuscript files are unchanged by this task.

- [ ] **Step 2: Deliver the PPTX and preview**

Report that the slide uses editable native PowerPoint objects, cite the final PPTX once as the output, and include the absolute PNG preview path for visual confirmation.
