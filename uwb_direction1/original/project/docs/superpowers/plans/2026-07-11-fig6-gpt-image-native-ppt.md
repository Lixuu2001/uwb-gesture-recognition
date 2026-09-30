# Paper 1 Figure 2 GPT Image Native PPT Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new single-slide, fully editable PowerPoint that closely reproduces the Chinese GPT Image version of Paper 1 Figure 2 without embedding the source PNG.

**Architecture:** Create a dedicated artifact-tool JavaScript builder that reuses the existing scientific panel specification but owns a new 1728:928 layout and new output names. Validate the builder with a Node structural test and artifact-tool end-to-end export test, then render the PPTX independently for visual and overflow QA.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, Node built-in test runner, PowerPoint DrawingML, bundled presentation render and QA tools.

---

## File Structure

- Create `paper1/figs/build_fig6_fusion_variants_gptimage_rebuilt.mjs`: dedicated one-slide native-shape builder with GPT Image geometry.
- Create `tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs`: topology, editability, naming, and export tests.
- Create `paper1/figs/fig6_fusion_variants_gptimage_rebuilt.pptx`: final editable PowerPoint.
- Create `paper1/figs/fig6_fusion_variants_gptimage_rebuilt.png`: final full-slide preview.
- Reuse `paper1/figs/fig6_fusion_variants_technical_spec.mjs`: panel order, Chinese titles, palette, four-token constraint, and banned-content validation.
- Read-only reference `paper1/figs/fig6_fusion_variants_technical_ai.png`: visual target only; never embed it in the slide.

### Task 1: Add the GPT Image rebuild contract test

**Files:**
- Create: `tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs`
- Read: `paper1/figs/fig6_fusion_variants_technical_spec.mjs`
- Test: `tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs`

- [ ] **Step 1: Write the failing structural test**

Create a Node test that requires a dedicated builder, checks the custom aspect ratio and output filenames, requires the Chinese labels, prohibits a source-image embed, and checks the named residual route and four bottleneck tokens in the exported inspection output:

```js
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const BUILDER = path.join(ROOT, "paper1", "figs", "build_fig6_fusion_variants_gptimage_rebuilt.mjs");
const SPEC = path.join(ROOT, "paper1", "figs", "fig6_fusion_variants_technical_spec.mjs");

test("builder declares the GPT Image paper-figure canvas and distinct outputs", () => {
  const source = fs.readFileSync(BUILDER, "utf8");
  assert.match(source, /width:\s*1728/);
  assert.match(source, /height:\s*928/);
  assert.match(source, /fig6_fusion_variants_gptimage_rebuilt\.pptx/);
  assert.match(source, /fig6_fusion_variants_gptimage_rebuilt\.png/);
});

test("all scientific labels remain Chinese and the source PNG is not embedded", () => {
  const source = fs.readFileSync(BUILDER, "utf8");
  for (const label of ["决策层聚合", "特征层融合", "瓶颈中层融合", "唯一跨视角通道"])
    assert.ok(source.includes(label), label);
  assert.doesNotMatch(source, /images\.add|technical_ai\.png|data:image/i);
});

test("artifact export preserves the residual route and exactly four bottleneck tokens", () => {
  const workspace = process.env.FIG6_GPTIMAGE_WORKSPACE;
  assert.ok(workspace, "FIG6_GPTIMAGE_WORKSPACE is required");
  fs.mkdirSync(workspace, { recursive: true });
  fs.copyFileSync(BUILDER, path.join(workspace, path.basename(BUILDER)));
  fs.copyFileSync(SPEC, path.join(workspace, path.basename(SPEC)));
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "fig6-gptimage-test-"));
  const result = spawnSync(process.execPath, [path.join(workspace, path.basename(BUILDER)), "--out-dir", out], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  for (const file of ["fig6_fusion_variants_gptimage_rebuilt.pptx", "fig6_fusion_variants_gptimage_rebuilt.png"])
    assert.ok(fs.statSync(path.join(out, file)).size > 500, file);
  const inspectPath = path.join(out, "fig6_fusion_variants_gptimage_rebuilt.inspect.ndjson");
  const records = fs.readFileSync(inspectPath, "utf8").trim().split(/\r?\n/).map(JSON.parse);
  const names = records.map((record) => record.name).filter(Boolean);
  assert.ok(names.includes("feature-residual-route"));
  assert.equal(names.filter((name) => name.startsWith("bottleneck-token-")).length, 4);
  assert.equal(names.some((name) => /classifier|accept|reject|oscr/i.test(name)), false);
});
```

- [ ] **Step 2: Run the test and confirm RED**

Run:

```powershell
$env:FIG6_GPTIMAGE_WORKSPACE='C:\Users\19252\AppData\Local\Temp\codex-presentations\019f4bb3-6e7e-7620-bc40-e738ffe7e5f3\fig6-gptimage-native-ppt\tmp'
node --test tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs
```

Expected: FAIL because `build_fig6_fusion_variants_gptimage_rebuilt.mjs` does not exist.

- [ ] **Step 3: Commit the failing contract test**

```powershell
git add tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs
git commit -m "test: define gpt image ppt rebuild contract"
```

### Task 2: Build the native-shape slide foundation

**Files:**
- Create: `paper1/figs/build_fig6_fusion_variants_gptimage_rebuilt.mjs`
- Read: `paper1/figs/build_fig6_fusion_variants_technical.mjs`
- Read: `paper1/figs/fig6_fusion_variants_technical_spec.mjs`
- Test: `tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs`

- [ ] **Step 1: Create the builder imports, output contract, and reusable primitives**

The new builder must import the existing panel specification and define distinct output names and a custom canvas:

```js
import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";
import { FIGURE, PANELS, PALETTE, validateTechnicalFigure } from "./fig6_fusion_variants_technical_spec.mjs";

const OUTPUTS = Object.freeze({
  pptx: "fig6_fusion_variants_gptimage_rebuilt.pptx",
  png: "fig6_fusion_variants_gptimage_rebuilt.png",
  layout: "fig6_fusion_variants_gptimage_rebuilt.layout.json",
  inspect: "fig6_fusion_variants_gptimage_rebuilt.inspect.ndjson",
});
const CANVAS = Object.freeze({ width: 1728, height: 928 });
const FONT = Object.freeze({ body: "Microsoft YaHei", math: "Times New Roman" });
const LINE = "#252A31";
```

Implement `addText`, `addBox`, `addAnchor`, `addFlow`, and `addDashedGroup` using native shapes. `addFlow` must use `tail` for the target arrow because the artifact runtime maps that property to PowerPoint `tailEnd`:

```js
function addFlow(slide, { name, x1, y1, x2, y2, color = LINE, width = 1.4, kind = "straight", arrow = true }) {
  const from = addAnchor(slide, `${name}-from`, x1, y1);
  const to = addAnchor(slide, `${name}-to`, x2, y2);
  return slide.shapes.connect(from, to, {
    kind,
    fromSide: "right",
    toSide: "left",
    line: { style: "solid", fill: color, width },
    head: { type: "none" },
    tail: arrow ? { type: "triangle", width: "sm", length: "sm" } : { type: "none" },
  });
}
```

- [ ] **Step 2: Add the slide shell and title geometry**

Use a single slide with three vertical bands centered around y = 140, 425, and 700. Titles use 27 px Chinese type and small colored tags. Keep margins close to the AI reference:

```js
const PANEL_Y = Object.freeze({ decision: 132, feature: 420, bottleneck: 706 });

function addPanelTitle(slide, { id, letter, title, tag, y }) {
  addText(slide, { name: `${id}-letter`, text: letter, left: 14, top: y - 115, width: 42, height: 30, fontSize: 25, bold: true, typeface: FONT.math });
  addText(slide, { name: `${id}-title`, text: title, left: 60, top: y - 115, width: 300, height: 32, fontSize: 27, bold: true });
  addText(slide, { name: `${id}-tag`, text: tag, left: 410, top: y - 113, width: 170, height: 28, fontSize: 22, bold: true, color: PALETTE[id] });
}
```

- [ ] **Step 3: Run the tests to expose the missing panels**

Run the same Node test. Expected: the builder starts but the inspection assertions fail because the residual route and bottleneck tokens have not been created.

- [ ] **Step 4: Commit the foundation**

```powershell
git add paper1/figs/build_fig6_fusion_variants_gptimage_rebuilt.mjs
git commit -m "feat: scaffold gpt image native ppt builder"
```

### Task 3: Rebuild panels (a) and (b)

**Files:**
- Modify: `paper1/figs/build_fig6_fusion_variants_gptimage_rebuilt.mjs`
- Test: `tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs`

- [ ] **Step 1: Draw panel (a) with independent L/T/R lanes**

Use connector-first ordering. Coordinates must reproduce the GPT Image rhythm: inputs at x 64, encoder group x 210–620, features x 650, inspector group x 785–1195, distances x 1260, score x 1510. Create all three lanes from a loop and merge only the distances:

```js
for (const [index, view] of FIGURE.views.entries()) {
  const y = panel.y + [-46, 0, 46][index];
  addFlow(slide, { name: `decision-input-${view}`, x1: 128, y1: y, x2: 235, y2: y });
  addFlow(slide, { name: `decision-encoder-${view}`, x1: 585, y1: y, x2: 650, y2: y });
  addFlow(slide, { name: `decision-inspector-${view}`, x1: 715, y1: y, x2: 805, y2: y });
  addFlow(slide, { name: `decision-distance-${view}`, x1: 1175, y1: y, x2: 1255, y2: y });
  addFlow(slide, { name: `decision-score-${view}`, x1: 1315, y1: y, x2: 1500, y2: panel.y, kind: "elbow" });
}
```

Then add the nodes over the connectors: input boxes, pale-blue encoder blocks, feature boxes, view-specific inspector rows, distance ellipses, and the final `s` ellipse.

- [ ] **Step 2: Draw panel (b) with cross-view attention and residual path**

Use purple for Q/K/V, attention, Add & Norm, fused features, and the final aggregation. Name the bypass connector exactly `feature-residual-route`:

```js
addFlow(slide, {
  name: "feature-residual-route",
  x1: 635, y1: panel.y + 55,
  x2: 995, y2: panel.y + 38,
  color: PALETTE.feature,
  width: 1.8,
  kind: "elbow",
});
```

Place Q/K/V at x 710, multi-head attention at x 800, Add & Norm at x 945, fused features at x 1105, inspector group x 1195–1485, distances x 1515, and score x 1660. Ensure no ordinary feature arrow bypasses the attention output except the named residual route.

- [ ] **Step 3: Run the Node test**

Expected: the Chinese labels and residual route pass; the test still fails until panel (c) creates four named bottleneck tokens.

- [ ] **Step 4: Commit panels (a) and (b)**

```powershell
git add paper1/figs/build_fig6_fusion_variants_gptimage_rebuilt.mjs
git commit -m "feat: rebuild decision and feature fusion panels"
```

### Task 4: Rebuild panel (c) and the legend

**Files:**
- Modify: `paper1/figs/build_fig6_fusion_variants_gptimage_rebuilt.mjs`
- Test: `tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs`

- [ ] **Step 1: Draw the independent token lanes**

Create L/T/R inputs, three independent blocks, and three stacked token nodes `TL/TT/TR`. Use black within-view arrows only. Add `×(4−K)` under the independent group.

- [ ] **Step 2: Draw exactly four shared bottleneck tokens**

Use four independent native rounded rectangles and stable names:

```js
for (let index = 0; index < 4; index += 1) {
  addBox(slide, {
    name: `bottleneck-token-${index + 1}`,
    left: 605 + index * 38,
    top: panel.y - 19,
    width: 29,
    height: 38,
    text: index === 0 ? "B" : "",
    fill: "#FFF9EE",
    stroke: PALETTE.bottleneck,
    strokeWidth: 1.5,
    radius: 5,
    fontSize: 16,
    bold: true,
  });
}
```

Connect each view-token lane to the first bottleneck token region and the bottleneck region to each fusion block. Do not add any L↔T, T↔R, or L↔R ordinary-token connectors.

- [ ] **Step 3: Add bottleneck fusion, inspectors, distances, score, and legend**

Place three amber bottleneck fusion blocks at x 840–1055, fused features at x 1115, inspector group x 1205–1485, distance ellipses x 1520, and score x 1660. Add the seven-item legend along the bottom with native samples and Chinese labels.

- [ ] **Step 4: Run the Node test and confirm GREEN**

Run:

```powershell
$env:FIG6_GPTIMAGE_WORKSPACE='C:\Users\19252\AppData\Local\Temp\codex-presentations\019f4bb3-6e7e-7620-bc40-e738ffe7e5f3\fig6-gptimage-native-ppt\tmp'
node --test tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs
```

Expected: all tests pass, including `feature-residual-route`, exactly four `bottleneck-token-*` objects, distinct output names, and no embedded source PNG.

- [ ] **Step 5: Commit the completed builder**

```powershell
git add paper1/figs/build_fig6_fusion_variants_gptimage_rebuilt.mjs tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs
git commit -m "feat: complete gpt image native ppt rebuild"
```

### Task 5: Export, render, inspect, and refine

**Files:**
- Create: `paper1/figs/fig6_fusion_variants_gptimage_rebuilt.pptx`
- Create: `paper1/figs/fig6_fusion_variants_gptimage_rebuilt.png`
- Modify if needed: `paper1/figs/build_fig6_fusion_variants_gptimage_rebuilt.mjs`

- [ ] **Step 1: Initialize the external artifact-tool workspace**

Run with `HOME=C:\Users\19252` so the bundled runtime resolves correctly:

```powershell
$env:HOME='C:\Users\19252'
node 'C:\Users\19252\.codex\plugins\cache\openai-primary-runtime\presentations\26.709.11516\skills\presentations\container_tools\setup_artifact_tool_workspace.mjs' `
  --workspace 'C:\Users\19252\AppData\Local\Temp\codex-presentations\019f4bb3-6e7e-7620-bc40-e738ffe7e5f3\fig6-gptimage-native-ppt\tmp'
```

- [ ] **Step 2: Export the final PPTX and builder preview**

Run the copied builder from the external scratch directory and copy only the PPTX and PNG into `paper1/figs/`.

Expected: both final files exist and are non-empty; the slide contains no image object corresponding to `fig6_fusion_variants_technical_ai.png`.

- [ ] **Step 3: Render with the independent PowerPoint renderer**

```powershell
$env:HOME='C:\Users\19252'
& 'C:\Users\19252\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'C:\Users\19252\.codex\plugins\cache\openai-primary-runtime\presentations\26.709.11516\skills\presentations\container_tools\render_slides.py' `
  'paper1\figs\fig6_fusion_variants_gptimage_rebuilt.pptx' `
  --output_dir 'C:\Users\19252\AppData\Local\Temp\codex-presentations\019f4bb3-6e7e-7620-bc40-e738ffe7e5f3\fig6-gptimage-native-ppt\rendered'
```

Open the resulting `slide-1.png` at full size and compare it with `fig6_fusion_variants_technical_ai.png`.

- [ ] **Step 4: Run overflow QA**

```powershell
& 'C:\Users\19252\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'C:\Users\19252\.codex\plugins\cache\openai-primary-runtime\presentations\26.709.11516\skills\presentations\container_tools\slides_test.py' `
  'paper1\figs\fig6_fusion_variants_gptimage_rebuilt.pptx'
```

Expected: `Test passed. No overflow detected.`

- [ ] **Step 5: Fix visual defects and repeat**

If any connector points left, text wraps unexpectedly, group labels collide with the dashed border, or panel (c) appears to contain more than four tokens, patch only the corresponding coordinates or style constant, regenerate, and repeat Steps 2–4.

- [ ] **Step 6: Commit final deliverables**

```powershell
git add paper1/figs/build_fig6_fusion_variants_gptimage_rebuilt.mjs `
  paper1/figs/fig6_fusion_variants_gptimage_rebuilt.pptx `
  paper1/figs/fig6_fusion_variants_gptimage_rebuilt.png `
  tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs
git commit -m "feat: deliver editable gpt image figure ppt"
```

### Task 6: Final verification and delivery

**Files:**
- Verify: `paper1/figs/fig6_fusion_variants_gptimage_rebuilt.pptx`
- Verify: `paper1/figs/fig6_fusion_variants_gptimage_rebuilt.png`
- Verify: `tests/test_fig6_fusion_variants_gptimage_rebuilt.mjs`

- [ ] **Step 1: Run the final structural test from a clean command**

Expected: all GPT Image rebuild tests pass.

- [ ] **Step 2: Run `slides_test.py` one final time**

Expected: no overflow.

- [ ] **Step 3: Verify Git scope**

Run `git status --short` and confirm the task changed only the dedicated builder, test, PPTX, preview, design specification, and implementation plan. Preserve every unrelated existing modification in the main worktree.

- [ ] **Step 4: Deliver one standalone PPTX link**

The final response must state that the slide was rebuilt from native PowerPoint objects, mention the successful structural and overflow QA, and include exactly one standalone Markdown link to `fig6_fusion_variants_gptimage_rebuilt.pptx`.
