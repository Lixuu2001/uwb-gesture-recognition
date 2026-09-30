# Paper 1 Figure 2 Fusion-Strategy Technical Schematic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publication-ready technical redraw of Paper 1's second figure, with an editable PowerPoint source, a GPT Image interpretation, a reusable prompt, and a verified side-by-side comparison.

**Architecture:** Store the scientific rules and visible labels in a dependency-free ES module, validate them with Node tests, and render one wide slide through `@oai/artifact-tool`. The PowerPoint version is canonical and exact; GPT Image receives the user reference for technical line-art style and the rendered PowerPoint for strict structure.

**Tech Stack:** Node.js ES modules, Node test runner, `@oai/artifact-tool`, built-in GPT Image generation, PowerPoint rendering/QA helpers, and PNG montage tooling.

---

## File map

- Create `paper1/figs/fig6_fusion_variants_technical_spec.mjs`: panel order, lane rules, notation, palette, banned content, and validation helpers.
- Create `tests/test_fig6_fusion_variants_technical.mjs`: structural and end-to-end export tests.
- Create `paper1/figs/build_fig6_fusion_variants_technical.mjs`: editable technical schematic authoring and export.
- Create `paper1/figs/fig6_fusion_variants_technical_prompt.txt`: reusable GPT Image prompt.
- Generate `paper1/figs/fig6_fusion_variants_technical.pptx`.
- Generate `paper1/figs/fig6_fusion_variants_technical_pptx.png`.
- Generate `paper1/figs/fig6_fusion_variants_technical_ai.png`.
- Generate `paper1/figs/fig6_fusion_variants_technical_comparison.png`.

Existing `fig6_fusion_variants.pdf`, Paper 1 Figure 1 files, and the manuscript include remain unchanged.

### Task 1: Lock the second-figure scientific contract with a failing test

**Files:**
- Create: `tests/test_fig6_fusion_variants_technical.mjs`
- Test target: `paper1/figs/fig6_fusion_variants_technical_spec.mjs`

- [ ] **Step 1: Write the failing structural test**

Create the test with this content:

```js
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import {
  FIGURE,
  PANELS,
  PALETTE,
  OUTPUTS,
  validateTechnicalFigure,
} from "../paper1/figs/fig6_fusion_variants_technical_spec.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test("targets Paper 1 Figure 2 and preserves the existing panel order", () => {
  assert.equal(FIGURE.sourceInclude, "fig6_fusion_variants.pdf");
  assert.deepEqual(PANELS.map((panel) => panel.id), ["decision", "feature", "bottleneck"]);
  assert.deepEqual(PANELS.map((panel) => panel.letter), ["(a)", "(b)", "(c)"]);
});

test("each strategy exposes only its real cross-view channel", () => {
  const decision = PANELS.find((panel) => panel.id === "decision");
  const feature = PANELS.find((panel) => panel.id === "feature");
  const bottleneck = PANELS.find((panel) => panel.id === "bottleneck");
  assert.equal(decision.firstExchange, "score");
  assert.deepEqual(decision.crossViewEdges, ["dL->score", "dT->score", "dR->score"]);
  assert.equal(feature.firstExchange, "post_encoder_attention");
  assert.equal(feature.preservesRangeTokens, true);
  assert.equal(bottleneck.firstExchange, "shared_bottleneck_tokens");
  assert.equal(bottleneck.bottleneckCount, 4);
  assert.equal(bottleneck.directOrdinaryTokenCrossEdges, 0);
});

test("technical styling is restrained and Figure 1-only content is banned", () => {
  assert.equal(PALETTE.background, "#FFFFFF");
  assert.equal(PALETTE.neutralFill, "#EAF1F8");
  assert.equal(PALETTE.feature, "#8B78E6");
  assert.equal(PALETTE.bottleneck, "#D89A32");
  assert.equal(PALETTE.decision, "#2B8793");
  assert.deepEqual(validateTechnicalFigure(), []);
  const copy = JSON.stringify({ FIGURE, PANELS });
  for (const banned of ["Closed-set classifier", "View dropout", "Accept", "Reject", "OSCR"])
    assert.equal(copy.includes(banned), false, banned);
});

test("declares every requested deliverable", () => {
  assert.deepEqual(Object.keys(OUTPUTS).sort(), ["ai", "comparison", "pptx", "pptxPng", "prompt"].sort());
});

test("artifact-tool builder exports the deck and QA artifacts", () => {
  const workspace = process.env.FIG6_TECH_WORKSPACE;
  assert.ok(workspace, "FIG6_TECH_WORKSPACE is required");
  const source = path.join(ROOT, "paper1", "figs", "build_fig6_fusion_variants_technical.mjs");
  const spec = path.join(ROOT, "paper1", "figs", "fig6_fusion_variants_technical_spec.mjs");
  assert.ok(fs.existsSync(source));
  fs.mkdirSync(workspace, { recursive: true });
  fs.copyFileSync(source, path.join(workspace, path.basename(source)));
  fs.copyFileSync(spec, path.join(workspace, path.basename(spec)));
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "fig6-tech-test-"));
  const result = spawnSync(process.execPath, [path.join(workspace, path.basename(source)), "--out-dir", out], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  for (const name of [OUTPUTS.pptx, OUTPUTS.pptxPng, "fig6_fusion_variants_technical.layout.json", "fig6_fusion_variants_technical.inspect.ndjson"])
    assert.ok(fs.statSync(path.join(out, name)).size > 500, name);
});
```

- [ ] **Step 2: Run the test and verify the missing-spec failure**

Run:

```powershell
node --test tests/test_fig6_fusion_variants_technical.mjs
```

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `fig6_fusion_variants_technical_spec.mjs`.

- [ ] **Step 3: Commit the failing test**

```powershell
git add tests/test_fig6_fusion_variants_technical.mjs
git commit -m "test: lock paper1 figure2 fusion structure"
```

### Task 2: Implement the dependency-free figure specification

**Files:**
- Create: `paper1/figs/fig6_fusion_variants_technical_spec.mjs`
- Test: `tests/test_fig6_fusion_variants_technical.mjs`

- [ ] **Step 1: Create the exact specification module**

```js
export const FIGURE = Object.freeze({
  sourceInclude: "fig6_fusion_variants.pdf",
  slide: Object.freeze({ width: 1800, height: 860 }),
  views: Object.freeze(["L", "T", "R"]),
  laneOffsets: Object.freeze([-28, 0, 28]),
  section: "4.4",
});

export const PALETTE = Object.freeze({
  background: "#FFFFFF",
  text: "#20252D",
  line: "#4D535B",
  muted: "#6C7480",
  neutralFill: "#EAF1F8",
  feature: "#8B78E6",
  featureFill: "#F2EEFC",
  bottleneck: "#D89A32",
  bottleneckFill: "#FFF3DD",
  decision: "#2B8793",
  decisionFill: "#EAF6F6",
});

export const PANELS = Object.freeze([
  Object.freeze({
    id: "decision", letter: "(a)", title: "Decision-level aggregation", tag: "Ours",
    firstExchange: "score",
    crossViewEdges: Object.freeze(["dL->score", "dT->score", "dR->score"]),
  }),
  Object.freeze({
    id: "feature", letter: "(b)", title: "Feature-level fusion", tag: "F1",
    firstExchange: "post_encoder_attention", preservesRangeTokens: true,
  }),
  Object.freeze({
    id: "bottleneck", letter: "(c)", title: "Bottleneck mid-level fusion", tag: "F2 · MBT-style",
    firstExchange: "shared_bottleneck_tokens", bottleneckCount: 4,
    directOrdinaryTokenCrossEdges: 0,
  }),
]);

export const OUTPUTS = Object.freeze({
  pptx: "fig6_fusion_variants_technical.pptx",
  pptxPng: "fig6_fusion_variants_technical_pptx.png",
  ai: "fig6_fusion_variants_technical_ai.png",
  comparison: "fig6_fusion_variants_technical_comparison.png",
  prompt: "fig6_fusion_variants_technical_prompt.txt",
});

export function validateTechnicalFigure() {
  const errors = [];
  if (PANELS.map((p) => p.id).join(",") !== "decision,feature,bottleneck") errors.push("panel order mismatch");
  if (PANELS[0].crossViewEdges.some((edge) => !edge.endsWith("->score"))) errors.push("early decision exchange");
  if (!PANELS[1].preservesRangeTokens) errors.push("F1 must preserve range tokens");
  if (PANELS[2].bottleneckCount !== 4 || PANELS[2].directOrdinaryTokenCrossEdges !== 0) errors.push("invalid bottleneck channel");
  if (FIGURE.slide.width / FIGURE.slide.height < 2) errors.push("figure is not wide enough");
  return errors;
}
```

- [ ] **Step 2: Run the structural tests**

Run:

```powershell
node --test --test-name-pattern "targets|strategy|styling|deliverable" tests/test_fig6_fusion_variants_technical.mjs
```

Expected: four tests PASS; the end-to-end builder test remains blocked because the builder does not exist.

- [ ] **Step 3: Commit the specification**

```powershell
git add paper1/figs/fig6_fusion_variants_technical_spec.mjs tests/test_fig6_fusion_variants_technical.mjs
git commit -m "feat: define paper1 figure2 technical schematic"
```

### Task 3: Build the editable PowerPoint technical schematic

**Files:**
- Create: `paper1/figs/build_fig6_fusion_variants_technical.mjs`
- Generate: `paper1/figs/fig6_fusion_variants_technical.pptx`
- Generate: `paper1/figs/fig6_fusion_variants_technical_pptx.png`

- [ ] **Step 1: Load the presentation runtime and create external scratch space**

Call `codex_app__load_workspace_dependencies`, set `SKILL_DIR` to the installed presentations skill, and choose an external temporary workspace. Initialize artifact-tool resolution:

```powershell
node "$SKILL_DIR\container_tools\setup_artifact_tool_workspace.mjs" --workspace "$TMP_DIR"
```

Expected: `$TMP_DIR/node_modules/@oai/artifact-tool` exists.

- [ ] **Step 2: Create the builder by adapting the verified local artifact-tool pattern**

Start from `paper1/figs/build_fig6_fusion_depth_axis.mjs` only for its `Presentation`, `PresentationFile`, `addText`, `addBox`, `addAnchor`, `addFlow`, export, and inspect patterns. The new builder must import the new spec and expose these panel functions:

```js
function addDecisionPanel(slide, connectors, panel) {
  addPanelTitle(slide, panel);
  addThreeViewInputs(slide, panel);
  addDashedGroup(slide, { name: "decision-encoder-group", label: "Shared-weight encoder", x: 250, y: panel.y - 68, w: 400, h: 136 });
  addRepeatedBlocks(slide, { prefix: "decision-encoder", x: 330, y: panel.y, countLabel: "×4", fill: PALETTE.neutralFill });
  addFeatureCodes(slide, panel, ["fL", "fT", "fR"]);
  addInspectorGroup(slide, panel, { mapping: "8928 → 1024 → 128" });
  addDistanceNodes(slide, panel, ["dL", "dT", "dR"]);
  addScoreNode(slide, panel, "s = mean(dv)/σ̄ + β·var(dv)/σ̄²");
  connectIndependentLanesThroughDistanceOnly(slide, connectors, panel);
}

function addFeaturePanel(slide, connectors, panel) {
  addPanelTitle(slide, panel);
  addThreeViewInputs(slide, panel);
  addDashedGroup(slide, { name: "feature-encoder-group", label: "Shared-weight encoder", x: 250, y: panel.y - 68, w: 360, h: 136 });
  addFeatureCodes(slide, panel, ["fL", "fT", "fR"]);
  addDashedGroup(slide, { name: "feature-attention-group", label: "Post-encoder cross-view attention", x: 735, y: panel.y - 80, w: 330, h: 160, stroke: PALETTE.feature });
  addQkvAttentionResidual(slide, panel, { labels: ["Q", "K", "V"], output: "f̃v", accent: PALETTE.feature });
  addInspectorGroup(slide, panel, { mapping: "8928 → 1024 → 128" });
  connectFeatureAttentionLanes(slide, connectors, panel);
}

function addBottleneckPanel(slide, connectors, panel) {
  addPanelTitle(slide, panel);
  addThreeViewInputs(slide, panel);
  addDashedGroup(slide, { name: "bottleneck-encoder-group", label: "Shared-weight encoder", x: 250, y: panel.y - 86, w: 880, h: 172, stroke: PALETTE.line });
  addRepeatedBlocks(slide, { prefix: "bottleneck-independent", x: 330, y: panel.y, countLabel: "×(4−K)", fill: PALETTE.neutralFill });
  addTokenStacks(slide, panel, ["TL", "TT", "TR"]);
  addSharedBottleneckTokens(slide, panel, { count: 4, label: "B", accent: PALETTE.bottleneck });
  addBottleneckReadWriteArrows(slide, connectors, panel);
  addRepeatedBlocks(slide, { prefix: "bottleneck-fusion", x: 930, y: panel.y, countLabel: "×K", fill: PALETTE.bottleneckFill });
  addInspectorGroup(slide, panel, { mapping: "8928 → 1024 → 128" });
  connectPostBottleneckIndependentLanes(slide, connectors, panel);
}
```

Define panel centers as `y = 180, 430, 680`, create connectors before visible nodes where possible, use 1.0–1.2 pt dashed group strokes, and keep every connector behind labels.

- [ ] **Step 3: Implement every specialized drawing helper used by the panel functions**

Place these helpers before `addDecisionPanel`. They rely only on the already-copied `addText`, `addBox`, and `addFlow` primitives:

```js
const VIEW_OFFSETS = [-28, 0, 28];

function panelLaneY(panel, index) { return panel.y + VIEW_OFFSETS[index]; }

function addPanelTitle(slide, panel) {
  const accent = PALETTE[panel.id];
  addText(slide, { name: `${panel.id}-letter`, text: panel.letter, left: 24, top: panel.y - 88, width: 58, height: 32, fontSize: 26, bold: true, typeface: "Times New Roman" });
  addText(slide, { name: `${panel.id}-title`, text: `${panel.title} · ${panel.tag}`, left: 82, top: panel.y - 88, width: 430, height: 32, fontSize: 25, bold: true, color: accent, typeface: "Arial" });
}

function addThreeViewInputs(slide, panel) {
  FIGURE.views.forEach((view, index) => addBox(slide, {
    name: `${panel.id}-input-${view}`, left: 110, top: panelLaneY(panel, index) - 15,
    width: 64, height: 30, text: view, fill: "#FFFFFF", stroke: PALETTE.line,
    fontSize: 19, bold: true, typeface: "Times New Roman",
  }));
}

function addDashedGroup(slide, { name, label, x, y, w, h, stroke = PALETTE.line }) {
  const group = slide.shapes.add({
    geometry: "roundRect", name, position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { style: "dash", fill: stroke, width: 1.2 },
    borderRadius: 16, shadow: "shadow-none",
  });
  addText(slide, { name: `${name}-label`, text: label, left: x + 12, top: y + 4, width: w - 24, height: 22, fontSize: 17, bold: true, color: PALETTE.text, typeface: "Arial", alignment: "right" });
  return group;
}

function addRepeatedBlocks(slide, { prefix, x, y, countLabel, fill }) {
  [0, 8, 16].forEach((dx, index) => addBox(slide, {
    name: `${prefix}-stack-${index}`, left: x + dx, top: y - 34 - dx / 2,
    width: 92, height: 68, text: index === 2 ? "Transformer\nblock" : "",
    fill, stroke: PALETTE.line, fontSize: 15, typeface: "Arial",
  }));
  addText(slide, { name: `${prefix}-count`, text: countLabel, left: x + 112, top: y - 15, width: 70, height: 30, fontSize: 18, typeface: "Times New Roman" });
}

function addFeatureCodes(slide, panel, labels) {
  labels.forEach((label, index) => addBox(slide, {
    name: `${panel.id}-feature-${index}`, left: 665, top: panelLaneY(panel, index) - 15,
    width: 68, height: 30, text: label, fill: "#FFFFFF", stroke: PALETTE.line,
    fontSize: 18, typeface: "Times New Roman",
  }));
}

function addInspectorGroup(slide, panel, { mapping }) {
  addDashedGroup(slide, { name: `${panel.id}-inspectors`, label: "View-specific inspectors", x: 1170, y: panel.y - 72, w: 290, h: 144 });
  FIGURE.views.forEach((view, index) => addBox(slide, {
    name: `${panel.id}-inspector-${view}`, left: 1200, top: panelLaneY(panel, index) - 16,
    width: 210, height: 32, text: `ηk̂^(${view})   ${mapping}`,
    fill: PALETTE.neutralFill, stroke: PALETTE.line, fontSize: 14, typeface: "Times New Roman",
  }));
}

function addDistanceNodes(slide, panel, labels) {
  labels.forEach((label, index) => addBox(slide, {
    name: `${panel.id}-distance-${index}`, left: 1480, top: panelLaneY(panel, index) - 15,
    width: 54, height: 30, text: label, fill: "#FFFFFF", stroke: PALETTE.decision,
    fontSize: 17, typeface: "Times New Roman", geometry: "ellipse",
  }));
}

function addScoreNode(slide, panel, formula) {
  addBox(slide, {
    name: `${panel.id}-score`, left: 1580, top: panel.y - 40, width: 168, height: 80,
    text: formula, fill: PALETTE.decisionFill, stroke: PALETTE.decision,
    fontSize: 15, typeface: "Times New Roman", geometry: "ellipse",
  });
}

function addQkvAttentionResidual(slide, panel, { labels, output, accent }) {
  labels.forEach((label, index) => addBox(slide, {
    name: `feature-${label}`, left: 770, top: panel.y - 55 + index * 38,
    width: 42, height: 28, text: label, fill: "#FFFFFF", stroke: accent,
    fontSize: 16, bold: true, typeface: "Times New Roman",
  }));
  addBox(slide, { name: "feature-mha", left: 835, top: panel.y - 50, width: 118, height: 66, text: "Multi-head\nattention", fill: PALETTE.featureFill, stroke: accent, fontSize: 15, typeface: "Arial" });
  addBox(slide, { name: "feature-addnorm", left: 980, top: panel.y - 50, width: 90, height: 66, text: "Add &\nNorm", fill: "#FFFFFF", stroke: accent, fontSize: 15, typeface: "Arial" });
  addText(slide, { name: "feature-output", text: output, left: 1080, top: panel.y - 16, width: 55, height: 30, fontSize: 19, color: accent, typeface: "Times New Roman" });
}

function addTokenStacks(slide, panel, labels) {
  labels.forEach((label, index) => {
    [0, 7, 14].forEach((dx, layer) => addBox(slide, {
      name: `bottleneck-${label}-${layer}`, left: 620 + dx, top: panelLaneY(panel, index) - 16 - dx / 3,
      width: 56, height: 26, text: layer === 2 ? label : "", fill: PALETTE.neutralFill,
      stroke: PALETTE.line, fontSize: 14, typeface: "Times New Roman",
    }));
  });
}

function addSharedBottleneckTokens(slide, panel, { count, label, accent }) {
  Array.from({ length: count }, (_, index) => addBox(slide, {
    name: `bottleneck-B-${index}`, left: 760 + index * 28, top: panel.y - 15,
    width: 22, height: 30, text: index === 0 ? label : "", fill: PALETTE.bottleneckFill,
    stroke: accent, fontSize: 13, bold: true, typeface: "Times New Roman",
  }));
  addText(slide, { name: "bottleneck-average", text: "average updated B copies", left: 744, top: panel.y + 24, width: 150, height: 20, fontSize: 12, color: accent, typeface: "Arial", alignment: "center" });
}

function connectIndependentLanesThroughDistanceOnly(slide, connectors, panel) {
  FIGURE.views.forEach((view, index) => {
    const y = panelLaneY(panel, index);
    addFlow(slide, connectors, { name: `decision-in-${view}`, x1: 174, y1: y, x2: 250, y2: y });
    addFlow(slide, connectors, { name: `decision-enc-${view}`, x1: 650, y1: y, x2: 665, y2: y });
    addFlow(slide, connectors, { name: `decision-insp-${view}`, x1: 733, y1: y, x2: 1170, y2: y });
    addFlow(slide, connectors, { name: `decision-dist-${view}`, x1: 1460, y1: y, x2: 1480, y2: y });
    addFlow(slide, connectors, { name: `decision-score-${view}`, x1: 1534, y1: y, x2: 1580, y2: panel.y + VIEW_OFFSETS[index] * 0.35, color: PALETTE.decision, kind: "curved" });
  });
}

function connectFeatureAttentionLanes(slide, connectors, panel) {
  FIGURE.views.forEach((view, index) => {
    const y = panelLaneY(panel, index);
    addFlow(slide, connectors, { name: `feature-in-${view}`, x1: 174, y1: y, x2: 250, y2: y });
    addFlow(slide, connectors, { name: `feature-code-${view}`, x1: 610, y1: y, x2: 665, y2: y });
    addFlow(slide, connectors, { name: `feature-qkv-${view}`, x1: 733, y1: y, x2: 770, y2: panel.y - 41 + index * 38, color: PALETTE.feature });
    addFlow(slide, connectors, { name: `feature-inspector-${view}`, x1: 1135, y1: y, x2: 1170, y2: y, color: PALETTE.feature });
  });
  addFlow(slide, connectors, { name: "feature-residual", x1: 770, y1: panel.y + 62, x2: 1025, y2: panel.y + 62, color: PALETTE.feature, kind: "elbow" });
}

function addBottleneckReadWriteArrows(slide, connectors, panel) {
  FIGURE.views.forEach((view, index) => {
    const y = panelLaneY(panel, index);
    addFlow(slide, connectors, { name: `bottleneck-read-${view}`, x1: 690, y1: y, x2: 760, y2: panel.y, color: PALETTE.bottleneck, kind: "curved" });
    addFlow(slide, connectors, { name: `bottleneck-write-${view}`, x1: 870, y1: panel.y, x2: 920, y2: y, color: PALETTE.bottleneck, kind: "curved" });
  });
}

function connectPostBottleneckIndependentLanes(slide, connectors, panel) {
  FIGURE.views.forEach((view, index) => {
    const y = panelLaneY(panel, index);
    addFlow(slide, connectors, { name: `bottleneck-in-${view}`, x1: 174, y1: y, x2: 250, y2: y });
    addFlow(slide, connectors, { name: `bottleneck-tokens-${view}`, x1: 520, y1: y, x2: 620, y2: y });
    addFlow(slide, connectors, { name: `bottleneck-late-${view}`, x1: 920, y1: y, x2: 1080, y2: y, color: PALETTE.bottleneck });
    addFlow(slide, connectors, { name: `bottleneck-inspector-${view}`, x1: 1130, y1: y, x2: 1170, y2: y, color: PALETTE.bottleneck });
  });
}
```

- [ ] **Step 4: Add exact export and inspection code**

After creating the slide and invoking the three panel functions, export with:

```js
const pngPath = path.join(outDir, OUTPUTS.pptxPng);
const layoutPath = path.join(outDir, "fig6_fusion_variants_technical.layout.json");
const inspectPath = path.join(outDir, "fig6_fusion_variants_technical.inspect.ndjson");
const pptxPath = path.join(outDir, OUTPUTS.pptx);

await writeBlob(pngPath, await presentation.export({ slide, format: "png", scale: 2 }));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(layoutPath, await layout.text(), "utf8");
const snapshot = await presentation.inspect({ kind: "slide,textbox,shape,layout", maxChars: 50000 });
await fs.writeFile(inspectPath, snapshot.ndjson, "utf8");
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(pptxPath);
console.log(JSON.stringify({ pptxPath, pngPath, layoutPath, inspectPath }));
```

- [ ] **Step 5: Run the builder smoke test**

Copy the builder and spec into `$TMP_DIR`, set the runtime search path, and run:

```powershell
$env:FIG6_TECH_WORKSPACE="$TMP_DIR"
node --test tests/test_fig6_fusion_variants_technical.mjs
```

Expected: five tests PASS.

- [ ] **Step 6: Export the canonical deliverables to `paper1/figs/`**

```powershell
node "$TMP_DIR\build_fig6_fusion_variants_technical.mjs" --out-dir "D:\uwb_gestures\paper1\figs"
```

Expected: the PPTX and PNG exist and are non-empty; the old PDF remains unchanged.

- [ ] **Step 7: Render and inspect the PowerPoint**

```powershell
python "$SKILL_DIR\container_tools\render_slides.py" "D:\uwb_gestures\paper1\figs\fig6_fusion_variants_technical.pptx"
python "$SKILL_DIR\container_tools\slides_test.py" "D:\uwb_gestures\paper1\figs\fig6_fusion_variants_technical.pptx"
```

Expected: one rendered slide, zero overflow findings. Inspect the full PNG and detailed crops with `view_image`; fix overlaps, incorrect arrow direction, overly small notation, or unintended cross-view edges before continuing.

- [ ] **Step 8: Commit the editable figure source and PPTX render**

```powershell
git add paper1/figs/fig6_fusion_variants_technical_spec.mjs paper1/figs/build_fig6_fusion_variants_technical.mjs tests/test_fig6_fusion_variants_technical.mjs paper1/figs/fig6_fusion_variants_technical.pptx paper1/figs/fig6_fusion_variants_technical_pptx.png
git commit -m "feat: build technical paper1 figure2 schematic"
```

### Task 4: Write the reusable GPT Image prompt and generate the AI interpretation

**Files:**
- Create: `paper1/figs/fig6_fusion_variants_technical_prompt.txt`
- Generate: `paper1/figs/fig6_fusion_variants_technical_ai.png`

- [ ] **Step 1: Write the exact prompt**

Use the approved design spec as the scientific contract. The prompt must identify the user screenshot as the style reference and the PPT render as the structure reference. Include these non-negotiable instructions:

```text
Create a publication-quality technical neural-network architecture schematic for Paper 1 Figure 2.

Use reference image 1 only for visual language: black and cool-grey technical line art, rounded dashed group containers, compact internal modules, explicit branches and residual routes, mathematical symbols, pure white background, and sparse pale blue/amber/lavender fills.

Use reference image 2 as the strict scientific structure. Preserve exactly three horizontal panels in this order:
(a) Decision-level aggregation (Ours): L/T/R remain independent through shared-weight encoders and view-specific inspectors; dL, dT, dR meet only at the final anomaly-score node.
(b) Feature-level fusion (F1): post-encoder L/T/R feature streams enter a Q/K/V multi-head cross-view attention module with an Add & Norm residual path before the inspectors.
(c) Bottleneck mid-level fusion (F2, MBT-style): ordinary L/T/R token streams remain separate; exactly four shared amber bottleneck tokens are the only cross-view channel inside the encoder; no direct ordinary-token cross-lane arrows.

Show enough internal detail to read as a research-paper architecture figure, not a business infographic. Use thin precise arrows, aligned lanes, repeated-block labels ×4, ×(4−K), and ×K, small tensor labels, and restrained typography. Do not add performance values, datasets, closed-set classification stages, acceptance/rejection boxes, decorative 3D objects, gradients, shadows, watermarks, logos, or invented modules.

Keep text short. Render only these labels where legible: L, T, R, Shared-weight encoder, Cross-view attention, Add & Norm, B, View-specific inspectors, dL, dT, dR, and s. Leave clean editable-looking spaces rather than inventing unreadable formula text.
```

- [ ] **Step 2: Generate with the built-in image tool**

Call the built-in image generator with:

- `referenced_image_paths[0]`: the user's supplied technical architecture screenshot.
- `referenced_image_paths[1]`: `paper1/figs/fig6_fusion_variants_technical_pptx.png`.
- `prompt`: the exact prompt file contents plus a final sentence requiring a wide 2.1:1 canvas.

Expected: one high-resolution wide raster image.

- [ ] **Step 3: Copy the generated file to the canonical path and inspect it**

Copy the selected generated image from the Codex generated-images location to:

`D:\uwb_gestures\paper1\figs\fig6_fusion_variants_technical_ai.png`

Inspect with `view_image`. Reject and regenerate if any of these occur: wrong panel order, early cross-view edges in (a), direct ordinary-token cross-edges in (c), invented decoder/classifier stages, unreadable central labels, or non-white background.

- [ ] **Step 4: Commit the prompt and accepted AI image**

```powershell
git add paper1/figs/fig6_fusion_variants_technical_prompt.txt paper1/figs/fig6_fusion_variants_technical_ai.png
git commit -m "feat: add GPT Image interpretation of paper1 figure2"
```

### Task 5: Build the comparison image and run final visual QA

**Files:**
- Generate: `paper1/figs/fig6_fusion_variants_technical_comparison.png`

- [ ] **Step 1: Create a two-image comparison montage**

Create a temporary folder containing only the PPT PNG and AI PNG, then run the bundled montage helper:

```powershell
python "$SKILL_DIR\container_tools\create_montage.py" --input_dir "$TMP_DIR\comparison-input" --output_file "D:\uwb_gestures\paper1\figs\fig6_fusion_variants_technical_comparison.png"
```

Expected: one side-by-side PNG containing both full figures at readable scale.

- [ ] **Step 2: Perform the scientific checklist**

Verify all statements are true:

```text
[ ] Exactly three panels in (a)/(b)/(c) order.
[ ] (a) has no cross-view edge before dL/dT/dR reach the final score node.
[ ] (b) contains Q/K/V or multi-head attention plus Add & Norm.
[ ] (b) preserves the range-path stream outside post-encoder attention.
[ ] (c) contains exactly four shared bottleneck tokens.
[ ] (c) has no direct ordinary-token cross-lane edge.
[ ] Figure 1-only classifier, view-drop, accept, and reject stages are absent.
[ ] No OSCR values or method-ranking claims appear.
[ ] PPT notation is editable and legible at manuscript width.
[ ] AI image adds no invented modules or arrow directions.
```

- [ ] **Step 3: Run the full verification commands**

```powershell
$env:FIG6_TECH_WORKSPACE="$TMP_DIR"
node --test tests/test_fig6_fusion_variants_technical.mjs
python "$SKILL_DIR\container_tools\slides_test.py" "D:\uwb_gestures\paper1\figs\fig6_fusion_variants_technical.pptx"
git status --short
```

Expected: all Node tests pass, slide QA reports no overflow, and only intended deliverables/source files are staged or untracked.

- [ ] **Step 4: Commit the comparison image**

```powershell
git add paper1/figs/fig6_fusion_variants_technical_comparison.png
git commit -m "docs: add paper1 figure2 visual comparison"
```

### Task 6: Delivery handoff

**Files:**
- Verify only; no manuscript modification.

- [ ] **Step 1: Confirm the manuscript include is unchanged**

Run:

```powershell
git diff -- paper1/mvosr_paper.tex
```

Expected: no new diff caused by this figure task.

- [ ] **Step 2: Deliver the canonical outputs**

Return links to the PPTX, PPT PNG, AI PNG, comparison PNG, prompt text, and builder source. State that the PPTX is the exact editable source of record and that replacing `fig6_fusion_variants.pdf` remains a separate user choice.
