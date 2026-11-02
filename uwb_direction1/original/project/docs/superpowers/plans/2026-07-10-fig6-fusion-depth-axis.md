# Fig. 6 Fusion-Depth-Axis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publication-ready, fully editable PPTX Fig. 6 plus an AI-generated visual interpretation and a side-by-side comparison under `paper1/figs/`.

**Architecture:** Store the scientific structure and layout constants in a dependency-free ES module, test that model first, then render it through `@oai/artifact-tool` from an external temporary workspace. The PowerPoint version is the exact source of record; the AI image is a style comparison using the same five-stage fusion-depth story.

**Tech Stack:** Node.js ES modules, Node test runner, `@oai/artifact-tool`, bundled slide QA tools, built-in `imagegen`, and the bundled montage utility.

---

## File map

- Create `paper1/figs/fig6_fusion_depth_axis_spec.mjs`: immutable stages, strategies, palette, dimensions, labels, and validation helpers.
- Create `paper1/figs/build_fig6_fusion_depth_axis.mjs`: artifact-tool authoring, PNG/layout export, and PPTX export.
- Create `tests/test_fig6_fusion_depth_axis.mjs`: structural tests and end-to-end builder smoke test.
- Modify `docs/superpowers/specs/2026-07-10-fig6-fusion-depth-axis-design.md`: correct the reproducible builder extension from `.py` to `.mjs` because the presentation runtime requires JavaScript artifact-tool authoring.
- Generate `paper1/figs/fig6_fusion_depth_axis.pptx`.
- Generate `paper1/figs/fig6_fusion_depth_axis_pptx.png`.
- Generate `paper1/figs/fig6_fusion_depth_axis_ai.png`.
- Generate `paper1/figs/fig6_fusion_depth_axis_comparison.png`.

### Task 1: Lock the scientific structure with a failing test

**Files:**
- Create: `tests/test_fig6_fusion_depth_axis.mjs`
- Create: `paper1/figs/fig6_fusion_depth_axis_spec.mjs`

- [ ] **Step 1: Write the failing structural test**

The first test imports the not-yet-created spec and asserts the agreed early-to-late ordering, exact stage labels, technical dimensions, palette, and banned content:

```js
import test from "node:test";
import assert from "node:assert/strict";
import {
  SLIDE,
  STAGES,
  STRATEGIES,
  PALETTE,
  validateFigureSpec,
} from "../paper1/figs/fig6_fusion_depth_axis_spec.mjs";

test("fusion depth advances from encoder to decision", () => {
  assert.deepEqual(STRATEGIES.map((d) => d.id), ["bottleneck", "feature", "decision"]);
  assert.ok(STRATEGIES[0].exchangeX < STRATEGIES[1].exchangeX);
  assert.ok(STRATEGIES[1].exchangeX < STRATEGIES[2].exchangeX);
});

test("shared scientific labels and dimensions are exact", () => {
  assert.deepEqual(STAGES.map((d) => d.id), ["input", "encoder", "feature", "inspector", "score"]);
  assert.equal(STAGES[0].notation, "xᵥ ∈ ℝ⁹⁰ˣ¹⁸⁹");
  assert.equal(STAGES[2].notation, "fᵥ ∈ ℝ⁸⁹²⁸");
  assert.equal(STAGES[3].notation, "ηₖ⁽ᵛ⁾");
  assert.equal(STAGES[4].notation, "s");
  assert.equal(SLIDE.width / SLIDE.height, 2);
});

test("figure contains structure only", () => {
  const text = JSON.stringify({ STAGES, STRATEGIES });
  for (const banned of ["OSCR", "本文采用", "稳健最优", "(a)", "(b)", "(c)"]) {
    assert.equal(text.includes(banned), false, banned);
  }
  assert.deepEqual(validateFigureSpec(), []);
  assert.equal(PALETTE.bottleneck, "#E18727");
  assert.equal(PALETTE.feature, "#E64B35");
  assert.equal(PALETTE.decision, "#00A087");
});
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
node --test tests/test_fig6_fusion_depth_axis.mjs
```

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `fig6_fusion_depth_axis_spec.mjs`.

- [ ] **Step 3: Implement the minimum dependency-free spec module**

Create a 1600×800 slide model with the five stage bands, three strategies, exact labels, NPG-style colours, and a `validateFigureSpec()` function that returns specific error strings for ordering, missing stages, out-of-bounds geometry, and banned copy.

```js
export const SLIDE = Object.freeze({ width: 1600, height: 800 });
export const PALETTE = Object.freeze({
  text: "#22252A", muted: "#667085", line: "#7B8494", border: "#D7DCE5",
  band: "#F3F5F8", white: "#FFFFFF", bottleneck: "#E18727",
  feature: "#E64B35", decision: "#00A087",
});
export const STAGES = Object.freeze([
  { id: "input", label: "输入 · Input", notation: "xᵥ ∈ ℝ⁹⁰ˣ¹⁸⁹", x: 255, width: 150 },
  { id: "encoder", label: "编码器 · Encoder", notation: "shared dual-path Transformer", x: 420, width: 390 },
  { id: "feature", label: "特征码 · Feature", notation: "fᵥ ∈ ℝ⁸⁹²⁸", x: 825, width: 185 },
  { id: "inspector", label: "检查器 · Inspector", notation: "ηₖ⁽ᵛ⁾", x: 1025, width: 210 },
  { id: "score", label: "分数 · Score", notation: "s", x: 1250, width: 190 },
]);
export const STRATEGIES = Object.freeze([
  { id: "bottleneck", label: "瓶颈中层融合", sublabel: "F2 · MBT-style", y: 225, exchangeX: 610, color: PALETTE.bottleneck },
  { id: "feature", label: "特征级融合", sublabel: "F1 · post-encoder", y: 420, exchangeX: 825, color: PALETTE.feature },
  { id: "decision", label: "决策级聚合", sublabel: "late aggregation", y: 615, exchangeX: 1345, color: PALETTE.decision },
]);
export function validateFigureSpec() {
  const errors = [];
  if (STRATEGIES.some((d, i) => i && d.exchangeX <= STRATEGIES[i - 1].exchangeX)) errors.push("fusion depth must increase");
  if (STAGES.map((d) => d.id).join(",") !== "input,encoder,feature,inspector,score") errors.push("stage order mismatch");
  const text = JSON.stringify({ STAGES, STRATEGIES });
  for (const banned of ["OSCR", "本文采用", "稳健最优", "(a)", "(b)", "(c)"]) if (text.includes(banned)) errors.push(`banned copy: ${banned}`);
  for (const d of STAGES) if (d.x < 0 || d.x + d.width > SLIDE.width) errors.push(`stage out of bounds: ${d.id}`);
  return errors;
}
```

- [ ] **Step 4: Run the test and verify GREEN**

Run: `node --test tests/test_fig6_fusion_depth_axis.mjs`

Expected: 3 tests pass, 0 fail.

### Task 2: Add the editable PPTX builder with an end-to-end failing test

**Files:**
- Modify: `tests/test_fig6_fusion_depth_axis.mjs`
- Create: `paper1/figs/build_fig6_fusion_depth_axis.mjs`

- [ ] **Step 1: Add the failing builder smoke test**

Append a test that checks the builder exists, copies the builder and spec into the initialized artifact-tool workspace, runs it, and verifies the four exact PowerPoint-side artifacts:

```js
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test("artifact-tool builder exports editable deck and QA artifacts", () => {
  const workspace = process.env.FIG6_ARTIFACT_WORKSPACE;
  assert.ok(workspace, "FIG6_ARTIFACT_WORKSPACE is required");
  const source = path.join(ROOT, "paper1", "figs", "build_fig6_fusion_depth_axis.mjs");
  assert.ok(fs.existsSync(source), "builder source must exist");
  const workBuilder = path.join(workspace, "build_fig6_fusion_depth_axis.mjs");
  const workSpec = path.join(workspace, "fig6_fusion_depth_axis_spec.mjs");
  fs.copyFileSync(source, workBuilder);
  fs.copyFileSync(path.join(ROOT, "paper1", "figs", "fig6_fusion_depth_axis_spec.mjs"), workSpec);
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "fig6-depth-axis-test-"));
  const result = spawnSync(process.execPath, [workBuilder, "--out-dir", out], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  for (const file of ["fig6_fusion_depth_axis.pptx", "fig6_fusion_depth_axis_pptx.png", "fig6_fusion_depth_axis.layout.json", "fig6_fusion_depth_axis.inspect.ndjson"]) {
    const target = path.join(out, file);
    assert.ok(fs.existsSync(target), file);
    assert.ok(fs.statSync(target).size > 500, file);
  }
});
```

- [ ] **Step 2: Initialize the external artifact-tool workspace and verify RED**

Run the bundled setup script against `%TEMP%\codex-presentations\fig6-fusion-depth-axis\tmp`, set `FIG6_ARTIFACT_WORKSPACE` to that directory, then run the Node test.

Expected: structural tests pass; builder test fails with `builder source must exist`.

- [ ] **Step 3: Implement the artifact-tool builder**

The builder must:

- parse `--out-dir`;
- create a 1600×800 presentation;
- create all connectors before entity nodes;
- add pale stage bands and shared headings once;
- render three L/T/R micro-lanes per strategy;
- show an amber bottleneck-token bridge inside the split encoder;
- show a coral cross-view-attention block after the encoder;
- converge only the decision track at the teal final score node;
- add a bottom `Fusion depth · early → late` axis;
- export PNG at scale 2, layout JSON, inspect NDJSON, and PPTX.

Use named objects such as `stage-band-encoder`, `strategy-bottleneck-label`, `exchange-feature-attention`, and `exchange-decision-score` so QA can locate every scientific component. Text styles use at least 32 px for stage/strategy labels and 22 px for body notation.

- [ ] **Step 4: Run the builder test and verify GREEN**

Run the same Node test with `FIG6_ARTIFACT_WORKSPACE` set.

Expected: 4 tests pass, 0 fail; the temporary output contains PPTX, PNG, layout JSON, and inspect NDJSON.

### Task 3: Export final PowerPoint artifacts and perform visual QA

**Files:**
- Generate: `paper1/figs/fig6_fusion_depth_axis.pptx`
- Generate: `paper1/figs/fig6_fusion_depth_axis_pptx.png`

- [ ] **Step 1: Run the tested builder against `paper1/figs/`**

Copy the builder and spec into the initialized temporary workspace and run:

```powershell
node <temporary-workspace>\build_fig6_fusion_depth_axis.mjs --out-dir D:\uwb_gestures\paper1\figs
```

Expected: exit 0 and the canonical PPTX plus high-resolution preview appear in `paper1/figs/`.

- [ ] **Step 2: Run automated slide QA**

Run the bundled `slides_test.py` against the final PPTX and re-run the Node test suite.

Expected: no slide-canvas overflow, 0 failed tests.

- [ ] **Step 3: Inspect the rendered PNG at original resolution**

Verify the stage headings are single-line, L/T/R lanes remain distinguishable, colour begins only after exchange, the three exchange locations move left to right, and no arrow crosses a label. If any defect appears, add a failing assertion when automatable, revise the builder, and rerun Tasks 2-3.

### Task 4: Generate and verify the AI visual interpretation

**Files:**
- Generate: `paper1/figs/fig6_fusion_depth_axis_ai.png`

- [ ] **Step 1: Generate with the built-in image tool**

Use the user's attached image and the current Fig. 6 preview as style references. Request a wide, white-background scientific deep-learning infographic with one five-stage depth axis, three unboxed strategy tracks, the same early-to-late exchange locations, restrained teal/coral/amber accents, smooth arrows, minimal English labels, and no results, metrics, panel letters, logos, watermarks, 3D decoration, skip connections, or invented modules.

- [ ] **Step 2: Save the generated image into the project**

Copy the selected generated output to `paper1/figs/fig6_fusion_depth_axis_ai.png` without altering the source PowerPoint files.

- [ ] **Step 3: Inspect the image at original resolution**

Confirm that it contains exactly three strategies, the five-stage progression is legible, the highlighted exchange positions advance from encoder-middle to post-encoder to final score, and no extra scientific claim appears. If structure or text is wrong, make one targeted regeneration and inspect again.

### Task 5: Build the side-by-side comparison and run the completion gate

**Files:**
- Generate: `paper1/figs/fig6_fusion_depth_axis_comparison.png`
- Modify: `docs/superpowers/specs/2026-07-10-fig6-fusion-depth-axis-design.md`

- [ ] **Step 1: Correct the builder extension in the approved spec**

Replace `build_fig6_fusion_depth_axis.py` with `build_fig6_fusion_depth_axis.mjs` and add one sentence that the PPTX is authored with `@oai/artifact-tool`.

- [ ] **Step 2: Create the two-image comparison**

Use the bundled `create_montage.py` on a temporary folder containing only the verified PPTX preview and AI image, writing the montage to `paper1/figs/fig6_fusion_depth_axis_comparison.png`.

- [ ] **Step 3: Run fresh verification**

Run:

```powershell
node --test tests/test_fig6_fusion_depth_axis.mjs
C:\Users\19252\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe C:\Users\19252\.codex\plugins\cache\openai-primary-runtime\presentations\26.709.11516\skills\presentations\container_tools\slides_test.py paper1\figs\fig6_fusion_depth_axis.pptx
```

Expected: all Node tests pass and slide QA reports no overflow.

- [ ] **Step 4: Verify deliverable inventory and manuscript isolation**

Confirm all five requested final files exist and have non-zero size. Confirm `paper1/figs/fig6_fusion_variants.pdf` and `paper1/mvosr_paper.tex` were not modified by this task.

- [ ] **Step 5: Report the result**

Return one standalone link to the final PPTX, links to both PNG versions and the comparison image, a concise recommendation about which version should remain canonical, and the exact final image-generation prompt.
