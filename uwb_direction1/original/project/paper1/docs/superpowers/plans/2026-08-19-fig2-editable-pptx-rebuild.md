# Figure 2 Editable PPTX Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the approved Figure 2 concept as a one-slide PowerPoint whose CIR thumbnails remain raster images and whose titles, modules, formulae, arrows, tokens, borders, and labels are native editable PowerPoint objects.

**Architecture:** Import the user-supplied PPTX with `@oai/artifact-tool`, preserve its single-slide master/layout hierarchy, explicitly inventory and classify every inherited object, then rebuild the slide locally using the approved three-panel design. Test-first structural assertions verify required text, named mechanisms, image count, banned legacy labels, and one-slide output before visual rendering and overflow QA.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled Node.js runtime, bundled PowerPoint renderer, `slides_test.py`.

---

### Task 1: Inventory the source deck and freeze the edit contract

**Files:**
- Read: `C:/Users/19252/Desktop/fig2_three_fusion.pptx`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_pptx_rebuild_20260819/inspect_source.mjs`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_pptx_rebuild_20260819/template-audit.txt`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_pptx_rebuild_20260819/template-frame-map.json`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_pptx_rebuild_20260819/deviation-log.txt`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_pptx_rebuild_20260819/source-notes.txt`

- [ ] **Step 1: Import and inspect the complete one-slide source deck**

The inspection module must import the PPTX, write the full inspect NDJSON, render slide 1, export its layout JSON, and query deletion APIs with `presentation.help`:

```js
const deck = await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const snapshot = await deck.inspect({
  kind: "slide,textbox,shape,image,notes,layout",
  include: "id,slide,name,title,text,textPreview,bbox,bboxUnit,alt,isPlaceholder,placeholders",
  maxChars: 200000,
});
await fs.writeFile(path.join(TMP, "template-inspect.ndjson"), snapshot.ndjson);
await writeBlob(path.join(TMP, "source-slide-1.png"), await deck.export({ slide: deck.slides.getItem(0), format: "png", scale: 1 }));
await fs.writeFile(path.join(TMP, "source-slide-1.layout.json"), await (await deck.slides.getItem(0).export({ format: "layout" })).text());
console.log(deck.help("*", { search: "shape\\.delete|image\\.delete", include: "index,examples,notes", maxChars: 12000 }).ndjson);
```

- [ ] **Step 2: Classify every inherited object explicitly**

`template-frame-map.json` must map output slide 1 to source slide 1 and list every source anchor under one of `delete`, `replace`, or `keep`. The rebuild intentionally deletes inherited local content while preserving the imported slide, layout, master, and notes relationship. New native primitives are authorized only within the full-slide content zone because the user approved a complete local redraw.

- [ ] **Step 3: Record provenance and deviations**

`source-notes.txt` records the user PPTX, the three project CIR assets, the local paper method section, and the approved GPT Image preview. `deviation-log.txt` records the corrected feature-level fusion, shared downstream decision path, normalized mean-variance formula, `m=1…M`, and simplified token towers.

### Task 2: Write structural tests before authoring

**Files:**
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_pptx_rebuild_20260819/test_fig2_rebuilt.mjs`
- Test target: `C:/Users/19252/Desktop/fig2_three_fusion_rebuilt_editable.pptx`

- [ ] **Step 1: Write a test that imports the expected output and checks behavior**

The test must assert one slide; exactly nine CIR images; named mechanisms `panel-a`, `panel-b`, `panel-c`, `attention-zone`, `bottleneck-zone`, `score-a`, `score-b`, and `score-c`; required Chinese titles and conclusions; correct `t=1…T` and `m=1…M`; and absence of the incorrect legacy strings.

```js
const REQUIRED_TEXT = [
  "(a) 决策级距离聚合（本文）",
  "(b) 编码后特征级融合",
  "(c) MBT 式瓶颈中层融合",
  "跨视角注意力（全特征）",
  "共享瓶颈 token（窄通道）",
  "归一化均值—方差异常分数",
  "仅在决策端汇合",
  "编码后全特征交互",
  "编码中层窄通道交换",
  "t=1…T",
  "m=1…M",
];
const BANNED_TEXT = [
  "时序 token（索引 t）",
  "距离 token（更新后）",
  "时间 token 全带宽交互；距离路径独立",
  "自连接",
  "互连接",
  "r=1…R",
];
```

The test derives image count and text from `presentation.inspect`, and derives named mechanism presence from slide layout JSON. It exits nonzero with a precise missing/forbidden-item message.

- [ ] **Step 2: Run the test and verify the RED state**

Run:

```powershell
& $env:RUNTIME_NODE test_fig2_rebuilt.mjs
```

Expected result: nonzero exit with `ENOENT` because `fig2_three_fusion_rebuilt_editable.pptx` does not exist yet.

### Task 3: Build the editable slide from the imported source

**Files:**
- Create: `D:/uwb_gestures/paper1/.codex_tmp/fig2_pptx_rebuild_20260819/build_fig2_rebuilt.mjs`
- Create: `C:/Users/19252/Desktop/fig2_three_fusion_rebuilt_editable.pptx`

- [ ] **Step 1: Mark the artifact edit exactly once**

Run immediately before authoring:

```powershell
& $env:RUNTIME_NODE "$env:SKILL_DIR/container_tools/mark_artifact_operation_started.mjs" --operation-kind edit --expected-output-count 1 --output-format pptx
```

Expected result: exit code 0.

- [ ] **Step 2: Import the source and apply the explicit edit plan**

The builder imports the source PPTX, resolves only anchors listed in the edit plan, and calls each resolved object's documented `delete()` method. It keeps the imported slide itself, its layout, its master, and any inherited non-empty notes relationship.

- [ ] **Step 3: Build the common visual grammar**

Use source-frame-relative geometry based on a 1755×896 design grid and scale all coordinates to the imported slide frame. Implement focused helpers:

```js
addText(slide, name, text, rect, style)
addBox(slide, name, rect, fill, line, text, textStyle)
addArrow(slide, name, from, to, options)
addCirTriplet(slide, panelX, panelKey)
addDualPathPair(slide, x, y, viewKey)
addInspectorDistance(slide, x, y, viewKey)
addScoreBlock(slide, panelX, panelKey)
```

Create connectors before their endpoint modules wherever possible. Use Microsoft YaHei for Chinese text, Cambria Math for formula runs where supported, white background, dark navy-gray primary lines, low-saturation yellow/blue/coral view colors, green only for exchange/score emphasis, and purple only for bottleneck tokens.

- [ ] **Step 4: Build panel (a)**

Create three parallel independent tracks from CIR through shared-parameter dual-path encoding, compact time/range blocks, view-wise inspectors, and `d_L/d_T/d_R`. No horizontal connector may appear before the three distances merge into `score-a`.

- [ ] **Step 5: Build panel (b)**

Create complete feature nodes `z_L/z_T/z_R`, one named green `attention-zone`, three updated feature nodes, and the same inspectors/distances/score. Draw cross-view links only inside the attention zone; do not add the redundant pre-attention `z_L↔z_T↔z_R` links shown in the raster concept.

- [ ] **Step 6: Build panel (c)**

Create shared-parameter encoder first and second-half bands with three independent aligned tracks. Place one named `bottleneck-zone` between them. Show each view writing to bottleneck tokens, a central `跨视角均值` label, and bottleneck write-back; add no direct L↔T, T↔R, or L↔R connector outside the bottleneck zone.

- [ ] **Step 7: Add notes, preview, layout export, and final PPTX export**

Set speaker notes to:

```text
[Sources]
- User source deck: C:/Users/19252/Desktop/fig2_three_fusion.pptx
- Method definition: D:/uwb_gestures/paper1/mvosr_paper_cn.tex, sections 2.3–2.4 and equation (score)
- CIR assets: D:/uwb_gestures/paper1/figs/assets/fig2_real_cir_L.png; fig2_real_cir_T.png; fig2_real_cir_R.png
- Visual direction: D:/uwb_gestures/paper1/figs/fig2_fusion_strategy_effect_preview_gpt_v1.png
[/Sources]
```

Export the rebuilt slide PNG at scale 1 and 2, its layout JSON, deck montage, and final PPTX.

### Task 4: Verify GREEN and perform visual QA

**Files:**
- Test: `D:/uwb_gestures/paper1/.codex_tmp/fig2_pptx_rebuild_20260819/test_fig2_rebuilt.mjs`
- Test: `C:/Users/19252/.codex/plugins/cache/openai-primary-runtime/presentations/26.813.12317/skills/presentations/container_tools/slides_test.py`
- Read: rendered slide PNG and layout JSON under the temporary build directory

- [ ] **Step 1: Run the structural test and verify GREEN**

Expected result: `PASS fig2 rebuilt structural checks` and exit code 0.

- [ ] **Step 2: Render the final PPTX independently**

Run bundled `render_slides.py` against the final PPTX and inspect the full-size slide PNG.

- [ ] **Step 3: Run overflow and boundary QA**

Run `slides_test.py` against the final PPTX. Expected result: no content outside the slide canvas and no reported text overflow.

- [ ] **Step 4: Inspect placeholder and source-note state**

Import the final PPTX again, inspect `isPlaceholder` and notes records, and fail if an inherited placeholder is empty or if the `[Sources]` block is absent.

- [ ] **Step 5: Compare against the approved concept and paper**

Confirm at full size that panel (a) has no early cross-view edge, panel (b) exchanges complete features only after encoding, panel (c) communicates only through bottleneck tokens, and all three use the same inspector/distance/score exit. Record the result in `qa-ledger.txt`.
