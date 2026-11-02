# MVOSR Figure 2 Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publication-grade Chinese Figure 2 that uses one real synchronized UWB gesture sample to explain the dual-path encoder and makes F2, F1, and decision-level aggregation distinguishable by fusion location.

**Architecture:** A Python asset exporter reads the fixed `HV1/G1/repetition 10` CSV rows, applies the repository's exact `abs_std` transform, and produces traceable CIR/slicing assets plus metadata. GPT Image supplies a low-text composition master and the acquisition-scene illustration; an `@oai/artifact-tool` JavaScript builder reconstructs the final one-slide figure with editable native shapes, while retaining real CIR rasters. Rendering and structural tests verify scientific content, text legibility, bounds, and unintended overlaps.

**Tech Stack:** Python 3, NumPy, pandas, Matplotlib, GPT Image, Node.js ES modules, `@oai/artifact-tool`, LibreOffice/PowerPoint-compatible PPTX rendering utilities.

---

## File map

- Create `paper1/fig_scripts/export_fig2_real_assets.py`: fixed-sample loader, exact `abs_std` transform, shared-scale render, slice overlay, and provenance JSON.
- Create `paper1/fig_scripts/test_export_fig2_real_assets.py`: numerical and provenance checks for sample selection, transform, dimensions, and shared scale.
- Create `paper1/figs/assets/fig2_cvpr_v2/`: generated CIR PNGs, slice PNG, provenance JSON, and GPT-generated acquisition scene.
- Create `paper1/figs/fig2_mvosr_cvpr_prompt.txt`: final GPT Image prompt used for the visual master.
- Create `paper1/figs/build_fig2_mvosr_cvpr_editable.mjs`: editable PowerPoint composition using artifact-tool.
- Create `paper1/figs/fig2_mvosr_cvpr_master.png`: GPT Image visual master.
- Create `paper1/figs/fig2_mvosr_cvpr_editable.pptx`: editable final figure.
- Replace `paper1/figs/fig2_mvosr_cvpr_preview.png`: fresh render of the new PPTX, replacing the previous rejected preview while retaining all older source files.
- Create `paper1/figs/fig2_mvosr_cvpr_preview.pdf` when export is stable.

### Task 1: Export one synchronized real-data evidence package

**Files:**
- Create: `paper1/fig_scripts/test_export_fig2_real_assets.py`
- Create: `paper1/fig_scripts/export_fig2_real_assets.py`
- Create: `paper1/figs/assets/fig2_cvpr_v2/cir_L.png`
- Create: `paper1/figs/assets/fig2_cvpr_v2/cir_T.png`
- Create: `paper1/figs/assets/fig2_cvpr_v2/cir_R.png`
- Create: `paper1/figs/assets/fig2_cvpr_v2/xv_slice_overlay.png`
- Create: `paper1/figs/assets/fig2_cvpr_v2/provenance.json`

- [ ] **Step 1: Write tests for the fixed sample and preprocessing contract**

```python
import json
from pathlib import Path
import numpy as np

from export_fig2_real_assets import abs_std, load_fixed_sample, robust_shared_limits


def test_abs_std_matches_repository_contract():
    x = np.arange(3 * 90 * 189, dtype=np.float32).reshape(3, 90, 189) - 1000
    got = abs_std(x)
    expected = np.abs(x)
    expected = (expected - expected.mean((1, 2), keepdims=True)) / (
        expected.std((1, 2), keepdims=True) + 1e-6
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_fixed_sample_is_three_synchronized_90_by_189_views():
    x, meta = load_fixed_sample()
    assert x.shape == (3, 90, 189)
    assert meta["participant"] == "HV1"
    assert meta["gesture"] == "G1"
    assert meta["repetition"] == 10
    assert meta["views"] == ["Left", "Top", "Right"]
    assert meta["row_range_zero_based"] == [810, 900]


def test_shared_limits_are_identical_for_all_views():
    x, _ = load_fixed_sample()
    lo, hi = robust_shared_limits(abs_std(x))
    assert np.isfinite([lo, hi]).all()
    assert lo < hi
```

- [ ] **Step 2: Run tests and confirm they fail because the exporter does not yet exist**

Run:

```powershell
& 'C:\Users\19252\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest paper1\fig_scripts\test_export_fig2_real_assets.py -q
```

Expected: import failure for `export_fig2_real_assets`.

- [ ] **Step 3: Implement the fixed-sample loader and exact transform**

The implementation must expose these stable interfaces:

```python
DATA_DIR = Path(r"D:\uwb_gestures\12652592\CSV_Format_HGR_DataSet_4participants\CSV_Format_HGR_DataSet_4participants\HV_01\ClutterRemovedData_HV01")
VIEW_FILES = {
    "Left": DATA_DIR / "HV1_G1_RadarLeft_ClutterRemoved_100samples.csv",
    "Top": DATA_DIR / "HV1_G1_RadarTop_ClutterRemoved_100samples.csv",
    "Right": DATA_DIR / "HV1_G1_RadarRight_ClutterRemoved_100samples.csv",
}
REPETITION = 10
T, M = 90, 189


def load_fixed_sample():
    start = (REPETITION - 1) * T
    stop = start + T
    views = [pd.read_csv(VIEW_FILES[name], header=None).to_numpy(np.float32)[start:stop]
             for name in ("Left", "Top", "Right")]
    x = np.stack(views, axis=0)
    return x, {
        "participant": "HV1", "gesture": "G1", "repetition": REPETITION,
        "views": ["Left", "Top", "Right"], "shape": list(x.shape),
        "row_range_zero_based": [start, stop],
        "normalization": "abs_std: abs(x), then per-sample/per-view mean and std over T,M",
    }


def abs_std(x):
    x = np.abs(np.asarray(x, dtype=np.float32))
    return (x - x.mean((1, 2), keepdims=True)) / (x.std((1, 2), keepdims=True) + 1e-6)


def robust_shared_limits(x):
    return tuple(float(v) for v in np.percentile(x, [1.0, 99.5]))
```

- [ ] **Step 4: Render all real-data assets from the same transformed array**

Use `magma` with a shared `vmin/vmax`, no synthetic smoothing, and 300 dpi. `xv_slice_overlay.png` must reuse the Top-view array and mark one representative time row and one representative range column selected by maximum centered energy, so the selection is deterministic and data-derived. Write the exact row/column indices, limits, source paths, and SHA-256 hashes into `provenance.json`.

- [ ] **Step 5: Run tests and inspect the four PNGs at original resolution**

Run the pytest command from Step 2, then inspect all PNGs. Expected: `3 passed`; each heatmap has shape 90×189 before rasterization; the overlay's row and column cross at a visible gesture-energy region.

- [ ] **Step 6: Commit only the exporter, tests, and generated evidence package**

```powershell
git add -- paper1/fig_scripts/export_fig2_real_assets.py paper1/fig_scripts/test_export_fig2_real_assets.py paper1/figs/assets/fig2_cvpr_v2
git commit -m "fig: export traceable Figure 2 UWB assets"
```

### Task 2: Generate the low-text GPT Image visual master

**Files:**
- Create: `paper1/figs/fig2_mvosr_cvpr_prompt.txt`
- Create: `paper1/figs/fig2_mvosr_cvpr_master.png`
- Create: `paper1/figs/assets/fig2_cvpr_v2/acquisition_scene.png`

- [ ] **Step 1: Build one local reference sheet**

Compose a temporary sheet containing the four real-data assets, the approved upper/lower wireframe, and the user's CVPR reference image. Keep the real heatmaps untouched and label their reserved slots only with `L`, `T`, and `R`.

- [ ] **Step 2: Save the final prompt text**

The prompt must request a 2.19:1 white scientific figure with a top main pipeline and three equal bottom comparison panels. It must explicitly state: real UWB three-radar geometry; thick navy main arrows; pale-blue local highlights; no rounded-card UI; no gradients, glow, 3D plastic, fake matrices, decorative neural textures, long Chinese text, or equations; blank label zones for later PPTX text; visual emphasis on the three different exchange locations.

- [ ] **Step 3: Generate the visual master with GPT Image**

Use the local reference sheet and the user's supplied CVPR reference as image references. Preserve the real-data regions as reserved rectangles and generate the acquisition scene as clean editorial vector-like art. Save the strongest result as `fig2_mvosr_cvpr_master.png`.

- [ ] **Step 4: Crop/export the acquisition scene**

If the visual master provides a clean acquisition scene, use the same image as reference for one focused GPT Image edit that returns only the scene on white/transparent background. Save it as `acquisition_scene.png`.

- [ ] **Step 5: Visual acceptance check**

Confirm: the visual hierarchy reads left-to-right; the three lower panels are equal; no text is garbled; no fake heatmaps appear; the blue emphasis marks encoder-middle, encoder-output, and distance-output positions respectively; the result resembles a paper figure rather than a product infographic.

### Task 3: Reconstruct the figure as an editable PPTX

**Files:**
- Create: `paper1/figs/build_fig2_mvosr_cvpr_editable.mjs`
- Create: `paper1/figs/fig2_mvosr_cvpr_editable.pptx`

- [ ] **Step 1: Read the artifact-tool API references required by this deck**

Read `API_QUICK_START.md`, `presentation.spec.md`, `slide.spec.md`, `shapes.spec.md`, `rich-text.spec.md`, `images.spec.md`, and `connectors.md`. Use only documented APIs.

- [ ] **Step 2: Create the exact presentation shell**

The builder must create one 13.33×6.10 in slide, white background, and fixed color constants:

```javascript
const C = {
  ink: "#17243A", blue: "#2F80C9", paleBlue: "#DDEFF8",
  temporal: "#C96B5C", range: "#3A7CA5",
  light: "#F2F3F5", white: "#FFFFFF",
};
const FONT_CN = "Microsoft YaHei";
const FONT_EN = "Arial";
```

- [ ] **Step 3: Place connectors before nodes and build the upper pipeline**

Create all data-flow and guide connectors first. Then add the acquisition raster, three CIR rasters, `abs-std`, the real `X_v` slice visual, parallel temporal/range paths, `D=32`, four Transformer layers, flatten/concatenate, `f_v`, the compact inspector-distance stage, and the known/unknown decision. Use native shapes and editable Chinese text for every label.

- [ ] **Step 4: Build the three scientifically distinct lower panels**

Implement three equal-width panels with identical L/T/R baselines:

- F2: both temporal and range paths show separate four-token bottleneck exchanges in the encoder middle.
- F1: only post-encoder temporal tokens exchange; the range path is visibly unconnected and bypasses unchanged.
- Ours: no cross-view line before `d_L,d_T,d_R`; only the final score node aggregates them.

Use one pale-blue local mechanism field per panel and no large rounded card background.

- [ ] **Step 5: Export the PPTX**

Run the builder with the bundled Node executable and set `NODE_PATH` to the bundled Node modules. Expected: `fig2_mvosr_cvpr_editable.pptx` exists, contains one slide, and opens without repair prompts.

### Task 4: Render, inspect, and repair the final artifact

**Files:**
- Replace: `paper1/figs/fig2_mvosr_cvpr_preview.png`
- Create: `paper1/figs/fig2_mvosr_cvpr_preview.pdf`

- [ ] **Step 1: Render the PPTX to PNG**

Run `render_slides.py` from the Presentations skill container tools. Expected: one slide PNG. Copy the slide render to `fig2_mvosr_cvpr_preview.png`.

- [ ] **Step 2: Run structural slide tests**

Run `slides_test.py` on the PPTX. Expected: no out-of-bounds objects and no unintended overlaps. Any flagged overlap between an arrow and its target node is acceptable only when it is the intended connector endpoint; all text-text and panel-panel overlaps must be zero.

- [ ] **Step 3: Inspect at full resolution and at paper scale**

At full resolution verify labels, arrow direction, raster sharpness, and panel alignment. At approximately 183 mm width verify that the three fusion locations remain visually separable and the minimum text is legible.

- [ ] **Step 4: Repair and repeat until clean**

Modify only `build_fig2_mvosr_cvpr_editable.mjs`, regenerate the PPTX, rerender, and rerun structural tests until all acceptance checks pass.

- [ ] **Step 5: Export PDF if stable**

Use LibreOffice conversion through the bundled runtime. Open/render the PDF and confirm its single page visually matches the PNG preview.

### Task 5: Final scientific audit and handoff

**Files:**
- Verify: all deliverables listed in the design specification.

- [ ] **Step 1: Audit provenance**

Compare `provenance.json` against the three source CSVs and confirm all assets derive from rows 810–899 of the same repetition.

- [ ] **Step 2: Audit scientific distinctions**

Check the final preview against `uwb_osr/model.py` and `mvosr_paper_cn.tex`: F1 temporal-only exchange, F2 both paths with separate four-token banks, and no cross-view exchange before distances in the proposed method.

- [ ] **Step 3: Verify output inventory**

Confirm the master PNG, editable PPTX, preview PNG, real-data assets, prompt, builder, and optional PDF exist and are non-empty. Do not modify the manuscript image reference.

- [ ] **Step 4: Present the visual result and editable file**

Show the preview directly in the response and link the editable PPTX, master PNG, preview PNG, prompt, and provenance JSON.

