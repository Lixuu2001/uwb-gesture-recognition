# Fig. 6 Fusion-Depth-Axis Redesign

## Objective

Redesign Paper 1 Fig. 6 so that readers can compare the three cross-view fusion structures on one shared network-depth coordinate system. The figure must show structural differences only. It must not contain experiment results, OSCR values, preference badges, or panel letters such as (a), (b), and (c).

## Scientific message

The sole variable is the first network depth at which information crosses views:

1. **Bottleneck mid-level fusion (F2, MBT-style):** each view exchanges information with shared bottleneck tokens inside the encoder.
2. **Feature-level fusion (F1):** three independently encoded feature codes enter a cross-view attention block after the encoder.
3. **Decision-level aggregation:** view-specific feature codes and inspectors remain independent; only their distances or anomaly scores are aggregated at the end.

The figure order is early-to-late fusion: bottleneck, feature, then decision. The highlighted exchange positions therefore move from left to right and form the figure's main visual rhythm.

## Layout

- Use a wide full-width manuscript figure, approximately 2:1 in aspect ratio.
- Use one shared header row for five stages: **Input**, **Encoder**, **Feature code**, **Inspector**, and **Score**.
- Use pale vertical stage bands to establish a common coordinate system.
- Use three unboxed strategy tracks, not three card-like panels.
- Place the strategy name at the left of each track, without letters or numbering.
- Represent the L/T/R views as three thin parallel micro-lanes within each track.
- Use a bottom arrow labelled **Fusion depth: early to late**.

## Modules and notation

Shared annotations, shown once:

- Input: three-view CIR, `x_v in R^(90 x 189)`.
- Encoder: shared-weight dual-path Transformer.
- Feature code: `f_v in R^8928`.
- Inspector: view-specific deep inspectors, `eta_k^(v)`.
- Score: aggregated anomaly score `s`.

Track-specific exchange modules:

- Bottleneck track: split the encoder into front and rear stages and place a narrow shared bottleneck-token bridge between them.
- Feature track: place a cross-view attention block directly after the shared-weight encoder and label its outputs as interacted features `f_tilde_v`.
- Decision track: keep the three paths separate through the inspectors and merge them only at the final score node.

Only real data paths are shown. No residual, U-Net-style skip, or multiscale connection is added unless it exists in the implemented model.

## Visual language

- Neutral structure: charcoal text, cool-grey strokes, white modules, pale-grey stage bands.
- Bottleneck exchange: amber (`#E18727`).
- Feature exchange: coral red (`#E64B35`).
- Decision aggregation: teal (`#00A087`).
- Before cross-view exchange, data-flow lines remain neutral grey. After exchange, the downstream path inherits the strategy colour.
- Use colour only for exchange position and interacted data flow; do not colour every module.
- Use Microsoft YaHei for Chinese labels and Arial or Times New Roman for English and mathematical notation.
- Keep line weights, corner radii, whitespace, and colours consistent with Paper 1 Fig. 1's restrained NPG-style palette.

## Deliverables

Create non-destructive files under `paper1/figs/`:

- `fig6_fusion_depth_axis.pptx`: canonical, fully editable PowerPoint figure.
- `fig6_fusion_depth_axis_pptx.png`: high-resolution render of the PowerPoint version.
- `fig6_fusion_depth_axis_ai.png`: AI-generated visual interpretation using the confirmed structure and the user's reference image as style guidance.
- `fig6_fusion_depth_axis_comparison.png`: side-by-side comparison of the PowerPoint and AI-image versions.
- `build_fig6_fusion_depth_axis.py`: reproducible PowerPoint build script.

Existing Fig. 6 files remain untouched. Replacing `fig6_fusion_variants.pdf` in the manuscript is outside this task until the user chooses a preferred version.

## PowerPoint implementation

- Use native PowerPoint shapes, text, and connectors so the figure remains editable.
- Create connectors before entity nodes so edges render behind modules.
- Use a blank slide with a custom wide aspect ratio matching the manuscript figure.
- Keep the exact technical notation editable rather than rasterized.
- Export a PNG preview at sufficient resolution for visual QA.

## AI-image implementation

- Treat the user's attached paper figure as a style reference, not an edit target.
- Generate a wide scientific infographic with a white background, restrained NPG-like colours, modular blocks, smooth data-flow curves, and minimal English labels.
- Preserve the same five-stage axis and early-to-late exchange locations.
- Do not introduce extra network blocks, skip connections, decorative 3D objects, results, metrics, panel letters, logos, or watermarks.
- The AI version is a visual comparison, not the source of record for exact notation. The PowerPoint version remains canonical because generated text may be imperfect.

## Verification

- Render the PPTX and inspect it visually at full-slide and cropped-detail scale.
- Confirm there are no unintended overlaps, clipped text, arrows crossing labels, or one-line headers wrapping.
- Confirm all three exchange positions match the paper description.
- Confirm no experimental result or preference statement appears.
- Confirm the AI image contains the same structural story and no invented modules.
- Produce the comparison image only after both individual outputs pass visual inspection.
