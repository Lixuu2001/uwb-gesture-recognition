# Paper 1 Figure 2 Fusion-Strategy Technical Schematic

## Scope

Redraw only the second figure in Paper 1, located under Section 4.4, "Systematic comparison of multi-view fusion strategies". In the LaTeX source this is the figure that includes `fig6_fusion_variants.pdf` and compares three strategies labelled (a), (b), and (c).

This task does not redraw or extend Figure 1, the overall MV-OSR architecture. The new figure must not reproduce the complete dual-path encoder, closed-set classification stage, view-drop training mechanism, acceptance/rejection outputs, or experimental results from Figure 1.

## Scientific message

The figure compares one variable: where cross-view information first exchanges.

1. **(a) Decision-level aggregation (Ours):** the three views remain independent through encoding and inspection. Only the per-view prototype distances or anomaly-score terms meet at the final aggregation node.
2. **(b) Feature-level fusion (F1):** independently encoded view features exchange information through post-encoder cross-view attention before entering the inspector stage.
3. **(c) Bottleneck mid-level fusion (F2, MBT-style):** the three view-token streams exchange a compressed summary through a small shared bottleneck-token set inside the encoder.

The comparison must make the exchange location visually obvious without requiring the caption. Panel order remains (a) decision, (b) feature, and (c) bottleneck so the existing caption and surrounding text remain valid.

## Composition

- Use a wide manuscript figure, approximately 2.0:1 to 2.2:1.
- Stack three horizontal strategy tracks, following the current (a)/(b)/(c) order.
- Use the user's attached technical architecture figure as the structural-style reference:
  - rounded dashed group containers;
  - compact internal modules;
  - explicit branch and residual routing;
  - mathematical symbols placed beside the relevant paths;
  - sparse pale fills rather than large coloured cards.
- Each strategy track uses the same left-to-right anchors: three-view input, shared-weight encoder, feature or token exchange, inspector stage, distance/score stage.
- Align equivalent stages vertically across the three tracks so the fusion point visibly shifts from right to left across (a), (b), and (c).
- Add a small shared bottom guide labelled `Fusion depth: late to early` only if it improves comparison without competing with panel labels.

## Shared visual grammar

### Inputs and view lanes

- Show L, T, and R as three thin parallel lanes.
- Use compact CIR matrix or token-stack glyphs, not generic input cards.
- Preserve the view identity labels throughout the path.

### Encoder group

- Use a rounded dashed container labelled `Shared-weight encoder`.
- The encoder is schematic in this figure: it shows repeated Transformer blocks or early/late block groups only to locate the fusion mechanism.
- Do not expand the full temporal and range paths from Figure 1.
- Use `×4` or `×K` multiplicity labels instead of drawing every repeated block.

### Inspector group

- Use a rounded dashed container labelled `View-specific inspectors` where applicable.
- Represent each inspector by the paper-faithful mapping `8928 → 1024 → 128` or a compact `MLP inspector` module with the mapping in small type.
- Show prototype-distance nodes `d_v = ||η_k̂^(v)(f_v) - φ_k̂^(v)||₂` only where they help explain decision aggregation.

### Lines and symbols

- Neutral within-view flow: dark grey solid arrows.
- Cross-view exchange: one restrained strategy accent colour.
- Residual or bypass paths: thin routed lines with clear junctions.
- Aggregation: circular `⊕` or `Σ` node.
- Avoid arrows crossing labels, group titles, or module boundaries.

## Panel-specific details

### (a) Decision-level aggregation (Ours)

- Three L/T/R lanes remain separate through the shared-weight encoder and view-specific inspectors.
- Show `f_L`, `f_T`, and `f_R`, followed by `η_k̂^(L)`, `η_k̂^(T)`, and `η_k̂^(R)`.
- Each lane produces a distance `d_L`, `d_T`, or `d_R`.
- Merge only at the final anomaly-score node:
  `s = mean_v(d_v)/σ̄ + β·var_v(d_v)/σ̄²`.
- Use a small `Ours` label beside the panel title, not a large badge or performance claim.
- No cross-view arrow may appear before the final score node.

### (b) Feature-level fusion (F1)

- Show three independent feature codes leaving the encoder.
- Expand the exchange mechanism inside its own dashed or lightly filled group:
  - three input feature/token streams;
  - Q/K/V or a compact multi-head cross-view attention symbol;
  - `Cross-view attention`;
  - `Add & Norm` residual output.
- The output remains view-indexed, `f̃_L`, `f̃_T`, and `f̃_R`, before entering the inspectors.
- Do not collapse the mechanism into a single generic red box.

### (c) Bottleneck mid-level fusion (F2, MBT-style)

- Split the encoder container into independent early blocks and late bottleneck-fusion blocks.
- Show ordinary L/T/R tokens as three separate token stacks.
- Show a small shared set of four bottleneck tokens `B`, visually distinct from ordinary tokens.
- Make the only cross-view channel explicit:
  - each view attends over `[own tokens; B]`;
  - updated bottleneck copies are averaged;
  - the shared bottleneck feeds the next fusion block.
- Ordinary view tokens must not have direct cross-lane arrows.
- Mark repeated bottleneck fusion layers as `×K` and keep total encoder depth schematic.

## Style and colour

- Pure white background.
- Primary language: black and cool-grey technical line art.
- Group containers: 1.0–1.2 pt dark-grey dashed outlines with modest corner radii.
- Modules: white or extremely pale fills.
- Neutral Transformer or inspector modules: pale blue-grey (`#EAF1F8`) with dark grey outlines.
- Feature-level exchange: muted lavender (`#8B78E6`, pale fill `#F2EEFC`).
- Bottleneck exchange: muted amber (`#D89A32`, pale fill `#FFF3DD`).
- Decision aggregation: muted teal (`#2B8793`, pale fill `#EAF6F6`).
- Text: charcoal (`#20252D`), with English/math in Times New Roman or Cambria and short Chinese labels in Microsoft YaHei if Chinese is retained.
- No gradients, strong shadows, glossy icons, decorative 3D objects, neon colours, or business-infographic styling.

## Text policy

- Use concise English module labels for publication consistency.
- Use at most two text sizes plus panel labels.
- Keep exact mathematical notation editable in the PowerPoint version.
- The AI-generated version may simplify labels, but it must not invent modules or equations.

## Deliverables

Create non-destructive outputs under `paper1/figs/`:

- `fig6_fusion_variants_technical.pptx`: canonical fully editable PowerPoint figure.
- `fig6_fusion_variants_technical_pptx.png`: high-resolution render of the PowerPoint version.
- `fig6_fusion_variants_technical_ai.png`: GPT Image visual interpretation using the same structure.
- `fig6_fusion_variants_technical_comparison.png`: side-by-side comparison.
- `fig6_fusion_variants_technical_prompt.txt`: final reusable image-generation prompt.
- `build_fig6_fusion_variants_technical.mjs`: reproducible PowerPoint build source.

Do not overwrite `fig6_fusion_variants.pdf`, modify the manuscript include, or change Figure 1 during this task.

## Verification

- Confirm the output contains exactly three panels in the existing (a)/(b)/(c) order.
- Confirm decision-level fusion has no cross-view path before the final score node.
- Confirm feature-level fusion shows a real attention-plus-residual mechanism rather than a label-only box.
- Confirm bottleneck fusion uses shared bottleneck tokens as the only cross-view channel.
- Confirm no Figure 1-only stages or experiment results appear.
- Render the PPTX and inspect the full figure plus detailed crops for clipping, line crossings, inconsistent lane spacing, unreadable formulae, and visual imbalance.
- Inspect the AI image for invented modules, corrupted labels, or incorrect arrow directions before creating the comparison image.
