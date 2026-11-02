# Figure 2 Panel (c) MBT Bottleneck Redesign

## Approved direction

Edit the existing one-slide PowerPoint in place. Retain panels (a) and (b), the
panel (c) frame/title, view labels, CIR images, shared encoder region, and the
bottom conclusion band. Replace only panel (c)'s middle mechanism region.

The user supplied a detailed Claude prompt as design input. Its HTML-only
delivery instructions are not adopted; the final deliverable remains the PPTX.
Its mechanism and layout corrections are adopted as the visual specification.

## Scientific semantics

- The current implementation processes each view's ordinary tokens together
  with a shared bottleneck set.
- Each view produces an updated bottleneck copy.
- The three updated copies are averaged along the view dimension.
- The bottleneck set is the only cross-view channel.
- For the current K=1 configuration, do not imply that the averaged bottleneck
  is immediately written back into ordinary tokens in the same layer.
- The temporal and range paths use separate bottleneck sets; show this with a
  compact `x2: temporal / range` annotation.

## Visual structure

- Six input token columns are grouped as L/T/R pairs: temporal 90 and range 189.
- Each group receives a low-saturation view-colored hairline.
- Twelve editable curves form a wide-to-narrow-to-wide silhouette:
  six bidirectional input-to-bottleneck curves and six output-directed curves.
- Four 40 x 17 purple rounded rectangles represent `N_B = 4` bottleneck tokens.
- The bottleneck frame uses muted purple-gray, not green.
- Green remains reserved for the bottom conclusion outline.
- Distance-token columns use a lighter shade of their own view color.
- Remove the old right-side mini legend and the old purple circles.
- Add an editable view-average formula and a concise note that the three token
  streams remain separately read out.

## Visual tone

Use a mature technical-paper treatment: white background, graphite lines,
muted view colors, restrained purple-gray bottleneck accents, small radii,
thin strokes, no shadows, no gradients, no illustration icons, and no glassy or
cartoon-like components.

## Editability and QA

All new content is created as native PowerPoint text, rectangles, lines, and
arrowheads. Preserve the source master/layout hierarchy and existing notes.
Render the complete slide, inspect it at full size, and run overflow checks.

