# MV-OSR figures — Python source

All six manuscript figures here are drawn from scratch in matplotlib.
This README maps each `.py` file to the figure it produces.

> Environment: `D:/D_conda/envs/pytorch/python.exe`
> (matplotlib 3.9.2, numpy 2.0.2). All scripts share `_style.py` for
> typography + palette so the figures read as one cohesive set.

## How to rebuild every figure

```bat
%PY% extract_case1_predictions.py --single   :: only the first time, ~28 min
%PY% extract_case1_predictions.py            :: only the first time, ~32 min
%PY% make_all.py                             :: renders fig1..fig6 + source-data CSVs
```

If you only want one figure, run that one script directly — they are
self-contained except for `_style.py`.

## File map

| Script | Produces | Story |
|---|---|---|
| [`_style.py`](_style.py) | (shared) | NMI-pastel palette + Arial/7pt rcParams + 300 dpi PNG / vector PDF / editable SVG export helper. Imported by every other script. |
| [`fig1_per_case_oscr.py`](fig1_per_case_oscr.py) | [`fig1_per_case_oscr.{png,pdf,svg}`](fig1_per_case_oscr.png) | **Hero.** Per-case OSCR for single-view vs three-view across Eom's 5 test cases; ΔOSCR multi-view gain annotated above each pair; Eom 2025 baseline (88.42) overlaid as dashed reference; 5-case mean±std bands shown as horizontal stripes. |
| [`fig2_ablation_heatmap.py`](fig2_ablation_heatmap.py) | [`fig2_ablation_heatmap.{png,pdf,svg}`](fig2_ablation_heatmap.png) | 2 × 2 ablation heatmap (pool ∈ {mean, flatten} × normalize ∈ {per_view_std, abs_std}), one panel per branch. Shared color scale; shows that normalization is the dominant factor (~10pt on single-view, ~4.5pt on three-view) and pooling is secondary. |
| [`fig3_normalisation_diag.py`](fig3_normalisation_diag.py) | [`fig3_normalisation_diag.{png,pdf,svg}`](fig3_normalisation_diag.png) | Four-mode normalization diagnostic on Case 0 single-view, 3 metrics per mode (INSPECTOR OSCR / AUROC / MSP OSCR). The abs_std bar group is outlined as "★ best". |
| [`fig4_gap_and_seed.py`](fig4_gap_and_seed.py) | [`fig4_gap_and_seed.{png,pdf,svg}`](fig4_gap_and_seed.png) | Two-panel: (a) closed-set vs OSCR gap for the four pool×normalize configs on three views; (b) Case-4 seed stability (1-view) showing MSP > INSPECTOR reproduces across 3 seeds — the failure mode that three-view fixes. |
| [`fig5_table_ii.py`](fig5_table_ii.py) | [`fig5_table_ii.{png,pdf,svg}`](fig5_table_ii.png) | **Eom *et al.* IEEE IoTJ 2025 Table II look-alike.** Four stacked sub-tables (a) Accuracy / (b) F1 / (c) AUROC / (d) OSCR. Rows = schemes (Proposed 3-view inspector, 3-view MSP, 1-view inspector, 1-view MSP, Eom 2025); columns = case 1–5 + Mean; "Proposed" row highlighted in teal. |
| [`fig6_confmat.py`](fig6_confmat.py) | [`fig6_confmat.{png,pdf,svg}`](fig6_confmat.png) | **Eom *et al.* Fig 9 + Fig 10 look-alike.** 2 × 2 confusion-matrix grid on Case 1 (known b/e/f/h/i/l + UG): top row = best-accuracy threshold (cf. Eom Fig 9), bottom row = 3-σ heuristic threshold (cf. Eom Fig 10); left column = single-view, right column = three-view. Reads `results/case1_{1view,3view}.npz`. |

## Pipeline scripts (not figures themselves)

| Script | Role |
|---|---|
| [`extract_case1_predictions.py`](extract_case1_predictions.py) | Retrains Case 1 with the project's standard pipeline (`uwb_osr.engine` + `uwb_osr.inspector`) and dumps `y_known / pred_known / conf_known / msp_known / y_unknown / pred_unknown / conf_unknown / thr_best / acc / f1 / auroc / oscr` into `D:/uwb_gestures/results/case1_{1view,3view}.npz`. `fig6_confmat.py` reads these. |
| [`source_data.py`](source_data.py) | Writes `source_data_fig1.csv` … `source_data_fig4b.csv`. The same numbers that appear on fig1–fig4. (Nature submission style: one CSV per figure for reviewer audit.) |
| [`make_all.py`](make_all.py) | One-shot driver: runs fig1 → fig6 + source-data export. Skips fig6 with a friendly message if the npz dumps are missing. |

## Companion text artifacts (not figures)

| File | Content |
|---|---|
| [`tables.md`](tables.md) | All result tables in Markdown (Tables I–VII), with verified per-case numbers and reproducibility pointers to the source log files. |
| [`tables.tex`](tables.tex) | LaTeX (IEEEtran + booktabs + multirow) versions of the same tables, ready to paste into a manuscript. |

## Style decisions (in case you want to re-skin)

* Width tokens (from `_style.py`): `W1 = 89 mm` (single-column), `W15 = 120 mm` (1.5-column), `W2 = 183 mm` (double-column).
* Palette tokens: `single` (neutral gray-blue), `three` (teal accent, headline), `eom` (salmon, Eom reference line), `gain` (muted green, only for ΔOSCR), `msp` (muted ochre).
* All text is editable in the SVG and PDF (`svg.fonttype="none"`, `pdf.fonttype=42`).
* No top / right spines on axes; 0.8 pt line weight; 7 pt body text.
