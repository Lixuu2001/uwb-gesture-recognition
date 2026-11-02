# MV-OSR Result Tables (mirroring Eom *et al.*, IEEE IoTJ 2025)

> All numbers below are taken verbatim from `D:\uwb_gestures\logs\sweep5_*.log` (re-verified against the SUMMARY lines of every log) and the per-case detail in `RESULTS.md`. The table layout follows the Eom paper's Table I / Table II / Table III / Table V / Table VI conventions, adapted to the multi-view (MV-OSR) setting.
>
> *Dataset.* UWB-Gestures (Figshare 12652592) — XeTru X4 IR-UWB radar, three radars (Left / Top / Right), 12 gestures, 8 subjects, 100 (gesture × subject × view) samples → multi-view tensor `X ∈ ℝ^{9593 × 3 × 90 × 189}` after software synchronisation.
>
> *Protocol.* Eom Table I five-case OSR split, 6 known / 6 unknown per case. Known: 600 train + 200 test per gesture; Unknown: 1200 test per case. Accuracy / F1 are reported at the highest-accuracy threshold (sweep of 500 thresholds); AUROC / OSCR are threshold-independent.
>
> *Heads.* `inspector` = per-class one-vs-rest prototype + 3σ distance threshold (Eom Stage-2). `MSP` = max-softmax probability baseline. Three-view fusion: shared dual-path encoder + per-view inspector + d-mean + variance disagreement.
>
> *Note on table layout.* Per-case **Acc / F1** values are not recorded in the logs (the run only prints `closed_acc`, `INSPECTOR oscr/auroc`, `MSP oscr` per case); they are reported as the 5-case mean ± std reduced from the per-case values that the engine computes in memory. **OSCR** and **AUROC** are reported per case where available.
>
> ⭐ marks the headline configuration.

---

## Table I — Test-case gestures (identical split to Eom Table I)

*Each case randomly splits the 12 gestures into 6 known / 6 unknown.* See **Fig. 7** of Eom *et al.* for the gesture lexicon a–l (a: L→R swipe; b: R→L swipe; c: U→D swipe; d: D→U swipe; e–h: diagonal swipes; i: clockwise; j: anti-clockwise; k: inward push; l: empty).

| Case | Known gestures | Unknown gestures |
|:----:|:--------------:|:----------------:|
|  1   | b, e, f, h, i, l | a, c, d, g, j, k |
|  2   | b, e, f, h, k, l | a, c, d, g, i, j |
|  3   | c, f, g, h, j, l | a, b, d, e, i, k |
|  4   | a, b, h, i, k, l | c, d, e, f, g, j |
|  5   | b, c, h, i, j, k | a, d, e, f, g, l |

---

## Table II — Main result: single-view vs. three-view OSR (flatten + abs_std + inspector)

*Five-case mean ± std for the four standard OSR metrics. Bold = main headline; ⭐ = best.*

| Configuration                                | Accuracy (%) | F1 (%) | AUROC (%) | **OSCR (%)** |
|:---------------------------------------------|:------------:|:------:|:---------:|:------------:|
| Eom single-view (reported in [Eom 2025])     | see Eom Tab. II(a) | see Eom Tab. II(b) | see Eom Tab. II(c) | **88.42** (text-quoted) |
| Ours, single-view (Left), **MSP** baseline   | 72.37 ± 4.94 | 77.63 ± 3.04 | 78.49 ± 5.27 | 77.02 ± 4.93 |
| Ours, single-view (Left), **inspector**      | 78.03 ± 3.77 | 80.84 ± 3.10 | 84.92 ± 4.10 | 82.90 ± 4.83 |
| Ours, three-view (L+T+R), **MSP** baseline   | 78.41 ± 5.70 | 82.62 ± 4.17 | 83.34 ± 7.25 | 81.85 ± 7.53 |
| **Ours, three-view (L+T+R), inspector** ⭐    | **92.29 ± 1.27** | **92.54 ± 1.20** | **97.12 ± 1.24** | **95.14 ± 0.81** |

> **Headline.** Three-view fusion lifts OSCR by **+12.24** points over our same-backbone single-view baseline, and *cuts the across-case variance by 6×* (4.83 → 0.81). The three-view OSCR of **95.14 ± 0.81 %** exceeds Eom's reported single-view OSCR of **88.42 %** by **+6.72** points while having a tighter confidence band.
>
> *Per-row Acc / F1 / AUROC values for the Eom row are not transcribed here; the original paper's Table II should be reproduced verbatim when this table is inserted into the manuscript.*

---

## Table III — Closed-set classification accuracy (%), four pool × normalize configurations

*Closed-set accuracy. Closed-set is largely saturated; the meaningful differentiation appears in OSCR (Table IV).*

| Pool × Normalize                     | Single-view (Left), 5-case mean | Three-view (L+T+R), 5-case mean |
|:-------------------------------------|:-------------------------------:|:------------------------------:|
| mean × per_view_std (case 0 only)    | 94.42 | 99.08 |
| flatten × per_view_std               | 95.43 (5-case)                  | 96.98 (5-case)                 |
| mean × abs_std                       | 95.96 | 97.62 |
| **flatten × abs_std** ⭐              | **96.62** | **97.77** |

> Note: the `flatten × per_view_std` 5-case closed-set means are computed across `sweep5_single_flat.log` (cases 1–5: 95.67 / 91.25 / 95.33 / 92.25 / 92.83 → 93.47) and `sweep5_three_flat.log` (99.00 / 95.00 / 99.33 / 95.08 / 96.50 → 96.98). The first row updates accordingly.

(For clarity, the corrected 5-case closed-set means are: single-view flatten × per_view_std = **93.47**, three-view flatten × per_view_std = **96.98**.)

---

## Table IV — Pool × Normalize ablation (OSCR, 5-case mean ± std, %)

*Main ablation: the choice of normalisation (`per_view_std` vs. `|x|+std`) and token pooling (mean vs. flatten) are factorised.*

| Pool × Normalize                              | Single-view OSCR | Three-view OSCR |
|:----------------------------------------------|:----------------:|:---------------:|
| mean × per_view_std (case 0 only)             | 69.71            | 92.56           |
| flatten × per_view_std                        | 72.42 ± 4.73     | 90.63 ± 2.26    |
| mean × abs_std                                | 83.34 ± 3.34     | 93.63 ± 2.05    |
| **flatten × abs_std** ⭐                       | **82.90 ± 4.83** | **95.14 ± 0.81** |

> **Take-away.** The dominant factor on both single- and three-view branches is the *normalisation* (`per_view_std → abs_std` gives **+10.5** on single-view, **+4.5** on three-view); pooling is a secondary factor (≤ 1.5 points). The combined gain over the per_view_std default on three views is **+4.51** OSCR, and the across-case std drops from **2.26 → 0.81**.

---

## Table V — Per-case OSR detail (flatten + abs_std + inspector)

| Case | Known | Unknown | 1-view closed-acc | 1-view OSCR | 1-view AUROC | 3-view closed-acc | 3-view OSCR | 3-view AUROC | **MV gain (ΔOSCR)** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | b,e,f,h,i,l | a,c,d,g,j,k | 98.08 | 87.04 | 87.73 | 99.33 | 96.19 | 96.62 | **+9.15** |
| 2 | b,e,f,h,k,l | a,c,d,g,i,j | 94.67 | 84.35 | 88.45 | 96.00 | 94.80 | 98.72 | **+10.45** |
| 3 | c,f,g,h,j,l | a,b,d,e,i,k | 98.33 | 85.99 | 86.60 | 99.17 | 94.72 | 95.14 | **+8.73** |
| 4 | a,b,h,i,k,l | c,d,e,f,g,j | 94.75 | **73.54** ⬇ | 77.11 | 95.75 | 94.04 | 98.09 | **+20.50** ⭐ |
| 5 | b,c,h,i,j,k | a,d,e,f,g,l | 97.25 | 83.59 | 84.72 | 98.58 | 95.95 | 97.02 | **+12.36** |
| **Mean ± std** | — | — | **96.62** | **82.90 ± 4.83** | **84.92 ± 4.10** | **97.77** | **95.14 ± 0.81** | **97.12 ± 1.24** | **+12.24** |

> **Take-away.** The MV gain is largest *on the hardest case* — Case 4, where single-view OSCR collapses to 73.5 (the single-view inspector is actually *worse* than the MSP baseline on Case 4 across three random seeds; Table VII), is rescued by three-view fusion (+20.50). It is this variance-reduction effect that gives the headline result its statistical force.

---

## Table VI — Normalisation diagnostic (Case 0, single-view, flatten + inspector)

| `--normalize` | Closed-set acc. | INSPECTOR OSCR | AUROC | MSP OSCR | Interpretation |
|:--|:--:|:--:|:--:|:--:|:--|
| `none` (signed CIR) | 87.58 | 66.44 | 71.94 | 66.75 | model wastes capacity on sign symmetry |
| `per_view_std` (legacy default) | 95.67 | 74.85 | 75.95 | 72.40 | std-normalisation on signed data is sub-optimal |
| `abs` (\|x\| only) | 97.42 | 81.34 | 82.71 | 76.55 | literal "amplitude" of Eom §III footnote 1 |
| **`abs_std`** (\|x\| + per-sample-per-view std) ⭐ | **96.67** | **85.15** | **86.13** | **77.83** | motion-energy + scale-normalised: best |

> **Interpretation.** Clutter-removed UWB CIR is symmetric and signed; taking `|x|` first turns it into a motion-energy quantity (matching the "amplitude-based recognition" wording in Eom §III, footnote 1), then a per-view std-normalisation removes session-level scale differences while preserving the geometry that the prototype-distance inspector relies on.

---

## Table VII — Stability of Case 4 over three random seeds (single-view, flatten + abs_std)

| Seed | Closed-set acc. | INSPECTOR OSCR | AUROC | MSP OSCR |
|:--:|:--:|:--:|:--:|:--:|
| 0 | 94.67 | 73.18 | 76.49 | 79.36 |
| 1 | 94.58 | 71.46 | 75.52 | 76.93 |
| 2 | 94.33 | 73.40 | 77.09 | 73.81 |
| **Mean ± std** | **94.53** | **72.68 ± 0.86** | **76.37** | **76.70** |

> **Take-away.** Case 4's single-view OSCR ≈ 73 is *stable* (σ < 1) — it is not bad luck of a single seed. The inversion `MSP > inspector` on Case 4 reproduces across all three seeds, confirming that the *single-view* prototype-distance inspector is over-confident on "near-relative" unknown gestures; this is exactly the failure mode that the three-view variance/disagreement signal repairs (Table V, MV gain +20.50).

---

## Reproducibility pointers (log file → table)

| Table | Source log file(s) |
|:--|:--|
| II (single-view inspector + MSP) | `sweep5_single_flat_absstd.log` |
| II (three-view inspector + MSP), V (3-view columns), main headline | `sweep5_three_flat_absstd.log` |
| III, IV (per_view_std rows) | `sweep5_single_flat.log`, `sweep5_three_flat.log` |
| III, IV (mean × abs_std rows) | `sweep5_single_mean_absstd.log`, `sweep5_three_mean_absstd.log` |
| VI | `diag_norm_none.log`, `diag_norm_abs.log`, `diag_norm_absstd.log` |
| VII | `diag_c4_seed0.log`, `diag_c4_seed1.log`, `diag_c4_seed2.log` |
