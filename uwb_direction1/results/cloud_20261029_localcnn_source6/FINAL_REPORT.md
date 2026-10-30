# Local CNN source-sixfold confirmation: completed and independently verified

The unchanged local ConvMixer CNN achieves **93.5944%** average held-user present-class macro recall versus **78.5336%** for the fixed default-scale MiniRocket control: **+15.0609 percentage points**, 6/6 positive users. The frozen descriptive continuation gate passes: mean gain>=1pp, >=4/6 positive users and no user worse than -2pp. This establishes a useful stronger baseline against this comparator. It does not establish superiority to every strong method, an improvement over the legacy eightfold result, independent validation or originality.

| Held source user | Local CNN (%) | Fixed MiniRocket (%) | Difference (pp) |
|---|---:|---:|---:|
| 1 | 96.8333 | 85.3333 | 11.5000 |
| 2 | 96.4000 | 84.8000 | 11.6000 |
| 3 | 89.5833 | 80.3333 | 9.2500 |
| 4 | 94.4167 | 78.3333 | 16.0833 |
| 5 | 95.3333 | 83.5833 | 11.7500 |
| 6 | 89.0000 | 58.8182 | 30.1818 |

## Frozen protocol and scope

Only users1–6,6900 quality-retained records; held h, validation h%6+1, four remaining training users. Each arm's macro recall averages only classes present in that held user; the six user scores have equal weight. Class exclusions are unchanged. All three signed clutter-removed views are resized64x128 then normalized by each record/view RMS, without fitted cross-record preprocessing. Local ConvMixer width256/depth4,350220parameters; AdamW lr0.001/decay0.0005, unweighted CE, batch32,24epochs,seed20261029,3CPUthreads, no augmentation/scheduler/smoothing. Selection uses validation macro, then lower CE, then earlier epoch.

Held1 is reused from the previously completed eligible pilot, selectedepoch23, rather than retrained. Exactly five new trajectories/120epochs were run; selected epochs for held1..6 are23,23,11,24,21,23. All five new unscored prediction sets froze before aggregate diagnostic scoring. MiniRocket uses the existing fixed9996-feature author implementation and original balanced LSQR readout; no reference refit, feature/alpha/seed search or outcome-based solver change. This comparator is the original LSQR version, not the direct-solve readout used in the separate MultiRocket paired pilot.

Registered commit1b8ffb10393682478718f8dee15f3c8c94528084; protocol SHA256b3633e607fc27365e6e7ab487ac6d0161b12cd112124bdf04a203236ca53d1df. The12 original frozen hashes remain unchanged. This is a known paper-informed local CNN baseline, not an official exact author reproduction or an original architecture.

## Verification and actual resources

The independent completed-run audit passes:69 artifact/input/code hashes; all exact roles and IDs; original initializations and first-update IDs; optimizer step counts/moments/reload shape compatibility; RNG/BN/finite states; validation-only epoch selection; all6900 validation and6900 diagnostic predictions functionally replayed; fixed MiniRocket IDs/labels/scores; all reported metrics and the original gate independently checked. Logit tolerance was fixed before outcomes at rtol1e-4/atol1e-5 with exact argmax equality. Nonselected historical epoch tensors were not retrained or functionally replayed. Auditor SHA256ab546a97ea56fe8155d1b48fed7baa1a77c6e819c69a2e506105a739a362acdd.

The training supervisor exited0 without timeout:33125.705182wall seconds (9.2016h),97709.758205CPU seconds (27.1416CPUh), below43200wall/129600CPU seconds. These costs cover only the five new trajectories, excluding the earlier reused held1 cost. The independent single-thread audit exited0 without timeout in529.831083wall/529.755558CPU seconds under its separate1790wall/1780soft1790hardCPU bounds. Original worker43382 exited; supervisor43381 is a zombie, not an active training job.

Evidence: SUMMARY.json, COMPLETE_AUDIT.json, TRAIN_TERMINATION.json, AUDIT_TERMINATION.json, ALL_NEW_PREDICTIONS_FROZEN.json and per-fold completion/log metadata. Model/optimizer/RNG/split-ID/prediction tensors remain outside Git.

## Interpretation and next authorized step

The strongest support is six consistent gains, including the difficult held6 (+30.1818pp); the smallest gain is held3 (+9.25pp). The limitation is equally important: this proves a descriptive advantage over MiniRocket in this source-only protocol. The old93.4417% uses eight users with six training users/one validation/one held user, so its number is not a matched comparator here. The prior held1 pilot helped select this baseline for expansion, and all eight users have substantial historical development. These are exploratory results, not new independent validation. Users7–8 labels were not used.

Keep the verified local CNN as a strong baseline. The exact24epoch full-fusion and fixed full-scale MultiRocket configs remain closed by their earlier gates; this result does not reopen them or prove their full paper schedules fail. The already prospectively registered TimeMIL64epoch full-model/GAP pilot is the next published-method accuracy experiment, with its original source-only split, seed, configuration, budget and gates. Its source-held1 CNN reference remains the unchanged prior96.8333% local/96% full pilot, not this different sixfold mean. TSLANet remains a later synthetic prototype, without a real-data protocol or fitting.

Complete private final preservation before launching TimeMIL, then verify its first actual update and report the true process/receipt. Do not claim a launch from preparation or a background watcher. No7–8-label selection, test tuning, rescue grids or originality claim is authorized by this result. The overall research-contribution goal remains unmet and continuation remains enabled.
