# Fixed class-conditional OAS-QDA source screen failed

Science preregistration `eb02e333e788cf982332e19d950400fc0f186fb3`; protocol SHA256 `5459673b08e1a9eed86a50969be47d62aec78ec40d37183bd87ad224470464b9`. A verification-only correction `8482674` accounts for the existing semantic-code baseline's common−.5 score offset; it changes no predictions, fitting or gate. The first attempt stopped before reporting a diagnostic metric; its logs and held3 artifacts are retained. Reproduction completed in a fresh `source-v2` directory. All held3 predictions before/after are bitwise identical.

| Arm | Mean source-user macro accuracy |
|---|---:|
| Original selected onehot ridge |93.063131%|
| Class OAS-QDA (primary) |92.424242%|
| Shared mean class covariance |92.998737%|
| Diagonal class covariance |92.275253%|
| Permuted class covariance |71.165404%|

QDA loses to ridge by0.638889pp (2/6 wins, worst−5pp), loses to pooled by0.574495pp (2/6 wins), and gains only0.148990pp over diagonal (required+.25pp). Only covariance-permutation comparison passes. Joint gate fails. Close this fixed estimator/jitter/normalization/permutation/seed configuration; no eightfold expansion or post-hoc control promotion. Failure does not rule out all generative classifiers. Large loss from covariance permutation only shows this particular parameter association matters; it is not evidence of improvement over a strong discriminator.

Users3–8, four train/next validation/held source diagnostic, same quality exclusions and frozen128dim source epoch24 averaged CNN features as earlier studies. Gaussian parameters use training IDs only; original training mean/scale is reused. Within each class, available users have equal sample counts (verified in all six folds), so plain class averages also equally weight its present users. Class3 can be absent for user6. Class priors are fixed uniform12, not estimated from diagnostic labels. No Gaussian hyperparameter selection uses validation or held users. All validation/held scores are saved before their evaluation labels are indexed for metrics. Global label array is loaded for training indexing, not sealed from program memory; the scientific claim is no target-label fitting/selection, not technical label isolation.

Per-class OAS shrinks ML covariance toward tr(S)/128 I with sklearn1.8's approximation that omits the original2/p term. Fixed1e−8 identity jitter stabilizes numerical inversion. Full Gaussian score includes log determinant. Controls retain class means: shared averages12 covariance matrices; diagonal preserves each class variance; permutation uses fixed seed20261007 to assign covariances to different classes. Pooled removes both correlation and scale differences, so its contrast is not a pure correlation test. This across-source-clip covariance differs from the previously failed within-clip spatial covariance descriptor.

These are classical QDA/LDA/shrinkage methods, not an original contribution. Official sklearn1.8 QDA/LDA math, covariance assumptions/OAS sections and `_oas` implementation were reviewed and pinned by blob SHA in `research/CLASS_SCATTER_REVIEW.md`. Chen etal2010 IEEE TSP is cited by those sources; original paper full text has not been read. Gaussian iid assumptions are not established for correlated human gesture repetitions or user shift, so no theoretical guarantee of target accuracy is claimed. Original project OSR proposal already discusses prototype/shrunk-covariance distances; novelty is not inferred from adapting them to closed-set classification.

Synthetic-only CPU cost check estimated14.746594s with margins; actual six-fold fitting, scoring and numerical verification5.989304s.72 class covariance fits were independently reconstructed analytically;576 class Gaussian score systems were checked against SciPy logpdf (maximum difference4.83124e−9, all argmax equal). Post-run audit verified45 artifact SHA, exact source roles/ID/truth alignment, per-user counts, means/gate and failed-attempt prediction equality. `PER_USER.csv`, `SUMMARY.json`, `AUDIT.json` and pinned configs provide the numerical record. No encoder training or background queue was started.

All source models, class scatter parameters, scores, IDs, evaluation arrays, error/recovery logs and literature snapshots go to private Drive; no large arrays in Git. Prior energy-area/semantic backup receipts preserve reused encoder/hidden/ridge recovery chains; original migration preserves raw data. Final receipt is `DRIVE_BACKUP_RECEIPT.json`. The archived report precedes final backup verification; latest Git copy/receipt is authoritative. Eightfold strong ridge94.897727% remains a separate protocol and cannot be compared directly to these six-source means. User goal remains unmet; automatic continuation remains enabled.

Final private backup:7 bundles/7 parts, 33,511,688 compressed bytes, 84 members. Owner-only metadata, authenticated raw redownload and all part/bundle/member hashes verified. No sharing changes.
