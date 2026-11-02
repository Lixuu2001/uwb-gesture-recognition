# GAP source-sixfold confirmation — final independently audited result,2026-10-30

Fixed known GAP baseline mean 93.667172% versus prior localCNN 93.594444%, paired mean +0.072727pp, 4/6 strict wins. The original descriptive gate fails: mean>=1pp,atleast4wins,worst>=-2pp. Close this exact64epoch GAP configuration; no seed/epoch/LR/normalization rescue grid or post-score replacement of controls. This is exploratory accuracy evidence, not original-contribution proof or independent validation.

|Held source user|GAP64epoch macro recall|LocalCNN24epoch macro recall|Difference|Selected GAP epoch|
|---|---:|---:|---:|---:|
|1|98.416667%|96.833333%|+1.583333pp|60|
|2|98.700000%|96.400000%|+2.300000pp|43|
|3|90.000000%|89.583333%|+0.416667pp|14|
|4|94.250000%|94.416667%|-0.166667pp|26|
|5|96.000000%|95.333333%|+0.666667pp|46|
|6|84.636364%|89.000000%|-4.363636pp|38|

Reused held1 from the audited TimeMIL/GAP pilot; onlyheld2–6 are new=5trajectories/320epochs. Source users1–6,heldH/validationH%6+1/otherfourtrain; all6900retainedsource records, originalqualitymask/exactIDs and seed20261029. GAP is the unchanged617612parameter author Inception feature extractor plus global average pooling,all189range bins/3views/90frames,per-record/view RMS then90x567 reshape. AdamWlr.001/weightdecay.0001+authorLookahead(.5,k6,lazy first slow state),batch64,clip2,unweighted12class OvRBCE,noaugmentation/scheduler. Each best checkpoint uses validationmacro/lowerBCE/earlierepoch. Allfive new diagnostic prediction files froze before aggregate scoring. No users7–8labels used.

Comparator uses previously independently audited localCNN24epoch resized-map pipelines with exactly matched source-user roles/qualitymask/IDs. This is fixed-pipeline accuracy comparison; architecture/inputresolution/64versus24epochs differ, so it is not a compute-matched or architecture-matched ablation. Legacy8fold93.4417% uses a different protocol/sixtrainingusers and cannot be presented as the reference score for an improvement claim here. Eight users have historical repeated development; even source-sixfold confirmation is exploratory and the reused positive held1pilot motivated expansion.

Independent completionaudit passes 97 artifact/input/code hashes, all6900NumPy source-preprocessing records, exactroles/counts/private IDs, authorinitial/first/final model/optimizer/RNG/BN states, validation-only selection, all6900selected-validation forwards and all6900diagnostic logits with exactargmax, BCE/macros/pairedmeans/wins/originalgate. Fixedpreprocess tolerances2e-5/2e-6,logits/BCE1e-4/1e-5,macro1e-12. CNN referencehashes/metrics and links to prior independent full-forward audit checked; CNN forwards not repeated. Historical nonselected GAP epochs not retrained/functionallyreplayed. AuditorSHA a6a77a8bdbfce9ca89bbf090331db7a65732d34f31ff72af57870f8223ea88b0.

Training supervisor exited0/no timeout: 12418.808975wall/12417.226442CPU seconds, below43200both, covering five new folds only. Independent one-thread audit exited0/no timeout: 45.581040wall/45.522751CPU seconds, below890wall/880soft890hardCPU. All model/optimizer/RNG/privateIDs/logs/predictions require final private persistence; final backup receipt records completion separately, without placing raw arrays/tensors/sample-ID values in Git.

Published source: Chenetal.ICML2024 TimeMIL PMLR235:7190–7206 https://proceedings.mlr.press/v235/chen24af.html,author7542c1c275aae3a9db35ee5c9bdfc0ccf386cd5c. This GAP control is a known temporal baseline, not a novel architecture or exact original InceptionTime ensemble/paper reproduction. Earlier TimeMIL64epoch pooling mechanism failed its frozen+.5pp gate despite positive held1accuracy and remains closed; no rescue or reinterpretation.

Overall original-contribution goal remains unresolved; continuation remains enabled. Next candidate TSLANet has full UEA core and author-default loss/stage operations verified synthetically, but still needs a single prospectively frozen source-only real protocol, complete runner and cost gate before any fit. Published prior art does not block useful accuracy transfer experiments. No users7–8selection, test-label tuning, additional seed or post-score threshold change is authorized by this report.
