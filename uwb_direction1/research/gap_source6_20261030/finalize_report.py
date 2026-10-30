"""Render existing independently audited result metadata; no array/model access."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def main():
 run=BASE/'run01';state=json.loads((run/'STATE.json').read_text());assert state['state']=='complete' and state['pid'] is None
 audit=json.loads((BASE/'COMPLETE_AUDIT.json').read_text());assert audit['state']=='pass'
 train=json.loads((BASE/'supervisor01/TERMINATION.json').read_text());verify=json.loads((BASE/'audit_supervisor01/TERMINATION.json').read_text())
 for term in (train,verify):assert term['exit_code']==0 and not term['wall_timeout']
 summary=json.loads((run/'SUMMARY.json').read_text());assert summary['descriptive_continuation_gate']==audit['descriptive_continuation_gate']
 rows=[]
 for row in summary['rows']:
  detail=next(x for x in audit['folds'] if x['held']==row['held'])
  rows.append(f"|{row['held']}|{row['gap']*100:.6f}%|{row['localcnn']*100:.6f}%|{row['difference']*100:+.6f}pp|{detail['selected_epoch']}|")
 passed=summary['descriptive_continuation_gate'];outcome='passes' if passed else 'fails'
 stop='This positive descriptive result licenses separately frozen future work only.' if passed else 'Close this exact64epoch GAP configuration; no seed/epoch/LR/normalization rescue grid or post-score replacement of controls.'
 text=f'''# GAP source-sixfold confirmation — final independently audited result,2026-10-30

Fixed known GAP baseline mean {summary['means']['gap']*100:.6f}% versus prior localCNN {summary['means']['localcnn']*100:.6f}%, paired mean {summary['mean_difference']*100:+.6f}pp, {summary['wins']}/6 strict wins. The original descriptive gate {outcome}: mean>=1pp,atleast4wins,worst>=-2pp. {stop} This is exploratory accuracy evidence, not original-contribution proof or independent validation.

|Held source user|GAP64epoch macro recall|LocalCNN24epoch macro recall|Difference|Selected GAP epoch|
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Reused held1 from the audited TimeMIL/GAP pilot; onlyheld2–6 are new=5trajectories/320epochs. Source users1–6,heldH/validationH%6+1/otherfourtrain; all6900retainedsource records, originalqualitymask/exactIDs and seed20261029. GAP is the unchanged617612parameter author Inception feature extractor plus global average pooling,all189range bins/3views/90frames,per-record/view RMS then90x567 reshape. AdamWlr.001/weightdecay.0001+authorLookahead(.5,k6,lazy first slow state),batch64,clip2,unweighted12class OvRBCE,noaugmentation/scheduler. Each best checkpoint uses validationmacro/lowerBCE/earlierepoch. Allfive new diagnostic prediction files froze before aggregate scoring. No users7–8labels used.

Comparator uses previously independently audited localCNN24epoch resized-map pipelines with exactly matched source-user roles/qualitymask/IDs. This is fixed-pipeline accuracy comparison; architecture/inputresolution/64versus24epochs differ, so it is not a compute-matched or architecture-matched ablation. Legacy8fold93.4417% uses a different protocol/sixtrainingusers and cannot be presented as the reference score for an improvement claim here. Eight users have historical repeated development; even source-sixfold confirmation is exploratory and the reused positive held1pilot motivated expansion.

Independent completionaudit passes {len(audit['artifact_hashes'])} artifact/input/code hashes, all6900NumPy source-preprocessing records, exactroles/counts/private IDs, authorinitial/first/final model/optimizer/RNG/BN states, validation-only selection, all6900selected-validation forwards and all6900diagnostic logits with exactargmax, BCE/macros/pairedmeans/wins/originalgate. Fixedpreprocess tolerances2e-5/2e-6,logits/BCE1e-4/1e-5,macro1e-12. CNN referencehashes/metrics and links to prior independent full-forward audit checked; CNN forwards not repeated. Historical nonselected GAP epochs not retrained/functionallyreplayed. AuditorSHA {audit['auditor_sha256']}.

Training supervisor exited0/no timeout: {train['supervisor_wall_seconds']:.6f}wall/{train['child_cpu_seconds']:.6f}CPU seconds, below43200both, covering five new folds only. Independent one-thread audit exited0/no timeout: {verify['supervisor_wall_seconds']:.6f}wall/{verify['child_cpu_seconds']:.6f}CPU seconds, below890wall/880soft890hardCPU. All model/optimizer/RNG/privateIDs/logs/predictions require final private persistence; final backup receipt records completion separately, without placing raw arrays/tensors/sample-ID values in Git.

Published source: Chenetal.ICML2024 TimeMIL PMLR235:7190–7206 https://proceedings.mlr.press/v235/chen24af.html,author7542c1c275aae3a9db35ee5c9bdfc0ccf386cd5c. This GAP control is a known temporal baseline, not a novel architecture or exact original InceptionTime ensemble/paper reproduction. Earlier TimeMIL64epoch pooling mechanism failed its frozen+.5pp gate despite positive held1accuracy and remains closed; no rescue or reinterpretation.

Overall original-contribution goal remains unresolved; continuation remains enabled. Next candidate TSLANet has full UEA core and author-default loss/stage operations verified synthetically, but still needs a single prospectively frozen source-only real protocol, complete runner and cost gate before any fit. Published prior art does not block useful accuracy transfer experiments. No users7–8selection, test-label tuning, additional seed or post-score threshold change is authorized by this report.
'''
 p=BASE/'FINAL_REPORT.md';assert not p.exists();p.write_text(text);print(json.dumps(dict(report=str(p),gate_passed=passed,means=summary['means'],mean_difference=summary['mean_difference'])))
if __name__=='__main__':main()
