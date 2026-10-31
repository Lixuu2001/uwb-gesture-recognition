"""Only score after all five new folds' dual predictions/full audits freeze."""
from pathlib import Path
import json,sys,time
from readiness import read,sha,ready
B=Path(__file__).resolve().parent
def gate(rows,cfg):
 import numpy as np
 out={}
 for key in ['localcnn','gap']:
  diff=[r['hinception']-r[key] for r in rows];mean=float(np.mean(diff));wins=sum(x>0 for x in diff);worst=min(diff);g=cfg['gate'];out[key]=dict(mean_difference=mean,wins=wins,worst_difference=worst,pass_gate=bool(mean+g['tolerance']>=g['mean_gain_over_each_frozen_pipeline_minimum'] and wins>=g['strict_wins_against_each_minimum'] and worst+g['tolerance']>=g['worst_user_difference_against_each_minimum']))
 return out,all(x['pass_gate'] for x in out.values())
def main():
 cfg=read(B/'PROTOCOL.json');ready(cfg);frozen={}
 # No NumPy/label/prediction load occurs until every completion is checked.
 for held in cfg['new_held_users']:
  fold=B/'run01'/('held'+str(held));a=read(fold/'COMPLETE_AUDIT.json');assert a['state']=='pass' and a['diagnostic_labels_read'] is False
  for p,d in a['artifact_hashes'].items():assert sha(p)==d,p
  for role,d in read(fold/'ALL_PREDICTIONS_FROZEN.json').items():assert sha(fold/(role+'_predictions_unscored.npz'))==d
  assert read(fold/'STATE.json')['diagnostic_labels_read'] is False
  for kind in ['train','audit']:
   t=read(B/'jobs'/('held'+str(held))/kind/'TERMINATION.json');assert t['exit_code']==0 and not t['wall_timeout']
  frozen[str(held)]={role:sha(fold/(role+'_predictions_unscored.npz')) for role in ['validation','diagnostic']}
 assert not (B/'run01/SUMMARY.json').exists();(B/'run01/ALL_NEW_PREDICTIONS_FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
 import numpy as np
 labels=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');local={r['held']:r['local'] for r in read(cfg['localcnn_summary'])['rows']};gap={r['held']:r['gap'] for r in read(cfg['gap_summary'])['rows']};rows=[]
 for held in range(1,7):
  fold=Path(cfg['reused_held1']['directory']) if held==1 else B/'run01'/('held'+str(held));pred=fold/'diagnostic_predictions_unscored.npz'
  with np.load(pred) as z:ids=z['ids'];decision=z['predictions'];assert np.array_equal(decision,z['probabilities'].argmax(1))
  for key in ['localcnn','gap']:
   ref=cfg['comparator_predictions'][key][str(held)]
   with np.load(ref['path']) as z:assert set(z['ids'].tolist())==set(ids.tolist())
  y=np.array(labels[ids],copy=True);score=float(np.mean([np.mean(decision[y==c]==c) for c in np.unique(y)]));rows.append(dict(held=held,hinception=score,localcnn=local[held],gap=gap[held]))
 checks,passed=gate(rows,cfg);result=dict(state='complete_exploratory_source_sixfold_confirmation',rows=rows,means={key:float(np.mean([r[key] for r in rows])) for key in ['hinception','localcnn','gap']},comparisons=checks,predeclared_gate_passed=passed,route='separately_registered_matched_ablation_and_eventual_locked_evaluation' if passed else 'close_exact_CPU100_source_confirmation_no_rescue',all_five_predictions_frozen_before_new_diagnostic_scoring=True,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,complete_private_modelset_verified=False)
 (B/'run01/SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(state=result['state'],gate_passed=passed)))
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
