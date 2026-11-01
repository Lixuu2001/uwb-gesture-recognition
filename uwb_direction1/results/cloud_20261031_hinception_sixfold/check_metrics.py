"""Post-preservation independent scoring check; no training or selection."""
from pathlib import Path
import hashlib,json,math
B=Path('/workspace/uwb-hinception-sixfold-20261031')
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 # Before NumPy, any prediction decode, or label mapping.
 proof=read(HERE/'READBACK_VERIFIED.json');receipt=read(HERE/'DRIVE_BACKUP_RECEIPT.json');manifest=read(HERE/'MANIFEST.json')
 assert proof['complete_modelset_verified'] and receipt['complete_modelset_verified']
 assert proof['archive_sha256']==receipt['archive_sha256']==manifest['archive_sha256']
 assert proof['chunks_verified']==41 and proof['archive_members']==478
 state=read(B/'STATE.json');assert state['pid'] is None and not state['active'] and not state['failed'] and sorted(state['completed'])==[2,3,4,5,6]
 cfg=read(B/'PROTOCOL.json');global_freeze=read(B/'run01/ALL_NEW_PREDICTIONS_FROZEN.json')
 assert set(global_freeze)=={'2','3','4','5','6'}
 for h in range(2,7):
  a=read(B/f'run01/held{h}/COMPLETE_AUDIT.json');assert a['state']=='pass' and not a['diagnostic_labels_read'] and not a['users7_8_labels_used']
  assert a['training_updates_replayed_exact']==100*math.ceil(cfg['counts'][str(h)][0]/64)
  for role,digest in global_freeze[str(h)].items():assert sha(B/f'run01/held{h}/{role}_predictions_unscored.npz')==digest
  for kind in ['train','audit']:
   t=read(B/f'jobs/held{h}/{kind}/TERMINATION.json');assert t['exit_code']==0 and not t['wall_timeout']
 assert sha(Path(cfg['source_manifest']))==cfg['source_manifest_sha256']
 summary=read(B/'run01/SUMMARY.json')
 assert summary['all_five_predictions_frozen_before_new_diagnostic_scoring'] and not summary['users7_8_labels_used'] and not summary['independent_validation']
 import numpy as np
 with np.load(cfg['source_manifest']) as z:source_ids=z['ids'];users=z['users']
 assert len(source_ids)==len(set(source_ids.tolist()))==6900 and set(users.tolist())==set(range(1,7))
 labels=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r')
 rows=[];total_correct={k:0 for k in ['hinception','localcnn','gap']};allids=[]
 for h in range(1,7):
  expected=set(source_ids[users==h].tolist());assert len(expected)==cfg['counts'][str(h)][2]
  row={'held':h,'samples':len(expected)}
  for method in ['hinception','localcnn','gap']:
   if method=='hinception':
    folder=Path(cfg['reused_held1']['directory']) if h==1 else B/f'run01/held{h}'
    path=folder/'diagnostic_predictions_unscored.npz';digest=cfg['reused_held1']['prediction_sha256'] if h==1 else global_freeze[str(h)]['diagnostic']
   else:
    ref=cfg['comparator_predictions'][method][str(h)];path=Path(ref['path']);digest=ref['sha256']
   assert sha(path)==digest
   with np.load(path) as z:
    ids=z['ids'];assert len(ids)==len(set(ids.tolist())) and set(ids.tolist())==expected
    if method=='hinception':
     p=z['probabilities'];pred=z['predictions'];assert np.array_equal(pred,p.argmax(axis=1))
    else:p=z['logits'];pred=p.argmax(axis=1)
    assert np.isfinite(p).all() and p.shape==(len(ids),12)
   # Only IDs already verified as belonging to this source user1..6 are indexed.
   y=np.asarray(labels[ids]);classes=np.unique(y);recalls=[float(np.count_nonzero(pred[y==c]==c))/int(np.count_nonzero(y==c)) for c in classes]
   balanced=math.fsum(recalls)/len(classes);correct=int(np.count_nonzero(pred==y))
   row[method]={'balanced_accuracy':balanced,'top1_accuracy':correct/len(ids),'correct':correct,'present_classes':len(classes),'prediction_sha256':digest}
   assert abs(balanced-summary['rows'][h-1][method])<=1e-12
   total_correct[method]+=correct
  rows.append(row);allids.extend(expected)
 assert len(allids)==len(set(allids))==6900
 means={k:math.fsum(r[k]['balanced_accuracy'] for r in rows)/6 for k in total_correct}
 for k,v in means.items():assert abs(v-summary['means'][k])<=1e-12
 checks={}
 for method in ['localcnn','gap']:
  diffs=[r['hinception']['balanced_accuracy']-r[method]['balanced_accuracy'] for r in rows]
  g=cfg['gate'];mean=math.fsum(diffs)/6;wins=sum(x>0 for x in diffs);worst=min(diffs)
  passed=mean+g['tolerance']>=g['mean_gain_over_each_frozen_pipeline_minimum'] and wins>=g['strict_wins_against_each_minimum'] and worst+g['tolerance']>=g['worst_user_difference_against_each_minimum']
  s=summary['comparisons'][method];assert passed==s['pass_gate'] and wins==s['wins'] and abs(mean-s['mean_difference'])<=1e-12 and abs(worst-s['worst_difference'])<=1e-12
  checks[method]={'mean_difference':mean,'wins':wins,'worst_difference':worst,'pass_gate':passed}
 assert all(x['pass_gate'] for x in checks.values())==summary['predeclared_gate_passed']
 result={'state':'pass_independent_post_preservation_metric_and_gate_crosscheck','source_rows':rows,'equal_user_mean_balanced_accuracy':means,'pooled_top1_accuracy':{k:n/6900 for k,n in total_correct.items()},'comparisons':checks,'predeclared_gate_passed':summary['predeclared_gate_passed'],'label_IDs_indexed_count':6900,'label_user_IDs_indexed':[1,2,3,4,5,6],'users7_8_labels_indexed':False,'original_summary_sha256':sha(B/'run01/SUMMARY.json'),'complete_private_receipt_sha256':sha(HERE/'DRIVE_BACKUP_RECEIPT.json'),'no_refitting_or_configuration_selection':True,'independent_validation':False,'novel_contribution_validated':False}
 (HERE/'METRIC_CROSSCHECK.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
