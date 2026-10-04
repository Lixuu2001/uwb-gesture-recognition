"""Descriptive source-loss history only; does not tune or fit a model."""
from pathlib import Path
import json,statistics,hashlib,datetime,argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();R=a.runs;O=a.out;O.mkdir(parents=True,exist_ok=True);q=json.loads((R/'queue_status.json').read_text());assert q['state']=='complete'
c=json.loads((R/'protocol.json').read_text());rows=[]
for h in c['users']:
 for arm in c['arms']:
  p=R/arm/f'held{h}'/'result.json';r=json.loads(p.read_text());epochs=r['history'][1:];assert len(epochs)==24
  row=dict(held=h,arm=arm,result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_final_ce=epochs[-1]['source_ce'],scale=epochs[-1]['scale'],seconds=sum(e['seconds'] for e in epochs))
  for label,start,end in [('early',2,6),('tail',13,24)]:
   es=[e for e in epochs if start<=e['epoch']<=end]
   for penalty in ['real_penalty','permuted_penalty']:
    row[label+'_'+penalty+'_scaled_to_ce_median']=statistics.median(e['scale']*e[penalty]/e['source_ce'] for e in es)
  rows.append(row)
out=dict(scope='descriptive source objective histories, not causal effect or gradient norm estimates',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,read_diagnostic_labels=False,read_diagnostic_accuracy=False,configuration_changed=False,notes=['Ratios use source loss epoch means, not per-batch ratio means.','Both EMA penalties are maintained even for ERM; counterfactual ratios there are descriptive.','A small penalty/CE value is not proof of small parameter-gradient effect.'])
destination=O/'SOURCE_HISTORY.json'
if destination.exists(): raise FileExistsError('Preserve existing source-history summary; use a fresh output directory')
destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':len(rows),'new_training_epoch_hours':sum(r['seconds'] for r in rows)/3600}))
