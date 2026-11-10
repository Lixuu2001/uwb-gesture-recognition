"""Post-training full8 gates and aggregation; no gesture-label I/O here."""
from pathlib import Path
import sys,json,hashlib,time,resource,shutil,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
ARMS=('native','tail','native_sourceBN')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def all_private():
 sys.path.insert(0,str(ROOT))
 from phase_supervisor import verify_previous_private
 ledger=read(ROOT/'PHASE_LEDGER.json')
 assert ledger['completed']==list(range(1,9)) and ledger['state']=='all8_audited_unscored_waiting_private_backup'
 for h in range(1,9):assert (ROOT/f'folds/held{h}/PRIVATE_FOLD_RELEASE.json').is_file()
 verify_previous_private(9) # Verify eight prior folds; this never launches a held9.
 for n,d in read(ROOT/'FROZEN_HASHES.json').items():assert sha(ROOT/n)==d,n
 return ledger
def resource_check(extra_CPU=0.):
 c=read(ROOT/'PROTOCOL.json');l=read(ROOT/'PHASE_LEDGER.json')
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==l['boot_id']
 wall=time.monotonic()-l['start_monotonic']
 external=sum(x['CPU_seconds'] for x in read(ROOT/'EXTERNAL_PHASE_CHARGES.json'))
 prior=sum(x['CPU_seconds'] for x in read(HERE/'POST_COSTS.json')) if (HERE/'POST_COSTS.json').exists() else 0.
 CPU=l['CPU']+external+prior+time.process_time()+extra_CPU
 assert wall<c['phase_wall_cap_seconds'] and CPU<c['phase_CPU_cap_seconds']
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=c['phase_RSS_cap_KiB'] and shutil.disk_usage(ROOT).free>=c['disk_reserve_bytes']
 return dict(wall=wall,raw_training_controller_CPU=l['CPU'],archive_external_CPU=external,post_prior_CPU=prior,current_process_CPU=time.process_time(),total_budget_CPU=CPU)
def verify_manifest():
 all_private();m=read(HERE/'ALL_EIGHT_FROZEN.json')
 assert m['held_users']==list(range(1,9)) and m['primary']=='tail' and m['baseline']=='native'
 for name,d in read(HERE/'SCORING_SOURCE_HASHES.json').items():assert sha(HERE/name)==d,name
 for path,d in m['files'].items():assert sha(path)==d,path
 return m
def aggregate(rows,arm):
 assert len(rows)==8 and sorted(r['held'] for r in rows)==list(range(1,9))
 return dict(equal_user_mean_accuracy=float(np.mean([r[arm]['accuracy'] for r in rows])),equal_user_mean_present_macro_recall=float(np.mean([r[arm]['present_class_macro_recall'] for r in rows])),worst_user_accuracy=min(r[arm]['accuracy'] for r in rows),worst_user_present_macro_recall=min(r[arm]['present_class_macro_recall'] for r in rows),pooled_sample_accuracy=sum(r[arm]['n']*r[arm]['accuracy'] for r in rows)/sum(r[arm]['n'] for r in rows))
def paired(rows,arm):
 assert len(rows)==8 and sorted(r['held'] for r in rows)==list(range(1,9))
 for r in rows:assert r['native']['n']==r[arm]['n'] and r['native']['class_support']==r[arm]['class_support']
 da=[r[arm]['accuracy']-r['native']['accuracy'] for r in rows];dm=[r[arm]['present_class_macro_recall']-r['native']['present_class_macro_recall'] for r in rows];tol=1e-12
 return dict(paired_accuracy_deltas=da,paired_present_macro_recall_deltas=dm,mean_accuracy_delta=float(np.mean(da)),mean_present_macro_recall_delta=float(np.mean(dm)),worst_paired_accuracy_delta=min(da),worst_paired_present_macro_recall_delta=min(dm),strict_accuracy_wins=sum(d>tol for d in da),accuracy_ties=sum(abs(d)<=tol for d in da),accuracy_losses=sum(d<-tol for d in da))
def gate(result,cfg):
 t=cfg['tolerance']
 return bool(result['mean_accuracy_delta']+t>=cfg['mean_accuracy_gain_min'] and result['strict_accuracy_wins']>=cfg['strict_improved_accuracy_users_min'] and result['worst_paired_accuracy_delta']+t>=cfg['worst_accuracy_paired_delta_min'])
def stage_costs():
 l=read(ROOT/'PHASE_LEDGER.json');hz=os.sysconf('SC_CLK_TCK');last=0.;rows=[]
 for h in range(1,9):
  for stage in ['train','replay','output_audit']:
   d=ROOT/f'folds/held{h}/execution'/stage;t=read(d/'TERMINATION.json');ident=read(d/'IDENTITY.json')
   assert t['exit_code']==0 and t['resource_stop'] is None
   rows.append(dict(held=h,stage=stage,worker_birth_to_wait_end_wall_seconds=l['start_monotonic']+t['wall']-ident['startticks']/hz,CPU_increment_including_controller=t['CPU']-last,cumulative=t));last=t['CPU']
 return dict(rows=rows,wall_definition='same-boot kernel startticks/HZ to controller waited end; includes loading/training/validation/checkpoint/output/sourceBN where applicable, up to one kernel tick and polling latency; excludes interfold idle wall',CPU_definition='differences of monotonically cumulative raw ledger CPU; includes controller/guards, excludes separately charged archives; never sum cumulative totals',totals_by_stage={s:dict(wall_seconds=sum(r['worker_birth_to_wait_end_wall_seconds'] for r in rows if r['stage']==s),CPU_seconds=sum(r['CPU_increment_including_controller'] for r in rows if r['stage']==s)) for s in ['train','replay','output_audit']})
