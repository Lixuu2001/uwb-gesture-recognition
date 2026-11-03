"""Private partial-checkpoint preservation only; never scores or resumes training."""
from pathlib import Path
import argparse,datetime,json,os,signal,sys,time
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',default='/workspace/uwb-shufflenetv2-x1-recovery-pilot-20261103');arg=a.parse_args();B=Path(arg.root).resolve();sys.path.insert(0,str(B))
 from common import read,write,sha
 from stream_archive import build
 import torch
 active=read(B/'ACTIVE_OBSERVATION.json');pid=read(B/'supervisor01/PID.json')['pid'];assert pid==active['worker_pid'];ticks=active['worker_start_ticks'];proc=Path(f'/proc/{pid}')
 assert proc.joinpath('stat').read_text().split()[21]==ticks
 assert str(B/'run_pilot.py').encode() in proc.joinpath('cmdline').read_bytes().split(b'\0')
 paused=False;start=time.monotonic();directory=None;epoch=None
 try:
  os.kill(pid,signal.SIGSTOP);paused=True
  for _ in range(100):
   if proc.joinpath('stat').read_text().split()[2]=='T':break
   time.sleep(.02)
  else:raise RuntimeError('Producer did not stop for consistent preservation')
  out=B/'run01';epoch=read(out/'STATE.json')['epoch'];assert 1<=epoch<40
  checkpoint=torch.load(out/'final_full.pt',map_location='cpu',weights_only=True)
  assert checkpoint['epoch']==epoch and checkpoint['updates']==314*epoch
  assert (out/f'epoch{epoch:02d}.json').exists()
  best=torch.load(out/'best_model.pt',map_location='cpu',weights_only=True);assert 1<=best['epoch']<=epoch
  last=json.loads((out/'steps.jsonl').read_text().splitlines()[-1])
  directory=B/f'producer_insurance_epoch{epoch:02d}';assert not directory.exists(),'Existing insurance requires ID/receipt inspection, no blind rebuild'
  scope=f'incomplete_producer_checkpoint_ShuffleNetV2_x1_recovery_epoch{epoch:02d}'
  info=dict(state='incomplete_checkpoint_preservation_not_a_scientific_result',scope=scope,checkpoint_epoch=epoch,checkpoint_updates=checkpoint['updates'],last_logged_update=last['update'],uncommitted_log_suffix_preserved=last['update']>checkpoint['updates'],automatic_resume_authorized=False,full_training_complete=False,full_replay_audit_complete=False,diagnostic_labels_scored=False,users7_8_labels_used=False,registration_commit=active['registration_commit'],frozen_sha256=sha(B/'FROZEN_HASHES.json'),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  note=B/f'PRODUCER_INSURANCE_EPOCH{epoch:02d}.json';assert not note.exists();write(note,info)
  files={'study/'+n:B/n for n in read(B/'FROZEN_HASHES.json')}
  for n in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','INITIAL_PRIVATE_INSURANCE.json']:files['study/'+n]=B/n
  files['PRESERVATION_STATE.json']=note;files['operations/preserve_active_checkpoint.py']=Path(__file__).resolve()
  for p in sorted(out.iterdir()):
   if p.is_file() and not p.name.endswith('.tmp'):files['run01/'+p.name]=p
  m=build(directory,f'shufflenetv2_x1_recovery40_producer_epoch{epoch:02d}_20261103',files,scope,chunk_size=7340032,limit=67108864)
  m.update(state='incomplete_producer_checkpoint_insurance_streamed',complete_modelset=False,full_training_complete=False,numeric_audit_certified=False);write(directory/'MANIFEST.json',m)
  print(json.dumps(dict(directory=str(directory),epoch=epoch,checkpoint_updates=checkpoint['updates'],last_logged_update=last['update'],archive_bytes=m['archive_bytes'],parts=len(m['chunks']),full_audit_claimed=False)))
 finally:
  if paused:
   assert proc.joinpath('stat').read_text().split()[21]==ticks
   os.kill(pid,signal.SIGCONT)
   result=dict(worker_pid=pid,worker_start_ticks=ticks,checkpoint_epoch=epoch,resumed=True,pause_seconds=time.monotonic()-start)
   write(B/'LAST_PRESERVATION_PAUSE.json',result)
   if directory is not None and directory.exists():write(directory/'PRESERVATION_PAUSE.json',result)
if __name__=='__main__':main()
