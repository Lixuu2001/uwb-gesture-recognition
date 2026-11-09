"""Frozen 16-update native training-only cost probe; no method-quality score."""
from pathlib import Path
import time,math,shutil,copy
import core as e
from role_data import RealData
from admission import require_probe
from worker import first_audit
np=e.np;torch=e.torch
def main():
    require_probe();cfg=e.read(e.ROOT/'PROTOCOL.json');out=e.ROOT/'probe';out.mkdir(exist_ok=False)
    data=RealData(1,probe=True);ids=np.array(data.roles['sample_ids']['train'],dtype=np.int64)
    order=e.order(len(ids),1);chosen=ids[order[:16*64]]
    np.savez(out/'source_ids.npz',ids=chosen,users=data.inputs.subjects[chosen])
    assert set(data.inputs.subjects[chosen])=={3,4,5,6}
    m,o,t=e.make();initial=e.capture(m,o,t,0);e.atomic_save(out/'initial.pt',initial)
    archived=torch.load(e.ROOT/'initial.pt',map_location='cpu',weights_only=True);assert e.semantic(initial)==e.semantic(archived)
    rows=[];start=time.monotonic();cpu=time.process_time()
    for i in range(16):
        batch=chosen[i*64:(i+1)*64];wt=time.monotonic();ct=time.process_time();x=data.batch('train',batch);y=data.labels.get('train',batch)
        loaded=time.monotonic();step=e.step(m,o,t,x,y,True)
        row=dict(step=step,load_wall_seconds=loaded-wt,wall_seconds=time.monotonic()-wt,CPU_seconds=time.process_time()-ct)
        rows.append(row)
        if i==0:
            audit=first_audit(initial,step,x,y)
            oldpath=Path('/workspace/uwb-timemil-20261029/run02/gap/first_update.pt')
            assert e.sha(oldpath)=='91c92836f8c9cb876608f13c655abbad83b6206654a890e711fbedec5576a5de'
            assert set(torch.serialization.get_unsafe_globals_in_checkpoint(oldpath))<={'torch.torch_version.TorchVersion'}
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):old=torch.load(oldpath,map_location='cpu',weights_only=True)
            assert e.semantic(old['model'])==step['model_sha256'] and e.semantic(old['optimizer'])==step['optimizer_sha256']
            np.testing.assert_array_equal(old['training_ids'],batch)
            audit['original_native_GAP_held1_first_model_optimizer_sourceIDs_exact']=True
            e.write(out/'FIRST_AUDIT.json',audit);e.atomic_save(out/'first_update.pt',e.capture(m,o,t,0))
        if time.monotonic()-start>cfg['probe_wall_cap_seconds'] or time.process_time()-cpu>cfg['probe_CPU_cap_seconds']:raise RuntimeError('Probe cap reached')
    final=e.capture(m,o,t,0,best_model=copy.deepcopy(m.state_dict()),best_epoch=0);e.atomic_save(out/'final_full.pt',final)
    # Evaluation-like forward timing uses only the same training samples, never validation/held inputs or labels.
    m.eval();forward=[]
    with torch.inference_mode():
        for i in range(4):
            batch=chosen[i*64:(i+1)*64];wt=time.monotonic();z=m(torch.from_numpy(data.batch('train',batch)));forward.append(time.monotonic()-wt)
            assert z.shape==(64,12) and torch.isfinite(z).all()
    # No six-source-user BN timing is fabricated from a four-user probe.
    # Six outer source users are not permitted here; do not weaken calibration guards just to time it.
    BN_cost_model='conservatively two full source-only forward passes per fold per production/replay; actual six-user BN timing pending production admission'
    helds=[data.inputs.roles(h) for h in range(1,9)]
    train_batches=sum(math.ceil(len(r['sample_ids']['train'])/64) for r in helds)
    val_batches=sum(math.ceil(len(r['sample_ids']['validation'])/64) for r in helds)
    test_batches=sum(math.ceil(len(r['sample_ids']['test'])/64) for r in helds)
    step95=float(np.quantile([r['wall_seconds'] for r in rows[2:]],.95));forward95=float(np.quantile(forward[1:],.95))
    full_train_replay=2*64*train_batches*step95
    full_val_outputs=(2*64*val_batches+2*3*(val_batches+test_batches)+4*train_batches)*forward95
    estimate=1.5*(full_train_replay+full_val_outputs)+3600
    full_bytes=(out/'final_full.pt').stat().st_size;first_bytes=(out/'first_update.pt').stat().st_size;initial_bytes=(out/'initial.pt').stat().st_size
    # Each side: two atomic slots + temporary slot + initial/first/final + selected3 models.
    # One temporary slot for the single active worker; source logs and predictions reserve64MiB.
    payload_bound=8*2*(3*full_bytes+first_bytes+initial_bytes+3*3*1024**2)+full_bytes+64*1024**2
    free=shutil.disk_usage(e.ROOT).free
    disk_need=payload_bound+cfg['disk_reserve_bytes']+cfg['transport_reserve_bytes']
    resource_pass=estimate<=min(cfg['phase_wall_cap_seconds'],cfg['phase_CPU_cap_seconds'])
    disk_pass=disk_need<=free
    e.write(out/'RESULT.json',dict(state='real16_native_updates_trainonly_cost_probe_completed_not_quality_evidence',rows=rows,forward_training_only_seconds=forward,updates=16,production_fold=False,source_users=[3,4,5,6],samples=1024,gesture_truth_values_read_only_from_users3_6=True,users7_8_truth_values_read=False,validation_or_test_labels_read=False,tail_collector_all16_for_overhead_only=True,held_scores_computed=0,model_files={p.name:e.sha(p) for p in out.glob('*.pt')},wall=time.monotonic()-start,CPU=time.process_time()-cpu))
    e.write(e.ROOT/'RESOURCE_GATE.json',dict(passed=resource_pass and disk_pass,resource_pass=resource_pass,disk_pass=disk_pass,measured_step_wall_p95_seconds=step95,measured_trainonly_forward_wall_p95_seconds=forward95,production_plus_replay_updates=2*64*train_batches,estimated_wall_and_singlethread_CPU_with_50pct_headroom_and_3600_reserve_seconds=estimate,phase_each_cap_seconds=129600,checkpoint_measured_bytes=full_bytes,initial_measured_bytes=initial_bytes,first_measured_bytes=first_bytes,full8_model_payload_disk_bound_bytes=payload_bound,required_disk_with_reserves_bytes=disk_need,observed_free_bytes=free,BN_cost_model=BN_cost_model,method_hyperparameters_changed=False,production_admitted=False,scope='Cost/storage gate alone never activates real8; require independent native/output audit and explicit production release',inputs={'PROTOCOL.json':e.sha(e.ROOT/'PROTOCOL.json'),'FROZEN_HASHES.json':e.sha(e.ROOT/'FROZEN_HASHES.json'),'probe/RESULT.json':e.sha(out/'RESULT.json')}))
    print('Cost probe complete; resource/disk gate:',resource_pass,disk_pass,'; full8 production remains unadmitted')
if __name__=='__main__':main()
