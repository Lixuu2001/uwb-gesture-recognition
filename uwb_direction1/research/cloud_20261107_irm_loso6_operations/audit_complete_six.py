"""Read-only completed-cohort native/ID audit. Does not train, infer or read gesture truth."""
from pathlib import Path
import sys,time,json,math
P=Path('/workspace/uwb-irm-loso6-20261107');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from controller import identity
from journal import load
from loso_roles_metrics import roles
import engine as e
import numpy as np

def main():
    started=time.monotonic();cpu=time.process_time();ready(False)
    old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['boot_id','startticks'])
    phase=read(P/'STATE.json');assert phase['state']=='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once' and phase['completed_folds']==list(range(1,7))
    assert phase['wall']<43200 and phase['cpu']<43200 and not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
    with np.load(read(P/'DEVELOPMENT_PROTOCOL.json')['source_ids']) as z:ids=z['ids'];users=z['users']
    assert len(ids)==len(users)==6900 and len(np.unique(ids))==6900 and set(np.unique(users))==set(range(1,7))
    rows=[]
    for h in range(1,7):
        fold=P/f'folds/held{h}';c=read(fold/'PROTOCOL.json');root=fold/'run';val,loc=roles(users,h)
        assert val==c['validation_user'] and list(map(len,loc.values()))==c['counts'] and set(users[loc['train']])==set(c['train_users'])
        for name in ['roles.npz','run/train/split.npz','run/replay/split.npz']:
            with np.load(fold/name) as z:
                for k,ix in loc.items():assert np.array_equal(z[k],ids[ix]),'Role-ID mismatch'
        assert sha(fold/'initial.pt')==read(P/'REAL_RELEASE.json')['initial_model_file_sha256']
        for path in [fold/'CONTROLLER_IDENTITY.json',fold/'LAUNCH_IDENTITY.json',*list((root/'execution').rglob('*IDENTITY.json'))]:
            a=read(path);q=identity(a['pid']);assert q is None or q['state']=='Z' or any(q[k]!=a[k] for k in ['boot_id','startticks'])
        assert read(fold/'PHASE_TERMINATION.json')['exit_code']==0
        for side in ['train','replay','output']:
            t=read(root/f'execution/{side}/TERMINATION.json');assert t['exit_code']==0 and t['resource_stop_reason'] is None
        complete=[read(root/f'{side}/COMPLETE.json') for side in ['train','replay']]
        for r in complete:assert r['epochs']==64 and r['updates']==c['expected_updates'] and not r['diagnostic_labels_read'] and not r['users7_8_labels_used']
        for k in complete[0]:
            if k!='state':assert complete[0][k]==complete[1][k],k
        for n in ['selected.pt','final_full.pt','validation_unscored.npz','diagnostic_unscored.npz','steps.jsonl']:
            assert sha(root/'train'/n)==sha(root/'replay'/n),n
        for epoch in range(65):
            for kind in ['epoch','journal']:assert sha(root/f'train/{kind}{epoch:03}.json')==sha(root/f'replay/{kind}{epoch:03}.json')
        for side in ['train','replay']:
            s,p=load(root/side);assert p['epoch']==64 and s['updates']==c['expected_updates'] and e.semantic(s)==complete[0]['final_state_sha256']
            raw=(root/side/'steps.jsonl').read_bytes();assert len(raw)==p['prefix_bytes'] and sha(root/side/'steps.jsonl')==p['prefix_sha256']
            steps=[json.loads(x) for x in raw.splitlines()];assert len(steps)==c['expected_updates']
            for i,r in enumerate(steps):
                assert r['update']==i+1 and r['completed_updates_before']==i and r['completed_updates_after']==i+1
                assert r['irm_penalty_weight']==(1. if i<500 else 100.) and r['optimizer_reset_this_update']==(i==500) and r['optimizer_reset_events']==([] if i<500 else [500])
                assert r['global_BCE']==r['classification_loss'] and all(math.isfinite(r[k]) for k in ['global_BCE','penalty','loss'])
                assert abs(r['loss']-(r['global_BCE']+r['irm_penalty_weight']*r['penalty']))<=1e-6*max(1.,abs(r['loss']))
            o=s['optimizer'];n=o['native_optimizer'];assert o['algorithm_updates']==c['expected_updates'] and o['reset_events']==[500]
            assert n['count']==c['expected_updates']-500 and all(float(v['step'])==n['count'] for v in n['fast']['state'].values())
            assert all(e.torch.isfinite(v).all() for v in s['model'].values())
        frozen=read(root/'PREDICTIONS_FROZEN.json')
        for k,v in frozen['files'].items():assert sha(root/k)==v
        ref=read(root/'train/FIRST_INDEPENDENT_REFERENCE.json');assert ref['parameters']==617612 and ref['training_labels_only'] and not ref['diagnostic_labels_read']
        assert read(root/'train/FIRST_RELEASE.json')['native_first_update_exact']
        rows.append(dict(held=h,validation_user=val,train_users=c['train_users'],role_counts=c['counts'],epochs_each=64,updates_each=c['expected_updates'],all65_epoch_records_and_all_steps_exact=True,selected_epoch_validation_only=complete[0]['selected_epoch'],selected_sha256=complete[0]['selected_sha256'],final_native_sha256=complete[0]['final_state_sha256'],first_independent_gradient_maxabs=ref['max_gradient_absolute_difference'],cost=read(fold/'STATE.json')))
    result=dict(state='all_six_complete_native_trajectories_roles_and_unscored_outputs_readonly_verified',rows=rows,complete_all_six=True,held_labels_read=False,gesture_truth_decoded=False,users7_8_labels_used=False,classification_metrics_computed=False,new_training_replay_or_inference_run=False,complete_stage_cost=phase,prior_costs_preserved=True,scientific_frozen_manifest_sha256=sha(P/'FROZEN_HASHES.json'),check_wall=time.monotonic()-started,check_CPU=time.process_time()-cpu)
    write(P/'operator/AUXILIARY_COMPLETE_SIX_AUDIT.json',result);print(json.dumps(result))

if __name__=='__main__':main()
