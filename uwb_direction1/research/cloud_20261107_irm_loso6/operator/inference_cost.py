"""One label-free paired inference timing pass after complete sixfold private freeze."""
from pathlib import Path
import sys,os,json,time,hashlib,signal,resource
P=Path('/workspace/uwb-irm-loso6-20261107');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready
from controller import identity

def preflight():
    old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid'])
    if now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks']):
        raise RuntimeError('active_original_phase_no_inference_or_model_data_loading')
    assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists(),'Timing protocol runs before cohort truth pairing'
    assert read(P/'STATE.json')['state']=='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once'
    registration=read(P/'REMOTE_PREREGISTRATION.json')
    assert registration['branch_head_verified'] and registration['force'] is False and registration['frozen_sha256']==sha(P/'FROZEN_HASHES.json')
    release=read(P/'FULL_PRIVATE_RELEASE.json');assert release['branch_head_verified'] and release['force'] is False
    backup=P/'full_backup';assert read(backup/'READBACK_VERIFIED.json')['complete_modelset_verified']
    assert read(backup/'MANIFEST.json')['archive_sha256']==release['complete_six_archive_sha256']
    ready(True);frozen=read(P/'ALL_SIX_PREDICTIONS_FROZEN.json')
    for row in frozen['files'].values():assert sha(row['path'])==row['sha256']
    cfg=read(P/'operator/INFERENCE_PROTOCOL.json')
    for row in cfg['baseline_artifacts'].values():
        assert sha(row['model'])==row['model_sha256'] and sha(row['prediction'])==row['prediction_sha256']
    assert not(P/'INFERENCE_ONCE.lock').exists() and not(P/'INFERENCE_COST.json').exists()
    return cfg,frozen

def main():
    started=time.monotonic();cpu=time.process_time()
    def timeout(sig,frame):raise TimeoutError('fixed inference-only wall or CPU resource cap')
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGXCPU,timeout)
    signal.setitimer(signal.ITIMER_REAL,300);resource.setrlimit(resource.RLIMIT_CPU,(300,305))
    cfg,frozen=preflight()
    fd=os.open(P/'INFERENCE_ONCE.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    try:
        import engine as e
        import numpy as np
        with np.load(cfg['source_ids']) as z:ids=z['ids'];users=z['users']
        assert len(ids)==6900 and set(np.unique(users))==set(range(1,7))
        maps=np.load(cfg['maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
        rows=[]
        for h in range(1,7):
            positions=np.flatnonzero(users==h);assert len(positions)==cfg['counts'][str(h)]
            for arm in (['baseline','method'] if h%2 else ['method','baseline']):
                if arm=='baseline':
                    model_path=Path(cfg['baseline_artifacts'][str(h)]['model']);pred_path=Path(cfg['baseline_artifacts'][str(h)]['prediction']);key='logits'
                else:
                    model_path=Path(frozen['files'][f'held{h}/selected.pt']['path'])
                    pred_path=Path(frozen['files'][f'held{h}/diagnostic_unscored.npz']['path']);key='scores'
                state=e.torch.load(model_path,map_location='cpu',weights_only=False)
                model,optimizer,generator=e.make();model.load_state_dict(state['model'],strict=True);model.eval()
                before=e.semantic(model.state_dict());assert sum(x.numel() for x in model.parameters())==617612
                warm_wall=time.monotonic();warm_cpu=time.process_time()
                e.evaluate(model,maps,positions[:64])
                warm_wall=time.monotonic()-warm_wall;warm_cpu=time.process_time()-warm_cpu
                wall=time.monotonic();proc=time.process_time();actual=e.evaluate(model,maps,positions)
                proc=time.process_time()-proc;wall=time.monotonic()-wall
                assert e.semantic(model.state_dict())==before,'Inference changed model parameters or buffers'
                with np.load(pred_path) as z:
                    np.testing.assert_array_equal(z['ids'],ids[positions]);np.testing.assert_array_equal(z[key],actual)
                row=dict(held=h,arm=arm,n=len(positions),batch=64,batches=(len(positions)+63)//64,parameters=617612,wall_seconds=wall,CPU_seconds=proc,wall_ms_per_sample=wall*1000/len(positions),CPU_ms_per_sample=proc*1000/len(positions),warmup_wall_seconds=warm_wall,warmup_CPU_seconds=warm_cpu,model_sha256=sha(model_path),model_semantic_sha256=before,logits_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),frozen_logits_bitwise_equal=True,model_and_buffers_unchanged=True,held_labels_read=False)
                rows.append(row);write(P/'INFERENCE_PROGRESS.json',dict(rows=rows,held_labels_read=False));del model,optimizer,generator,state,actual
        result=dict(state='one_label_free_frozen_sixfold_paired_inference_cost_pass',rows=rows,protocol_sha256=sha(P/'operator/INFERENCE_PROTOCOL.json'),total_worker_wall_seconds=time.monotonic()-started,total_worker_CPU_seconds=time.process_time()-cpu,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,held_labels_read=False,users7_8_labels_used=False,new_training_or_optimizer_updates=False,accuracy_or_macro_computed=False,method_selection_or_gate_changed=False)
        assert result['max_RSS_KiB']<=12582912
        write(P/'INFERENCE_COST.json',result);print(json.dumps(result))
    except BaseException as ex:
        write(P/'INFERENCE_FAILURE.json',dict(error=repr(ex),scientific_negative=False,no_automatic_retry=True,partial_results_preserved=True,held_labels_read=False,wall=time.monotonic()-started,CPU=time.process_time()-cpu));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    assert sys.argv[1:]==['--run'];main()
