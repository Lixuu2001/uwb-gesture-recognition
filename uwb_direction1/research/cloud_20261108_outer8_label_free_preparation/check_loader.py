"""One real input pass and generated firewalls; never fit or score models."""
from pathlib import Path
import json,hashlib,time,resource,shutil
import numpy as np
import torch
from lazy_outer_data import OuterInputs,rms_batch,sha,EXPECTED

B=Path(__file__).resolve().parent
def dump(name,obj):
    p=B/name
    if p.exists(): raise FileExistsError(p)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def main():
    start=time.monotonic();cpu=time.process_time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    root=Path('/workspace/uwb-restored-20261103')
    d=OuterInputs(root/'cache/X_clutterremoved.npy',root/'cache/subj_clutterremoved.npy',
                  '/workspace/uwb-accuracy-loop-20261029/run01/source_ids.npz')
    maps_path=Path('/workspace/uwb-timemil-20261029/run02/source_maps.npy')
    if sha(maps_path)!=EXPECTED['development_maps']: raise ValueError('Development cache SHA mismatch')
    maps=np.load(maps_path,mmap_mode='r',allow_pickle=False)
    roles=[d.roles(h) for h in range(1,9)]
    assert sorted(i for r in roles for i in r['sample_ids']['test'])==d.ids.tolist()
    # Membership errors must fail before touching raw data. No label path exists.
    blocked=[]
    for name,call in [
        ('cross_role',lambda:d.batch(1,'train',roles[0]['sample_ids']['test'][:1])),
        ('unknown_role',lambda:d.batch(1,'fitting_test',d.ids[:1])),
        ('negative_id',lambda:d.batch(1,'test',np.array([-1]))),
        ('float_id',lambda:d.batch(1,'test',d.ids[:1].astype(float))),
        ('gesture_labels',lambda:d.labels(1,'train')),
        ('duplicate_id',lambda:rms_batch(d.raw,np.array([0,0]))),
        ('oversized_batch',lambda:rms_batch(d.raw,np.arange(65))),
    ]:
        try:call()
        except (ValueError,PermissionError):blocked.append(name)
        else:raise AssertionError('Firewall accepted '+name)
    # Generated signed/zero channels independently checked, with reduction tolerance.
    rng=np.random.default_rng(20261029);g=rng.normal(size=(2,3,90,189)).astype(np.float32);g[0,0]=0
    got=rms_batch(g,np.arange(2));den=np.maximum(np.sqrt(np.mean(g*g,axis=(2,3),keepdims=True)),1e-8)
    reference=(g/den).transpose(0,2,1,3).reshape(2,90,567)
    np.testing.assert_allclose(got,reference,rtol=5e-7,atol=5e-7)
    assert np.all(got[0,:,:189]==0) and np.any(got<0)
    stream=hashlib.sha256();n6=0;total=0;batch_seconds=[]
    for begin in range(0,len(d.ids),64):
        t=time.monotonic();ids=d.ids[begin:begin+64];x=rms_batch(d.raw,ids)
        batch_seconds.append(time.monotonic()-t);stream.update(x.tobytes());total+=len(ids)
        m=d.subjects[ids]<=6
        if m.any():
            positions=np.searchsorted(d.dev_ids,ids[m]);np.testing.assert_array_equal(d.dev_ids[positions],ids[m])
            np.testing.assert_array_equal(x[m],maps[positions]);n6+=int(m.sum())
        if time.monotonic()-start>180 or time.process_time()-cpu>180:raise RuntimeError('Preparation time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>6*1024*1024:raise RuntimeError('Preparation RSS cap')
    assert n6==6900 and total==9300
    # Role-scoped API produces same exact input; not a model/output audit.
    for r in roles:
        for role in ('train','validation','test'):
            ids=np.array(r['sample_ids'][role][:1],dtype=np.int64)
            np.testing.assert_array_equal(d.batch(r['held'],role,ids),rms_batch(d.raw,ids))
    private=B/'outer8_ids_and_roles.npz'
    if private.exists():raise FileExistsError(private)
    np.savez(private,ids=d.ids,users=d.users,**{f'held{r["held"]}_{k}':np.array(v,dtype=np.int64) for r in roles for k,v in r['sample_ids'].items()})
    dump('INPUT_CHECK.json',dict(state='label_free_real_outer8_inputs_verified_no_fit_no_score',
        files_sha256=EXPECTED,private_ids_archive_sha256=sha(private),users=8,samples=total,
        subject_counts={str(h):int((d.users==h).sum()) for h in range(1,9)},
        folds=[dict(held=r['held'],validation_user=r['validation_user'],users=r['users'],counts={k:len(v) for k,v in r['sample_ids'].items()}) for r in roles],
        full8_RMS_bytes_sha256=stream.hexdigest(),full8_inputs_finite=True,signed_RMS=True,
        development6900_maps_elementwise_bitwise_equal=True,all8_test_ID_coverage_exactly_once=True,
        generated_numpy_RMS_tolerance=5e-7,generated_zero_and_signed_input_pass=True,blocked=blocked,
        gesture_label_arrays_loaded=False,legacy_NPZ_labels_key_accessed=False,new_model_fits=0,test_scores_computed=0,
        no_full_map_cache_created=True,full_map_cache_would_require_bytes=9300*90*567*4+128,
        free_disk_bytes=shutil.disk_usage(B).free,input_pass_wall_seconds=sum(batch_seconds),
        batch64_median_wall_seconds=float(np.median(batch_seconds[:-1])),
        wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
        max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        interpretation='Input/metadata audit only. No gesture-label supports/missing classes inferred for users7/8; no model training/inference cost estimate from loader time alone.'))
    print('PASS: 9300 inputs; 6900 native-cache exact; 8 disjoint folds; labels/fit/score=0')

if __name__=='__main__':main()
