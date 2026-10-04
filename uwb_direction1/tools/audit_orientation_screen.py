"""Independent full source-screen FFT/correlation/eigendecomposition replay."""
import json,time
from pathlib import Path
import numpy as np
from audit_restored import digest,dump
from check_orientation_screen import reference_tensor


def independent_image(raw):
    x=np.asarray(raw,dtype=np.float64);n=x.shape[-1];h=np.zeros(n);h[0]=1;h[1:(n+1)//2]=2
    if n%2==0:h[n//2]=1
    analytic=np.fft.ifft(np.fft.fft(x,axis=-1)*h,axis=-1);envelope=np.sqrt(analytic.real**2+analytic.imag**2)
    reduced=np.stack([envelope[...,3*j:3*j+3].sum(-1)/3 for j in range(63)],-1)
    maximum=reduced.max(axis=(-1,-2),keepdims=True)
    return np.sqrt(np.divide(reduced,maximum,out=np.zeros_like(reduced),where=maximum>0))


def independent_measure(z,threshold):
    matrix=np.empty(z.shape[:-1]+(2,2));matrix[...,0,0]=z[...,0];matrix[...,0,1]=matrix[...,1,0]=z[...,1];matrix[...,1,1]=z[...,2]
    eigen,vec=np.linalg.eigh(matrix);total=eigen.sum(-1);gap=eigen[...,1]-eigen[...,0];coh=np.divide(gap,total,out=np.zeros_like(total),where=total>1e-12)
    # Principal eigenvector components, not the production discriminant formula.
    oblique=2*abs(vec[...,0,1]*vec[...,1,1]);eligible=(z[...,0]+z[...,2]>=threshold)&(z[...,0]+z[...,2]>1e-12)
    usable=eligible&(coh>=.6)&(oblique>=.3)
    return float(usable.sum()/max(eligible.sum(),1)),int(eligible.sum())


def main():
    R=Path('/workspace/uwb-orientation-screen-20261004');P=Path('/workspace/uwb-temporal-publish-20261003/uwb_direction1');start=time.perf_counter();c=json.loads((R/'PROTOCOL.json').read_text());s=json.loads((R/'SUMMARY.json').read_text());identity=json.loads((R/'IDENTITY.json').read_text());manifest=json.loads((R/'FILES.json').read_text())
    for n,sha in manifest.items():assert digest(R/n)==sha
    assert digest(P/'research/ORIENTATION_SCREEN_PROTOCOL.json')==identity['protocol_sha256']
    for n,sha in identity['code_sha256'].items():assert digest(P/'tools'/n)==sha
    raw=np.load('/workspace/uwb-restored-20260930/cache/X_clutterremoved.npy',mmap_mode='r');users=np.load('/workspace/uwb-restored-20260930/cache/subj_clutterremoved.npy');labels=np.load('/workspace/uwb-restored-20260930/cache/y_clutterremoved.npy');ids=np.load(R/'ids.npy');assert digest(R/'ids.npy')==c['ids_sha256']
    assert set(users[ids])==set(range(3,9)) and len(set(ids))==568
    for u in range(3,9):
        for y in range(12):assert np.sum((users[ids]==u)&(labels[ids]==y))==(0 if (u,y)==(6,3) else 8)
    with np.load(R/'tensors.npz') as f:original=f['tensors'];perms=f['permutations'];np.testing.assert_array_equal(f['ids'],ids)
    scores=np.zeros((568,3,2));eligible=np.zeros((568,3,2),dtype=int);maxerror=0.;locations=0
    for n,sid in enumerate(ids):
        im=independent_image(raw[int(sid)]);perm=np.random.default_rng(np.random.SeedSequence([c['seed'],int(sid)])).permutation(90);np.testing.assert_array_equal(perm,perms[n])
        for view in range(3):
            tensors=[]
            for arm,x in enumerate([im[view],im[view][perm]]):
                z=reference_tensor(x)[10:80:5,10:53:5].reshape(-1,3);error=float(abs(z-original[n,view,arm]).max());maxerror=max(error,maxerror);assert error<1e-12;tensors.append(z);locations+=len(z)
            threshold=np.quantile(tensors[0][:,0]+tensors[0][:,2],.75)
            for arm in range(2):scores[n,view,arm],eligible[n,view,arm]=independent_measure(tensors[arm],threshold)
        if (n+1)%100==0:print(json.dumps({'audited_clips':n+1}),flush=True)
    rows=[]
    for u in range(3,9):
        ix=users[ids]==u
        for view in range(3):
            mean=scores[ix,view].mean(0);gate=bool(mean[0]>=c['gate']['min_original_usable_fraction'] and mean[0]-mean[1]>=c['gate']['min_gain_over_shuffled']);record=next(z for z in s['rows'] if z['user']==u and z['view']==view)
            np.testing.assert_allclose([record['original'],record['shuffled']],mean,atol=1e-12);np.testing.assert_array_equal(record['eligible'],eligible[ix,view].sum(0));assert gate==record['passed'];rows.append({'user':u,'view':view,'passed':gate})
    viewpass=[sum(z['passed'] for z in rows if z['view']==v)>=4 for v in range(3)];assert viewpass==s['view_pass'];assert (sum(viewpass)>=2)==s['gate_passed']
    result={'state':'passed','all_source_clips_recomputed':568,'all_tensor_locations_recomputed':locations,'all_tensor_components_recomputed':locations*3,'max_tensor_error':maxerror,'artifact_hashes':len(manifest),'source_roles_and_sampling_verified':True,'all_18_user_view_statistics_verified':True,'gate_reproduced':True,'seconds':time.perf_counter()-start}
    dump(R/'AUDIT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
