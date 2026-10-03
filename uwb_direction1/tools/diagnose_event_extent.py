"""Source-only temporal concentration diagnostic; no onset labels or classifier.

Reuses the two historical motion proxies without training or claiming a new
method. An interval retaining proxy mass is NOT a true gesture boundary.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.signal import hilbert
from audit_restored import digest, dump, quality_mask


def evidence(raw):
    n = raw.shape[-1]
    h = np.zeros(n); h[0] = 1; h[1:(n+1)//2] = 2
    if n % 2 == 0: h[n//2] = 1
    z = np.fft.ifft(np.fft.fft(raw.astype(np.float64), axis=-1)*h, axis=-1)
    z -= z.mean(-2, keepdims=True)
    d = z[..., 1:, :]*z[..., :-1, :].conj()
    phase = (np.sqrt(np.abs(d))*np.abs(np.angle(d))).sum(-1)
    amp = np.abs(np.diff(np.abs(z), axis=-2)).sum(-1)
    # 89 real transitions, without the historical duplicate first increment.
    return np.stack((phase, amp), axis=-2)


def normalized(e):
    den = e.sum(-1, keepdims=True)
    valid = den[..., 0] > 1e-20
    p = np.divide(e, den, out=np.zeros_like(e), where=den > 1e-20)
    return p, valid


def shortest(p, mass=.9):
    """p: [proxy,time]. Return earliest shortest joint interval [start,end).

    All proxy masses must be retained. End is exclusive on transition indices;
    corresponding raw frame interval is [start,end] (end-start+1 frames).
    Caller handles all-zero evidence, which has no interpretable localization.
    """
    c = np.concatenate((np.zeros((len(p), 1)), np.cumsum(p, axis=-1)), axis=-1)
    c[:, -1] = 1.
    ends = np.stack([np.searchsorted(row, row[:-1]+mass-1e-12, side='left') for row in c]).max(0)
    widths = ends-np.arange(p.shape[-1]); widths[ends > p.shape[-1]] = p.shape[-1]+1
    start = int(np.argmin(widths))
    return np.array([start, int(ends[start])], dtype=np.int16)


def brute(p, mass=.9):
    for width in range(1, p.shape[-1]+1):
        for start in range(p.shape[-1]-width+1):
            if np.all(p[:, start:start+width].sum(-1) >= mass-1e-12):
                return np.array([start, start+width], dtype=np.int16)
    raise AssertionError('No interval')


def checks():
    rng = np.random.default_rng(20261003)
    cases = [np.ones((2, 89))/89, np.eye(89)[[0,88]],
             np.eye(89)[[45,45]], rng.dirichlet(np.ones(89), size=2)]
    for p in cases:
        np.testing.assert_array_equal(shortest(p), brute(p))
    assert tuple(shortest(cases[0])) == (0,81)
    assert tuple(shortest(cases[1])) == (0,89)
    assert tuple(shortest(cases[2])) == (45,46)
    p, valid = normalized(np.zeros((2,89)))
    assert not valid.any() and not p.any()
    return {'uniform_impulse_disjoint_random_intervals_checked': True,
            'zero_evidence_marked_invalid': True}


def describe(ids, users, p, valid, bounds, config):
    # bounds axes: record, view, phase/amp/joint, start/end.
    frames = bounds[..., 1]-bounds[..., 0]+1
    out = {}
    for v, name in enumerate(config['views']):
        by_user = {}
        for u in config['source_users']:
            sel = users == u
            good = valid[sel,v].all(-1)
            eligible = good & (frames[sel,v,2] <= config['screen']['max_frames'])
            by_user[str(u)] = {'records': int(sel.sum()), 'both_proxies_valid': int(good.sum()),
                'joint_short_fraction_all_records': float(eligible.mean()),
                'frame_length_quantiles_phase_amp_joint': np.quantile(frames[sel,v], [.1,.5,.9], axis=0).tolist()}
        passed_users = [int(u) for u, row in by_user.items()
                        if row['joint_short_fraction_all_records'] >= config['screen']['min_fraction']]
        a, b = bounds[:,v,0], bounds[:,v,1]
        intersection = np.maximum(0, np.minimum(a[:,1],b[:,1])-np.maximum(a[:,0],b[:,0]))
        union = np.maximum(a[:,1],b[:,1])-np.minimum(a[:,0],b[:,0])
        agreement = np.divide(intersection, union, out=np.zeros(len(ids)), where=union>0)
        out[name] = {'by_user':by_user, 'screen_passed_users':passed_users,
            'screen_passed':len(passed_users)>=config['screen']['min_users'],
            'phase_amp_interval_iou_quantiles':np.quantile(agreement,[.1,.5,.9]).tolist(),
            'phase_amp_start_absolute_difference_median':float(np.median(np.abs(a[:,0]-b[:,0]))),
            'phase_amp_end_absolute_difference_median':float(np.median(np.abs(a[:,1]-b[:,1]))),
            'first_last9_transition_mass_mean_phase_amp':np.stack((p[:,v,:,:9].sum(-1).mean(0),p[:,v,:,-9:].sum(-1).mean(0))).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/EVENT_EXTENT_PROTOCOL.json')
    a=ap.parse_args(); config=json.loads(a.protocol.read_text())
    if a.out.exists(): raise ValueError('Use a new output directory')
    checked=checks()
    paths={k:a.data/'cache'/n for k,n in [('X','X_clutterremoved.npy'),('y','y_clutterremoved.npy'),('users','subj_clutterremoved.npy')]}
    hashes={k:digest(v) for k,v in paths.items()}
    assert hashes==config['input_sha256']
    y=np.load(paths['y']); users=np.load(paths['users']); raw=np.load(paths['X'],mmap_mode='r')
    ids=np.flatnonzero(quality_mask(y,users)&np.isin(users,config['source_users']))
    assert len(ids)==config['records'] and not np.isin(users[ids],config['excluded_users']).any()
    assert raw.shape==(9593,3,90,189)
    start=time.perf_counter(); ps=[]; masks=[]; max_error=0.
    for i in range(0,len(ids),16):
        x=np.asarray(raw[ids[i:i+16]],dtype=np.float64); e=evidence(x)
        if i==0:
            z=hilbert(x,axis=-1); z-=z.mean(-2,keepdims=True)
            d=z[...,1:,:]*z[...,:-1,:].conj()
            ref=np.stack(((np.sqrt(np.abs(d))*np.abs(np.arctan2(d.imag,d.real))).sum(-1),
                          np.abs(np.abs(z[...,1:,:])-np.abs(z[...,:-1,:])).sum(-1)),axis=-2)
            np.testing.assert_allclose(e,ref,rtol=1e-11,atol=1e-9)
            max_error=float(np.max(np.abs(e-ref)))
        p,valid=normalized(e); ps.append(p); masks.append(valid)
    p=np.concatenate(ps); valid=np.concatenate(masks)
    assert np.isfinite(p).all() and (p>=0).all()
    bounds=np.zeros((len(ids),3,3,2),dtype=np.int16)
    bounds[...,1]=89
    for i in range(len(ids)):
        for v in range(3):
            for j in range(2):
                if valid[i,v,j]: bounds[i,v,j]=shortest(p[i,v,j:j+1],config['retained_mass'])
            if valid[i,v].all(): bounds[i,v,2]=shortest(p[i,v],config['retained_mass'])
    # Independent exhaustive search on the first16 real clips, each view/proxy/joint.
    for i in range(16):
        for v in range(3):
            for j in range(3):
                pp=p[i,v,j:j+1] if j<2 else p[i,v]
                if (valid[i,v,j] if j<2 else valid[i,v].all()):
                    np.testing.assert_array_equal(bounds[i,v,j],brute(pp,config['retained_mass']))
    a.out.mkdir(parents=True)
    np.savez_compressed(a.out/'per_clip.npz',ids=ids,users=users[ids],probability=p,valid=valid,bounds=bounds)
    result={'state':'completed_source_proxy_extent_diagnostic','records':len(ids),'new_trained_models':0,
        'views':describe(ids,users[ids],p,valid,bounds,config),'invalid_proxy_cases':int((~valid).sum()),
        'checks':dict(checked,first16_scipy_evidence_max_absolute_error=max_error,first16_exhaustive_intervals=144),
        'seconds_excluding_input_hashes':time.perf_counter()-start,'input_sha256':hashes,
        'protocol_sha256':digest(a.protocol),'script_sha256':digest(__file__),
        'per_clip_sha256':digest(a.out/'per_clip.npz'),
        'interpretation':'Signal-change proxies, not gesture/background boundaries. Screen only for energy-preserving crop planning, not classification or novelty.'}
    dump(a.out/'SUMMARY.json',result);dump(a.out/'protocol.json',config)
    print(json.dumps({'state':result['state'],'seconds':result['seconds_excluding_input_hashes'],
        'screen':{k:v['screen_passed'] for k,v in result['views'].items()}}),flush=True)


if __name__=='__main__': main()
