"""Independent exhaustive-window and source-ID audit of temporal extent outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_restored import digest, dump, quality_mask


def exhaustive_windows(p, valid, mass):
    # Unlike production's CDF search for each start, enumerate every width and
    # every start; first success per record is shortest, first true start is earliest.
    result=np.zeros((len(p),2),dtype=np.int16); result[:,1]=89
    done=~valid.all(-1)
    c=np.pad(np.cumsum(p,axis=-1),((0,0),(0,0),(1,0)))
    for width in range(1,90):
        windows=c[...,width:]-c[...,:-width]
        ok=(windows>=mass-1e-12).all(1)
        found=ok.any(-1)&~done
        starts=ok.argmax(-1)
        result[found,0]=starts[found];result[found,1]=starts[found]+width
        done|=found
    assert done.all()
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); summary=json.loads((a.out/'SUMMARY.json').read_text());cfg=json.loads((a.out/'protocol.json').read_text())
    assert digest(a.out/'per_clip.npz')==summary['per_clip_sha256']
    source=Path(__file__).resolve().parents[1]
    frozen=source/'research/EVENT_EXTENT_PROTOCOL.json'
    assert digest(frozen)==summary['protocol_sha256'] and json.loads(frozen.read_text())==cfg
    assert digest(source/'tools/diagnose_event_extent.py')==summary['script_sha256']
    z=np.load(a.out/'per_clip.npz'); ids=z['ids'];u=z['users'];p=z['probability'];valid=z['valid'];bounds=z['bounds']
    y=np.load(a.data/'cache/y_clutterremoved.npy');users=np.load(a.data/'cache/subj_clutterremoved.npy')
    wanted=np.flatnonzero(quality_mask(y,users)&np.isin(users,cfg['source_users']))
    np.testing.assert_array_equal(ids,wanted);np.testing.assert_array_equal(u,users[ids])
    assert len(ids)==len(set(ids.tolist()))==7100
    assert p.shape==(7100,3,2,89) and valid.shape==(7100,3,2) and bounds.shape==(7100,3,3,2)
    assert np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.sum(-1)[valid],1,atol=1e-14)
    assert (p.sum(-1)[~valid]==0).all()
    checked=0
    for start in range(0,len(ids),128):
        pp=p[start:start+128].reshape(-1,2,89);vv=valid[start:start+128].reshape(-1,2)
        for j in range(3):
            ref=exhaustive_windows(pp[:,j:j+1] if j<2 else pp,vv[:,j:j+1] if j<2 else vv,cfg['retained_mass'])
            np.testing.assert_array_equal(ref,bounds[start:start+128,:,j].reshape(-1,2));checked+=len(ref)
    frames=bounds[...,1]-bounds[...,0]+1; screen={}
    for v,name in enumerate(cfg['views']):
        row=summary['views'][name];pass_users=[]
        for h in cfg['source_users']:
            mask=u==h;reference=row['by_user'][str(h)]
            good=valid[mask,v].all(-1);short=good&(frames[mask,v,2]<=cfg['screen']['max_frames'])
            assert reference['records']==int(mask.sum()) and reference['both_proxies_valid']==int(good.sum())
            rate=float(short.sum()/mask.sum());assert rate==reference['joint_short_fraction_all_records']
            np.testing.assert_allclose(np.percentile(frames[mask,v], [10,50,90],axis=0),reference['frame_length_quantiles_phase_amp_joint'],atol=1e-12)
            if rate>=cfg['screen']['min_fraction']: pass_users.append(h)
        assert pass_users==row['screen_passed_users']
        screen[name]=len(pass_users)>=cfg['screen']['min_users'];assert screen[name]==row['screen_passed']
        aa=bounds[:,v,0].astype(int);bb=bounds[:,v,1].astype(int)
        intersection=np.clip(np.minimum(aa[:,1],bb[:,1])-np.maximum(aa[:,0],bb[:,0]),0,None)
        union=(aa[:,1]-aa[:,0])+(bb[:,1]-bb[:,0])-intersection
        np.testing.assert_allclose(np.percentile(intersection/union,[10,50,90]),row['phase_amp_interval_iou_quantiles'],atol=1e-14)
        assert np.median(abs(aa[:,0]-bb[:,0]))==row['phase_amp_start_absolute_difference_median']
        assert np.median(abs(aa[:,1]-bb[:,1]))==row['phase_amp_end_absolute_difference_median']
        np.testing.assert_allclose(np.stack((p[:,v,:,:9].sum((0,2))/len(ids),p[:,v,:,-9:].sum((0,2))/len(ids))),row['first_last9_transition_mass_mean_phase_amp'],atol=1e-14)
    assert summary['invalid_proxy_cases']==int((~valid).sum())
    report={'state':'passed','records':len(ids),'independent_exhaustive_intervals':checked,
        'all_summary_quantiles_boundaries_edge_mass_and_screens_recomputed':True,'screen':screen,
        'per_clip_sha256':digest(a.out/'per_clip.npz'),'audit_script_sha256':digest(__file__),
        'limits':'Full evidence extraction not independently rerun; first16 raw clips checked against scipy in production. Exhaustive interval audit covers every saved probability series.'}
    dump(a.out/'POST_AUDIT.json',report);print(json.dumps(report))


if __name__=='__main__':main()
