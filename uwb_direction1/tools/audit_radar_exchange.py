"""Independent replay using actual input-slot exchange and functional CNN."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from semantic_code_eightfold_prepare import functional_hidden
from circular_signal import get_batch
from audit_restored import digest,dump
from energy_area import source_roles


def main(root):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();c=json.loads((root/'PROTOCOL.json').read_text());identity=json.loads((root/'IDENTITY.json').read_text());assert digest(root/'PROTOCOL.json')==identity['protocol_sha256'];manifest=json.loads((root/'FILES.json').read_text())
    for n,h in manifest.items():assert digest(root/n)==h
    for n,h in identity['code_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    files={k:Path(c['data'])/n for k,n in [('raw','X_clutterremoved.npy'),('labels','y_clutterremoved.npy'),('users','subj_clutterremoved.npy')]};files['prepared']=Path(c['prepared'])
    for k,p in files.items():assert digest(p)==c['input_sha256'][k]
    y=np.load(files['labels']);u=np.load(files['users']);cache=np.load(files['prepared'],mmap_mode='r');old=Path(c['readout_reference']);assert digest(old/'FILES.json')==c['readout_manifest_sha256'];rm=json.loads((old/'FILES.json').read_text());q=np.array([1,0,2,3,6,7,4,5,9,8,10,11]);scores={k:[] for k in c['arms']};max_hidden=0.;max_score=0.;count=0
    for h in c['users']:
        d=root/f'held{h}'
        for n in ['models.npz','selection.json','split.json','predictions.npz']:assert digest(old/f'held{h}'/n)==rm[f'held{h}/{n}']
        f=np.load(d/'features.npz');pred=np.load(d/'predictions.npz');truth=np.load(d/'evaluation.npz');m=np.load(old/f'held{h}/models.npz');sel=json.loads((old/f'held{h}/selection.json').read_text())['onehot'];assert json.loads((d/'selection.json').read_text())==dict(inherited_onehot=sel,new_selection=False)
        parts=source_roles(y,u,h);saved=json.loads((d/'split.json').read_text())
        for k,v in parts.items():np.testing.assert_array_equal(v,saved[k])
        ck=Path(c['models'])/f'held{h}/segment_24.pt';assert digest(ck)==json.loads((d/'result.json').read_text())['checkpoint_sha256'];record=torch.load(ck,map_location='cpu',weights_only=False);state={k.removeprefix('module.'):v for k,v in record['average']['weights'].items() if k.startswith('module.')}
        for role in ['val','diagnostic']:
            ids=f[role+'_ids'];np.testing.assert_array_equal(ids,parts[role]);np.testing.assert_array_equal(ids,pred[role+'_ids']);np.testing.assert_array_equal(truth[role+'_y'],y[ids]);aa=[];bb=[]
            for i in range(0,len(ids),64):
                x=get_batch(cache,ids[i:i+64],'legacy')
                with torch.no_grad():a=functional_hidden(x,state).numpy();b=functional_hidden(x[:,[2,1,0]],state).numpy()
                aa.append(a);bb.append(b)
            a=np.concatenate(aa).astype(np.float64);b=np.concatenate(bb).astype(np.float64);count+=len(ids)
            for actual,key in [(a,'hidden'),(b,'swapped_hidden')]:
                max_hidden=max(max_hidden,float(abs(actual-f[role+'_'+key]).max()));np.testing.assert_array_equal(actual,f[role+'_'+key])
            # Independent affine evaluation, no production score/arms/decision helper.
            coef=m['coefs'][sel['selected']];a=((a-m['mean'])/m['scale'])@coef+m['bias'];b=((b-m['mean'])/m['scale'])@coef+m['bias'];out={'ridge':a,'equivariant':.5*a+.5*b[:,q],'invariant':.5*a+.5*b,'mapped_swap':b[:,q]}
            np.testing.assert_array_equal(.5*b+.5*a[:,q],out['equivariant'][:,q])
            for k,v in out.items():
                max_score=max(max_score,float(abs(v-pred[role+'_'+k]).max()));np.testing.assert_allclose(v,pred[role+'_'+k],atol=1e-12,rtol=1e-12)
                if role=='diagnostic':scores[k].append(float(np.mean([np.mean(v.argmax(1)[y[ids]==cl]==cl) for cl in np.unique(y[ids])])))
    summary=json.loads((root/'SUMMARY.json').read_text());passes=[]
    for k,v in scores.items():np.testing.assert_allclose(v,summary['scores'][k],atol=1e-15,rtol=0);assert abs(np.mean(v)-summary['means'][k])<1e-15
    for k,minimum in c['gate']['minimum_mean_gain'].items():
        delta=np.array(scores[c['primary']])-scores[k];r={'mean_gain':float(delta.mean()),'wins':int((delta>1e-12).sum()),'minimum_gain':float(delta.min()),'passed':bool(delta.mean()>=minimum and (delta>1e-12).sum()>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline'])};assert r==summary['comparisons'][k];passes.append(r['passed'])
    assert all(passes)==summary['gate_passed'];out=dict(state='passed',artifact_sha256_checks=len(manifest),clips_two_forward_passes=count,hidden_max_difference=max_hidden,prediction_max_difference=max_score,actual_swapped_input_functional_replay=True,semantic_action_and_gate_recomputed=True,seconds=time.perf_counter()-start);dump(root/'AUDIT.json',out);print(json.dumps(out))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):main(a.run)
