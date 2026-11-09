"""Rebuild all output states from final journal; held truth remains inaccessible."""
import argparse,copy
from pathlib import Path
import core as e
from admission import require_production
from role_data import RealData
from journal import load
from lazy_bn import calibrate
torch=e.torch;np=e.np
def main():
    p=argparse.ArgumentParser();p.add_argument('--held',type=int,required=True);a=p.parse_args();require_production(a.held)
    data=RealData(a.held,production_release=e.ROOT/'PRODUCTION_RELEASE.json');out=e.ROOT/'folds'/f'held{a.held}'
    st,_=load(out/'train');fresh,_=load(out/'replay');assert st['epoch']==fresh['epoch']==64 and e.semantic(st)==e.semantic(fresh)
    assert st['updates']==64*((len(data.roles['sample_ids']['train'])+63)//64)
    assert st['tail']['n']==st['updates']//2
    states={'native':st['best_model'],'tail':st['tail']['model'],'native_sourceBN':st['best_model']};files={}
    for arm,weights in states.items():
        model,_,_=e.make();model.load_state_dict(weights);model.eval()
        if arm!='native':calibrate(model,data)
        for side in ['train','replay']:
            path=out/side/f'selected_{arm}.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
            assert e.semantic(saved['model'])==e.semantic(model.state_dict());files[str(path)]=e.sha(path)
        for role in ['validation','test']:
            ids=np.array(data.roles['sample_ids'][role],dtype=np.int64);model.eval();zs=[]
            with torch.inference_mode():
                for i in range(0,len(ids),64):zs.append(model(torch.from_numpy(data.batch(role,ids[i:i+64]))).numpy())
            scores=np.concatenate(zs)
            for side in ['train','replay']:
                f=out/side/f'{arm}_{role}_unscored.npz'
                with np.load(f) as z:np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['scores'],scores)
                files[str(f)]=e.sha(f)
    assert not data.labels.reads # even validation labels unnecessary in output-only audit.
    for side in ['train','replay']:
        for n in ['initial.pt','first_update.pt','final_full.pt','COMPLETE.json','FIRST_AUDIT.json']:
            f=out/side/n;files[str(f)]=e.sha(f)
    e.write(out/'PREDICTIONS_FROZEN.json',dict(all3_arms_prod64_fresh64_audited_unscored=True,held=a.held,files=files,all_output_audit_gesture_label_reads=0,held_truth_read=False,accuracy_or_macro_scores_computed=False,training_journal_final_state_exact=True))
if __name__=='__main__':main()
