"""Correctly attest final native state without changing frozen worker or artifacts."""
from pathlib import Path
import sys,copy,json,time
S=Path('/workspace/uwb-mixup-gap64-20261105');sys.path.insert(0,str(S))
from common import read,write,sha,ready
import engine as e
from journal import load
def verify(stage):
    assert stage in ['train','replay'];ready(True);d=S/'run'/stage
    original=read(d/'COMPLETE.json');original_hash=sha(d/'COMPLETE.json');state,p=load(d)
    assert original['epochs']==state['epoch']==64 and original['updates']==state['updates']==4736
    assert sha(d/'final_full.pt')==p['checkpoint_sha256']
    canonical=e.semantic(state);assert canonical==read(d/'epoch064.json')['state_sha256']
    m,o,g=e.make();e.restore(state,m,o,g)
    actual=e.capture(m,o,g,64,4736,state['best_key'],state['best_model'],state['best_epoch']);assert e.semantic(actual)==canonical
    selected=e.torch.load(d/'selected.pt',map_location='cpu',weights_only=True)
    assert selected['selected_epoch']==state['best_epoch']==original['selected_epoch']
    assert e.semantic(selected['model'])==e.semantic(state['best_model']) and original['selected_sha256']==sha(d/'selected.pt')
    for role in ['validation','diagnostic']:assert original[role+'_outputs_sha256']==sha(d/(role+'_unscored.npz'))
    records=[read(d/f'epoch{j:03}.json') for j in range(1,65)]
    best=max(records,key=lambda r:(r['validation_macro'],-r['validation_loss'],-r['epoch']))
    assert best['epoch']==state['best_epoch'] and best['model_sha256']==e.semantic(selected['model'])
    transient=copy.deepcopy(state);transient['model']=copy.deepcopy(selected['model'])
    alias_digest=e.semantic(transient)
    assert original['final_state_sha256']==alias_digest,'Not the diagnosed metadata alias; stop'
    raw=(d/'steps.jsonl').read_bytes();assert len(raw.splitlines())==4736 and len(raw)==p['prefix_bytes'] and __import__('hashlib').sha256(raw).hexdigest()==p['prefix_sha256']
    term=read(S/f'run/execution/{stage}/TERMINATION.json');assert term['exit_code']==0 and term['resource_stop_reason'] is None
    result=dict(state='pass_actual_native_final_state_and_selected_alias_metadata_explained',stage=stage,observed_unix=time.time(),epochs=64,updates=4736,selected_epoch=state['best_epoch'],actual_final_native_state_sha256=canonical,actual_checkpoint_file_sha256=p['checkpoint_sha256'],original_completion_record_sha256=original_hash,original_misnamed_final_state_sha256=original['final_state_sha256'],reconstructed_selected_weight_alias_sha256=alias_digest,actual_final_matches_epoch64_journal_record=True,native_model_optimizer_all_RNG_recapture_exact=True,inner_validation_best_model_verified=True,selected_outputs_original_hashes_verified=True,actual_termination=term,original_completion_record_modified=False,models_or_scientific_code_modified=False,new_forward_or_optimizer_update=False,diagnostic_labels_read=False,users7_8_labels_used=False)
    target=d/'COMPLETION_STATE_METADATA_CORRECTION.json'
    if target.exists():
        old=read(target)
        for k,v in result.items():
            if k!='observed_unix':assert old[k]==v,k
        result=old
    else:write(target,result)
    assert sha(d/'COMPLETE.json')==original_hash
    return result
if __name__=='__main__':print(json.dumps(verify(sys.argv[1])))
