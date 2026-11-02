"""Small actual-model continuation/recovery comparison; no formal fit."""
import copy
import json
from pathlib import Path
import torch
from runner import T,AugmentReader,prepare_cache,mild_epoch,HERE

def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    path=HERE/'smoke/AUG/fold1/warmup/progress.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    identity=saved['info']['identity']; config=identity['config']
    cache,_=prepare_cache(); reader,y,s,parts=T.protocol.open_fold(1)
    reader=AugmentReader(reader,cache,True)
    matrix,_=T.smoothing.prepare_matrix(1,T.HERE/'runs_s2026/matrices',identity['source']['data'])
    matrix=torch.as_tensor(matrix,device='cuda')
    criterion=lambda z,labels:T.smoothing.classification_loss(z,labels,matrix)
    model=T.models.build_model(config['seed'],'cuda'); opt=T.optimizer_for(model,'warmup',config)
    sampler=T.state.ShuffleSampler(parts['train'],config['seed']+1)
    T.state.load_progress(path,model,opt,sampler,identity)
    # Run a real extra epoch, save, then continue another without interruption.
    row1=mild_epoch(model,opt,sampler,reader,y,criterion,config)
    checkpoint=HERE/'smoke/recovery_checkpoint.pt'
    T.state.save_progress(checkpoint,model,opt,sampler,{'identity':identity})
    row_a=mild_epoch(model,opt,sampler,reader,y,criterion,config)
    hash_a=T.state.state_hash(model.state_dict())
    logits_a,_=T.evaluate(model,reader,y,parts['val'],'val',32)
    # Recreate all objects, restore the saved state, repeat that same next epoch.
    del model,opt
    model=T.models.build_model(config['seed'],'cuda'); opt=T.optimizer_for(model,'warmup',config)
    sampler=T.state.ShuffleSampler(parts['train'],config['seed']+1)
    T.state.load_progress(checkpoint,model,opt,sampler,identity)
    row_b=mild_epoch(model,opt,sampler,reader,y,criterion,config)
    hash_b=T.state.state_hash(model.state_dict())
    logits_b,_=T.evaluate(model,reader,y,parts['val'],'val',32)
    import numpy as np
    result={'state_hash_equal':hash_a==hash_b,'next_epoch_row_equal':row_a==row_b,
            'validation_max_abs_error':float(np.max(np.abs(logits_a-logits_b))),
            'uninterrupted_hash':hash_a,'resumed_hash':hash_b,
            'augmentation_trace':row_a['augmentation_trace'],'augmentation_counts':row_a['augmentation_counts']}
    assert result['state_hash_equal'] and result['next_epoch_row_equal'] and result['validation_max_abs_error']==0
    T.protocol.dump(HERE/'smoke/RECOVERY_CHECK.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
