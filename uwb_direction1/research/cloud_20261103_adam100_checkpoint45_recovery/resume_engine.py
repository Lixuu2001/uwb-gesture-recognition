"""Registered operational continuation only; no change to the Adam100 recipe."""
from pathlib import Path
import json,shutil
from native_adam100 import np,torch,make,optimizer,step,evaluate,validate_roles
from adam_engine import raw

def restore(saved):
    model=make();opt=optimizer(model)
    model.load_state_dict(saved['model'],strict=True)
    opt.load_state_dict(saved['optimizer'])
    sampler=torch.Generator();sampler.set_state(saved['order_rng'])
    torch.set_rng_state(saved['torch_rng'])
    return model,opt,sampler

def copy_prefix(c,out):
    old=Path(c['operational_recovery']['original_root'])/'run01'
    lines=(old/'steps.jsonl').read_text().splitlines(keepends=True)
    assert len(lines)==10604
    rows=[json.loads(x) for x in lines]
    assert all(x['update']==i+1 for i,x in enumerate(rows))
    assert rows[10574]['epoch']==45 and rows[10575]['epoch']==46
    for p in sorted(old.iterdir()):
        if p.is_file() and p.name!='steps.jsonl':shutil.copy2(p,out/p.name)
    (out/'steps.jsonl').write_text(''.join(lines[:10575]))
    saved=torch.load(out/'final_full.pt',map_location='cpu',weights_only=True)
    assert saved['epoch']==45 and saved['updates']==10575
    return saved,rows[10575:]

def check_suffix(row,suffix):
    k=row['update']-10576
    assert k>=0,'Non-contiguous recovery update'
    if k<len(suffix):assert row==suffix[k],'Interrupted uncommitted suffix mismatch'

def continue_run(maps,loc,y,v,users,saved,on_step,on_epoch,expected_epoch=45):
    validate_roles(users[loc['train']],users[loc['validation']])
    assert saved['epoch']==expected_epoch and saved['updates']==235*expected_epoch
    assert 1<=expected_epoch<100
    model,opt,sampler=restore(saved);updates=saved['updates']
    for state in opt.state.values():assert int(state['step'])==updates
    for epoch in range(expected_epoch+1,101):
        model.train();order=torch.randperm(4700,generator=sampler).numpy()
        for offset in range(0,4700,20):
            positions=order[offset:offset+20]
            loss=step(model,opt,raw(maps,loc['train'][positions]),y[positions]);updates+=1
            on_step(dict(epoch=epoch,offset=offset,update=updates,loss=loss,
                order_sha256=__import__('hashlib').sha256(positions.tobytes()).hexdigest()))
        scores,macro=evaluate(model,
            lambda start,end:raw(maps,loc['validation'][start:end]),v)
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),
            order_rng=sampler.get_state(),torch_rng=torch.get_rng_state(),
            epoch=epoch,updates=updates)
        on_epoch(dict(epoch=epoch,updates=updates,validation_macro=macro,
            selection='fixed_final_epoch100_not_best_validation',
            diagnostic_labels_read=False,users7_8_labels_used=False),scores,state)
    assert updates==23500
    return state
