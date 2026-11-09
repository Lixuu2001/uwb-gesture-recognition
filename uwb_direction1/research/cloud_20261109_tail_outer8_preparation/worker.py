"""Native full8 production/fresh replay and sealed predictions; never held truth."""
from pathlib import Path
import copy,fcntl,json,hashlib,sys,time,math
import core as e
from journal import StepLog,commit,load as journal_load
from lazy_bn import calibrate
np=e.np;torch=e.torch
def evaluate(m,data,role):
    ids=np.asarray(data.roles['sample_ids'][role],dtype=np.int64);m.eval();chunks=[]
    with torch.inference_mode():
        for i in range(0,len(ids),64):chunks.append(m(torch.from_numpy(data.batch(role,ids[i:i+64]))).numpy())
    z=np.concatenate(chunks)
    if z.shape!=(len(ids),12) or not np.isfinite(z).all():raise ValueError('Output shape/finite failure')
    if role!='validation':return z,None,None
    y=data.labels.get('validation',ids);macro=float(np.mean([np.mean(z.argmax(1)[y==k]==k) for k in np.unique(y)]))
    return z,macro,float(e.bce(torch.from_numpy(z),y))
def first_audit(initial,actual,x,y):
    """Independent BCE derivative; separate fresh model and all global RNG restored."""
    py=e.random.getstate();nr=copy.deepcopy(np.random.get_state())
    with torch.random.fork_rng(devices=[]):
        m,o,t=e.make();e.restore(initial,m,o,t);m.train();z=m(torch.from_numpy(x));zz=z.detach().numpy().astype('float64')
        hot=np.eye(12)[y];loss=float((np.logaddexp(0,zz)-hot*zz).mean());gradient=(1/(1+np.exp(-zz))-hot)/(len(y)*12)
        grads=torch.autograd.grad(z,tuple(m.parameters()),grad_outputs=torch.as_tensor(gradient,dtype=z.dtype))
        # Compare independent clipped derivative with a separate native BCE backward.
        e.restore(initial,m,o,t);m.train();native_z=m(torch.from_numpy(x));o.zero_grad();e.bce(native_z,y).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.)
        # Recompute independent clipped gradients; first grads tuple remains unclipped.
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g,2) for g in grads]),2);scale=torch.clamp(2./(norm+1e-6),max=1.)
        differences=[]
        for p,g in zip(m.parameters(),grads):
            torch.testing.assert_close(p.grad,g*scale,rtol=2e-4,atol=2e-6);differences.append(float((p.grad-g*scale).abs().max()))
        assert abs(loss-actual['loss'])<2e-6
        o.step();assert e.semantic(m.state_dict())==actual['model_sha256'] and e.semantic(o.state_dict())==actual['optimizer_sha256']
    e.random.setstate(py);np.random.set_state(nr)
    return dict(state='NumPy_first_BCE_derivative_and_native_optimizer_replay_verified',max_gradient_difference=max(differences),reference_loss=loss,parameters=617612,held_truth_read=False,does_not_exclude_all_shared_errors=True)
def save_outputs(out,m,t,state,data,replay_ref,generated):
    arms={'native':state['best_model'],'tail':t.model.state_dict(),'native_sourceBN':state['best_model']}
    costs={};files={}
    for arm,weights in arms.items():
        model=copy.deepcopy(m);model.load_state_dict(weights);model.eval()
        if arm!='native':costs[arm]=calibrate(model,data)
        chosen=dict(model=copy.deepcopy(model.state_dict()),arm=arm,held=data.held,native_selected_epoch=state['best_epoch'],tail_start=t.start,tail_end=t.end,tail_count=t.n)
        path=out/f'selected_{arm}.pt';e.atomic_save(path,chosen)
        if replay_ref:
            old=torch.load(replay_ref/path.name,map_location='cpu',weights_only=True);assert e.semantic(old)==e.semantic(chosen)
        files[path.name]=e.sha(path)
        for role in ['validation','test']:
            # Validation here is already selected; output-only pass needn't request labels.
            ids=np.asarray(data.roles['sample_ids'][role],dtype=np.int64);model.eval();zs=[]
            with torch.inference_mode():
                for i in range(0,len(ids),64):zs.append(model(torch.from_numpy(data.batch(role,ids[i:i+64]))).numpy())
            z=np.concatenate(zs);assert np.isfinite(z).all()
            f=out/f'{arm}_{role}_unscored.npz';np.savez_compressed(f,ids=ids,scores=z);files[f.name]=e.sha(f)
            if replay_ref:
                with np.load(replay_ref/f.name) as old:np.testing.assert_array_equal(old['ids'],ids);np.testing.assert_array_equal(old['scores'],z)
    e.write(out/'CALIBRATION_COST.json',costs);return files
def fit(out,data,epochs=64,tail_start=33,replay_ref=None,resume=False,generated=False):
    if not generated and (epochs!=64 or tail_start!=33):raise ValueError('Real training protocol fixed')
    out=Path(out);out.mkdir(parents=True,exist_ok=resume)
    if (out/'COMPLETE.json').exists():raise FileExistsError('Completed fold cannot be refit')
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m,o,t=e.make();ids=np.array(data.roles['sample_ids']['train'],dtype=np.int64);y=data.labels.get('train',ids)
    state,p=journal_load(out);checkpoint_cost=[];reference=(Path(replay_ref)/'steps.jsonl').open() if replay_ref else None
    if state is None:
        state=e.capture(m,o,t,0);e.atomic_save(out/'initial.pt',state);log=StepLog(out/'steps.jsonl')
        commit(out,state,dict(epoch=0,updates=0,state_sha256=e.semantic(state)),log.prefix())
    else:
        if not resume:raise ValueError('Existing journal needs explicit recovery')
        e.restore(state,m,o,t);log=StepLog(out/'steps.jsonl',state['updates'],p['prefix_bytes'],p['prefix_sha256'])
    if reference:
        for _ in range(state['updates']):assert reference.readline()
    best_key=state['best_key'];best_model=state['best_model'];best_epoch=state['best_epoch']
    try:
        for epoch in range(state['epoch']+1,epochs+1):
            order=e.order(len(ids),epoch);total=0.
            for offset in range(0,len(ids),64):
                pos=order[offset:offset+64];batch_ids=ids[pos];x=data.batch('train',batch_ids)
                result=e.step(m,o,t,x,y[pos],epoch>=tail_start)
                row=dict(epoch=epoch,offset=offset,training_ids_sha256=hashlib.sha256(batch_ids.tobytes()).hexdigest(),**result)
                if reference:assert json.loads(reference.readline())==row,'Fresh replay step mismatch'
                log.append(row);total+=result['loss']*len(pos)
                if o.count==1:
                    e.atomic_save(out/'first_update.pt',e.capture(m,o,t,0))
                    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
                    audit=first_audit(initial,result,x,y[pos]);e.write(out/'FIRST_AUDIT.json',audit)
            z,vm,vl=evaluate(m,data,'validation');key=(vm,-vl,-epoch)
            if best_key is None or key>tuple(best_key):best_key=key;best_model=copy.deepcopy(m.state_dict());best_epoch=epoch
            state=e.capture(m,o,t,epoch,best_key,best_model,best_epoch)
            rec=dict(epoch=epoch,updates=o.count,train_global_BCE=total/len(ids),validation_present_macro=vm,validation_global_BCE=vl,
                validation_logits_sha256=hashlib.sha256(z.tobytes()).hexdigest(),state_sha256=e.semantic(state))
            if reference:assert rec==e.read(Path(replay_ref)/f'epoch{epoch:03}.json'),'Fresh replay epoch mismatch'
            wt=time.monotonic();ct=time.process_time();commit(out,state,rec,log.prefix());checkpoint_cost.append(dict(wall=time.monotonic()-wt,CPU=time.process_time()-ct))
            print(json.dumps(dict(epoch=epoch,updates=o.count,held=data.held,stage='fresh' if reference else 'production')),flush=True)
        assert o.count==epochs*math.ceil(len(ids)/64) and t.n==(epochs-tail_start+1)*math.ceil(len(ids)/64)
        if reference:assert reference.read()==''
        terminal=e.semantic(state);e.atomic_save(out/'final_full.pt',state)
        files=save_outputs(out,m,t,state,data,replay_ref,generated)
        assert e.semantic(e.capture(m,o,t,epochs,best_key,best_model,best_epoch))==terminal
        e.write(out/'CHECKPOINT_COST.json',checkpoint_cost)
        e.write(out/'COMPLETE.json',dict(state='frozen_unscored_fresh_exact' if reference else 'frozen_unscored_production',epochs=epochs,updates=o.count,tail_count=t.n,tail_start_update=t.start,tail_end_update=t.end,selected_native_epoch=best_epoch,final_state_sha256=terminal,files=files,held_truth_read=False,generated_fixture=generated))
    finally:
        log.close();lock.close()
        if reference:reference.close()
def main():
    from role_data import RealData
    from admission import require_production
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','replay']);p.add_argument('--held',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    require_production(a.held);data=RealData(a.held,production_release=e.ROOT/'PRODUCTION_RELEASE.json');base=e.ROOT/'folds'/f'held{a.held}'
    fit(base/a.mode,data,replay_ref=base/'train' if a.mode=='replay' else None,resume=a.resume)
if __name__=='__main__':main()
