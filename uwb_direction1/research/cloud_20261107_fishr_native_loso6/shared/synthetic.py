"""Generated fullnative checks: independent all-head autograd, full state, real reset boundary."""
from pathlib import Path
from collections import OrderedDict
import ast,copy,json,tempfile,time,os,subprocess,sys,signal
from common import B,PHASE,ready,write,sha
import engine as e
from journal import StepLog,commit,load
from controller import identity
from loso_roles_metrics import roles,classification_metrics,paired_sixfold
t=e.torch

def defaults():
    tree=ast.parse((PHASE/'author_reference/domainbed/hparams_registry.py').read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='algorithm' and len(n.test.comparators)==1 and isinstance(n.test.comparators[0],ast.Constant) and n.test.comparators[0].value=='Fishr')
    vals={n.args[0].value:ast.literal_eval(n.args[1]) for statement in branch.body for n in ast.walk(statement) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_hparam'}
    assert vals==dict(**{'lambda':1000.,'ema':.95,'penalty_anneal_iters':1500})
    assert e.c['fishr_lambda']==1000 and e.c['fishr_ema']==.95 and e.c['fishr_reset_at_completed_updates']==1500
    return vals

def native_warmup_and_active():
    m,o,g=e.make();assert e.semantic(m.state_dict())==e.c['matched_initial_model_state_sha256']
    rng=e.np.random.default_rng(931);x=rng.normal(size=(8,90,567)).astype('float32');y=e.np.arange(8)%12
    domains=e.np.array([e.c['train_users'][i] for i in [0,0,0,1,1,2,2,3]]);labels=t.as_tensor(y,dtype=t.long)
    initial=copy.deepcopy(e.capture(m,o,g,0,0));m.train()
    logits=m(t.from_numpy(x));loss=e.native_bce(logits,labels);o.zero_grad();loss.backward();t.nn.utils.clip_grad_norm_(m.parameters(),2.);grads=[p.grad.detach().clone() for p in m.parameters()];o.step()
    expected=copy.deepcopy(e.capture(m,o,g,0,1))
    e.restore(initial,m,o,g);m.train();row=e.step(m,o,x,y,1,domains)
    assert row['fishr_penalty_weight']==0 and row['completed_updates_after']==1 and not row['optimizer_reset_this_update']
    for a,b in zip(grads,[p.grad for p in m.parameters()]):assert t.equal(a,b),'native warmup clipped gradient changed'
    actual=copy.deepcopy(e.capture(m,o,g,0,1));ema=actual['optimizer'].pop('fishr_EMA');expected['optimizer'].pop('fishr_EMA')
    assert e.semantic(actual)==e.semantic(expected),'warmup model/BN/nativeopt/allRNG differs'
    assert ema['updates']==[1]*4
    # Independent per-example autograd on the complete stochastic native forward,
    # all four classifier tensors. This is a test only, never the production loop.
    e.restore(initial,m,o,g);o.algorithm_updates=1500;o.inner.count=1500;m.train()
    before=copy.deepcopy(e.capture(m,o,g,0,1500))
    z=m(t.from_numpy(x));example=t.nn.functional.binary_cross_entropy_with_logits(z,t.nn.functional.one_hot(labels,12).float(),reduction='none').mean(1)
    head=list(m.head.named_parameters());individual=[t.autograd.grad(v,[p for _,p in head],create_graph=True,retain_graph=True) for v in example]
    gradients=OrderedDict((n,t.stack([gs[k].reshape(-1) for gs in individual])) for k,(n,_) in enumerate(head))
    # Execute the pinned author's actual EMA and flattened-MSE helpers.
    tr=ast.parse((PHASE/'author_reference/domainbed/lib/misc.py').read_text());nodes=[n for n in tr.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['MovingAverage','l2_between_dicts']]
    ns=dict(torch=t,collections=__import__('collections'),OrderedDict=OrderedDict)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_author_misc_for_fullnative_check','exec'),ns)
    moving=[ns['MovingAverage'](.95) for _ in range(4)];mom=[]
    for group in range(4):
        ix=t.as_tensor(domains==e.c['train_users'][group]);cur=OrderedDict()
        for n,v in gradients.items():
            a=v[ix];cur[n]=(a-a.mean(0)).pow(2).mean(0)
        mom.append(moving[group].update(cur))
    mean=OrderedDict((n,t.stack([d[n] for d in mom]).mean(0)) for n in gradients)
    penalty=t.stack([ns['l2_between_dicts'](d,mean) for d in mom]).mean()
    classification=e.native_bce(z,labels);objective=classification+1000*penalty
    pg=t.autograd.grad(penalty,list(m.feature_extractor.parameters()),retain_graph=True,allow_unused=True)
    assert any(v is not None and v.abs().sum()>0 for v in pg),'Fishr penalty disconnected from full backbone'
    assert o.reset_if_due();o.zero_grad();objective.backward();t.nn.utils.clip_grad_norm_(m.parameters(),2.);reference=[p.grad.detach().clone() for p in m.parameters()];o.step()
    model_expected=e.semantic(m.state_dict());rng_expected=t.get_rng_state().clone()
    e.restore(before,m,o,g);m.train();result=e.step(m,o,x,y,1,domains)
    assert result['fishr_penalty_weight']==1000 and result['optimizer_reset_this_update'] and o.count==1 and o.reset_events==[1500]
    assert abs(result['penalty']-float(penalty.detach()))<=1e-11
    differences=[]
    for a,b in zip(reference,[p.grad for p in m.parameters()]):
        t.testing.assert_close(a,b,rtol=2e-4,atol=2e-6);differences.append(float((a-b).abs().max()))
    assert t.equal(t.get_rng_state(),rng_expected)
    for k in mean:t.testing.assert_close(o.ema.data[0][k].reshape(-1),moving[0].ema_data[k].reshape(-1),rtol=2e-5,atol=1e-10)
    return dict(warmup_native_bitwise=True,full_native_active_independent_classifier_autograd_and_pinned_author_EMA=True,max_clipped_gradient_abs_difference=max(differences),penalty=float(penalty.detach()),penalty_reaches_backbone=True,active_model_semantic_bitwise_same=e.semantic(m.state_dict())==model_expected,reference_is_tolerance_check_not_claim_entire_author_protocol_reproduced=True)

def reset_and_resume():
    p=t.nn.Parameter(t.tensor([.3,-.8]));o=e.ScheduledNativeOptimizer([p])
    for j in range(1501):
        o.ema.update(0,OrderedDict(a=p.square()))
        oldema=e.semantic(o.ema.state_dict());weight=o.penalty_weight();reset=o.reset_if_due()
        assert weight==(0 if j<1500 else 1000) and reset==(j==1500)
        assert e.semantic(o.ema.state_dict())==oldema
        if reset:assert o.count==0 and not o.slow and not o.inner.fast.state
        o.zero_grad();(p.square().sum()).backward();o.step()
        if j==1499:
            saved=copy.deepcopy(o.state_dict());q=t.nn.Parameter(p.detach().clone());oo=e.ScheduledNativeOptimizer([q]);oo.load_state_dict(saved)
            assert e.semantic(oo.state_dict())==e.semantic(saved) and oo.reset_if_due() and oo.count==0
    assert o.algorithm_updates==1501 and o.count==1 and o.reset_events==[1500] and o.ema.updates==[1501,0,0,0]
    q=t.nn.Parameter(p.detach().clone());oo=e.ScheduledNativeOptimizer([q]);oo.load_state_dict(copy.deepcopy(o.state_dict()));assert not oo.reset_if_due()
    # Full native slow step6, disk restore, next stochastic step exact incl EMA/BN/all RNG.
    m,o,g=e.make();rng=e.np.random.default_rng(812);x=rng.normal(size=(8,90,567)).astype('float32');y=e.np.arange(8)%12;u=e.np.array(e.c['train_users'])[e.np.arange(8)%4]
    for j in range(6):m.train();e.step(m,o,x,y,1,u)
    assert o.count==6 and o.slow and o.ema.updates==[6]*4
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);state=copy.deepcopy(e.capture(m,o,g,0,6));log=StepLog(d/'steps.jsonl')
        for i in range(1,7):log.append(dict(update=i,generated_only=True))
        rec=e.record(state,0,6);commit(d,state,rec,log.prefix())
        m.train();a=e.step(m,o,x,y,2,u);expected=copy.deepcopy(e.capture(m,o,g,1,7));log.append(dict(update=7,generated_only=True));log.close()
        stored,progress=load(d);e.restore(stored,m,o,g)
        suffix_before=(d/'steps.jsonl').read_bytes();log=StepLog(d/'steps.jsonl',6,progress['prefix_bytes'],progress['prefix_sha256']);m.train();b=e.step(m,o,x,y,2,u)
        assert a==b and e.semantic(e.capture(m,o,g,1,7))==e.semantic(expected)
        log.append(dict(update=7,generated_only=True));assert (d/'steps.jsonl').read_bytes()==suffix_before
        commit(d,expected,e.record(expected,1,7),log.prefix());log.close();again,_=load(d);assert e.semantic(again)==e.semantic(expected)
    # Active path disk restore around boundary; includes EMA histories of all4.
    o.algorithm_updates=1500;o.inner.count=1500
    before=copy.deepcopy(e.capture(m,o,g,1,1500));m.train();a=e.step(m,o,x,y,2,u);expected=copy.deepcopy(e.capture(m,o,g,1,1501))
    e.restore(before,m,o,g);m.train();b=e.step(m,o,x,y,2,u);assert a==b and e.semantic(e.capture(m,o,g,1,1501))==e.semantic(expected)
    # Native tail20 and single-domain singleton, absence retains other histories.
    x=rng.normal(size=(20,90,567)).astype('float32');m.train();e.step(m,o,x,e.np.arange(20)%12,3,e.np.array(e.c['train_users'])[e.np.arange(20)%4])
    previous=copy.deepcopy(o.ema.state_dict());m.train();r=e.step(m,o,x[:1],e.np.array([0]),3,e.np.array([e.c['train_users'][0]]))
    assert r['penalty']==0 and o.ema.updates[0]==previous['updates'][0]+1 and o.ema.updates[1:]==previous['updates'][1:]
    assert e.semantic(o.ema.data[1:])==e.semantic(previous['data'][1:])
    m.eval();before=e.semantic(e.capture(m,o,g,2,o.algorithm_updates))
    with t.inference_mode():out=m(t.from_numpy(x[:8]))
    assert out.shape==(8,12) and e.semantic(e.capture(m,o,g,2,o.algorithm_updates))==before
    return dict(actual_scalar_optimizer_updates=1501,reset_entire_fast_and_slow_once=True,EMA_preserved_at_reset=True,fullnative_disk_EMA_BN_allRNG_recovery_bitwise=True,two_slot_journal_prefix_and_uncommitted_suffix_checked=True,active_boundary_fullnative_resume_exact=True,tail20_and_singleton_missing_groups=True,inference_model_buffers_unchanged=True)

def firewall_and_supervisor():
    # Generated metadata, every sixfold role and exact expected counts.
    users=e.np.repeat(e.np.arange(1,7),[1200,1000,1200,1200,1200,1100]);allids=e.np.arange(len(users))
    for h in range(1,7):
        val,loc=roles(users,h);cfg=json.loads((PHASE/f'folds/held{h}/PROTOCOL.json').read_text());assert val==cfg['validation_user'] and [len(v) for v in loc.values()]==cfg['counts']
        for a,b in [('train','validation'),('train','diagnostic'),('validation','diagnostic')]:assert not set(allids[loc[a]])&set(allids[loc[b]])
        assert 64*((len(loc['train'])+63)//64)==cfg['expected_updates']
    # Real loader's label slicing AST reads train/validation only.
    tree=ast.parse((PHASE/'shared/data.py').read_text());truths=[n for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='truth']
    assert len(truths)==2 and {'train','validation'}=={n.value for sub in truths for n in ast.walk(sub) if isinstance(n,ast.Constant) and n.value in ['train','validation','diagnostic']}
    metric=classification_metrics(e.np.array([0,0,0,1]),e.np.array([0,0,1,1]));assert metric['accuracy']!=metric['present_class_macro_recall'] and metric['class_recall'][2] is None
    # Own child identity/boot/start guard and process-group cap-stop fixture.
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    original=identity(child.pid);now=identity(child.pid);assert original and all(original[k]==now[k] for k in ['pid','startticks','boot_id'])
    assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=5);assert child.returncode==-signal.SIGTERM
    assert identity(os.getpid())['boot_id']==original['boot_id']
    return dict(generated_all6_roles_counts_disjoint=True,loader_train_val_truth_slices_only=True,accuracy_and_presentmacro_distinct_and_missingnull=True,owned_child_identity_and_group_stop=True,no_real_ids_or_arrays_or_labels_opened=True)

def main():
    ready();start=time.monotonic();cpu=time.process_time();checks={}
    for name,fn in [('pinned_defaults',defaults),('fullnative_independent_reference',native_warmup_and_active),('real_optimizer_boundary_and_fullnative_journal_resume',reset_and_resume),('all6_firewall_metric_supervisor',firewall_and_supervisor)]:checks[name]=fn()
    result=dict(state='pass',all_passed=True,checks=checks,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,users7_8_labels_used=False)
    write(B/'SYNTHETIC_RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
