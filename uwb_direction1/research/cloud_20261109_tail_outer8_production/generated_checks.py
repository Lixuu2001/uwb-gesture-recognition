"""Full native generated replay/BN/roles/journal checks; no UWB truth values."""
from pathlib import Path
import copy,importlib.util,hashlib
import core as e
from role_data import RoleLabels
from outer_subject_roles import plan_roles
from worker import fit
from journal import StepLog,load as journal_load
np=e.np;torch=e.torch
class Fixture:
    def __init__(self):
        self.held=1;users=np.repeat(np.arange(1,9),2);ids=np.arange(100,116)
        self.roles=plan_roles(users.tolist(),ids.tolist(),1,8)
        self.x=np.random.default_rng(71).normal(size=(16,90,567)).astype('float32')
        self.calls=[]
        def labels(ids):self.calls.append(ids.tolist());return (ids%12).astype('int64')
        self.labels=RoleLabels(self.roles,labels)
    def batch(self,role,ids):
        if not set(ids.tolist())<=set(self.roles['sample_ids'][role]):raise PermissionError('Fixture role')
        return np.array(self.x[ids-100],copy=True)
def main():
    import sys
    label=sys.argv[1] if len(sys.argv)>1 else 'generated_native64'
    out=e.ROOT/label;out.mkdir(exist_ok=False);d=Fixture();blocked=[];before=list(d.calls)
    for name,role,ids in [('held_truth','test',[100]),('crossrole_truth','train',[100]),('float_ID','train',[104.])]:
        try:d.labels.get(role,np.array(ids))
        except (ValueError,PermissionError):blocked.append(name)
        else:raise AssertionError(name)
    assert d.calls==before
    # Literal pinned averaging component and unchanged native update trajectory.
    path=Path('/workspace/uwb-swad-loso6-20261108/author_reference/domainbed/lib/swa_utils.py')
    receipt=e.read('/workspace/uwb-swad-loso6-20261108/AUTHOR_SOURCE_RECEIPT.json')
    expected=next(row['sha256'] for row in receipt['files'] if row['path']=='domainbed/lib/swa_utils.py')
    assert e.sha(path)==expected
    # Verify immutable component via the full existing frozen manifest instead of file-order assumptions.
    f=e.read('/workspace/uwb-swad-loso6-20261108/FROZEN_HASHES.json')
    for n,h in f.items():assert e.sha(Path('/workspace/uwb-swad-loso6-20261108')/n)==h
    spec=importlib.util.spec_from_file_location('literal_average',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    m,o,t=e.make();literal=mod.AveragedModel(m)
    with torch.no_grad():
        for i in range(3):
            for p in m.parameters():p.add_(.001*(i+1))
            snapshot=e.semantic(m.state_dict());rng=torch.get_rng_state().clone();t.add(m,i+1);literal.update_parameters(m,step=i+1)
            assert snapshot==e.semantic(m.state_dict()) and torch.equal(rng,torch.get_rng_state())
            for a,b in zip(t.model.parameters(),literal.module.parameters()):assert torch.equal(a,b)
    m,o,t=e.make();x=d.batch('train',np.array([104,105]));y=d.labels.get('train',np.array([104,105]))
    row=e.step(m,o,t,x,y,True);state=e.semantic(m.state_dict());opt=e.semantic(o.state_dict())
    plain,po,_=e.make();plain.train();z=plain(torch.from_numpy(x));po.zero_grad();e.bce(z,y).backward();torch.nn.utils.clip_grad_norm_(plain.parameters(),2.);po.step()
    assert state==e.semantic(plain.state_dict()) and opt==e.semantic(po.state_dict())
    fit(out/'train',d,epochs=64,tail_start=33,generated=True)
    fit(out/'replay',d,epochs=64,tail_start=33,replay_ref=out/'train',generated=True)
    a,pa=journal_load(out/'train');b,pb=journal_load(out/'replay');assert e.semantic(a)==e.semantic(b)
    # Journal refuses damaged committed prefix; a noncommitted matching suffix is reused exactly.
    logpath=out/'suffix.jsonl';log=StepLog(logpath);log.append(dict(update=1,value='committed'));prefix=log.prefix();log.append(dict(update=2,value='uncommitted'));log.close()
    log=StepLog(logpath,1,prefix['prefix_bytes'],prefix['prefix_sha256']);log.append(dict(update=2,value='uncommitted'));log.close()
    log=StepLog(logpath,1,prefix['prefix_bytes'],prefix['prefix_sha256'])
    try:log.append(dict(update=2,value='changed'))
    except AssertionError:blocked.append('changed_uncommitted_journal_suffix')
    else:raise AssertionError('journal suffix')
    finally:log.close()
    try:fit(out/'train',d,epochs=64,tail_start=33,generated=True)
    except FileExistsError:blocked.append('completed_fold_refit')
    else:raise AssertionError('duplicate fit')
    assert all(set(ids)<=set(d.roles['sample_ids']['train']+d.roles['sample_ids']['validation']) for ids in d.calls)
    e.write(e.ROOT/(label.upper()+'_CHECK.json'),dict(passed=True,sources={n:e.sha(e.ROOT/n) for n in ['core.py','role_data.py','worker.py','lazy_bn.py','journal.py','generated_checks.py','lazy_outer_data.py','outer_subject_roles.py']},blocked=blocked,native_parameters=617612,full_native_generated_train_and_fresh_replay=True,epochs_each=64,initial_and_all_step_optimizer_tail_RNG_journal_and_3arm_outputs_exact=True,pinned_average_arithmetic_exact=True,passive_average_native_trajectory_exact=True,source_only_BN_fullpass_no_parameter_change_and_RNG_restore=True,NumPy_first_gradient_audit=True,actual_UWB_truth_read=False,actual_outer8_training_started=False,production_admitted=False))
    print('PASS: generated full-native replay, first gradient, passive mean, BN, truth firewall and journal')
if __name__=='__main__':main()
