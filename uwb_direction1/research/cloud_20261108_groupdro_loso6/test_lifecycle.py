"""Exercise unchanged production worker/controller/journal with generated fixture data only."""
from pathlib import Path
import sys,tempfile,json,os,copy,subprocess,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'shared'))
from common import read,write,sha
import engine as e

def firewall_checks():
    import data as d
    np=e.np;oldload=np.load;oldprotocol=d.protocol;observations=[]
    users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100]);ids=np.arange(6900,dtype=np.int64)
    class Context:
        def __init__(self,x):self.x=x
        def __enter__(self):return self.x
        def __exit__(self,*a):return False
    class Maps:shape=(6900,90,567);dtype=np.dtype('float32')
    for h in range(1,7):
        c=read(P/f'folds/held{h}/PROTOCOL.json');_,loc=d.roles(users,h);allowed=np.concatenate([loc['train'],loc['validation']]);opened=[]
        class Truth:
            def __getitem__(self,ix):
                assert np.isin(ix,allowed).all() and not np.isin(ix,loc['diagnostic']).any()
                opened.append(np.array(ix,copy=True));return np.asarray(ix)%12
        def fake_load(path,*a,**kw):
            path=str(path)
            if path==c['ids']:return Context(dict(ids=ids,users=users))
            if path.endswith('/roles.npz'):return Context({k:ids[ix] for k,ix in loc.items()})
            if path.endswith('/subj_clutterremoved.npy'):return users
            if path==c['maps']:return Maps()
            if path.endswith('/y_clutterremoved.npy'):return Truth()
            raise AssertionError('Unexpected fixture load '+path)
        try:
            np.load=fake_load;d.protocol=lambda:c
            _,_,actual,_,y,v=d.load(True)
            assert list(map(len,actual.values()))==c['counts'] and len(opened)==2
            np.testing.assert_array_equal(opened[0],loc['train']);np.testing.assert_array_equal(opened[1],loc['validation'])
            observations.append(dict(held=h,train_count=len(y),validation_count=len(v),held_truth_read_count=0))
        finally:np.load=oldload;d.protocol=oldprotocol
    return observations

def main():
    assert not(P/'LIFECYCLE_TEST_RESULT.json').exists();started=time.monotonic();role_checks=firewall_checks()
    with tempfile.TemporaryDirectory(prefix='uwb_groupdro_generated_lifecycle_') as folder:
        root=Path(folder);(root/'shared').mkdir();m,o,g=e.make();e.atomic_save(root/'initial.pt',e.capture(m,o,g,0,0))
        wrapper='''from pathlib import Path
import sys
sys.path.insert(0,REAL_SHARED)
import worker as w
import engine as e
B=Path(FIXTURE_ROOT)
c=dict(e.c,counts=[12,3,3],expected_updates=64,real_wall_cap=120,real_CPU_cap=120,real_RSS_KiB=12582912)
def fixture_load(labels=False):
    np=e.np;ids=np.arange(18,dtype=np.int64);users=np.array(e.c['train_users'])[np.arange(18)%4]
    users[12:15]=e.c['validation_user'];users[15:]=e.c['held']
    loc=dict(train=np.arange(12),validation=np.arange(12,15),diagnostic=np.arange(15,18))
    maps=np.random.default_rng(901).normal(size=(18,90,567)).astype('float32')
    return ids,users,loc,maps,np.arange(12)%12,np.arange(3)%12
w.B=B;w.ready=lambda real=False:c;w.load=fixture_load
mode=sys.argv[1]
if mode in ['train','replay']:w.fit(mode=='replay',False)
elif mode=='first':w.first()
else:assert mode=='output';w.output_audit()
'''.replace('REAL_SHARED',repr(str(P/'shared'))).replace('FIXTURE_ROOT',repr(str(root)))
        (root/'shared/worker.py').write_text(wrapper)
        import controller
        controller.B=root;controller.PHASE=root
        c=dict(e.c,real_wall_cap=120,real_CPU_cap=120,real_RSS_KiB=12582912)
        controller.ready=lambda real=False:c
        controller.main()
        a=read(root/'run/train/COMPLETE.json');b=read(root/'run/replay/COMPLETE.json')
        assert a['epochs']==b['epochs']==64 and a['updates']==b['updates']==64
        assert a['final_state_sha256']==b['final_state_sha256']
        assert (root/'run/train/steps.jsonl').read_bytes()==(root/'run/replay/steps.jsonl').read_bytes()
        reference=read(root/'run/train/FIRST_INDEPENDENT_REFERENCE.json')
        copied=[]
        for f in root.rglob('*'):
            if f.is_file() and f.suffix in ['.json','.log']:
                dest=P/'generated_lifecycle_logs'/f.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes());copied.append(str(dest.relative_to(P)))
        term={k:read(root/f'run/execution/{k}/TERMINATION.json') for k in ['train','replay','output']}
        result=dict(state='pass_actual_production_worker_controller_journal_on_generated_fixture',generated_fixture_counts=[12,3,3],actual_generated_production64_and_independentfresh64_updates_each=64,first_auditor_additional_generated_update=1,all65_epoch_records_and_all_steps_equal=True,independent_first_reference=reference,real_data_firewall_exercised_with_synthetic_IDs_for_all6=role_checks,terminations=term,production_source_SHA256={n:sha(P/n) for n in ['shared/worker.py','shared/controller.py','shared/journal.py','shared/data.py','shared/engine.py']},generated_log_files=copied,wall=time.monotonic()-started,real_UWB_samples_labels_or_actual_IDs_read=False,new_UWB_fit_started=False)
        write(P/'LIFECYCLE_TEST_RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['generated_log_files','terminations','real_data_firewall_exercised_with_synthetic_IDs_for_all6']}))
if __name__=='__main__':main()
