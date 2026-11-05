"""Synthetic operational checkpoint/suffix recovery across native slow update6."""
from pathlib import Path
import sys,copy,tempfile,json,time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from common import B,ready,write
import engine as e
from journal import StepLog,commit,load
def main():
    ready();start=time.monotonic();out=B/'journal_probe_execution';out.mkdir(exist_ok=False)
    rng=e.np.random.default_rng(447);x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);m,o,g=e.make();log=StepLog(d/'steps.jsonl');initial=copy.deepcopy(e.capture(m,o,g,0,0));commit(d,initial,e.record(initial,0,0),log.prefix())
        for update in range(1,6):
            m.train();v=e.step(m,o,x,y,1);log.append(dict(update=update,**v))
        saved=copy.deepcopy(e.capture(m,o,g,1,5));immutable=e.semantic(saved);commit(d,saved,e.record(saved,1,5),log.prefix())
        for update in [6,7]:
            m.train();v=e.step(m,o,x,y,1);log.append(dict(update=update,**v))
        log.close();expected=copy.deepcopy(e.capture(m,o,g,1,7));assert o.slow and o.count==7 and e.semantic(saved)==immutable
        stored,p=load(d);assert stored['updates']==5 and e.semantic(stored)==immutable
        original_bytes=(d/'steps.jsonl').read_bytes();n,no,ng=e.make();e.restore(stored,n,no,ng);recover=StepLog(d/'steps.jsonl',5,p['prefix_bytes'],p['prefix_sha256']);assert recover.original==7
        for update in [6,7]:
            n.train();v=e.step(n,no,x,y,1);recover.append(dict(update=update,**v))
        assert no.count==7 and no.slow and e.semantic(e.capture(n,no,ng,1,7))==e.semantic(expected)
        assert (d/'steps.jsonl').read_bytes()==original_bytes and e.semantic(saved)==immutable
        # Commit the recovered next epoch and verify the entire native journal chain.
        final=copy.deepcopy(e.capture(n,no,ng,2,7));commit(d,final,e.record(final,2,7),recover.prefix());recover.close();loaded,pr=load(d);assert loaded['epoch']==2 and loaded['updates']==7 and e.semantic(loaded)==e.semantic(final)
    q=dict(state='pass_synthetic_full_native_journal_and_uncommitted_suffix_recovery',committed_updates=5,captured_uncommitted_rows=2,native_slow_buffer_boundary_update=6,all_recovered_rows_model_optimizer_Python_NumPy_Torch_order_RNG_exact=True,original_suffix_bytes_and_saved_fixture_immutable=True,two_slot_journal_chain_recaptured=True,scientific20_changed=False,new_UWB_fit_started=False,RegMixup_diagnostic_read=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-start)
    write(out/'RESULT.json',q);print(json.dumps(q))
if __name__=='__main__':main()
