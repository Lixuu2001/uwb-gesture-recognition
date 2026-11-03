"""Independent actual first update, full native model/optimizer/RNG comparison."""
import sys,hashlib
from common import B,require_initial,read,write,sha
from data import load
from checks import member
import engine as e
def main():
    held=int(sys.argv[1]);arm=sys.argv[2];c,p=require_initial();out=member(held,arm);d=out/'first_audit';d.mkdir(exist_ok=False)
    m,o,g=e.make(arm);initial=e.capture(m,o,g,0,0)
    assert e.semantic(initial)==read(B/'SEED_FILES.json')[arm]['state_sha256']
    ids,users,loc,maps,y,v=load(held,labels=True);batch=64 if arm=='timemil' else 20
    order=e.order(arm,len(y),1,g);positions=order[:batch]
    loss=e.step(arm,m,o,e.raw(arm,maps,loc['train'][positions]),y[positions],1)
    row=dict(epoch=1,offset=0,update=1,loss=loss,positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),training_ids_sha256=hashlib.sha256(ids[loc['train'][positions]].tobytes()).hexdigest())
    expected=read(out/'FIRST_READY.json');assert expected['step']==row
    assert expected['checkpoint_sha256']==sha(out/'first_update.pt')
    actual=e.torch.load(out/'first_update.pt',map_location='cpu',weights_only=True)
    assert e.semantic(actual)==expected['state_sha256']==e.semantic(e.capture(m,o,g,0,1))
    write(d/'CERTIFICATE.json',dict(state='pass',held=held,arm=arm,first_state_sha256=e.semantic(actual),first_checkpoint_sha256=sha(out/'first_update.pt'),complete_model_optimizer_order_Torch_Python_Numpy_RNG_exact=True,diagnostic_labels_read=False,users7_8_labels_used=False))
if __name__=='__main__':main()
