"""Freeze all three arm predictions after member audits; never reads truth."""
import sys,math
from common import B,require_initial,read,write,sha
from checks import verify_audits,member
from data import load
from vote import np,combine
def independent(a,b):
    result=[]
    for za,zb in zip(a,b):
        la=[-math.log1p(math.exp(-float(z))) if z>=0 else float(z)-math.log1p(math.exp(float(z))) for z in za]
        aa=[math.exp(x-max(la)) for x in la];bb=[math.exp(float(x)-float(max(zb))) for x in zb]
        result.append([.5*x/math.fsum(aa)+.5*y/math.fsum(bb) for x,y in zip(aa,bb)])
    return np.array(result)
def main():
    held=int(sys.argv[1]);require_initial();verify_audits(held);root=B/'folds'/f'held{held}';dest=root/'fusion';dest.mkdir(exist_ok=False)
    ids,users,loc,maps,_,_=load(held,labels=False)
    for role in ['validation','diagnostic']:
        with np.load(member(held,'timemil')/(role+'_unscored.npz')) as z:ai=z['ids'];a=z['scores']
        with np.load(member(held,'shuffle')/(role+'_unscored.npz')) as z:bi=z['ids'];b=z['scores']
        np.testing.assert_array_equal(ai,ids[loc[role]]);np.testing.assert_array_equal(bi,ai)
        probs,pred=combine(ai,a,bi,b);reference=independent(a,b)
        np.testing.assert_allclose(probs,reference,rtol=0,atol=1e-14);np.testing.assert_array_equal(pred,reference.argmax(1))
        np.savez_compressed(dest/(role+'_unscored.npz'),ids=ai,probabilities=probs,predictions=pred,timemil_predictions=a.argmax(1),shuffle_predictions=b.argmax(1))
    write(dest/'FOLD_FROZEN.json',dict(state='three_arms_frozen_before_any_new_held_truth',held=held,files={n:sha(dest/n) for n in ['validation_unscored.npz','diagnostic_unscored.npz']},source_member_manifests={arm:sha(member(held,arm)/'ARM_COMPLETE.json') for arm in ['timemil','shuffle']},diagnostic_labels_read=False,users7_8_labels_used=False))
if __name__=='__main__':main()
