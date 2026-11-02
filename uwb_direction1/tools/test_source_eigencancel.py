import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from circular_signal import SignalNet,encode,get_batch
from source_eigencancel import extract,filtered_raw
from semantic_code_eightfold_prepare import functional_hidden


def main():
    torch.set_num_threads(1);torch.manual_seed(7);net=SignalNet().eval();rng=np.random.default_rng(23);raw=rng.normal(size=(3,3,90,189)).astype(np.float32);cache=encode(raw);ids=np.arange(3);m={}
    for v in range(3):
        m[f'v{v}_basis']=np.linalg.qr(rng.normal(size=(189,3)))[0];m[f'v{v}_random_basis']=np.linalg.qr(rng.normal(size=(189,3)))[0]
    maximum=0.
    for mask in [[False]*3,[True,False,True],[True]*3]:
        m['selected_views']=np.array(mask);pred=extract(net,raw,cache,ids,m,3);original=get_batch(cache,ids,'legacy')
        for arm,random,only in [('selective',False,True),('all_views',False,False),('random_selected',True,True)]:
            x=get_batch(encode(filtered_raw(raw,m,random=random,selected_only=only)),ids,'legacy')
            if only:x[:,~m['selected_views']]=original[:,~m['selected_views']]
            with torch.no_grad():ref=functional_hidden(x,net.state_dict()).numpy()
            maximum=max(maximum,float(abs(ref-pred[arm]).max()));np.testing.assert_allclose(ref,pred[arm],atol=2e-5,rtol=2e-5)
    print(json.dumps(dict(state='passed',all_three_selection_masks=True,maximum_hidden_difference=maximum)))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
