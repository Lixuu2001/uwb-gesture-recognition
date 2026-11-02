"""Differential masking, update, dropout RNG and resume checks on synthetic data."""
import copy,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from source_rsc import select_mask,source_step,auxiliary


def reference(f,head,y):
    z=f.detach().clone().requires_grad_(True);p=auxiliary(head,z);g=torch.autograd.grad(p[torch.arange(len(y)),y].sum(),z)[0];a=g.numpy();keep=a<np.percentile(a,(1-1/3)*100,axis=1).astype(a.dtype)[:,None]
    with torch.no_grad():
        d=(p.softmax(1)[torch.arange(len(y)),y]-auxiliary(head,z*torch.from_numpy(keep)).softmax(1)[torch.arange(len(y)),y]).numpy()
    unmasked=d<np.float32(np.percentile(d,(1-1/3)*100));return torch.from_numpy(keep|unmasked[:,None])


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(6,5),nn.Tanh());self.head=nn.Sequential(nn.Flatten(),nn.Linear(15,9),nn.GELU(),nn.Dropout(.2),nn.Linear(9,4))
    def forward(self,x):return self.head(self.encoder(x.reshape(len(x)*3,1,2,3)).reshape(len(x),-1))


def main():
    torch.set_num_threads(1);torch.manual_seed(48);head=nn.Sequential(nn.Linear(17,13),nn.GELU(),nn.Dropout(.2),nn.Linear(13,4));f=torch.randn(11,17);y=torch.arange(11)%4;rng=torch.get_rng_state().clone();m,st=select_mask(f,head,y);assert torch.equal(m,reference(f,head,y));assert torch.equal(rng,torch.get_rng_state());assert all(p.grad is None for p in head.parameters())
    rm,_=select_mask(f,head,y,random_coordinates=True,mask_seed=75);assert torch.equal((~m).sum(1),(~rm).sum(1));assert not torch.equal(m,rm)
    zero=nn.Linear(17,4);nn.init.zeros_(zero.weight);nn.init.zeros_(zero.bias);mask,_=select_mask(f,nn.Sequential(zero),y);assert not mask.any() # Declared strict-percentile tie behavior.
    for arm in ['shared128','rsc24','random24']:
        torch.manual_seed(24);net=Tiny();opt=torch.optim.AdamW(net.parameters(),lr=.001);x=torch.randn(7,3,1,2,3);y=torch.arange(7)%4
        refnet=copy.deepcopy(net);refopt=torch.optim.AdamW(refnet.parameters(),lr=.001);state=torch.get_rng_state();source_step(net,opt,x,y,arm,mask_seed=77);after=torch.get_rng_state();torch.set_rng_state(state)
        if arm=='shared128':loss=F.cross_entropy(refnet(x),y)
        else:
            z=refnet.encoder(x.reshape(21,1,2,3)).reshape(7,-1);mask=reference(z,refnet.head,y)
            if arm=='random24':
                gen=torch.Generator().manual_seed(77);rank=torch.rand(mask.shape,generator=gen).argsort(1).argsort(1);mask=rank>=(~mask).sum(1)[:,None]
            loss=F.cross_entropy(refnet.head(z*mask),y)
        refopt.zero_grad();loss.backward();refopt.step();assert torch.equal(after,torch.get_rng_state())
        for k,v in net.state_dict().items():torch.testing.assert_close(v,refnet.state_dict()[k],rtol=0,atol=0)
        # Checkpoint full AdamW and global dropout RNG, then continue same next mask seed.
        snap=(copy.deepcopy(net.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state().clone());source_step(net,opt,x,y,arm,mask_seed=78);continued=copy.deepcopy(net.state_dict());final_rng=torch.get_rng_state().clone()
        resumed=Tiny();resopt=torch.optim.AdamW(resumed.parameters(),lr=.001);resumed.load_state_dict(snap[0]);resopt.load_state_dict(snap[1]);torch.set_rng_state(snap[2]);source_step(resumed,resopt,x,y,arm,mask_seed=78)
        assert torch.equal(torch.get_rng_state(),final_rng)
        for k,v in continued.items():assert torch.equal(v,resumed.state_dict()[k])
    # Full runner round-trip: actual serialization includes scheduler, dense tail,
    # validation-best state, epoch order, optimizer and both sources of randomness.
    import source_rsc_pilot as runner
    import tempfile,contextlib,io
    from pathlib import Path
    runner.make_model=lambda arm,seed: (torch.manual_seed(seed),Tiny())[1]
    runner.get_batch=lambda cache,ids,kind: cache[ids]
    def tiny_predict(net,cache,ids,kind,batch):
        net.eval()
        with torch.no_grad():return torch.cat([net(cache[ids[k:k+batch]]) for k in range(0,len(ids),batch)])
    runner.predict=tiny_predict
    labels=np.tile(np.arange(4),6);users=np.repeat(np.arange(3,9),4)
    torch.manual_seed(80);cache=torch.randn(24,3,1,2,3)
    cfg=dict(seed=2026,epochs=3,tail_start_epoch=2,batch=5,lr=.001,weight_decay=.001,min_lr=1e-5,feature_drop=1/3,batch_drop=1/3,mask_seed_base=4072026)
    def equal_tree(a,b):
        if torch.is_tensor(a):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal_tree(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal_tree(x,y)
        else:assert a==b
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        for arm in ['rsc24','random24']:
            direct=Path(tmp)/arm/'direct';restart=Path(tmp)/arm/'restart';identity={'synthetic':True,'arm':arm}
            runner.fit(direct,cfg,identity,cache,labels,users,3,arm)
            runner.fit(restart,cfg,identity,cache,labels,users,3,arm,stop_after=1)
            runner.fit(restart,cfg,identity,cache,labels,users,3,arm)
            a=torch.load(direct/'progress.pt',weights_only=False);b=torch.load(restart/'progress.pt',weights_only=False)
            for k in ['model','optimizer','scheduler','rng','tail','best','best_ce','best_epoch','step']:equal_tree(a[k],b[k])
            for x,y in zip(a['history'],b['history']):
                for k in set(x)-{'seconds','segment_sha256'}:equal_tree(x[k],y[k])
            with np.load(direct/'predictions.npz') as x,np.load(restart/'predictions.npz') as y:
                for k in x.files:np.testing.assert_array_equal(x[k],y[k])
    print(json.dumps({'passed':True,'mask_rule_numpy_differential':True,'same_row_removal_counts':True,'auxiliary_preserves_dropout_rng':True,'updates_bitwise_equal':True,'resume_bitwise_equal':True,'strict_zero_gradient_ties_checked':True,'full_runner_epoch_resume_both_arms_bitwise_equal':True}))

if __name__=='__main__':main()
