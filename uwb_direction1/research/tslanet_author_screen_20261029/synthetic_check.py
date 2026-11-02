"""Full UEA-scale author-core synthetic compatibility check, not a real trial."""
from author_core import make_model,BASE
import copy
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    model,namespace = make_model()
    x = torch.randn(16,567,90)
    target = torch.arange(16)%12
    assert model.patch_embed.num_patches == 21
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    # Full-width forward/backward; classification's dormant input_layer retained.
    model.train();optimizer.zero_grad(set_to_none=True)
    tic=time.monotonic();pred=model(x)
    loss=F.cross_entropy(pred,target,label_smoothing=.1);loss.backward()
    assert pred.shape == (16,12) and torch.isfinite(pred).all() and torch.isfinite(loss)
    unused = [name for name,p in model.named_parameters() if p.grad is None]
    assert set(unused) == {'input_layer.weight','input_layer.bias'}
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step();classification_seconds=time.monotonic()-tic
    # Author pretraining target intentionally remains attached, including dropout.
    optimizer.zero_grad(set_to_none=True)
    tic=time.monotonic();reconstructed,reference=model.pretrain(x)
    mask=model.mask
    assert reconstructed.shape==reference.shape==(16,21,256)
    assert mask.shape==(16,21) and (mask.sum(1)==9).all()
    assert reference.requires_grad
    preloss=(((reconstructed-reference)**2).mean(-1)*mask).sum()/mask.sum()
    assert torch.isfinite(preloss);preloss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    pretrain_unused=[name for name,p in model.named_parameters() if p.grad is None]
    assert set(pretrain_unused)=={'input_layer.weight','input_layer.bias','head.weight','head.bias'}
    optimizer.step();pretraining_seconds=time.monotonic()-tic
    # Independent NumPy ASB forward: energy mask, complex weights and inverse FFT.
    block=model.tsla_blocks[0].asb
    z=torch.randn(4,21,256)
    observed=block(z).detach().numpy()
    raw=np.fft.rfft(z.numpy(),axis=1,norm='ortho')
    energy=(np.abs(raw)**2).sum(-1)
    binary=(energy/(np.median(energy,axis=1,keepdims=True)+1e-6)>float(block.threshold_param.detach())).astype('float32')
    wg=torch.view_as_complex(block.complex_weight.detach()).numpy()
    wh=torch.view_as_complex(block.complex_weight_high.detach()).numpy()
    expected=np.fft.irfft(raw*wg+raw*binary[...,None]*wh,n=21,axis=1,norm='ortho')
    np.testing.assert_allclose(observed,expected,rtol=3e-5,atol=2e-6)
    # Model/AdamW/RNG resume across actual dropout and stochastic-depth operations.
    def update(m,opt):
        m.train();opt.zero_grad(set_to_none=True)
        value=F.cross_entropy(m(x),target,label_smoothing=.1)
        value.backward();opt.step()
    saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()))
    update(model,optimizer)
    clone,_=make_model();other=torch.optim.AdamW(clone.parameters(),lr=.001,weight_decay=.0001)
    clone.load_state_dict(saved['model']);other.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    update(clone,other)
    assert all(torch.equal(value,clone.state_dict()[key]) for key,value in model.state_dict().items())
    result=dict(state='pass',scope='synthetic_full_UEA_author_core_only',real_data_values_or_labels_read=False,
                real_training_started=False,real_protocol_registered=False,lightning_driver_reproduction=False,
                parameters=sum(p.numel() for p in model.parameters()),batch_size=16,channels=567,time_points=90,
                embedding=256,depth=3,patch_size=8,patches=21,dropout=.15,mask_ratio=.4,
                classification_unused=unused,pretraining_unused=pretrain_unused,
                independent_numpy_spectral_forward_verified=True,author_attached_pretraining_target_preserved=True,
                full_model_stochastic_resume_bitwise_equal=True,classification_batch_seconds=classification_seconds,
                pretraining_batch_seconds=pretraining_seconds,wall_seconds=time.monotonic()-started)
    dest=BASE/'SYNTHETIC_CHECK.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
