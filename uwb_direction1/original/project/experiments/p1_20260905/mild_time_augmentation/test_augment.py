import numpy as np
import torch
import pytest
from augment import raw_warp, prepare_input, epoch_schedule, AugmentReader, legacy_transform

def test_warp_identity_monotone_endpoints_and_no_modality_mixing():
    x=np.random.default_rng(1).normal(size=(2,1,90,189)).astype('float32')
    assert np.array_equal(raw_warp(x,0),x)
    for a in [-2,2]:
        z=raw_warp(x,a)
        assert np.array_equal(z[:,:,0],x[:,:,0]) and np.array_equal(z[:,:,-1],x[:,:,-1])
        np.testing.assert_array_equal(raw_warp(x[:1],a),z[:1])
    with pytest.raises(ValueError): raw_warp(x,30)

def test_zero_warp_matches_original_preprocessing_and_recomputes_phase():
    x=np.random.default_rng(2).normal(size=(2,1,90,189)).astype('float32')
    np.testing.assert_array_equal(prepare_input(x,0),legacy_transform(x,'concat').astype('float16'))
    np.testing.assert_array_equal(prepare_input(x,2),legacy_transform(raw_warp(x,2),'concat').astype('float16'))
    assert np.isfinite(prepare_input(x,2)).all()

def test_schedule_replays_without_touching_torch_rng():
    before=torch.get_rng_state().clone(); ids=np.arange(1000)
    s=epoch_schedule(ids,123)
    assert set(s)=={-1,0,1} and .43<np.mean(s!=0)<.57
    np.testing.assert_array_equal(s,epoch_schedule(ids,123))
    assert not np.array_equal(s,epoch_schedule(ids,124))
    assert torch.equal(before,torch.get_rng_state())

def test_reader_uses_only_source_plan_and_keeps_validation_clean():
    class Fake:
        partitions={'train':np.array([0,1]),'val':np.array([2]),'test':np.array([3])}
        amplitude=np.zeros((4,3,90,189))
        def load_ids(self,ids,role,mode,device='cpu'):
            if not np.isin(ids,self.partitions[role]).all(): raise ValueError('Wrong role')
            return torch.zeros((len(ids),1,90,378),dtype=torch.float16)
    cache={-1:np.ones((4,1,90,378),dtype='float16'),1:np.ones((4,1,90,378),dtype='float16')*2}
    r=AugmentReader(Fake(),cache,True); r.configure(12); r.plan[:2]=[-1,1]
    x=r.load_ids(np.array([0,1]),'train','R','cpu')
    assert x[0].mean()==1 and x[1].mean()==2
    assert r.load_ids(np.array([2]),'val','R','cpu').sum()==0
    assert r.load_ids(np.array([3]),'test','R','cpu').sum()==0
    with pytest.raises(ValueError): r.load_ids(np.array([2]),'train','R','cpu')
    r.enabled=False
    assert r.load_ids(np.array([0]),'train','R','cpu').sum()==0
