import numpy as np
import pytest
import torch
from core import time_map, warp, dtw_map, style_mix, partners

def test_warp_identity_and_monotone_endpoints():
    x=torch.randn(3,1,90,378)
    assert torch.equal(warp(x,time_map(90,0)),x)
    for a in [-4,4]:
        g=time_map(90,a)
        assert g[0]==0 and abs(g[-1]-89)<1e-5 and np.all(np.diff(g)>0)
    with pytest.raises(ValueError): time_map(90,30)

def test_dtw_recovers_known_slowdown_better_than_time_index():
    t=np.arange(90,dtype=float)
    g=time_map(90,4)
    c=(t[:,None]-g[None,:])**2
    p=dtw_map(c,12)
    error=np.abs(np.interp(p,t,g)-t)[5:-5].mean()
    assert error < .6 and error < np.abs(g-t)[5:-5].mean()/3

def test_dtw_identical_ramp_and_bad_cost():
    t=np.arange(10)
    np.testing.assert_array_equal(dtw_map((t[:,None]-t[None,:])**2,3),t)
    with pytest.raises(ValueError): dtw_map(np.full((10,10),np.nan),3)

def test_style_endpoints_and_content_normalization():
    a=torch.randn(4,90,32); b=torch.randn_like(a)*3+4
    assert torch.equal(style_mix(a,b,0),a)
    c=style_mix(a,b,1)
    torch.testing.assert_close(c.mean(1),b.mean(1),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(c.std(1,unbiased=False),b.std(1,unbiased=False),atol=2e-5,rtol=1e-5)

def test_partner_constraints_and_impossible_group():
    ids=np.arange(12); y=np.repeat([0,1],6); s=np.tile(np.repeat([1,2,3],2),2)
    p=partners(ids,y,s,7,False)
    assert np.all(p!=ids) and np.all(y[p]==y[ids]) and np.all(s[p]==s[ids])
    p=partners(ids,y,s,7,True)
    assert np.all(y[p]==y[ids]) and np.all(s[p]!=s[ids])
    with pytest.raises(ValueError): partners(np.array([0]),y,s,7,False)
