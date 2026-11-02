"""Exact algebraic tests, independent Chen recursion, and source-role isolation."""
import json
import numpy as np
from energy_area import areas,energy_path,describe,source_roles


def chen_area(p):
    delta=np.zeros(p.shape[-1]);second=np.zeros((len(delta),len(delta)))
    for d in np.diff(p,axis=0):
        second+=np.outer(delta,d)+.5*np.outer(d,d);delta+=d
    logsecond=second-.5*np.outer(delta,delta)
    return logsecond[np.triu_indices(len(delta),1)]


def run():
    rng=np.random.default_rng(721);p=rng.normal(size=(5,90,7));maxerr=0.
    for path in p:
        ref=chen_area(path);maxerr=max(maxerr,float(abs(ref-areas(path)).max()))
        np.testing.assert_allclose(areas(path),ref,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(areas(path[::-1]),-areas(path),atol=1e-12)
        np.testing.assert_allclose(areas(path+12),areas(path),atol=1e-12)
        refined=np.stack([path[:-1],.37*path[:-1]+.63*path[1:]],axis=1).reshape(-1,7)
        refined=np.concatenate([refined,path[-1:]])
        np.testing.assert_allclose(areas(refined),areas(path),atol=1e-12)
    assert areas(np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]))[0]==1
    raw=rng.normal(size=(2,3,90,189));ep=energy_path(raw)
    np.testing.assert_allclose(ep,energy_path(-3*raw),atol=1e-14)
    assert not energy_path(np.zeros_like(raw)).any()
    assert not areas(np.ones((90,7))).any()
    a=describe(ep,[14,17],381);b=describe(ep[::-1],[17,14],381)
    for k in a:np.testing.assert_allclose(a[k],b[k][::-1],atol=1e-14)
    y=np.tile(np.arange(12),8);u=np.repeat(np.arange(1,9),12)
    for h in range(3,9):
        roles=source_roles(y,u,h);v=h+1 if h<8 else 3
        assert set(u[roles['train']])==set(range(3,9))-{h,v}
        assert set(u[roles['val']])=={v} and set(u[roles['diagnostic']])=={h}
    return {'state':'passed','chen_max_error':maxerr,'synthetic_paths':5,'checks':['Chen level2','unit square','reversal','translation','segment subdivision','RF gain/sign invariance','zeros','stable sample-ID shuffle','source roles'],'scope':'Synthetic only; no recognition result'}


if __name__=='__main__':print(json.dumps(run(),indent=2))
