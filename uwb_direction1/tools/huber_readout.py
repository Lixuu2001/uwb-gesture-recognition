"""Fixed-scale source-only Huber objective; no target data access."""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve


def objective(theta,x,y,w,delta,lam):
    residual=x@theta-y;ab=abs(residual);loss=np.where(ab<=delta,.5*residual**2,delta*(ab-.5*delta));grad=x.T@(w*np.clip(residual,-delta,delta));grad[:-1]+=lam*theta[:-1]
    return float(w@loss+.5*lam*(theta[:-1]@theta[:-1])),grad


def weighted_fit(x,y,w,lam):
    pen=np.eye(x.shape[1])*lam;pen[-1,-1]=0
    return solve(x.T@(w[:,None]*x)+pen,x.T@(w*y),assume_a='pos')


def fit(x,y,u,w,initial,lam,held,c):
    target=np.eye(12)[y];residual=x@initial-target;mad=np.median(abs(residual-np.median(residual,axis=0)),axis=0)/c['normal_mad'];delta=c['epsilon']*np.maximum(mad,c['scale_floor']);models={k:np.empty_like(initial) for k in ['huber','one_step','shuffled']};qsave=np.empty_like(target);qperm=np.empty_like(target);logs=[]
    for k in range(12):
        start=initial[:,k];q0=np.minimum(1,delta[k]/np.maximum(abs(residual[:,k]),1e-300));models['one_step'][:,k]=weighted_fit(x,target[:,k],w*q0,lam)
        result=minimize(objective,start,args=(x,target[:,k],w,delta[k],lam),method='L-BFGS-B',jac=True,options=dict(gtol=1e-10,ftol=1e-15,maxiter=c['max_iter'],maxls=50));theta=result.x;loss,g=objective(theta,x,target[:,k],w,delta[k],lam);gn=float(abs(g).max());logs.append(dict(output=k,iterations=int(result.nit),success=bool(result.success),message=str(result.message),loss=loss,gradient_inf=gn,initial_loss=objective(start,x,target[:,k],w,delta[k],lam)[0]));assert gn<=c['gradient_tolerance'],logs[-1];assert loss<=logs[-1]['initial_loss']+1e-12
        models['huber'][:,k]=theta;q=np.minimum(1,delta[k]/np.maximum(abs(x@theta-target[:,k]),1e-300));qq=q.copy()
        for uu in np.unique(u):
            for yy in np.unique(y[u==uu]):
                ix=np.flatnonzero((u==uu)&(y==yy));rng=np.random.default_rng(np.random.SeedSequence([c['seed'],held,k,int(uu),int(yy)]));qq[ix]=q[rng.permutation(ix)]
        qsave[:,k]=q;qperm[:,k]=qq;models['shuffled'][:,k]=weighted_fit(x,target[:,k],w*qq,lam)
    return models,dict(delta=delta,mad=mad,influence=qsave,shuffled_influence=qperm,source_weights=w),logs
