"""Independent raw FFT/scalar NCC spot checks and full decision/statistic replay."""
import argparse,json,time
from pathlib import Path
import numpy as np
from audit_restored import digest,dump


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);a=p.parse_args();start=time.perf_counter();R=a.runs
 c=json.loads((R/'PROTOCOL.json').read_text());identity=json.loads((R/'IDENTITY.json').read_text());summary=json.loads((R/'SUMMARY.json').read_text())
 for name,h in json.loads((R/'FILES.json').read_text()).items():assert digest(R/name)==h
 for name,h in identity['raw_files_sha256'].items():assert digest(a.data/'cache'/name)==h
 for name,h in identity['code_sha256'].items():assert digest(Path(__file__).parent/name)==h
 u=np.load(a.data/'cache/subj_clutterremoved.npy');y=np.load(a.data/'cache/y_clutterremoved.npy');raw=np.load(a.data/'cache/X_clutterremoved.npy',mmap_mode='r');ids=np.load(R/'ids.npy')
 expected=[]
 for user in c['source_users']:
  for label in np.unique(y[(u==user)&~((u==6)&(y==3))]):
   pool=np.flatnonzero((u==user)&(y==label)&~((u==6)&(y==3)));rng=np.random.default_rng(np.random.SeedSequence([c['seed'],int(user),int(label)]));expected.extend(sorted(rng.choice(pool,min(len(pool),c['per_user_class']),replace=False)))
 np.testing.assert_array_equal(ids,expected);assert len(ids)==len(set(ids.tolist())) and (u[ids]>=3).all()
 with np.load(R/'correspondences.npz') as z:
  np.testing.assert_array_equal(z['ids'],ids);rows=[];scalar_checks=0;max_error=0.
  # Independently reconstruct every mask/shift from saved complete cost volumes.
  replay={}
  for arm in ['adjacent','distant']:
   cost=z[arm+'_cost'];backcost=z[arm+'_reverse_cost'];order=np.argsort(cost,axis=-1);ro=np.argsort(backcost,axis=-1)
   # Original argmax picks first tie; indices are verified using max and equality explicitly.
   winner=(cost==cost.max(-1,keepdims=True)).argmax(-1);rw=(backcost==backcost.max(-1,keepdims=True)).argmax(-1)
   shift=winner-c['max_shift'];back=rw-c['max_shift'];cycle=np.abs(shift+back)
   peak=cost.max(-1);gap=peak-np.take_along_axis(cost,order[...,-2:-1],axis=-1)[...,0]
   bp=backcost.max(-1);bg=bp-np.take_along_axis(backcost,ro[...,-2:-1],axis=-1)[...,0]
   good=z[arm+'_eligible']&(peak>=c['min_correlation'])&(gap>=c['min_gap'])&(bp>=c['min_correlation'])&(bg>=c['min_gap'])&(cycle<=c['max_cycle_error'])&(abs(shift)<c['max_shift'])&(abs(back)<c['max_shift'])
   for name,value in [('shift',shift),('back',back),('peak',peak),('gap',gap),('cycle',cycle),('reliable',good),('moving',good&(shift!=0))]:np.testing.assert_array_equal(value,z[arm+'_'+name])
   replay[arm]=good
  # One predetermined view/time/center per record, both arms, every candidate lag and direction.
  for j,idx in enumerate(ids):
   v=j%3;ti=j%len(c['frame_starts']);ci=j%len(c['centers']);t=c['frame_starts'][ti];center=c['centers'][ci]
   x=np.asarray(raw[idx,v],dtype=np.float64);n=x.shape[-1];mult=np.zeros(n);mult[0]=1;mult[1:(n+1)//2]=2
   if n%2==0:mult[n//2]=1
   e=np.abs(np.fft.ifft(np.fft.fft(x,axis=-1)*mult,axis=-1))
   for arm,t2 in [('adjacent',t+1),('distant',(t+31)%90)]:
    aa=e[t];bb=e[t2];s=int(z[arm+'_shift'][j,v,ti,ci]);r=c['patch_radius'];M=c['max_shift']
    pa=aa[center-r:center+r+1];pn=np.linalg.norm(pa-pa.mean());eligible=aa.std()>1e-12/np.sqrt(2*r+1) and pn>=c['texture_fraction']*np.sqrt(2*r+1)*aa.std();assert eligible==z[arm+'_eligible'][j,v,ti,ci]
    for A,B,C,key in [(aa,bb,center,'cost'),(bb,aa,center+s,'reverse_cost')]:
     for k,lag in enumerate(range(-M,M+1)):
      xx=A[C-r:C+r+1];yy=B[C+lag-r:C+lag+r+1];xx=xx-xx.mean();yy=yy-yy.mean();den=np.linalg.norm(xx)*np.linalg.norm(yy);actual=np.dot(xx,yy)/den if den>1e-24 else -2.;saved=z[arm+'_'+key][j,v,ti,ci,k];error=abs(actual-saved);max_error=max(max_error,float(error));assert error<1e-9;scalar_checks+=1
  for original in summary['rows']:
   user=original['source_user'];v=original['view'];ix=np.flatnonzero(u[ids]==user);row={}
   for arm in ['adjacent','distant']:
    fractions=[];moving=[];valid=0
    for i in ix:
     den=int(z[arm+'_eligible'][i,v].sum());valid+=den>0
     fractions.append(int(replay[arm][i,v].sum())/den if den else 0.)
     moving.append(int((replay[arm][i,v]&(z[arm+'_shift'][i,v]!=0)).sum())/den if den else 0.)
    assert valid==original[arm+'_eligible_records']
    row[arm+'_reliable_fraction']=sum(fractions)/len(ix);row[arm+'_moving_fraction']=sum(moving)/len(ix)
   for k,val in row.items():assert abs(val-original[k])<1e-14
   gap=row['adjacent_reliable_fraction']-row['distant_reliable_fraction'];passed=row['adjacent_reliable_fraction']>=c['gate']['min_reliable'] and gap>=c['gate']['min_gap_over_distant'] and row['adjacent_moving_fraction']>=c['gate']['min_moving'];assert passed==original['passed'];rows.append((v,passed))
  wins={str(v):sum(p for view,p in rows if view==v) for v in range(3)};gate=sum(n>=c['gate']['users_per_view'] for n in wins.values())>=c['gate']['views_required'];assert wins==summary['passing_users_per_view'] and gate==summary['gate_passed']
 dump(R/'AUDIT.json',dict(state='passed',files_sha256_verified=True,source_ids_and_role_verified=True,full_masks_shifts_statistics_replayed=True,independent_raw_scalar_ncc_checks=scalar_checks,max_scalar_cost_error=max_error,raw_fft_scope='one predetermined view/time/center per record, both arms and directions, every lag; not all raw patch pairs',gate_passed=gate,classification_fitted=False,seconds=time.perf_counter()-start));print((R/'AUDIT.json').read_text())


if __name__=='__main__':main()
