"""Future subject-only outer-LOSO role planner; accepts no gesture labels or model state."""
def plan_roles(subjects,sample_ids,held,n_users):
 subjects=list(subjects);sample_ids=list(sample_ids)
 assert n_users>=3 and type(held) is int and 1<=held<=n_users
 assert len(subjects)==len(sample_ids)>0
 assert all(type(x) is int and 1<=x<=n_users for x in subjects)
 assert set(subjects)==set(range(1,n_users+1))
 assert all(type(x) is int and x>=0 for x in sample_ids) and len(set(sample_ids))==len(sample_ids)
 val=held%n_users+1
 positions={'train':[],'validation':[],'test':[]}
 for i,u in enumerate(subjects):positions['test' if u==held else 'validation' if u==val else 'train'].append(i)
 ids={k:[sample_ids[i] for i in ix] for k,ix in positions.items()}
 users={k:sorted({subjects[i] for i in ix}) for k,ix in positions.items()}
 assert users['test']==[held] and users['validation']==[val] and len(users['train'])==n_users-2
 for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not(set(ids[a])&set(ids[b])) and not(set(users[a])&set(users[b]))
 assert sorted(i for ix in positions.values() for i in ix)==list(range(len(subjects)))
 return dict(held=held,validation_user=val,users=users,positions=positions,sample_ids=ids)

def generated_checks():
 subjects=[h for h,c in enumerate([3,4,2,5,1,6,4,3],1) for _ in range(c)];ids=list(range(2000,2000+len(subjects)))
 rows=[plan_roles(subjects,ids,h,8) for h in range(1,9)]
 assert rows[-1]['validation_user']==1 and all(len(r['users']['train'])==6 for r in rows)
 assert sorted(x for r in rows for x in r['sample_ids']['test'])==ids
 blocked=[]
 for name,ss,ii in [('duplicate_ID',subjects,[ids[0]]+ids[:-1]),('mismatched_lengths',subjects,ids[:-1]),('missing_subject8',[7 if u==8 else u for u in subjects],ids)]:
  try:plan_roles(ss,ii,1,8)
  except AssertionError:blocked.append(name)
  else:raise AssertionError('Scientific role firewall accepted '+name)
 return dict(state='subject_only_outer8_role_planner_generated_checks_pass',generated_users=8,generated_samples=len(ids),all8_test_roles_cover_each_ID_once=True,each_fold_train6_validation1_test1_disjoint=True,held8_wraps_validation1=True,blocked_invalid_inputs=blocked,actual_outer8_data_loader_or_worker_not_completed=True,real_user7_8_subjects_or_gesture_labels_read=False,new_model_fits_started=0,method_or_window_selection_performed=False)
