"""Synthetic checkpoint audit of an explicitly supplied, pinned third-party SAM source.
Usage: python check_sam_resume_review.py /path/to/sam.py
This does not fetch code or read UWB data. JSON is printed to stdout.
"""
import importlib.util,torch,copy,json,argparse,hashlib
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('source',type=Path)
args=parser.parse_args()
data=args.source.read_bytes()
assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()=='df9bcb7d97791f2f3036c2b173b12f8c572d0d80', 'Unexpected third-party source blob'

spec=importlib.util.spec_from_file_location('review_sam',str(args.source));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1);torch.manual_seed(22)
x=torch.tensor([[1.,2.],[-1.,.5],[.3,-.8]]);y=torch.tensor([[.1],[.7],[-.2]])
a=torch.nn.Linear(2,1);oa=m.SAM(a.parameters(),torch.optim.AdamW,lr=.001,weight_decay=.001)
def step(net,opt):
 opt.zero_grad();((net(x)-y)**2).mean().backward();opt.first_step(zero_grad=True);((net(x)-y)**2).mean().backward();opt.second_step(zero_grad=True)
for _ in range(3):step(a,oa)
saved=copy.deepcopy(oa.state_dict());b=torch.nn.Linear(2,1);b.load_state_dict(a.state_dict());ob=m.SAM(b.parameters(),torch.optim.AdamW,lr=.001,weight_decay=.001);ob.load_state_dict(saved)
count_before=len(oa.base_optimizer.state);count_after=len(ob.base_optimizer.state);assert count_before==2 and count_after==0
step(a,oa);step(b,ob);difference=max(float((pa-pb).detach().abs().max()) for pa,pb in zip(a.parameters(),b.parameters()));assert difference>0
record={'synthetic_only':True,'implementation':'davda54/sam sam.py blob df9bcb7d97791f2f3036c2b173b12f8c572d0d80','optimizer':'AdamW','steps_before_resume':3,'uninterrupted_base_state_entries':count_before,'restored_base_state_entries_from_wrapper_state_dict_only':count_after,'next_step_max_parameter_difference':difference,'wrapper_saved_parameter_state_keys':sorted(set(k for s in saved['state'].values() for k in s)),'finding':'Saving only this wrapper state_dict omits base AdamW moment state; explicit nested base state persistence required','not_a_uwb_experiment':True}
print(json.dumps(record,indent=2))
