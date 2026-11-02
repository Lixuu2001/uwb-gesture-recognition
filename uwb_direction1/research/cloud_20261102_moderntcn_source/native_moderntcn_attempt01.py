"""Full released waveGesture preset; source modules unchanged, current CPU runtime."""
import os,sys,ast,shlex,random
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
torch.set_num_threads(1)
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'vendor'))
from models.ModernTCN import Model
def source_configuration():
    values={}
    for node in ast.walk(ast.parse((B/'run_upstream.py').read_text())):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args:
            key=ast.literal_eval(node.args[0]).removeprefix('--')
            for kw in node.keywords:
                if kw.arg=='default':values[key]=ast.literal_eval(kw.value)
    line=next(line for line in (B/'classification_upstream.sh').read_text().splitlines() if '--model_id waveGesture ' in line)
    words=shlex.split(line);i=words.index('--task_name')
    while i<len(words):
        key=words[i].removeprefix('--');i+=1;raw=[]
        while i<len(words) and not words[i].startswith('--'):raw.append(words[i]);i+=1
        prior=values.get(key)
        if isinstance(prior,list):value=[int(v) for v in raw]
        elif isinstance(prior,bool):value=raw[0].lower()=='true'
        elif isinstance(prior,int):value=int(raw[0])
        elif isinstance(prior,float):value=float(raw[0])
        else:value=raw[0]
        values[key]=value
    return values
def config():
    values=source_configuration();values.update(seq_len=90,enc_in=567,num_class=12,pred_len=0,random_seed=20261029,use_gpu=False)
    return SimpleNamespace(**values)
def make(meta=False):
    c=config();torch.manual_seed(c.random_seed);np.random.seed(c.random_seed);random.seed(c.random_seed)
    if meta:
        with torch.device('meta'):m=Model(c).float()
    else:m=Model(c).float()
    return m,c
def assert_structure(m,c):
    assert c.num_blocks==[1,1] and c.large_size==[51,49] and c.small_size==[5,5]
    assert c.dims==[128,256] and c.ffn_ratio==1 and c.patch_size==c.patch_stride==1
    assert c.batch_size==32 and c.train_epochs==100 and c.patience==10 and c.learning_rate==.001
    assert c.class_dropout==.1 and c.dropout==.3 and c.head_dropout==0 and c.use_multi_scale is False
    assert c.enc_in==567 and c.seq_len==90 and c.num_class==12 and c.pred_len==0
    assert len(m.model.stages)==2 and m.model.head_class.in_features==567*256*45
    assert m.model.head_class.out_features==12
def step(m,opt,x,y):
    opt.zero_grad();z=m(x,None,None,None);loss=torch.nn.functional.cross_entropy(z,y)
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),max_norm=4.0);opt.step()
    return float(loss.detach())
