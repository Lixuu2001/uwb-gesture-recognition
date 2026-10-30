"""Literal public CLI model defaults; distinct from the 64-width core screen.

Only task dimensions/classes and seed are adapted. This is an official default
configuration, not an accuracy-selected rescue width. No datasets are loaded.
"""
import ast
import torch
from author_core import BASE,definitions
def cli_defaults():
 tree=ast.parse((BASE/'vendor/ConvTran/main.py').read_text());defaults={}
 for node in tree.body:
  if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
   call=node.value
   if isinstance(call.func,ast.Attribute) and call.func.attr=='add_argument' and call.args:
    for keyword in call.keywords:
     if keyword.arg=='default': defaults[ast.literal_eval(call.args[0]).lstrip('-')]=ast.literal_eval(keyword.value)
 return defaults
def make_model(seed=20261029):
 defaults=cli_defaults();assert {k:defaults[k] for k in ('emb_size','dim_ff','num_heads','dropout','Fix_pos_encode','Rel_pos_encode','batch_size','epochs','lr')}==dict(emb_size=16,dim_ff=256,num_heads=8,dropout=.01,Fix_pos_encode='tAPE',Rel_pos_encode='eRPE',batch_size=16,epochs=100,lr=.001)
 torch.manual_seed(seed);ns=definitions();cfg=dict(Data_shape=(16,567,90),**{k:defaults[k] for k in ('emb_size','dim_ff','num_heads','dropout','Fix_pos_encode','Rel_pos_encode')})
 return ns['ConvTran'](cfg,12),ns,cfg
