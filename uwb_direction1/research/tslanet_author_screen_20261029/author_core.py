"""Executable author class bodies with an explicit framework-only CPU adapter.

Not the Lightning training driver or an exact original-environment reproduction.
Only named AST definitions execute; no dataset loader or reporting code executes.
TSLANet MIT source ca0e88416d3ae49fd50e399c44ae94868378a94d.
timm Apache2 helpers v1.0.22 (78c3724f4545c0067d0b4d2f7df6337196be01ce).
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import ast
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

BASE = Path(__file__).resolve().parent
CORE_NAMES = ('ICB','PatchEmbed','Adaptive_Spectral_Block','TSLANet_layer','TSLANet')

def selected_definitions(path, names, namespace):
    parsed = ast.parse(path.read_text())
    selected = [node for node in parsed.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    # Bodies, expressions and calls remain the author's original AST.
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)

def make_model(seed=20261029):
    torch.manual_seed(seed)
    config = SimpleNamespace(seq_len=90,num_channels=567,num_classes=12,emb_dim=256,depth=3,
                             patch_size=8,dropout_rate=.15,masking_ratio=.4,
                             ICB=True,ASB=True,adaptive_filter=True)
    ns = dict(torch=torch,nn=nn,math=math,warnings=warnings,
              L=SimpleNamespace(LightningModule=nn.Module),args=config)
    selected_definitions(BASE/'vendor/timm_pinned/drop.py',('drop_path','DropPath'),ns)
    selected_definitions(BASE/'vendor/timm_pinned/weight_init.py',('_trunc_normal_','trunc_normal_'),ns)
    selected_definitions(BASE/'vendor/TSLANet/Classification/utils.py',('random_masking_3D',),ns)
    selected_definitions(BASE/'vendor/TSLANet/Classification/TSLANet_classification.py',CORE_NAMES,ns)
    model = ns['TSLANet']()
    return model,ns
