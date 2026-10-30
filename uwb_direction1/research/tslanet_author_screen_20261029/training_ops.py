"""Author-default loss/update adapters; no data loader or real training launch.

TSLANet source ca0e88416d3ae49fd50e399c44ae94868378a94d; timm loss
78c3724f4545c0067d0b4d2f7df6337196be01ce, original class body unchanged.
Original default: AdamW1e-3/weight_decay1e-4 for both stages, smoothing0.1.
"""
import torch
from torch import nn
from torch.nn import functional as F
from author_core import BASE,selected_definitions

def classification_criterion():
    namespace=dict(torch=torch,nn=nn,F=F)
    selected_definitions(BASE/'vendor/timm_pinned/cross_entropy.py',('LabelSmoothingCrossEntropy',),namespace)
    return namespace['LabelSmoothingCrossEntropy']()

def pretraining_loss(model,data):
    predictions,target=model.pretrain(data)
    # Preserve the author's attached learned/dropout target and Boolean mask.
    loss=(predictions-target)**2
    loss=loss.mean(dim=-1)
    return (loss*model.mask).sum()/model.mask.sum()

def classification_loss(model,data,labels,criterion=None):
    criterion=classification_criterion() if criterion is None else criterion
    return criterion(model(data),labels.to(torch.int64))

def stage_optimizer(model):
    return torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
