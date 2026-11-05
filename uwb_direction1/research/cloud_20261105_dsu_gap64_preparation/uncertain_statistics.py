"""Equation7 of Li et al., ICLR2022, with released-code variance and RNG conventions.

Independent component implementation; no image backbone or author benchmark claim.
Reference commit82a1de036351b72620abef1a9c4cc561751264a2, root dsu.py.
"""
import numpy as np
import torch
class UncertainStatistics(torch.nn.Module):
 def __init__(self,probability=.5,eps=1e-6,factor=1.):
  super().__init__();self.probability=probability;self.eps=eps;self.factor=factor
 def forward(self,features):
  if not self.training:return features
  if np.random.random()>self.probability:return features
  assert features.ndim==4 and features.shape[0]>1 and features.shape[2]*features.shape[3]>1
  center=features.mean((2,3))
  spread=(features.var((2,3),correction=1)+self.eps).sqrt()
  location_scope=(center.var(0,keepdim=True,correction=1)+self.eps).sqrt().expand_as(center).contiguous()
  scale_scope=(spread.var(0,keepdim=True,correction=1)+self.eps).sqrt().expand_as(spread).contiguous()
  shifted_center=center+(torch.randn_like(location_scope)*self.factor)*location_scope
  shifted_spread=spread+(torch.randn_like(scale_scope)*self.factor)*scale_scope
  view=lambda value:value[:,:,None,None]
  standardized=(features-view(center))/view(spread)
  return standardized*view(shifted_spread)+view(shifted_center)
