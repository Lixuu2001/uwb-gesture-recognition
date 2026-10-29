"""Paper-informed MV-GLFNet reimplementation; see PROTOCOL for unresolved choices.
Xue, Yang, Zhu2026 DOI10.3389/fnins.2026.1864448. Not official author code.
"""
import torch
from torch import nn
from torch.nn import functional as F
class Mixer(nn.Module):
 def __init__(self):
  super().__init__();self.depth=nn.Sequential(nn.Conv2d(256,256,3,padding=1,groups=256),nn.GELU(),nn.BatchNorm2d(256));self.point=nn.Sequential(nn.Conv2d(256,256,1),nn.GELU(),nn.BatchNorm2d(256))
 def forward(self,x):return self.point(x+self.depth(x))
class Local(nn.Module):
 def __init__(self):
  super().__init__();self.embed=nn.Sequential(nn.Conv2d(3,256,2,2),nn.GELU(),nn.BatchNorm2d(256));self.blocks=nn.Sequential(*[Mixer() for _ in range(4)]);self.fc=nn.Linear(256,256)
 def forward(self,x):return self.fc(self.blocks(self.embed(x)).mean((-2,-1)))
class Global(nn.Module):
 def __init__(self):
  super().__init__();self.stem=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2));self.project=nn.Conv2d(64,256,1);self.embed=nn.Conv2d(256,256,2,2);self.norm_in=nn.LayerNorm(256);self.pos=nn.Parameter(torch.zeros(1,128,256));nn.init.trunc_normal_(self.pos,std=.02);self.drop=nn.Dropout(.1);self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(256,8,1024,.1,activation='gelu',batch_first=True,norm_first=True) for _ in range(4)]);self.norm_out=nn.LayerNorm(256)
 def forward(self,x):
  z=self.embed(self.project(self.stem(x))).flatten(2).transpose(1,2);assert z.shape[1:]==(128,256)
  z=self.drop(self.norm_in(z)+self.pos)
  for block in self.blocks:z=block(z)
  return self.norm_out(z).mean(1)
class MVGLF(nn.Module):
 def __init__(self,arm):
  super().__init__();assert arm in ['local','full'];self.arm=arm;self.local=Local()
  if arm=='full':
   self.global_branch=Global();self.pl=nn.Linear(256,128);self.pg=nn.Linear(256,128);self.bilinear=nn.Bilinear(128,128,32);self.gate=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,2));self.head=nn.Linear(288,12)
  else:self.head=nn.Linear(256,12)
 def forward(self,x):
  l=self.local(x)
  if self.arm=='local':return self.head(l)
  g=self.global_branch(x);w=self.gate(torch.cat((l,g),1)).softmax(1);first=w[:,:1]*l+w[:,1:]*g;second=self.bilinear(self.pl(l),self.pg(g));return self.head(torch.cat((first,second),1))
def preprocess(raw):
 """Provided clutter-removed maps; no second MTI. Label-free per-record transform."""
 assert raw.ndim==4 and raw.shape[1:]==(3,90,189)
 z=F.interpolate(raw.float(),size=(64,128),mode='bilinear',align_corners=False)
 rms=z.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-8)
 return z/rms
