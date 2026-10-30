"""Pinned library-maintainer gesture preset with fixed label-blind sample ordering."""
import sys
from pathlib import Path
import numpy as np
import torch
SCREEN=Path('/workspace/uwb-lightts-screen-20261030');sys.path.insert(0,str(SCREEN))
from author_adapter import make,step,early_stopper,snapshot,restore,equal,source_check
def training_order(epoch,seed=20261029):
 assert 1<=epoch<=100
 return torch.randperm(4700,generator=torch.Generator().manual_seed(seed+epoch)).numpy()
def evaluation_order(count,role,seed=20261029):
 assert (count,role) in [(1000,'validation'),(1200,'diagnostic')]
 return np.random.RandomState(seed+(101 if role=='validation' else 102)).permutation(count)
def predictions(logits):return torch.softmax(torch.from_numpy(logits),dim=1).argmax(1).numpy()
