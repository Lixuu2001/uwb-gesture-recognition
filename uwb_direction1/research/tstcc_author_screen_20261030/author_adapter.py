"""Pinned TS-TCC/HAR default model+optimizer with explicit UWB shape-only adapter.

No real dataset, data loader or launcher. Original training step/loss/modules stay
unchanged. Full HAR default batch128 and40epochs are retained for later screening.
Author augmentations are not silently repaired or claimed physically faithful.
"""
import hashlib,importlib.util,json,random,sys
from pathlib import Path
import numpy as np
import torch
B=Path(__file__).resolve().parent;SOURCE=B/'vendor/TS-TCC'
EINOPS=Path('/workspace/uwb-convtran-screen-20261030/vendor/site-packages')
sys.path[:0]=[str(SOURCE),str(EINOPS)]
def source_check():
 for row in json.loads((B/'SOURCE_INDEX.json').read_text())['files']:
  data=(SOURCE/row['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256']
 from config_files.HAR_Configs import Config
 return Config
def make_models(seed=20261029):
 Config=source_check();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 from models.model import base_Model
 from models.TC import TC
 cfg=Config();assert cfg.batch_size==128 and cfg.num_epoch==40 and cfg.TC.timesteps==6 and cfg.TC.hidden_dim==100
 cfg.input_channels=567;cfg.num_classes=12;cfg.features_len=13
 model=base_Model(cfg);temporal=TC(cfg,torch.device('cpu'))
 opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,betas=(cfg.beta1,cfg.beta2),weight_decay=3e-4)
 t_opt=torch.optim.Adam(temporal.parameters(),lr=cfg.lr,betas=(cfg.beta1,cfg.beta2),weight_decay=3e-4)
 return cfg,model,temporal,opt,t_opt
def author_one_batch(model,temporal,opt,t_opt,cfg,aug1,aug2):
 from trainer.trainer import model_train
 return model_train(model,temporal,opt,t_opt,torch.nn.CrossEntropyLoss(),[(aug1,torch.zeros(len(aug1),dtype=torch.long),aug1,aug2)],cfg,torch.device('cpu'),'self_supervised')
