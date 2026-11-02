"""Future full-layer ResNet18 fine-tuning prototype;not yet a UWB experiment."""
import sys
sys.path.insert(0,'/workspace/uwb-resnet18-imagenet-source-20261102')
from native_resnet18 import np,torch,nn,F,make as pretrained,preprocess,encoder_digest
def make():
 m=pretrained(True);torch.manual_seed(20261029);m.fc=nn.Linear(512,12);m.requires_grad_(True);m.train();assert sum(p.numel() for p in m.parameters())==11182668 and all(p.requires_grad for p in m.parameters());return m
def optimizer(m):return torch.optim.SGD(m.parameters(),lr=3e-4,momentum=.9,weight_decay=1e-4,dampening=0,nesterov=False)
def augmented_pixels(raw,generator):
 z=preprocess(raw);mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1);std=torch.tensor([.229,.224,.225]).reshape(1,3,1,1);u=(z*std+mean).clamp(0,1)
 n=len(z);scale=.8+.4*torch.rand(n,generator=generator);shift=torch.randint(-32,33,(n,2),generator=generator);theta=torch.zeros((n,2,3));theta[:,0,0]=1/scale;theta[:,1,1]=1/scale;theta[:,0,2]=-shift[:,0]/(112*scale);theta[:,1,2]=-shift[:,1]/(112*scale)
 grid=F.affine_grid(theta,u.shape,align_corners=False);aug=F.grid_sample(u,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
 return (aug-mean)/std
def step(m,opt,raw,labels,generator):
 assert m.training and all(p.requires_grad for p in m.parameters());opt.zero_grad(set_to_none=True);logits=m(augmented_pixels(raw,generator));loss=F.cross_entropy(logits,torch.as_tensor(labels,dtype=torch.long));assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step();assert all(torch.isfinite(p).all() for p in m.parameters());return float(loss.detach())
