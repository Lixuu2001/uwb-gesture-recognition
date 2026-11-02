"""Synthetic default-loss/stage-transfer checks against author method ASTs."""
import ast,copy,types,unittest
import torch
from torch import nn
from torch.nn import functional as F
from author_core import BASE,make_model
import training_ops as ops

def author_method(class_name,method_name):
    tree=ast.parse((BASE/'vendor/TSLANet/Classification/TSLANet_classification.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==class_name)
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==method_name)
    scope=dict(torch=torch,optim=torch.optim,args=types.SimpleNamespace(pretrain_lr=1e-3,train_lr=1e-3))
    exec(compile(ast.Module(body=[method],type_ignores=[]),'author-method','exec'),scope)
    return scope[method_name]

class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def test_default_smoothed_loss_and_gradient_vs_pytorch(self):
        torch.manual_seed(17);a=torch.randn(7,12,requires_grad=True);b=a.detach().clone().requires_grad_();y=torch.arange(7)%12
        criterion=ops.classification_criterion();self.assertEqual(criterion.smoothing,.1)
        loss=criterion(a,y);ref=F.cross_entropy(b,y,label_smoothing=.1);loss.backward();ref.backward()
        torch.testing.assert_close(loss,ref,rtol=1e-6,atol=1e-7);torch.testing.assert_close(a.grad,b.grad,rtol=1e-6,atol=1e-7)
    def compare_author(self,stage):
        model,_=make_model();clone,_=make_model();model.train();clone.train();torch.manual_seed(99);x=torch.randn(4,567,90);y=torch.arange(4)%12;rng=torch.get_rng_state()
        criterion=ops.classification_criterion()
        ours=ops.pretraining_loss(model,x) if stage=='pretraining' else ops.classification_loss(model,x,y,criterion)
        ours.backward();torch.set_rng_state(rng)
        wrapper=types.SimpleNamespace(model=clone,criterion=criterion,f1=lambda p,y:p.new_tensor(0.),log=lambda *a,**kw:None)
        original=author_method('model_'+stage,'_calculate_loss')(wrapper,(x,y));original.backward()
        self.assertTrue(torch.equal(ours,original));self.assertTrue(torch.isfinite(ours))
        for (name,p),(other,q) in zip(model.named_parameters(),clone.named_parameters()):
            self.assertEqual(name,other);self.assertEqual(p.grad is None,q.grad is None)
            if p.grad is not None:self.assertTrue(torch.equal(p.grad,q.grad),name)
    def test_pretrain_loss_gradients_exact_author_body(self):self.compare_author('pretraining')
    def test_classification_loss_gradients_exact_author_body(self):self.compare_author('training')
    def test_optimizer_default_matches_both_author_stages(self):
        model=nn.Linear(3,2)
        for stage in ('pretraining','training'):
            ref=author_method('model_'+stage,'configure_optimizers')(model);actual=ops.stage_optimizer(model)
            self.assertEqual(actual.state_dict()['param_groups'],ref.state_dict()['param_groups'])
    def test_pretrain_transfer_resets_optimizer_preserves_head(self):
        model,_=make_model();self.assertEqual(sum(p.numel() for p in model.parameters()),4132879);original=copy.deepcopy(model.state_dict());opt=ops.stage_optimizer(model)
        x=torch.randn(4,567,90);model.train();opt.zero_grad();loss=ops.pretraining_loss(model,x);loss.backward();opt.step()
        self.assertTrue(torch.equal(original['head.weight'],model.state_dict()['head.weight']))
        self.assertTrue(torch.equal(original['head.bias'],model.state_dict()['head.bias']))
        clone,_=make_model();clone.load_state_dict(model.state_dict());fresh=ops.stage_optimizer(clone);self.assertEqual(len(fresh.state),0)
        self.assertGreater(len(opt.state),0);self.assertTrue(all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items()))
        clone.train();fresh.zero_grad();value=ops.classification_loss(clone,x,torch.arange(4));value.backward();fresh.step();self.assertTrue(torch.isfinite(value))
    def test_pretraining_target_attached_and_mask_boolean(self):
        model,_=make_model();model.train();pred,target=model.pretrain(torch.randn(4,567,90));self.assertTrue(target.requires_grad);self.assertIsNotNone(target.grad_fn);self.assertEqual(model.mask.dtype,torch.bool);self.assertGreater(int(model.mask.sum()),0);self.assertEqual(pred.shape,target.shape)

if __name__=='__main__':unittest.main()
