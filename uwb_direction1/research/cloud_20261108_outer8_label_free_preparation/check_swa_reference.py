"""Generated component checks from immutable BSD author code; no real data."""
from pathlib import Path
import ast,json,hashlib,torch
B=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(1)
    source=(B/'author_reference/utils.py').read_text()
    # Python3 reserved async keyword affects parsing only; extract functions below.
    tree=ast.parse(source.replace('async=True','non_blocking=True'))
    names={'moving_average','_check_bn','check_bn','reset_bn','_get_momenta','_set_momenta','bn_update'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(functions)==len(names)
    namespace={'torch':torch}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned-author-generated-check>','exec'),namespace)
    a=torch.nn.BatchNorm1d(2,dtype=torch.float64);b=torch.nn.BatchNorm1d(2,dtype=torch.float64)
    old=a.running_mean.clone();states=[]
    for i,values in enumerate([(1.,3.),(5.,7.),(9.,11.)]):
        with torch.no_grad():b.weight.copy_(torch.tensor(values,dtype=torch.float64));b.bias.fill_(i)
        states.append(b.weight.detach().clone());namespace['moving_average'](a,b,1/(i+1))
    torch.testing.assert_close(a.weight,torch.stack(states).mean(0),rtol=1e-15,atol=1e-15)
    assert torch.equal(a.running_mean,old) # parameters only; source-BN pass required.
    # Execute author's BN refresh on CPU by replacing only its explicit .cuda call.
    class CPUInput(torch.Tensor):
        def cuda(self,**kwargs):return self.as_subclass(torch.Tensor)
    batches=[torch.tensor([[1.,2.],[3.,4.],[5.,8.]],dtype=torch.float64),torch.tensor([[7.,6.],[9.,10.]],dtype=torch.float64)]
    loader=[(x.as_subclass(CPUInput),torch.zeros(len(x),dtype=torch.int64)) for x in batches]
    m=torch.nn.BatchNorm1d(2,affine=False,dtype=torch.float64);momentum=m.momentum
    namespace['bn_update'](loader,m)
    expected_mean=sum(len(x)*x.mean(0) for x in batches)/5
    expected_var=sum(len(x)*x.var(0,unbiased=True) for x in batches)/5
    torch.testing.assert_close(m.running_mean,expected_mean,rtol=1e-15,atol=1e-15)
    torch.testing.assert_close(m.running_var,expected_var,rtol=1e-15,atol=1e-15)
    assert m.momentum==momentum and m.training and int(m.num_batches_tracked)==2
    training=ast.parse((B/'author_reference/train.py').read_text())
    schedule=next(n for n in training.body if isinstance(n,ast.FunctionDef) and n.name=='schedule')
    from types import SimpleNamespace
    ns={'args':SimpleNamespace(swa=True,swa_start=161,epochs=200,swa_lr=.05,lr_init=.1)}
    exec(compile(ast.Module(body=[schedule],type_ignores=[]),'<pinned-schedule>','exec'),ns)
    assert ns['schedule'](0)==.1 and ns['schedule'](161)==.05
    epochs=[e+1 for e in range(200) if (e+1)>=161 and (e+1-161)%1==0]
    assert epochs==list(range(161,201))
    result=dict(state='pinned_SWA_author_components_generated_checks_pass',author_commit='411b2fcad59bec60c6c9eb1eb19ab906540e5ea2',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((B/'author_reference').iterdir())},
        author_parameters_only_equal_weight_average_verified=True,author_sample_weighted_BN_refresh_verified=True,
        default_200_epoch_start161_end200_collection_count=40,source_schedule_AST_extracted=True,
        adaptations=['reserved async keyword replaced only for parsing','generated CPU tensor cuda stub; no author CUDA benchmark executed'],
        original_SWA_benchmark_reproduced=False,real_data_read=False,models_fit=0,held_labels_read=False,
        native_AdamW_dense_tail_not_equivalent_to_original_SGD_epoch_end_SWA=True)
    import sys
    p=B/(sys.argv[1] if len(sys.argv)>1 else 'SWA_COMPONENT_CHECK.json');assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: pinned SWA averaging, BN refresh and schedule components; generated data only')
if __name__=='__main__':main()
