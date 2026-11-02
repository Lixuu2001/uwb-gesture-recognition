from pathlib import Path
import json,hashlib,sys
from native_moderntcn import np,torch,make,assert_structure,source_configuration
B=Path(__file__).resolve().parent
def main():
    m,c=make(meta=True);assert_structure(m,c)
    with torch.device('meta'):
        x=torch.empty(32,90,567);m.train();z=m(x,None,None,None)
    assert tuple(z.shape)==(32,12)
    n=sum(p.numel() for p in m.parameters());tensors=len(list(m.parameters()))
    independent_ffn=sum(2*567*d*d+2*d*567*567 for d in [128,256])
    actual_ffn=sum(p.numel() for name,p in m.named_parameters() if 'ffn' in name)
    expected_ffn=independent_ffn+sum(4*567*d for d in [128,256]);assert actual_ffn==expected_ffn
    # Native OneCycleLR constructor changes lr before any update; training source never steps scheduler.
    tiny=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.Adam([tiny],lr=.001)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,steps_per_epoch=147,pct_start=.3,epochs=100,max_lr=.001)
    assert abs(opt.param_groups[0]['lr']-.001/25)<1e-15
    r=dict(state='pass_meta_structure_and_scheduler_semantics_only',full_released_waveGesture=True,
       total_parameters=n,trainable_tensors=tensors,FP32_parameter_bytes=n*4,
       minimum_FP32_weights_grad_Adam_m_v_bytes=n*16,meta_input_shape=[32,90,567],meta_output_shape=list(z.shape),
       independent_FFN_weight_and_bias_count_verified=True,FFN_parameters=actual_ffn,
       native_initial_OneCycleLR_lr=opt.param_groups[0]['lr'],native_scheduler_stepped_in_training=False,
       real_numerical_forward_or_optimizer_on_full_model=False,CPU_cost_measured=False,UWB_values_labels_fit=False,
       runtime_versions=dict(torch=torch.__version__,numpy=np.__version__),author_preset=source_configuration(),adapter_config=vars(c))
    (B/'META_STRUCTURE_CHECK.json').write_text(json.dumps(r,indent=2)+'\n')
    files={}
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name and Path(name).is_file() and (name.endswith('.py') or '.so' in name):files[str(Path(name).resolve())]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    (B/'DEPENDENCIES.json').write_text(json.dumps(dict(files=files,scope='Imported Python/binary closure;not whole OS or all dynamically linked libraries'),indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ['author_preset','adapter_config']}))
if __name__=='__main__':main()
