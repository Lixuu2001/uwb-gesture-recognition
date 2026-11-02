import math
def select(p,steps,last,ev,saves,loads,size,rss,free):
 assert (p['epochs'],p['batch_size'],p['trainable_parameters'],p['seed'],p['training_updates'])==(40,15,11182668,20261029,12560)
 assert len(steps)==6 and len(last)==1 and len(ev)==len(saves)==len(loads)==3
 assert all(math.isfinite(x) and x>0 for x in steps+last+ev+saves+loads+[size,rss])
 seconds=2*12560*max(steps+last)+2*2680*max(ev)+2*147*max(ev)+8*max(saves)+4*max(loads)+180
 full=size;model=(size+1)//2+1048576;meta=32*1048576
 # Conservative gzip overhead bound, to be enforced by actual streaming writer.
 archive_upper=math.ceil((p['weights_bytes']+full+model+meta)*1.01)+1048576
 checkpoint_peak=model+2*full
 archive_peak=model+full+archive_upper+3*7*1048576
 disk=max(checkpoint_peak,archive_peak)+64*1048576
 return dict(resource_gate_passed=seconds<=43200 and 4*seconds<=172800 and rss+2097152<=14680064 and disk<=free,estimated_seconds=seconds,estimated_hours=seconds/3600,estimated_CPU_seconds=4*seconds,estimated_max_RSS_KiB=rss+2097152,required_additional_workspace_bytes=disk,workspace_free_bytes=free,full_checkpoint_bytes=size,model_only_upper_bytes=model,stream_archive_upper_bytes=archive_upper,streaming_backup_implementation_pending=True,actual_UWB_fit_started=False,accuracy_measured=False)
