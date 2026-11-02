"""Registered conservative full100E producer + full replay timing bound."""
import math
def decision(train,ev,save,load,peak,checkpoint_bytes,free):
    values=[*train,*ev,*save,*load]
    if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Nonfinite/nonpositive cost')
    if any(len(v)!=3 for v in [train,ev,save,load]):raise ValueError('Three registered measurements required')
    wall=2*(14700*max(train)+3485*max(ev)+101*max(save)+2*max(load))+180
    # Conservative CPU bound uses four cores for all elapsed work.
    cpu=4*wall
    margin_peak=peak+2097152
    storage=8*checkpoint_bytes+67108864
    reasons=[]
    if wall>43200:reasons.append('full_producer_and_replay_wall_cost')
    if cpu>172800:reasons.append('full_producer_and_replay_CPU_cost')
    if margin_peak>14680064:reasons.append('memory_plus_2GiB_map_margin')
    if storage>free:reasons.append('eight_checkpoint_sets_plus_64MiB_workspace_storage')
    return dict(resource_gate_passed=not reasons,reasons=reasons,estimated_pair_wall_seconds=wall,estimated_pair_CPU_seconds=cpu,peak_with_margin_KiB=margin_peak,required_checkpoint_workspace_bytes=storage,checkpoint_bytes=checkpoint_bytes,workspace_free_bytes=free)
