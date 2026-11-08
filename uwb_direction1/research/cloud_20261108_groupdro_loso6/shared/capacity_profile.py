"""Verify original fixed GroupDRO timing proof; never retime unchanged code."""
from common import PHASE,ready,read,sha
import json
def main():
    ready();a=read(PHASE/'core_preparation/GENERATED_CORE_RESULT.json');g=read(PHASE/'RESOURCE_GATE.json')
    assert g['gate_passed'] and a['capacity_gate_passed'] and a['profile_protocol_fixed_before_measure']
    assert g['source_result_sha256']==sha(PHASE/'core_preparation/GENERATED_CORE_RESULT.json')
    assert g['source_protocol_sha256']==sha(PHASE/'core_preparation/PROTOCOL.json')
    plan=read(PHASE/'DEVELOPMENT_PROTOCOL.json');assert sha(PHASE/'groupdro_core.py')==plan['component_source']['core_source_sha256']
    value=1.25*31189.589142517*max(1.,a['conservative_ratio'])+600
    assert value==g['reserve_seconds']==a['projected_full6_train_replay_reserve_seconds'] and value<=43200
    print(json.dumps(dict(state='original_fixed_profile_verified_no_retiming_or_real_fit',gate_passed=True,reserve_seconds=value,GroupDRO_native_ratio=a['conservative_ratio'])))
if __name__=='__main__':main()
