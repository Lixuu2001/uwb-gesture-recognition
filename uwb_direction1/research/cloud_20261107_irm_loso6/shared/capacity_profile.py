"""Verify/reuse the explicitly published cost calibration; do not repeat unchanged timing."""
from common import PHASE,ready,read,sha
import json
def main():
    ready();a=read(PHASE/'COST_ESTIMATOR_CALIBRATION_V2.json');g=read(PHASE/'RESOURCE_GATE.json')
    assert g['gate_passed'] and a['calibrated_resource_gate_passed']
    assert g['source_calibration_sha256']==sha(PHASE/'COST_ESTIMATOR_CALIBRATION_V2.json')
    assert sha(a['prior_cost_receipt'])==a['prior_cost_receipt_sha256']
    assert sha(a['historical_Fishr_cost_formula_source'])==a['historical_formula_source_sha256']
    value=1.25*a['actual_prior_VREx6_CPU_reference']*max(1.,a['relative_ratio'])+600
    assert value==g['reserve_seconds']==a['new_estimate_seconds'] and value<=43200
    print(json.dumps(dict(state='existing_calibration_verified_no_new_profile_or_real_fit',gate_passed=True,reserve_seconds=value,IRM_native_ratio=a['relative_ratio'])))
if __name__=='__main__':main()
