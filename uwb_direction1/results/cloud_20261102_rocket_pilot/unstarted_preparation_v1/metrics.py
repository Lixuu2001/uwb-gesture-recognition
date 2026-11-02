def gate(score,reference=.9866666666666667,minimum=.005,tolerance=1e-12):
    import math
    assert all(math.isfinite(v) for v in [score,reference,minimum,tolerance]) and 0<=score<=1
    delta=score-reference
    return dict(delta_macro=delta,predeclared_gate_passed=delta+tolerance>=minimum,
       route='separately_register_source6_confirmation' if delta+tolerance>=minimum else 'close_exact_ROCKET10000_native_RidgeCV_seed_configuration_no_rescue')
