import copy
import pytest
from runner import assert_control_match


def fixture():
    source = {'code': {'trainer.py': 'hash'}, 'data': {'y': 'labels'}, 'runtime': {'torch': 'same'}}
    config = {'seed': 2026, 'warmup_lr': 1e-4}
    manifest = {'source': copy.deepcopy(source), 'config': dict(config, augmentation={'mode': 'PAIRED'})}
    result = {'identity': {'source': copy.deepcopy(source), 'config': dict(manifest['config'], augmentation_fold=1, active_mode='CONTROL'), 'fold': 1}}
    return source, config, manifest, result


def test_control_contract_accepts_disabled_time_reference():
    assert_control_match(*fixture())


@pytest.mark.parametrize('field', ['weights_source', 'config', 'mode', 'manifest_binding'])
def test_control_contract_rejects_unmatched_reference(field):
    source, config, manifest, result = fixture()
    if field == 'weights_source': source['code']['trainer.py'] = 'changed'
    elif field == 'config': config['warmup_lr'] = 2e-4
    elif field == 'mode': result['identity']['config']['active_mode'] = 'AUG'
    else: result['identity']['source']['data']['y'] = 'other'
    with pytest.raises(ValueError): assert_control_match(source, config, manifest, result)
