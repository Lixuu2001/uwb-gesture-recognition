"""CPU-only completeness gate for the formal sixteen-run deployment replay."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def module():
    spec = importlib.util.spec_from_file_location('e2e_deployment_check', HERE / 'verify_deployment.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def complete_records():
    return [dict(mode=mode, fold=fold, seed=2026, passed=True)
            for mode in ('R', 'MV') for fold in range(1, 9)]


def test_import_ignores_foreign_deploy_module_in_same_process(monkeypatch):
    foreign = types.ModuleType('deploy')
    foreign.marker = 'unrelated parent deployment module'
    monkeypatch.setitem(sys.modules, 'deploy', foreign)
    check = module()
    assert check.HERE == HERE
    assert Path(check.load_deployment.__code__.co_filename).resolve() == HERE / 'deploy.py'
    assert sys.modules['deploy'] is foreign


def test_complete_unique_seed2026_replays_pass():
    result = module().summarize_replays(complete_records())
    assert result['complete_eight_fold_pairing'] is True
    assert result['passed'] is True


@pytest.mark.parametrize('fault', ['empty', 'missing', 'duplicate_extra', 'duplicate_replacement',
                                  'wrong_seed', 'all_wrong_seed', 'missing_seed'])
def test_incomplete_duplicate_or_wrong_seed_replays_cannot_pass(fault):
    records = complete_records()
    if fault == 'empty':
        records = []
    elif fault == 'missing':
        records.pop()
    elif fault == 'duplicate_extra':
        records.append(records[0].copy())
    elif fault == 'duplicate_replacement':
        records[-1] = records[0].copy()
    elif fault == 'wrong_seed':
        records[0]['seed'] = 2027
    elif fault == 'all_wrong_seed':
        for record in records:
            record['seed'] = 2027
    else:
        del records[0]['seed']
    result = module().summarize_replays(records)
    assert result['complete_eight_fold_pairing'] is False
    assert result['passed'] is False


def test_one_failed_parity_blocks_otherwise_complete_replay():
    records = complete_records()
    records[-1]['passed'] = False
    result = module().summarize_replays(records)
    assert result['complete_eight_fold_pairing'] is True
    assert result['passed'] is False
