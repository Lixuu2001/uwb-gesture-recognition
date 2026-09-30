"""Strict CPU audit of saved trainable-encoder runs and user-paired effects."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
BASE = PARENT.parent
sys.path.insert(0, str(HERE))


def _load(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_report = _load(PARENT / 'report.py', '_e2e_parent_report_helpers')
_model = _load(HERE / 'model.py', '_e2e_audit_model')
_require, _same, _close = _report._require, _report._same, _report._close
_read, _digest, _ce = _report._read, _report._digest, _report._ce


class _StateView:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


def _state_hash(state, prefix=''):
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    _require(bool(selected), f'No saved state for prefix {prefix}')
    return _model.state_hash(_StateView(selected))


def _source_audit(directory, config):
    snapshots = directory.parent / 'source_code'
    for raw_path, expected in config['source_code'].items():
        source = Path(raw_path)
        if snapshots.is_dir():
            prefix = 'e2e_' if source.parent.resolve() == HERE else 'parent_'
            source = snapshots / (prefix + source.name)
        _require(source.is_file() and _digest(source) == expected,
                 f'Saved training source changed or missing: {source}')


def _training_stats(features):
    # Match train_e2e.main -> setup before performing float32 reductions.
    torch.set_num_threads(4)
    return _model._data.fit_scale(features)


def _verify_project(config, arrays, deployment, lineage):
    sys.path.insert(0, str(BASE))
    from pipeline import split
    y = np.load(BASE / 'prepared/y.npy')
    subjects = np.load(BASE / 'prepared/subj.npy')
    _require(config['split_seed'] == 2026, 'Expected split seed 2026')
    tr, va, te = split(y, subjects, config['fold'], 2026)
    for key, expected in [('train_ids', tr), ('val_ids', va), ('ids', te), ('y', y[te]), ('val_y', y[va])]:
        _same(arrays[key], expected, f'{key} differs from exact project split/labels')
    for raw_path, expected in config['source_identity'].items():
        _require(_digest(raw_path) == expected, f'Raw data/encoder source changed: {raw_path}')
    cache_meta = _report._find_cache(config, config['source_identity'])
    cache_path = cache_meta.with_suffix('.npz')
    _require(_digest(cache_path) == config['cache_sha256'], 'Frozen normalization cache content changed')
    with np.load(cache_path, allow_pickle=False) as cache:
        for key, cached in [('ids', 'test_ids'), ('val_ids', 'val_ids'), ('train_ids', 'train_ids'),
                            ('y', 'test_y'), ('val_y', 'val_y'), ('base_logits', 'test_base')]:
            _same(arrays[key], cache[cached], f'Saved {key} differs from source cache')
        mu, sd = _training_stats(torch.from_numpy(cache['train_x']))
    _require(torch.equal(mu, deployment['state']['mu']) and torch.equal(sd, deployment['state']['sd']),
             'Saved normalization differs from training-only cache statistics')
    model, original = _model.build_model(config['fold'], config['seed'], mu, sd, device='cpu')
    _require(original == lineage, 'Rebuilt initialization differs from recorded source/seed hashes')
    initial_weights = {key: value.detach().clone() for key, value in model.encoder.named_parameters()}
    model.load_state_dict(deployment['state'], strict=True)
    changed = sum(not torch.equal(initial_weights[key], value) for key, value in model.encoder.named_parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    final_hash = _model.state_hash(model.encoder)
    return {'updated_encoder_tensors': changed, 'actual_trainable_parameters': trainable,
            'verified_final_encoder_sha256': final_hash}


def audit_run(directory, verify_project=True):
    """Recompute all reported metrics and deployed encoder status from artifacts."""
    directory = Path(directory)
    status = _read(directory / 'status.json')
    _require(status.get('status') == 'complete', f'Run is incomplete: {directory}')
    config = _read(directory / 'config.json')
    history = _read(directory / 'history.json')
    lineage = _read(directory / 'initialization.json')
    for key in ('fold', 'mode', 'seed'):
        _require(status[key] == config[key], f'Status/config mismatch: {key}')
    _require(config['mode'] in ('R', 'MV'), 'Unknown training mode')
    for filename in ('deployment.pt', 'predictions.npz'):
        _require(_digest(directory / filename) == status['artifacts'][filename], f'{filename} hash mismatch')
    _source_audit(directory, config)
    deployment = torch.load(directory / 'deployment.pt', map_location='cpu', weights_only=False)
    _require(deployment['config'] == config and deployment['initialization'] == lineage,
             'Deployment config/initialization mismatch')
    _require(deployment['best_epoch'] == status['best_epoch'], 'Deployment best epoch mismatch')
    _require(lineage['source_fold'] == config['fold'] and lineage['source_seed'] == 2026
             and lineage['model_seed'] == config['seed'], 'Initialization fold/seed mismatch')
    _require(status['initial_encoder_sha256'] == lineage['initial_encoder_sha256'], 'Initial encoder hash mismatch')
    state = deployment['state']
    mu, sd = state['mu'], state['sd']
    _require(mu.shape == sd.shape == (1, 3, 1664) and torch.isfinite(mu).all()
             and torch.isfinite(sd).all() and (sd > 0).all(), 'Invalid saved normalization')
    _require(all(torch.isfinite(value).all() for value in state.values()), 'Nonfinite deployment weights')
    final_encoder_hash = _state_hash(state, 'encoder.')
    _require(final_encoder_hash == status['final_encoder_sha256'], 'Saved encoder hash differs from reported final hash')
    changed = final_encoder_hash != lineage['initial_encoder_sha256']
    _require(changed == status['encoder_changed'], 'Reported encoder_changed does not match saved weights')
    with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    partitions = []
    for key in ('train_ids', 'val_ids', 'ids'):
        ids = arrays[key]
        _require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids) > 0
                 and (ids >= 0).all() and len(np.unique(ids)) == len(ids), f'Invalid/duplicate {key}')
        partitions.append(set(ids.tolist()))
    _require(not any(partitions[a] & partitions[b] for a, b in ((0, 1), (0, 2), (1, 2))),
             'Train/validation/test overlap')
    for label_key, id_key, logits_keys in [('y', 'ids', ('logits', 'base_logits')),
                                         ('val_y', 'val_ids', ('val_logits',))]:
        labels = arrays[label_key]
        _require(labels.shape == arrays[id_key].shape and np.issubdtype(labels.dtype, np.integer)
                 and ((labels >= 0) & (labels < 12)).all(), f'Invalid {label_key}')
        for key in logits_keys:
            _require(arrays[key].shape == (len(labels), 12) and np.isfinite(arrays[key]).all(),
                     f'Invalid/nonfinite {key}')
    prediction = arrays['logits'].argmax(1)
    _same(prediction, arrays['pred'], 'Saved prediction differs from logits argmax')
    accuracy = float((prediction == arrays['y']).mean())
    base_accuracy = float((arrays['base_logits'].argmax(1) == arrays['y']).mean())
    macro_f1 = float(f1_score(arrays['y'], prediction, labels=np.unique(arrays['y']), average='macro', zero_division=0))
    for key, actual in [('accuracy', accuracy), ('base_accuracy', base_accuracy), ('macro_f1', macro_f1)]:
        _close(actual, status[key], f'Recomputed {key} differs from status')
    _require(status['n_test'] == len(prediction), 'Test count mismatch')
    _require(history and [row['epoch'] for row in history] == list(range(len(history))), 'Nonconsecutive epoch history')
    _require(all(np.isfinite(row['val_ce']) for row in history), 'Nonfinite validation CE history')
    _require(status['epochs'] == len(history) - 1 and len(status['sampling_hashes']) == status['epochs'],
             'Epoch/sampling count mismatch')
    _require(status['epochs'] <= config['epochs'], 'Training exceeded configured epoch budget')
    selected = min(history, key=lambda row: row['val_ce'])
    _require(selected['epoch'] == status['best_epoch'], 'Selected model is not first minimum validation CE')
    reached_budget = status['epochs'] == config['epochs']
    valid_early_stop = (status['epochs'] >= config['minimum_epochs']
                        and status['epochs'] - status['best_epoch'] >= config['patience'])
    _require(reached_budget or valid_early_stop,
             'Completed run neither reached its epoch budget nor met the configured early-stop rule')
    val_ce = _ce(arrays['val_logits'], arrays['val_y'])
    _close(val_ce, selected['val_ce'], 'Saved validation CE differs from selected history')
    _close(val_ce, status['best_val_ce'], 'Saved validation CE differs from selected status')
    _close(history[0]['val_ce'], float(np.log(12)), 'Epoch 0 is not the zero-initialized direct classifier')
    _close(float((arrays['val_logits'].argmax(1) == arrays['val_y']).mean()),
           selected['val_accuracy'], 'Selected validation accuracy mismatch')
    gradients = [row.get('max_encoder_gradient_norm', 0.) for row in history]
    _require(all(np.isfinite(value) and value >= 0 for value in gradients), 'Invalid encoder gradient history')
    gradient_observed = any(value > 0 for value in gradients)
    _require(gradient_observed == status['encoder_gradient_observed'], 'Encoder gradient flag differs from history')
    if status['best_epoch'] == 0:
        _require(not changed, 'Epoch 0 must retain the initial encoder')
        _require(np.count_nonzero(arrays['logits']) == 0 and np.count_nonzero(arrays['val_logits']) == 0,
                 'Epoch 0 must have zero direct-classifier logits')
    project = {'updated_encoder_tensors': None, 'actual_trainable_parameters': None}
    if verify_project:
        project = _verify_project(config, arrays, deployment, lineage)
        _require(project['actual_trainable_parameters'] == status['trainable_parameters'], 'Trainable parameter count mismatch')
        _require(project['verified_final_encoder_sha256'] == final_encoder_hash, 'Instantiated encoder hash mismatch')
    resume_identity_verified = False
    progress_path = directory / 'progress.pt'
    if progress_path.exists():
        progress = torch.load(progress_path, map_location='cpu', weights_only=False)
        info = progress['info']
        if 'run_identity' in info:
            expected_identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            _require(info['run_identity'] == expected_identity, 'Resume checkpoint run/config identity mismatch')
            resume_identity_verified = True
        _require(info['history'] == history and info['sampling_hashes'] == status['sampling_hashes']
                 and info['epoch'] == status['epochs'] and info['best_epoch'] == status['best_epoch'],
                 'Resume checkpoint metadata differs from final history/selection')
        _close(info['best'], val_ce, 'Resume checkpoint best CE mismatch')
        _require(_state_hash(info['best_state']) == _state_hash(state), 'Resume best state differs from deployment')
    return {**status, **project, 'accuracy': accuracy, 'base_accuracy': base_accuracy, 'macro_f1': macro_f1,
            'val_ce': val_ce, 'encoder_changed': changed, 'encoder_gradient_observed': gradient_observed,
            'resume_identity_verified': resume_identity_verified,
            'initial_fuser_sha256': lineage['initial_fuser_sha256'], 'normalization_sha256': _state_hash({'mu': mu, 'sd': sd}),
            'config': config, 'maximum_epochs': config['epochs'], 'cache_sha256': config['cache_sha256'],
            'source_identity': config['source_identity'], 'directory': str(directory.resolve()),
            'project_verified': verify_project,
            'partition_hashes': {key: _report._array_hash(arrays[key]) for key in
                                ('train_ids', 'val_ids', 'ids', 'y', 'val_y', 'base_logits')}}


def summarize_records(records):
    indexed, common = {}, {}
    for record in records:
        key = (record['mode'], record['seed'], record['fold'])
        _require(key not in indexed, f'Duplicate run: {key}')
        indexed[key] = record
        pair = (record['seed'], record['fold'])
        if pair in common:
            first = common[pair]
            for field in ('initial_encoder_sha256', 'initial_fuser_sha256', 'normalization_sha256',
                          'cache_sha256', 'source_identity'):
                _require(first[field] == record[field], f'Paired arms differ in {field}')
            _close(first['base_accuracy'], record['base_accuracy'], 'Paired anchor accuracy differs')
            prefix = min(len(first['sampling_hashes']), len(record['sampling_hashes']))
            _require(first['sampling_hashes'][:prefix] == record['sampling_hashes'][:prefix], 'Paired sampler prefix mismatch')
            if 'config' in first and 'config' in record:
                _require({k: v for k, v in first['config'].items() if k != 'mode'} ==
                         {k: v for k, v in record['config'].items() if k != 'mode'}, 'Paired training/provenance config mismatch')
            if 'partition_hashes' in first and 'partition_hashes' in record:
                _require(first['partition_hashes'] == record['partition_hashes'], 'Paired split/label/anchor mismatch')
        common[pair] = record
    summaries = []
    for mode, seed in sorted({(r['mode'], r['seed']) for r in records}):
        rows = sorted([r for r in records if r['mode'] == mode and r['seed'] == seed], key=lambda r: r['fold'])
        effects = []
        for row in rows:
            reference = indexed.get(('R', seed, row['fold']))
            if reference:
                effects.append({'fold': row['fold'], 'accuracy': row['accuracy'],
                                'control_accuracy': reference['accuracy'], 'base_accuracy': row['base_accuracy'],
                                'gain_pp': 100 * (row['accuracy'] - reference['accuracy'])})
        gains = np.array([r['gain_pp'] for r in effects])
        mean_gain = float(gains.mean()) if len(gains) else None
        ci = None
        if len(gains):
            rng = np.random.default_rng(9999)
            bootstrap = gains[rng.integers(0, len(gains), size=(10000, len(gains)))].mean(1)
            ci = np.quantile(bootstrap, [.025, .975]).tolist()
        accuracy = float(np.mean([r['accuracy'] for r in rows]))
        anchor = float(np.mean([r['base_accuracy'] for r in rows]))
        complete = {r['fold'] for r in rows} == set(range(1, 9)) and len(effects) == 8
        improved = int((gains > 0).sum())
        budget_limited = [r['fold'] for r in rows if r['epochs'] >= r['maximum_epochs']
                          and r['best_epoch'] >= max(1, r['maximum_epochs'] - 2)]
        summaries.append({'mode': mode, 'seed': seed, 'n_folds': len(rows), 'folds': [r['fold'] for r in rows],
                          'accuracy': accuracy, 'macro_f1': float(np.mean([r['macro_f1'] for r in rows])),
                          'base_accuracy': anchor, 'anchor_gain_pp': 100 * (accuracy - anchor),
                          'paired_gain_pp': mean_gain, 'paired_effects_pp': effects, 'bootstrap_ci95_pp': ci,
                          'n_paired_folds': len(effects), 'users_improved': improved, 'complete': complete,
                          'gate': bool(mode == 'MV' and complete and mean_gain >= 2 and improved >= 6 and accuracy > anchor),
                          'encoder_changed_runs': sum(r['encoder_changed'] for r in rows),
                          'encoder_gradient_runs': sum(r['encoder_gradient_observed'] for r in rows),
                          'resume_identity_verified_runs': sum(r.get('resume_identity_verified', False) for r in rows),
                          'updated_encoder_tensors': {str(r['fold']): r['updated_encoder_tensors'] for r in rows},
                          'trainable_parameters': sorted({r['trainable_parameters'] for r in rows}),
                          'best_epoch_zero_count': sum(r['best_epoch'] == 0 for r in rows),
                          'best_epochs': {str(r['fold']): r['best_epoch'] for r in rows},
                          'budget_limited_folds': budget_limited,
                          'elapsed_training_seconds': sum(r['elapsed_training_seconds'] for r in rows)})
    return {'summaries': summaries, 'records': records,
            'aggregation': 'equal held-user weights; model seeds reported separately',
            'bootstrap': {'unit': 'held_user', 'draws': 10000, 'seed': 9999, 'interval': 'percentile_95'}}


def _markdown(audit):
    lines = ['# 可训练编码器的缺失视角融合审计', '',
             '训练中编码器和融合头均可更新；验证、测试及部署仅接受 Right 雷达输入。', '',
             '| 方法 | 种子 | 用户数 | 准确率 | 相对匹配 R 对照/百分点 | 改善用户 | 编码器更新运行 | 判定 |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for row in audit['summaries']:
        gain = '缺少对照' if row['paired_gain_pp'] is None else f'{row["paired_gain_pp"]:+.3f}'
        decision = '不完整' if not row['complete'] else '对照' if row['mode'] == 'R' else '通过投入门槛' if row['gate'] else '未过投入门槛'
        lines.append(f'| E2E_{row["mode"]} | {row["seed"]} | {row["n_folds"]}/8 | {100*row["accuracy"]:.3f}% | '
                     f'{gain} | {row["users_improved"]}/{row["n_paired_folds"]} | '
                     f'{row["encoder_changed_runs"]}/{row["n_folds"]} | {decision} |')
    lines.append('')
    for row in audit['summaries']:
        ci = row['bootstrap_ci95_pp']
        interval = '无配对区间' if ci is None else f'配对提升的用户 bootstrap 95% 区间 [{ci[0]:+.3f}, {ci[1]:+.3f}] 个百分点'
        lines.append(f'- E2E_{row["mode"]} / {row["seed"]}：冻结原始分类器 {100*row["base_accuracy"]:.3f}%，'
                     f'相对原始分类器 {row["anchor_gain_pp"]:+.3f} 个百分点；{interval}。'
                     f'可训练参数 {row["trainable_parameters"]}；梯度记录非零 {row["encoder_gradient_runs"]}/{row["n_folds"]}；'
                     f'恢复身份已核验 {row["resume_identity_verified_runs"]}/{row["n_folds"]}。')
        lines.append(f'  各折实际改变的编码器参数张量数：{row["updated_encoder_tensors"]}；'
                     f'各折选中轮次：{row["best_epochs"]}。')
        if row['budget_limited_folds']:
            lines.append(f'  用户 {row["budget_limited_folds"]} 在预算最后 3 轮内选中最优模型；'
                         '这提示训练预算可能限制结果，不能据此声称已经充分收敛。')
    lines.extend(['', f'完整运行 {len(audit["records"])} 个；未完成运行 {len(audit.get("incomplete_runs", []))} 个。',
                  '恢复身份仅在检查点身份摘要与本次配置匹配时计为已核验；缺失身份标记的旧记录不计为已核验。',
                  '投入门槛：完整 8 位用户、相对同初始化 R 对照平均至少 +2 个百分点、至少 6 位用户改善，'
                  '同时准确率超过冻结原始分类器。门槛不是显著性检验。',
                  '采用固定 2026 划分、训练用户内部验证，以 Right 硬交叉熵选择轮次，包括零初始化分类头的第 0 轮。'
                  '单种子筛选不能代表跨初始化稳定；这 8 位用户已用于开发，bootstrap 区间不构成外部人群泛化证明。',
                  '本轮只验证可训练编码器上的缺失视角联合分类；结果不能概括为其他两条路线或全部多视角方法有效/无效。',
                  '逐折指标、初始化/数据/模型哈希、改变的权重数量与训练预算记录见 audit.json。'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=HERE / 'runs')
    args = parser.parse_args()
    _require(args.runs.is_dir(), f'Run directory does not exist: {args.runs}')
    records, incomplete = [], []
    for directory in sorted(args.runs.iterdir()):
        if not directory.is_dir() or not (directory / 'config.json').exists():
            continue
        status = _read(directory / 'status.json') if (directory / 'status.json').exists() else {'status': 'missing'}
        if status.get('status') != 'complete':
            incomplete.append({'directory': str(directory.resolve()), 'status': status})
            continue
        try:
            records.append(audit_run(directory))
        except Exception as exc:
            raise ValueError(f'E2E integrity audit failed for {directory.name}: {exc}') from exc
    audit = summarize_records(records)
    audit['incomplete_runs'] = incomplete
    for filename, content in [('audit.json', json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)),
                              ('SUMMARY.md', _markdown(audit))]:
        path = args.runs / filename
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(path)
    print(f'E2E_AUDIT_OK complete_runs={len(records)} incomplete_runs={len(incomplete)} report={args.runs / "SUMMARY.md"}')


if __name__ == '__main__':
    main()
