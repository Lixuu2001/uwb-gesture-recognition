"""Strict saved-artifact audit and equal-user summaries for frozen adapters."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
STAGES = {1: ('R_ONLY', 'RANDOM_ZERO', 'JOINT_ZERO', 'JOINT_TOKEN', 'JOINT_CONTRAST'),
          2: ('SS_R_ONLY', 'SS_JOINT', 'SS_ALIGN'),
          3: ('PROTO_R_ONLY', 'PROTO_MULTI', 'PROTO_MLP')}
CONTROLS = {1: 'R_ONLY', 2: 'SS_R_ONLY', 3: 'PROTO_R_ONLY'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@lru_cache(maxsize=256)
def _digest_cached(path, size, mtime_ns):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _digest(path):
    path = Path(path)
    stat = path.stat()
    return _digest_cached(str(path), stat.st_size, stat.st_mtime_ns)


def _array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def _same(actual, expected, message):
    _require(np.array_equal(actual, expected), message)


def _close(actual, expected, message, tolerance=1e-6):
    _require(np.isfinite(actual) and np.isfinite(expected)
             and np.isclose(actual, expected, atol=tolerance, rtol=tolerance), message)


def _ce(logits, labels):
    logits = logits.astype(np.float64)
    offset = logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(logits - offset).sum(axis=1)) + offset[:, 0]
    return float((logsumexp - logits[np.arange(len(labels)), labels]).mean())


def _stage(arm):
    for stage, arms in STAGES.items():
        if arm in arms:
            return stage
    raise ValueError(f'Unknown arm: {arm}')


def _find_cache(config, encoder_identity):
    for path in sorted(HERE.glob(f'cache*/fold{config["fold"]}.json')):
        meta = _read(path)
        if meta.get('cache_sha256') == config['cache_sha256'] and meta.get('identity') == encoder_identity:
            return path
    raise ValueError('No cache manifest matches both run content hash and encoder identity')


def _verify_project(config, arrays, deployment):
    sys.path.insert(0, str(BASE))
    from pipeline import split
    y = np.load(BASE / 'prepared/y.npy')
    subjects = np.load(BASE / 'prepared/subj.npy')
    _require(config['split_seed'] == 2026, 'Expected fixed split seed 2026')
    tr, va, te = split(y, subjects, config['fold'], 2026)
    for key, expected in [('train_ids', tr), ('val_ids', va), ('ids', te),
                          ('y', y[te]), ('val_y', y[va])]:
        _same(arrays[key], expected, f'{key} does not match project split/labels')
    meta_path = _find_cache(config, deployment['encoder_identity'])
    meta = _read(meta_path)
    cache_path = meta_path.with_suffix('.npz')
    _require(_digest(cache_path) == config['cache_sha256'], 'Cache content hash changed')
    _require(deployment['encoder_identity'] == meta['identity'], 'Deployment encoder lineage differs from cache')
    for path, expected in meta['identity'].items():
        _require(_digest(path) == expected, f'Encoder/cache source changed: {path}')
    # Recorded training code is compared across runs; current working copies may
    # legitimately evolve after a completed experiment. Never substitute them.
    _require(_digest(BASE / f'label_smoothing_results/global_matrix_h{config["fold"]}.npz')
             == config['matrix_sha256'], 'Label matrix changed')
    with np.load(cache_path, allow_pickle=False) as cache:
        for key, cached in [('train_ids', 'train_ids'), ('val_ids', 'val_ids'), ('ids', 'test_ids'),
                            ('y', 'test_y'), ('val_y', 'val_y'), ('base_logits', 'test_base'),
                            ('val_base_logits', 'val_base')]:
            _same(arrays[key], cache[cached], f'Saved {key} differs from cache')


def audit_run(directory, verify_project_split=True):
    """Return independently recomputed metrics, or raise on any integrity failure."""
    directory = Path(directory)
    status = _read(directory / 'status.json')
    _require(status.get('status') == 'complete', f'Run is incomplete: {directory}')
    config = _read(directory / 'config.json')
    readout = config.get('readout', 'residual')
    _require(readout in ('residual', 'direct'), 'Unknown readout mode')
    snapshot_directory = directory.parent / 'source_code'
    if snapshot_directory.is_dir():
        for filename, expected in config['code_sha256'].items():
            source = snapshot_directory / filename
            _require(source.is_file() and _digest(source) == expected,
                     f'Saved training source hash mismatch: {filename}')
    history = _read(directory / 'history.json')
    for field in ('arm', 'fold', 'seed', 'shuffle'):
        _require(status[field] == config[field], f'Status/config mismatch: {field}')
    for filename, key in [('predictions.npz', 'predictions_sha256'), ('deployment.pt', 'deployment_sha256')]:
        _require(_digest(directory / filename) == status[key], f'{filename} hash mismatch')
    deployment = torch.load(directory / 'deployment.pt', map_location='cpu', weights_only=False)
    _require(deployment['config'] == config, 'Deployment/config mismatch')
    with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    partitions = []
    for key in ('train_ids', 'val_ids', 'ids'):
        ids = arrays[key]
        _require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer)
                 and len(ids) > 0 and (ids >= 0).all(), f'Invalid {key}')
        _require(len(np.unique(ids)) == len(ids), f'Duplicate IDs in {key}')
        partitions.append(set(ids.tolist()))
    _require(not any(partitions[a] & partitions[b] for a, b in ((0, 1), (0, 2), (1, 2))),
             'Train/validation/test ID overlap')
    for label_key, id_key, logits_keys in [('y', 'ids', ('logits', 'base_logits')),
                                         ('val_y', 'val_ids', ('val_logits', 'val_base_logits'))]:
        labels = arrays[label_key]
        _require(labels.shape == arrays[id_key].shape and np.issubdtype(labels.dtype, np.integer)
                 and ((labels >= 0) & (labels < 12)).all(), f'Invalid {label_key}')
        for key in logits_keys:
            _require(arrays[key].shape == (len(labels), 12) and np.isfinite(arrays[key]).all(),
                     f'Invalid/nonfinite {key}')
    pred = arrays['logits'].argmax(axis=1)
    _same(arrays['pred'], pred, 'Saved pred differs from logits argmax')
    accuracy = float((pred == arrays['y']).mean())
    base_accuracy = float((arrays['base_logits'].argmax(1) == arrays['y']).mean())
    macro_f1 = float(f1_score(arrays['y'], pred, labels=np.unique(arrays['y']),
                             average='macro', zero_division=0))
    for key, value in [('accuracy', accuracy), ('base_accuracy', base_accuracy), ('macro_f1', macro_f1)]:
        _close(value, status[key], f'Recomputed {key} differs from status')
    _require(status['n_test'] == len(pred), 'n_test differs from predictions')
    _require(len(history) > 0 and [row['epoch'] for row in history] == list(range(len(history))),
             'History must contain consecutive epochs beginning at 0')
    _require(all(np.isfinite(row['val_ce']) for row in history), 'Nonfinite validation history')
    selected = min(history, key=lambda row: row['val_ce'])
    _require(status['best_epoch'] == selected['epoch'], 'Best epoch is not first minimum validation CE')
    _require(status['epochs'] == len(history) - 1 and len(status['sampling_hashes']) == status['epochs'],
             'Epoch count or sampling history length mismatch')
    val_ce = _ce(arrays['val_logits'], arrays['val_y'])
    _close(val_ce, status['best_val_ce'], 'Saved validation CE differs from selected status')
    _close(val_ce, selected['val_ce'], 'Saved validation CE differs from selected history')
    initial_val_logits = arrays['val_base_logits'] if readout == 'residual' else np.zeros_like(arrays['val_logits'])
    _close(_ce(initial_val_logits, arrays['val_y']), history[0]['val_ce'], 'Epoch 0 CE differs from initial model')
    _close(float((arrays['val_logits'].argmax(1) == arrays['val_y']).mean()),
           selected['val_accuracy'], 'Saved validation accuracy differs from selected history')
    if status['best_epoch'] == 0:
        initial_test_logits = arrays['base_logits'] if readout == 'residual' else np.zeros_like(arrays['logits'])
        _same(arrays['logits'], initial_test_logits, 'Epoch 0 test logits differ from initial model')
        _same(arrays['val_logits'], initial_val_logits, 'Epoch 0 validation logits differ from initial model')
    if verify_project_split:
        _verify_project(config, arrays, deployment)
    return {**status, 'directory': str(directory.resolve()), 'stage': _stage(config['arm']), 'readout': readout,
            'accuracy': accuracy, 'base_accuracy': base_accuracy, 'macro_f1': macro_f1,
            'val_ce': val_ce, 'cache_sha256': config['cache_sha256'], 'config': config,
            'partition_hashes': {key: _array_hash(arrays[key]) for key in
                                ('train_ids', 'val_ids', 'ids', 'y', 'val_y', 'base_logits', 'val_base_logits')},
            'project_split_verified': verify_project_split}


def _method(record):
    return record['arm'] + ('_SHUF' if record.get('shuffle', False) else '')


def summarize_records(records):
    """Use equal weight per held subject, never pooled sample accuracy."""
    readouts = {r.get('readout', r.get('config', {}).get('readout', 'residual')) for r in records}
    _require(len(readouts) <= 1, 'Mixed direct/residual readouts require separate run roots')
    indexed = {}
    common = {}
    for record in records:
        key = (_method(record), record['seed'], record['fold'])
        _require(key not in indexed, f'Duplicate run: {key}')
        indexed[key] = record
        comparison_key = (record['seed'], record['fold'])
        if comparison_key in common:
            first = common[comparison_key]
            _require(first['cache_sha256'] == record['cache_sha256'], 'Compared runs use different caches')
            _close(first['base_accuracy'], record['base_accuracy'], 'Compared anchor accuracies differ')
            shared_epochs = min(len(first['sampling_hashes']), len(record['sampling_hashes']))
            _require(first['sampling_hashes'][:shared_epochs] == record['sampling_hashes'][:shared_epochs],
                     'Compared sampler prefixes differ')
            if 'partition_hashes' in first and 'partition_hashes' in record:
                _require(first['partition_hashes'] == record['partition_hashes'], 'Compared partitions/anchors differ')
            if 'config' in first and 'config' in record:
                exclude = {'arm', 'shuffle'}
                first_config = {k: v for k, v in first['config'].items() if k not in exclude}
                next_config = {k: v for k, v in record['config'].items() if k not in exclude}
                _require(first_config == next_config, 'Compared optimization/provenance settings differ')
            # Retain the longest sampler history to check every overlapping epoch.
            if len(record['sampling_hashes']) > len(first['sampling_hashes']):
                common[comparison_key] = record
        else:
            common[comparison_key] = record
    groups = sorted({(record['stage'], _method(record), record['seed']) for record in records})
    summaries = []
    for stage, arm, seed in groups:
        rows = sorted([r for r in records if r['stage'] == stage and _method(r) == arm and r['seed'] == seed],
                      key=lambda r: r['fold'])
        control = CONTROLS[stage]
        effects = []
        for row in rows:
            reference = indexed.get((control, seed, row['fold']))
            if reference is not None:
                effects.append({'fold': row['fold'], 'accuracy': row['accuracy'],
                                'control_accuracy': reference['accuracy'], 'base_accuracy': row['base_accuracy'],
                                'gain_pp': 100 * (row['accuracy'] - reference['accuracy']),
                                'anchor_gain_pp': 100 * (row['accuracy'] - row['base_accuracy'])})
        gains = np.array([effect['gain_pp'] for effect in effects])
        mean_gain = float(gains.mean()) if len(gains) else None
        if len(gains):
            rng = np.random.default_rng(9999)
            boot = gains[rng.integers(0, len(gains), size=(10000, len(gains)))].mean(axis=1)
            ci = np.quantile(boot, [.025, .975]).tolist()
        else:
            ci = None
        complete = {r['fold'] for r in rows} == set(range(1, 9)) and len(effects) == 8
        accuracy = float(np.mean([r['accuracy'] for r in rows]))
        anchor = float(np.mean([r['base_accuracy'] for r in rows]))
        improved = int((gains > 0).sum())
        summaries.append({'stage': stage, 'arm': arm, 'seed': seed, 'control': control,
                          'n_folds': len(rows), 'folds': [r['fold'] for r in rows],
                          'n_paired_folds': len(effects), 'accuracy': accuracy,
                          'macro_f1': float(np.mean([r['macro_f1'] for r in rows])),
                          'base_accuracy': anchor, 'anchor_gain_pp': 100 * (accuracy - anchor),
                          'paired_gain_pp': mean_gain, 'paired_effects_pp': effects,
                          'bootstrap_ci95_pp': ci, 'users_improved': improved,
                          'best_epoch_zero_count': sum(r['best_epoch'] == 0 for r in rows),
                          'parameters': sorted({r['parameters'] for r in rows}),
                          'trainable_parameters': sorted({r['trainable_parameters'] for r in rows}),
                          'seconds': sum(r['seconds'] for r in rows), 'complete': complete,
                          'gate': bool(arm != control and complete and mean_gain >= 2
                                       and improved >= 6 and accuracy > anchor)})
    return {'summaries': summaries, 'records': records, 'readout': next(iter(readouts), 'residual'),
            'bootstrap': {'unit': 'held_user', 'seed': 9999, 'draws': 10000, 'interval': 'percentile_95'},
            'aggregation': 'unweighted held-user means, separately by model seed'}


def _markdown(audit):
    lines = ['# 单雷达部署：三条冻结特征路线实验审计', '',
             '各项准确率为留出用户等权平均；配对差值比较同阶段、同模型种子的 Right-only 对照。', '',
             '| 阶段 | 方法 | 种子 | 用户数 | 准确率 | 配对提升/百分点 | 改善用户 | 初始模型回退 | 判定 |',
             '|---|---|---|---:|---:|---:|---:|---:|---|']
    for row in audit['summaries']:
        gain = '无对应对照' if row['paired_gain_pp'] is None else f'{row["paired_gain_pp"]:+.3f}'
        verdict = ('不完整' if not row['complete'] else
                   '对照' if row['arm'] == row['control'] else '通过投入门槛' if row['gate'] else '未过投入门槛')
        lines.append(f'| {row["stage"]} | {row["arm"]} | {row["seed"]} | {row["n_folds"]}/8 | '
                     f'{100*row["accuracy"]:.3f}% | {gain} | {row["users_improved"]}/{row["n_paired_folds"]} | '
                     f'{row["best_epoch_zero_count"]} | {verdict} |')
    lines.extend(['', '冻结锚点及配对区间：'])
    for row in audit['summaries']:
        ci = row['bootstrap_ci95_pp']
        interval = '无配对区间' if ci is None else f'配对提升的用户 bootstrap 95% 区间 [{ci[0]:+.3f}, {ci[1]:+.3f}] 个百分点'
        lines.append(f'- {row["arm"]} / {row["seed"]}：对应冻结 Right 锚点 {100*row["base_accuracy"]:.3f}%，'
                     f'相对锚点 {row["anchor_gain_pp"]:+.3f} 个百分点；{interval}。')
    lines.extend(['', f'通过完整性审计的运行：{len(audit["records"])}；未完成运行：{len(audit.get("incomplete_runs", []))}。',
                  '门槛要求完整 8 用户、相对匹配对照平均至少 +2 个百分点、至少 6 用户改善，且超过冻结锚点。'
                  '这是继续投入的规则，不是显著性检验。', '',
                  '本实验复用仅在 Right 上训练的冻结编码器，测试小型融合头；三条路线分别是缺失视角联合训练、'
                  '共享/私有特征分解和类别原型补全的适配，不是对应论文的端到端复现。',
                  '验证集来自训练用户内部，依据 Right-only 硬交叉熵选轮次并包括第 0 轮。'
                  '各模型种子单独汇报；只有一个种子的结果不能声称跨初始化稳定。'
                  '这 8 位用户已用于项目开发；用户 bootstrap 只描述这组用户的变化，不证明对外部人群的泛化。'
                  '未过门槛的结论仅覆盖本次冻结特征和适配器，不排除端到端方法。',
                  '逐折指标、配对差值、参数量、耗时、数据与模型哈希保存在 audit.json。'])
    lines.insert(3, ('本目录为直接分类读出：第 0 轮是零初始化分类头，回退不等于回到冻结原始分类器。'
                     if audit.get('readout') == 'direct' else
                     '本目录为残差读出：第 0 轮修正量为零，回退对应冻结原始 Right 分类器。'))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=HERE / 'runs')
    parser.add_argument('--stage', type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    _require(args.runs.is_dir(), f'Run directory does not exist: {args.runs}')
    records, incomplete = [], []
    for directory in sorted(args.runs.iterdir()):
        if not directory.is_dir() or not (directory / 'config.json').exists():
            continue
        config = _read(directory / 'config.json')
        if args.stage is not None and _stage(config['arm']) != args.stage:
            continue
        status = _read(directory / 'status.json') if (directory / 'status.json').exists() else {'status': 'missing'}
        if status.get('status') != 'complete':
            incomplete.append({'directory': str(directory), 'status': status})
            continue
        try:
            records.append(audit_run(directory))
        except Exception as exc:
            raise ValueError(f'Integrity audit failed for {directory.name}: {exc}') from exc
    audit = summarize_records(records)
    audit['incomplete_runs'] = incomplete
    audit['stage_filter'] = args.stage
    for filename, content in [('audit.json', json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)),
                              ('SUMMARY.md', _markdown(audit))]:
        path = args.runs / filename
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(content, encoding='utf-8')
        temp.replace(path)
    print(f'AUDIT_OK complete_runs={len(records)} incomplete_runs={len(incomplete)} report={args.runs / "SUMMARY.md"}')


if __name__ == '__main__':
    main()
