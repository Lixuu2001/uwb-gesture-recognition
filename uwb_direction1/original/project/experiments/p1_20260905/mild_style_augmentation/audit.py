"""Independent fixed-style audit; no imports of runner or style producer code.

CPU auditing is read-only except its reports. --replay explicitly runs the original
standalone Right classifier on clean validation/test input and saves new arrays.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = BASE/'cross_user_meta/runs_s2026'
CONTROL_ROOT = BASE/'mild_time_augmentation/runs'
MODES = ('CONTROL', 'STYLE')
STYLE_CONFIG = {'mode': 'STYLE', 'probability': .25, 'share': .1, 'seed': 20260909,
                'donor_policy': 'same_class_cross_user_uniform_user_then_instance',
                'statistics': 'float32_detached_population_std_clamp1e-6'}


def load_auditor(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_auditor('_style_independent_original_audit', BASE/'cross_user_meta/audit.py')
time_audit = load_auditor('_style_independent_time_audit', BASE/'mild_time_augmentation/audit.py')
require = legacy.require


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_config(config, original):
    expected = dict(original, style=STYLE_CONFIG)
    if config.get('smoke'):
        expected.update(smoke=True, warmup_epochs=1, warmup_min=1, epochs=1, min_epochs=1, smoke_steps=3)
    require(config == expected, 'fixed configuration differs from original config plus approved style settings')


def audit_schedule(result, train_ids, labels, subjects, saved_sampler):
    """Reconstruct donors and masks from the pre-shuffle PCG64 state each epoch."""
    config = result['identity']['config']; fold = result['identity']['fold']
    require(config['active_mode'] == 'STYLE' and config['style_fold'] == fold, 'style mode/fold mismatch')
    train_ids = np.asarray(train_ids)
    require(np.array_equal(train_ids, np.unique(train_ids)), 'style training IDs must be sorted and unique')
    require(np.array_equal(saved_sampler['ids'], train_ids), 'style sampler IDs differ')
    rng = np.random.default_rng(config['seed']+fold)
    # Pool construction uses only the six-user training subset, never val/test labels.
    pools = {}
    for sample in train_ids:
        label, user = int(labels[sample]), int(subjects[sample])
        pools.setdefault(label, {}).setdefault(user, []).append(int(sample))
    totals = {'clean': 0, 'style': 0}
    for row in result['history'][1:]:
        payload = json.dumps({'sampler': rng.bit_generator.state, 'fold': fold,
                              'augmentation_seed': 20260909}, sort_keys=True).encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
        require(row['style_seed'] == seed, 'style seed differs from independent sampler state')
        style_rng = np.random.Generator(np.random.PCG64(seed))
        masks = style_rng.random(len(train_ids)) < .25
        donors = np.empty_like(train_ids)
        for index, anchor in enumerate(train_ids):
            pool = pools[int(labels[anchor])]
            users = sorted(u for u in pool if u != int(subjects[anchor]))
            require(bool(users), 'no same-class cross-user training donor')
            user = int(style_rng.choice(users))
            donors[index] = style_rng.choice(pool[user])
        require(np.all(labels[donors] == labels[train_ids]) and
                np.all(subjects[donors] != subjects[train_ids]) and
                np.all(np.isin(donors, train_ids)), 'donors violate training-only same-class cross-user constraint')
        shuffled = rng.permutation(train_ids)
        steps = math.ceil(len(train_ids)/config['warmup_batch'])
        if config['smoke_steps']:
            steps = min(steps, config['smoke_steps'])
        consumed = shuffled[:steps*config['warmup_batch']]
        indices = np.searchsorted(train_ids, consumed)
        triples = np.column_stack((consumed, donors[indices], masks[indices].astype(np.int64))).astype(np.int64)
        counts = {'clean': int((~masks[indices]).sum()), 'style': int(masks[indices].sum())}
        require(row['style_trace'] == hashlib.sha256(triples.tobytes()).hexdigest(), 'style donor/mask/sample trace differs')
        require(row['style_counts'] == counts, 'style counts differ')
        require(row['source_rows'] == len(consumed), 'style source row count differs')
        require(row['steps'] == steps, 'style optimizer step count differs')
        for key in totals:
            totals[key] += counts[key]
    require(saved_sampler['rng'] == rng.bit_generator.state, 'style sampler final RNG differs')
    return {'independent_schedule_verified': True, 'training_counts': totals,
            'donor_partition': 'same class, different user, training IDs only', 'numpy_rng': 'PCG64'}


def audit_pair(results):
    if set(results) != set(MODES):
        return
    left, right = (results[m] for m in MODES)
    for key in ('partitions', 'matrix_sha256', 'initial_state_sha256'):
        require(left['identity'][key] == right['identity'][key], f'paired {key} mismatch')
    for a, b in zip(left['history'], right['history']):
        for key in ('epoch', 'episode_hash', 'steps'):
            require(a[key] == b[key], f'paired {key} differs at epoch {a["epoch"]}')
    require(left['history'][0]['val_ce'] == right['history'][0]['val_ce'], 'paired initial validation differs')


def audit_source(source, original, hashes, snapshot_root):
    require(source['data'] == original['data'], 'prepared data identity changed')
    require(source['runtime'] == original['runtime'], 'runtime identity changed')
    require(source['runtime']['torch'] == torch.__version__ and source['runtime']['cuda'] == torch.version.cuda,
            'audit Torch/CUDA build differs from training')
    for path, digest in original['code'].items():
        require(source['code'].get(path) == digest, f'original code identity changed: {path}')
    for group in ('code', 'data'):
        for path, digest in source[group].items():
            hashes.verify(path, digest)
    for index, (path, digest) in enumerate(source['code'].items()):
        hashes.verify(snapshot_root/'source'/f'{index:02d}_{Path(path).name}', digest)


def audit_pretrain(path, ledger, source, hashes):
    hashes.verify(path, ledger['pretrain_check_sha256'])
    certificate = read_json(path)
    require(certificate.get('status') == 'verified' and certificate.get('source_code') == source['code'],
            'pretrain certificate is not verified for these exact frozen sources')


def audit_manifest(root, hashes):
    original = read_json(OLD/'manifest.json')
    manifest = read_json(root/'manifest.json')
    source = manifest['source']
    audit_config(manifest['config'], original['config'])
    expected_code = set(original['source']['code']) | {str(HERE/'runner.py'), str(HERE/'style.py')}
    require(set(source['code']) == expected_code, 'style code provenance has missing or unexpected files')
    require(set(source) == {'code', 'data', 'runtime', 'style'}, 'unexpected style source identity fields')
    audit_source(source, original['source'], hashes, root)
    require(source['style'] == {'numpy': np.__version__,
                               'control_manifest_sha256': legacy.file_hash(CONTROL_ROOT/'manifest.json')},
            'style NumPy/control manifest provenance differs')
    control_manifest = read_json(CONTROL_ROOT/'manifest.json')
    audit_source(control_manifest['source'], original['source'], hashes, CONTROL_ROOT)
    control_base_config = {k: v for k, v in control_manifest['config'].items() if k != 'augmentation'}
    require(control_base_config == original['config'], 'referenced CONTROL core configuration differs')
    require(control_manifest['config']['augmentation'] == {
        'mode': 'PAIRED', 'probability': .5, 'raw_warp_amounts': [-2, 2],
        'derive_phase_after_warp': True,
        'seed_policy': 'SHA256 of source sampler state/fold/fixed seed20260908'},
        'referenced CONTROL augmentation wrapper configuration differs')
    require(control_manifest['source']['augmentation']['source']['numpy'] == np.__version__,
            'referenced CONTROL NumPy differs')
    ledger = read_json(root/'CONTROL_REUSE.json')
    require(ledger['controls_reused'] is True, 'expected previously verified CONTROL reuse')
    require(ledger['numpy_version'] == np.__version__, 'control ledger NumPy differs')
    hashes.verify(CONTROL_ROOT/'manifest.json', ledger['control_manifest_sha256'])
    old_environment = BASE/'mild_time_augmentation/ENVIRONMENT.json'
    hashes.verify(old_environment, ledger['environment_sha256'])
    for environment in (read_json(old_environment), read_json(root/'ENVIRONMENT.json')):
        for key, expected in [('python', sys.version), ('numpy', np.__version__),
                              ('torch', torch.__version__), ('cuda', torch.version.cuda)]:
            require(environment[key] == expected, f'control/style current environment differs: {key}')
    if not manifest['config']['smoke']:
        audit_pretrain(HERE/'PRETRAIN_CHECK.json', ledger, source, hashes)
    entries = ledger['controls']
    require(len(entries) == 8 and {x['fold'] for x in entries} == set(range(1, 9)),
            'control ledger requires exactly eight distinct folds')
    for entry in entries:
        expected = CONTROL_ROOT/'CONTROL'/f'fold{entry["fold"]}'/'warmup'/'result.json'
        require(Path(entry['path']).resolve() in (expected.resolve(), expected.parent.resolve()),
                'control ledger path leaves frozen control location')
        hashes.verify(expected, entry['result_sha256'])
        result = read_json(expected)
        require(entry['initial_state_sha256'] == result['identity']['initial_state_sha256'],
                'control ledger initial state differs')
        require(entry['selected_state_sha256'] == result['selected_state_sha256'],
                'control ledger selected state differs')
        require(entry['files'] == result['files'], 'control ledger files differ')
    return manifest, control_manifest, ledger


def audit_run(root, fold, mode, manifest, control_manifest, ledger, hashes, random_hashes, initial_states):
    mode_root = (CONTROL_ROOT if mode == 'CONTROL' else root)/mode
    directory = mode_root/f'fold{fold}'/'warmup'
    result = read_json(directory/'result.json')
    selected_manifest = control_manifest if mode == 'CONTROL' else manifest
    expected = dict(selected_manifest['config'], active_mode=mode)
    expected['augmentation_fold' if mode == 'CONTROL' else 'style_fold'] = fold
    require(result['identity']['config'] == expected and result['identity']['source'] == selected_manifest['source'],
            'result differs from frozen source/configuration manifest')
    original_matrix = legacy.audit_matrix
    def matrix_adapter(unused_root, *args):
        return original_matrix(OLD, *args)
    legacy.audit_matrix = matrix_adapter
    try:
        row = legacy.audit_run(mode_root, fold, 'warmup', hashes, random_hashes)
    finally:
        legacy.audit_matrix = original_matrix
    data = {Path(p).name: p for p in manifest['source']['data']}
    labels = np.load(data['y.npy']); subjects = np.load(data['subj.npy'])
    parts = legacy.partitions(labels, subjects, fold)
    split = read_json(mode_root/f'fold{fold}'/'split.json')
    require(split == {'ids': {k: v.tolist() for k, v in parts.items()},
                      'training_subjects': np.unique(subjects[parts['train']]).tolist(),
                      'val': fold % 8+1, 'test': fold}, 'saved whole-user split manifest differs')
    entry = next(x for x in ledger['controls'] if x['fold'] == fold)
    require(result['identity']['initial_state_sha256'] == entry['initial_state_sha256'], 'initialization differs from referenced control')
    progress = torch.load(directory/'progress.pt', map_location='cpu', weights_only=False)
    seed = expected['seed']
    if seed not in initial_states:
        model = legacy._load_module('model').build_model(seed, 'cpu')
        initial_states[seed] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        del model
    row.update(mode=mode, last_update_verified=True,
               last_parameter_updates=legacy.audit_last_updates(progress['state'], initial_states[seed], 'A0'))
    if mode == 'STYLE':
        row['style'] = audit_schedule(result, parts['train'], labels, subjects, progress['sampler'])
    else:
        row['control_schedule'] = time_audit.audit_schedule(result, parts['train'], progress['sampler'])
    states = progress['optimizer']['state']
    steps = sum(x['steps'] for x in result['history'])
    require(bool(states) and all(int(v['step']) == steps for v in states.values()),
            'Adam actual updates differ from completed optimizer steps')
    groups = progress['optimizer']['param_groups']
    require(len(groups) == 1 and groups[0]['lr'] == expected['warmup_lr'] and
            groups[0]['weight_decay'] == expected['weight_decay'], 'saved Adam settings differ')
    # Cross-check the legacy NumPy metric against independently maintained sklearn.
    from sklearn.metrics import accuracy_score, f1_score
    with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
        for role, logits_key, label_key, metrics_key in [('test', 'logits', 'y', 'metrics'),
                                                       ('val', 'val_logits', 'val_y', 'validation')]:
            y = saved[label_key]; pred = saved[logits_key].argmax(1)
            require(abs(accuracy_score(y, pred)-row[metrics_key]['accuracy']) < 1e-12 and
                    abs(f1_score(y, pred, labels=np.unique(y), average='macro', zero_division=0)-
                        row[metrics_key]['macro_f1']) < 1e-12, f'{role} sklearn metric cross-check differs')
    row.update(optimizer_steps_verified=steps, sklearn_metrics_verified=True,
               result_json_sha256=legacy.file_hash(directory/'result.json'),
               verified_artifact_sha256=result['files'])
    return row


def summarize_scores(scores, repetitions=10000, seed=20260909, smoke=False, errors=(), missing=()):
    out = {'complete_runs': sum(len(scores.get(m, {})) for m in MODES), 'means': {}, 'gate': None}
    for mode in MODES:
        rows = list(scores.get(mode, {}).values())
        if rows:
            out['means'][mode] = {k: float(np.mean([r[k] for r in rows])) for k in ('accuracy', 'macro_f1')}
    if any(set(scores.get(m, {})) != {str(h) for h in range(1, 9)} for m in MODES):
        out['gate_reason'] = 'Requires all eight audited STYLE runs and eight referenced CONTROL runs'
        return out
    resamples = np.random.default_rng(seed).integers(0, 8, size=(repetitions, 8))
    out['paired_effect'] = {}
    for metric in ('accuracy', 'macro_f1'):
        delta = np.array([scores['STYLE'][str(h)][metric]-scores['CONTROL'][str(h)][metric] for h in range(1, 9)])
        out['paired_effect'][metric] = {'mean_pp': float(delta.mean()*100), 'per_user_pp': (delta*100).tolist(),
            'improved_users': int((delta > 0).sum()),
            'bootstrap_95_percentile_pp': np.quantile(delta[resamples].mean(1)*100, [.025, .975]).tolist()}
    primary = out['paired_effect']['accuracy']
    if smoke or errors or missing:
        out['gate_reason'] = 'Audit errors, incomplete runs, or smoke configuration prevent formal gate evaluation'
    else:
        out['gate'] = {'passed': primary['mean_pp'] >= 2 and primary['improved_users'] >= 6,
                       'required_mean_pp': 2, 'required_improved_users': 6,
                       'primary_metric': 'clean outer-test accuracy, equal weight per user'}
    out['bootstrap'] = {'unit': 'paired outer user', 'users': 8, 'repetitions': repetitions, 'seed': seed,
                        'interpretation': 'descriptive fixed-protocol uncertainty; overlapping training folds and one model seed'}
    return out


def replay_right(root, fold, mode):
    """No style hooks or donors: strict original Right load and clean input only."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    directory = (CONTROL_ROOT if mode == 'CONTROL' else root)/mode/f'fold{fold}'/'warmup'
    model = legacy._load_module('model').Classifier([2], 'concat', 'mean')
    model.load_state_dict(torch.load(directory/'right.pt', map_location='cpu', weights_only=False), strict=True)
    model = model.cuda().eval()
    reader, labels, subjects, parts = legacy._load_module('protocol').open_fold(fold)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    checks = {}; arrays = {}
    try:
        with np.load(directory/'predictions.npz', allow_pickle=False) as saved, torch.no_grad():
            for role, saved_key in [('val', 'val_logits'), ('test', 'logits')]:
                outputs = []
                for start in range(0, len(parts[role]), 32):
                    x = reader.load_ids(parts[role][start:start+32], role, 'R')
                    require(tuple(x.shape[1:]) == (1, 90, 378), 'standalone replay must use only Right input')
                    with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(x)[0]
                    outputs.append(logits.float().cpu().numpy())
                actual = np.concatenate(outputs); expected = saved[saved_key]
                checks[role] = {'max_abs_logit_error': float(np.abs(actual-expected).max()),
                                'argmax_equal': bool(np.array_equal(actual.argmax(1), expected.argmax(1))),
                                'array_equal': bool(np.array_equal(actual, expected))}
                require(checks[role]['array_equal'], 'standalone clean replay logits are not exactly equal')
                arrays.update({role+'_logits': actual, role+'_pred': actual.argmax(1), role+'_ids': parts[role],
                               role+'_y': labels[parts[role]], role+'_subjects': subjects[parts[role]]})
        path = root/'audit_replays'/mode/f'fold{fold}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return {'passed': True, 'clean_checks': checks, 'batch': 32, 'exact_logits_required': True,
                'prediction_file': str(path), 'prediction_sha256': legacy.file_hash(path),
                'donors_used': False, 'used_for_checkpoint_selection': False}
    finally:
        del model
        torch.cuda.empty_cache()


def audit_all(root, folds=range(1, 9), replay=False):
    root = Path(root)
    hashes = legacy.HashVerifier(); random_hashes = {}; initial_states = {}
    errors = []; missing = []; reports = []; scores = {m: {} for m in MODES}
    try:
        manifest, control_manifest, ledger = audit_manifest(root, hashes)
    except (ValueError, KeyError, OSError, RuntimeError, StopIteration) as error:
        return {'status': 'failed', 'errors': [{'run': 'manifest/control', 'error': str(error)}],
                'missing': [], 'runs': [], 'summary': {'gate': None}, 'replay_requested': replay}
    for fold in folds:
        results = {}
        for mode in MODES:
            name = f'{mode}/fold{fold}/warmup'
            directory = (CONTROL_ROOT if mode == 'CONTROL' else root)/name
            if not (directory/'result.json').exists():
                missing.append(name); continue
            try:
                row = audit_run(root, fold, mode, manifest, control_manifest, ledger, hashes, random_hashes, initial_states)
                results[mode] = row.pop('result')
                if replay:
                    row['replay'] = replay_right(root, fold, mode)
                reports.append(row); scores[mode][str(fold)] = row['metrics']
            except (ValueError, KeyError, OSError, RuntimeError, StopIteration) as error:
                errors.append({'run': name, 'error': str(error)})
        if not manifest['config']['smoke']:
            try:
                audit_pair(results)
            except ValueError as error:
                errors.append({'run': f'paired/fold{fold}', 'error': str(error)})
    summary = summarize_scores(scores, smoke=manifest['config']['smoke'], errors=errors, missing=missing)
    return {'status': 'failed' if errors else 'incomplete' if missing or summary['complete_runs'] != 16 else 'verified',
            'errors': errors, 'missing': missing, 'runs': reports, 'summary': summary,
            'config': manifest['config'], 'controls_reused': True, 'replay_requested': replay,
            'source_hashes_verified': len(hashes.cache), 'audit_code_sha256': legacy.file_hash(__file__),
            'legacy_audit_sha256': legacy.file_hash(BASE/'cross_user_meta/audit.py'),
            'control_audit_sha256': legacy.file_hash(BASE/'mild_time_augmentation/audit.py')}


def write_results(report, destination):
    summary = report['summary']
    lines = ['# 固定低强度统计增强：独立审计结果', '',
             f'审计状态：{report["status"]}；有效完成运行：{summary.get("complete_runs", 0)}/16。', '',
             'STYLE 使用 25% 启用概率和 10% 同类跨用户供体统计混合；两个投影分支共用供体与掩码。',
             'CONTROL 复用上一轮当前环境训练的八个干净 Right 对照；审计核对数据、原代码、运行环境、NumPy、配置、完整用户划分、OOF 矩阵、初始化和采样。',
             '主指标为八名外层测试用户等权平均的干净输入准确率；宏 F1 对各用户实际存在的类别求平均。', '',
             '| 干净输入 | 用户等权准确率 | 用户等权宏 F1 |', '|---|---:|---:|']
    for mode in MODES:
        means = summary.get('means', {}).get(mode)
        if means:
            lines.append(f'| {mode} | {100*means["accuracy"]:.3f}% | {100*means["macro_f1"]:.3f}% |')
    if summary.get('complete_runs', 0) != 16:
        lines.extend(['', '暂列已审计运行的均值；完整八用户配对前不能据此判断增强效果。'])
    rows = {(r['mode'], r['fold']): r for r in report['runs']}
    lines.extend(['', '| 测试用户 | CONTROL 准确率 | STYLE 准确率 | 差值 pp | CONTROL 宏 F1 | STYLE 宏 F1 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for fold in range(1, 9):
        if all((m, fold) in rows for m in MODES):
            a, b = (rows[m, fold]['metrics'] for m in MODES)
            lines.append(f'| {fold} | {100*a["accuracy"]:.3f}% | {100*b["accuracy"]:.3f}% | {100*(b["accuracy"]-a["accuracy"]):+.3f} | {100*a["macro_f1"]:.3f}% | {100*b["macro_f1"]:.3f}% |')
    lines.extend(['', '| 测试用户 | CONTROL 选中 / 最终 epoch | STYLE 选中 / 最终 epoch |', '|---|---:|---:|'])
    for fold in range(1, 9):
        if any((m, fold) in rows for m in MODES):
            values = [f'{rows[m, fold]["best_epoch"]} / {rows[m, fold]["last_epoch"]}' if (m, fold) in rows else '未完成' for m in MODES]
            lines.append(f'| {fold} | {values[0]} | {values[1]} |')
    lines.extend(['', '仅按干净验证 CE 的首次最小值选择模型；epoch 0 保留为随机初始化候选。'])
    for metric, label in [('accuracy', '准确率'), ('macro_f1', '宏 F1')]:
        if metric in summary.get('paired_effect', {}):
            x = summary['paired_effect'][metric]; lo, hi = x['bootstrap_95_percentile_pp']
            lines.extend(['', f'{label}配对平均变化 {x["mean_pp"]:+.3f} pp；用户配对 bootstrap 95% 区间 [{lo:+.3f}, {hi:+.3f}] pp；{x["improved_users"]}/8 用户改善。'])
    gate = summary.get('gate')
    if gate is None:
        lines.extend(['', '当前不能判定预设开发门槛；原因：'+summary.get('gate_reason', '审计尚未完整通过')+'。'])
    elif gate['passed']:
        lines.extend(['', '达到“平均至少 +2 pp 且至少 6/8 用户改善”的固定开发门槛。仅作为后续第二种子验证的信号。'])
    else:
        lines.extend(['', '未达到“平均至少 +2 pp 且至少 6/8 用户改善”的固定开发门槛，停止这一固定设置的扩展调参。结论仅限当前幅度、数据和协议。'])
    if report.get('replay_requested'):
        replay_count = sum(bool(r.get('replay', {}).get('passed')) for r in report['runs'])
        lines.extend(['', f'独立原始 Right 干净验证/测试重放通过 {replay_count}/16；通过项 logits 逐值一致。新重放数组位于本轮 audit_replays 目录，文件哈希记录在审计 JSON。'])
    else:
        lines.extend(['', '尚未运行独立 Right 部署重放，不能声称部署输出已重放验证。'])
    lines.extend(['', '供体仅来自六位训练用户的同类、不同用户样本；验证和测试保持干净输入。本轮不进行依赖测试标签选择供体的风格压力测试。',
                  '这是一项特征统计干预，不将统计变化等同于纯用户风格或物理可解释量。',
                  '模型种子为 2026，训练折彼此重叠且只有八名用户；bootstrap 是固定协议下的描述性不确定性，不能替代更多用户、种子或现场验证。'])
    if report['errors']:
        lines.extend(['', '审计错误：', *[f'- {x["run"]}: {x["error"]}' for x in report['errors']]])
    Path(destination).write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE/'runs')
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(1, 9)))
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    require(len(set(args.folds)) == len(args.folds) and all(h in range(1, 9) for h in args.folds), 'invalid/duplicate folds')
    report = audit_all(args.out, args.folds, args.replay)
    path = args.report or args.out/('AUDIT_REPLAY.json' if args.replay else 'AUDIT.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_results(report, path.with_name('RESULTS.md'))
    print(json.dumps({'status': report['status'], 'errors': report['errors'], 'missing': report['missing'],
                      'summary': report['summary'], 'report': str(path)}, ensure_ascii=False))
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
