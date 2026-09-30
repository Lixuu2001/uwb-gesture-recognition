"""Independent paired audit. CPU by default; --replay explicitly uses the GPU.

Reuses only the old independent auditor and model/protocol, never the trainer or
augmentation schedule implementation. The OOF matrix root is explicitly separate.
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
MODES = ('CONTROL', 'AUG')
spec = importlib.util.spec_from_file_location('_mild_independent_legacy_audit', BASE/'cross_user_meta/audit.py')
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
require = legacy.require


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_schedule(result, train_ids, saved_sampler):
    """Reproduce every epoch from the pre-shuffle RNG, independently of augment.py."""
    config = result['identity']['config']
    fold = result['identity']['fold']
    mode = config['active_mode']
    require(mode in MODES and config['augmentation_fold'] == fold, 'augmentation mode/fold mismatch')
    rng = np.random.default_rng(config['seed'] + fold)
    total_counts = {'-1': 0, '0': 0, '1': 0}
    for row in result['history'][1:]:
        encoded = json.dumps({'sampler': rng.bit_generator.state, 'fold': fold,
                              'augmentation_seed': 20260908}, sort_keys=True).encode()
        seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'little')
        require(row['augmentation_seed'] == seed, 'augmentation seed differs from sampler state')
        aug_rng = np.random.default_rng(seed)
        selected = aug_rng.random(len(train_ids)) < .5
        directions = aug_rng.choice(np.array([-1, 1], dtype=np.int8), size=len(train_ids))
        choices = np.where(selected, directions, 0).astype(np.int8)
        if mode == 'CONTROL':
            choices.fill(0)
        plan = dict(zip(train_ids.tolist(), choices.tolist()))
        shuffled = rng.permutation(train_ids)
        steps = math.ceil(len(train_ids)/config['warmup_batch'])
        if config['smoke_steps']:
            steps = min(steps, config['smoke_steps'])
        consumed = shuffled[:steps*config['warmup_batch']]
        trace = hashlib.sha256()
        counts = {'-1': 0, '0': 0, '1': 0}
        for sample in consumed:
            sign = plan[int(sample)]
            trace.update(np.array([sample, sign], dtype=np.int64).tobytes())
            counts[str(sign)] += 1
        require(row['augmentation_trace'] == trace.hexdigest(), 'augmentation sample/sign trace differs')
        require(row['augmentation_counts'] == counts, 'augmentation counts differ')
        require(row['source_rows'] == len(consumed), 'source row count differs')
        for key in counts:
            total_counts[key] += counts[key]
    require(np.array_equal(saved_sampler['ids'], train_ids), 'augmentation sampler IDs differ')
    require(saved_sampler['rng'] == rng.bit_generator.state, 'augmentation sampler final RNG differs')
    return {'independent_schedule_verified': True, 'training_counts': total_counts,
            'sign_to_raw_amount': {'-1': -2, '0': 0, '1': 2}}


def audit_pair(results):
    if set(results) != set(MODES):
        return
    left, right = (results[m] for m in MODES)
    for key in ('source', 'partitions', 'matrix_sha256', 'initial_state_sha256'):
        require(left['identity'][key] == right['identity'][key], f'paired {key} mismatch')
    for a, b in zip(left['history'], right['history']):
        for key in ('epoch', 'episode_hash', 'steps'):
            require(a[key] == b[key], f'paired {key} differs at epoch {a["epoch"]}')
        if a['epoch']:
            require(a['augmentation_seed'] == b['augmentation_seed'], 'paired augmentation schedule seeds differ')
    require(left['history'][0]['val_ce'] == right['history'][0]['val_ce'], 'paired initial validation differs')


def audit_cache(source, hashes, cache_root):
    meta = source['augmentation']
    require(read_json(cache_root/'manifest.json') == meta, 'cache manifest differs from frozen source')
    src = meta['source']
    require(src['amounts'] == [-2, 2] and src['fitted_parameters'] is False, 'cache warp/fitting identity differs')
    require(meta['identity_max_abs_error'] == 0 and meta['identity_differing_values'] == 0,
            'zero-warp control was not exactly equal to original preparation')
    require(src['numpy'] == np.__version__, 'audit NumPy differs from training/cache runtime')
    hashes.verify(BASE.parents[1]/'cache/X_clutterremoved.npy', src['raw'])
    for name, key in [('y', 'labels'), ('subj', 'subjects')]:
        hashes.verify(BASE/f'prepared/{name}.npy', src[key])
    for name, digest in src['prepared'].items():
        hashes.verify(BASE/f'prepared/{name}.npy', digest)
    for path, digest in src['code'].items():
        hashes.verify(path, digest)
        require(source['code'][path] == digest, 'cache transform and trainer source differ')
    require(set(meta['files']) == {'minus2.npy', 'plus2.npy'}, 'cache pair incomplete')
    for name, digest in meta['files'].items():
        hashes.verify(cache_root/name, digest)
        array = np.load(cache_root/name, mmap_mode='r')
        require(array.shape == (meta['samples'], 1, 90, 378) and array.dtype == np.float16,
                'cache shape/dtype mismatch')
    return {'verified': True, 'manifest_sha256': legacy.file_hash(cache_root/'manifest.json'),
            'identity_max_abs_error': 0, 'fitted_parameters': False}


def audit_manifest(root, hashes, cache_root):
    manifest = read_json(root/'manifest.json')
    source, config = manifest['source'], manifest['config']
    old = read_json(OLD/'manifest.json')
    ledger = read_json(root/'CONTROL_REUSE.json')
    require(ledger['controls_reused'] is False, 'new paired experiment must rerun controls')
    require(ledger['numpy_version'] == np.__version__, 'control ledger current NumPy differs')
    require(source['runtime'] == old['source']['runtime'], 'declared original runtime differs')
    require(source['runtime']['torch'] == torch.__version__ and source['runtime']['cuda'] == torch.version.cuda,
            'audit Torch/CUDA build differs from training')
    for group in ('code', 'data'):
        for path, digest in old['source'][group].items():
            require(source[group].get(path) == digest, f'original {group} identity changed: {path}')
        for path, digest in source[group].items():
            hashes.verify(path, digest)
    for index, (path, digest) in enumerate(source['code'].items()):
        hashes.verify(root/'source'/f'{index:02d}_{Path(path).name}', digest)
    for key, value in old['config'].items():
        if config.get('smoke') and key in ('warmup_epochs', 'warmup_min', 'epochs', 'min_epochs', 'smoke_steps', 'smoke'):
            continue
        require(config.get(key) == value, f'original configuration changed: {key}')
    aug = config['augmentation']
    require(aug['mode'] in (*MODES, 'PAIRED') and aug['probability'] == .5 and
            aug['raw_warp_amounts'] == [-2, 2] and aug['derive_phase_after_warp'] is True,
            'fixed augmentation configuration differs')
    require(len(ledger['controls']) == 8 and {x['fold'] for x in ledger['controls']} == set(range(1, 9)),
            'old control provenance requires exactly eight distinct folds')
    for entry in ledger['controls']:
        path = OLD/f'fold{entry["fold"]}/warmup/result.json'
        hashes.verify(path, entry['result_sha256'])
        original = read_json(path)
        require(entry['initial_state_sha256'] == original['identity']['initial_state_sha256'],
                'old provenance initialization differs')
    return manifest, ledger, audit_cache(source, hashes, cache_root)


def audit_run(root, fold, mode, manifest, ledger, hashes, random_hashes, initial_states):
    directory = root/mode/f'fold{fold}'/'warmup'
    result = read_json(directory/'result.json')
    expected = dict(manifest['config'], augmentation_fold=fold, active_mode=mode)
    require(result['identity']['config'] == expected and result['identity']['source'] == manifest['source'],
            'result differs from frozen paired manifest')
    # The existing auditor resolves run artifacts below MODE, but OOF matrices
    # remain the separately frozen old matrix directory. Restore after each call.
    original_matrix = legacy.audit_matrix
    def matrix_adapter(unused_root, *args):
        return original_matrix(OLD, *args)
    legacy.audit_matrix = matrix_adapter
    try:
        row = legacy.audit_run(root/mode, fold, 'warmup', hashes, random_hashes)
    finally:
        legacy.audit_matrix = original_matrix
    data_paths = {Path(p).name: p for p in manifest['source']['data']}
    labels = np.load(data_paths['y.npy']); subjects = np.load(data_paths['subj.npy'])
    parts = legacy.partitions(labels, subjects, fold)
    split = read_json(root/mode/f'fold{fold}'/'split.json')
    require(split == {'ids': {k: v.tolist() for k, v in parts.items()},
                      'training_subjects': np.unique(subjects[parts['train']]).tolist(),
                      'val': fold % 8 + 1, 'test': fold}, 'saved split manifest differs')
    original = next(x for x in ledger['controls'] if x['fold'] == fold)
    require(result['identity']['initial_state_sha256'] == original['initial_state_sha256'],
            'fresh initialization differs from original control')
    progress = torch.load(directory/'progress.pt', map_location='cpu', weights_only=False)
    seed = expected['seed']
    if seed not in initial_states:
        model = legacy._load_module('model').build_model(seed, 'cpu')
        initial_states[seed] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        del model
    updates = legacy.audit_last_updates(progress['state'], initial_states[seed], 'A0')
    row.update(last_parameter_updates=updates, last_update_verified=True, mode=mode,
               augmentation=audit_schedule(result, parts['train'], progress['sampler']))
    states = progress['optimizer']['state']
    steps = sum(x['steps'] for x in result['history'])
    require(bool(states) and all(int(v['step']) == steps for v in states.values()),
            'Adam actual update counters differ from completed optimizer steps')
    require(len(progress['optimizer']['param_groups']) == 1, 'unexpected optimizer groups')
    group = progress['optimizer']['param_groups'][0]
    require(group['lr'] == expected['warmup_lr'] and group['weight_decay'] == expected['weight_decay'],
            'saved Adam settings differ from frozen config')
    row['optimizer_steps_verified'] = steps
    row['result_json_sha256'] = legacy.file_hash(directory/'result.json')
    row['verified_artifact_sha256'] = result['files']
    return row


def summarize_scores(scores, repetitions=10000, seed=20260908):
    out = {'complete_runs': sum(len(scores.get(m, {})) for m in MODES), 'means': {}, 'gate': None}
    for mode in MODES:
        rows = list(scores.get(mode, {}).values())
        if rows:
            out['means'][mode] = {k: float(np.mean([x[k] for x in rows])) for k in ('accuracy', 'macro_f1')}
    if any(set(scores.get(m, {})) != {str(h) for h in range(1, 9)} for m in MODES):
        out['gate_reason'] = 'Requires all 16 audited paired runs covering eight outer users'
        return out
    resamples = np.random.default_rng(seed).integers(0, 8, size=(repetitions, 8))
    out['paired_effect'] = {}
    for metric in ('accuracy', 'macro_f1'):
        delta = np.array([scores['AUG'][str(h)][metric]-scores['CONTROL'][str(h)][metric] for h in range(1, 9)])
        out['paired_effect'][metric] = {'mean_pp': float(delta.mean()*100),
                                      'per_user_pp': (delta*100).tolist(),
                                      'improved_users': int((delta > 0).sum()),
                                      'bootstrap_95_percentile_pp': np.quantile(delta[resamples].mean(1)*100, [.025, .975]).tolist()}
    primary = out['paired_effect']['accuracy']
    out['gate'] = {'passed': primary['mean_pp'] >= 2 and primary['improved_users'] >= 6,
                   'required_mean_pp': 2, 'required_improved_users': 6,
                   'primary_metric': 'clean outer-test accuracy, equal weight per user'}
    out['bootstrap'] = {'unit': 'paired outer user', 'users': 8, 'repetitions': repetitions, 'seed': seed,
                        'interpretation': 'descriptive fixed-protocol uncertainty; overlapping training folds and one model seed'}
    return out


def replay_right(root, fold, mode, cache_root):
    """Standalone original Right Classifier, clean val/test plus post-selection stress."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    directory = root/mode/f'fold{fold}'/'warmup'
    model = legacy._load_module('model').Classifier([2], 'concat', 'mean')
    model.load_state_dict(torch.load(directory/'right.pt', map_location='cpu', weights_only=False), strict=True)
    model = model.cuda().eval()
    reader, labels, subjects, parts = legacy._load_module('protocol').open_fold(fold)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    checks = {}; arrays = {'ids': parts['test'], 'y': labels[parts['test']],
                           'subjects': subjects[parts['test']], 'val_ids': parts['val'], 'val_y': labels[parts['val']]}
    stress_metrics = {}
    with np.load(directory/'predictions.npz', allow_pickle=False) as saved, torch.no_grad():
        for condition, role, filename in [('val', 'val', None), ('clean', 'test', None),
                                           ('minus2', 'test', 'minus2.npy'), ('plus2', 'test', 'plus2.npy')]:
            cache = np.load(cache_root/filename, mmap_mode='r') if filename else None
            output = []
            for start in range(0, len(parts[role]), 32):
                ids = parts[role][start:start+32]
                x = (reader.load_ids(ids, role, 'R') if cache is None else
                     torch.from_numpy(np.array(cache[ids], copy=True)).cuda())
                require(tuple(x.shape[1:]) == (1, 90, 378), 'standalone replay requires Right input only')
                with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)[0]
                output.append(logits.float().cpu().numpy())
            actual = np.concatenate(output)
            arrays[condition+'_logits'] = actual
            arrays[condition+'_pred'] = actual.argmax(1)
            if filename is None:
                expected = saved['val_logits' if role == 'val' else 'logits']
                checks[condition] = {'max_abs_logit_error': float(np.abs(actual-expected).max()),
                                     'argmax_equal': bool(np.array_equal(actual.argmax(1), expected.argmax(1))),
                                     'allclose': bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-5))}
                require(checks[condition]['argmax_equal'] and checks[condition]['allclose'], 'standalone clean replay differs')
            else:
                stress_metrics[condition] = legacy.metrics(actual, labels[parts['test']])
    path = directory/'audit_replay_predictions.npz'
    np.savez_compressed(path, **arrays)
    del model; torch.cuda.empty_cache()
    return {'passed': True, 'clean_checks': checks, 'stress_metrics': stress_metrics,
            'prediction_file': str(path), 'prediction_sha256': legacy.file_hash(path),
            'batch': 32, 'atol': 1e-5, 'rtol': 1e-5, 'used_for_checkpoint_selection': False}


def audit_all(root, folds=range(1, 9), replay=False, cache_root=None):
    root = Path(root); cache_root = Path(cache_root or HERE/'cache')
    hashes = legacy.HashVerifier(); random_hashes = {}; initial_states = {}
    errors = []; missing = []; reports = []; scores = {m: {} for m in MODES}
    stress = {c: {m: {} for m in MODES} for c in ('minus2', 'plus2')}
    try:
        manifest, ledger, cache_report = audit_manifest(root, hashes, cache_root)
    except (ValueError, KeyError, OSError, RuntimeError, StopIteration) as error:
        return {'status': 'failed', 'errors': [{'run': 'manifest/cache', 'error': str(error)}],
                'missing': [], 'runs': [], 'summary': {'gate': None}, 'replay_requested': replay}
    for fold in folds:
        results = {}
        for mode in MODES:
            name = f'{mode}/fold{fold}/warmup'
            if not (root/name/'result.json').exists():
                missing.append(name); continue
            try:
                row = audit_run(root, fold, mode, manifest, ledger, hashes, random_hashes, initial_states)
                results[mode] = row.pop('result')
                if replay:
                    row['replay'] = replay_right(root, fold, mode, cache_root)
                    for condition in stress:
                        stress[condition][mode][str(fold)] = row['replay']['stress_metrics'][condition]
                reports.append(row); scores[mode][str(fold)] = row['metrics']
            except (ValueError, KeyError, OSError, RuntimeError, StopIteration) as error:
                errors.append({'run': name, 'error': str(error)})
        try:
            audit_pair(results)
        except ValueError as error:
            errors.append({'run': f'paired/fold{fold}', 'error': str(error)})
    summary = summarize_scores(scores)
    if errors or missing or manifest['config'].get('smoke'):
        summary['gate'] = None
        summary['gate_reason'] = 'Errors, incomplete runs, or smoke configuration prevent formal gate evaluation'
    stress_summary = {}
    if replay:
        for condition, values in stress.items():
            stress_summary[condition] = summarize_scores(values)
            stress_summary[condition]['gate'] = None
            stress_summary[condition]['gate_reason'] = 'Post-selection robustness only; never a clean accuracy development gate'
    return {'status': 'failed' if errors else 'incomplete' if missing or summary['complete_runs'] != 16 else 'verified',
            'errors': errors, 'missing': missing, 'runs': reports, 'summary': summary,
            'stress_summary': stress_summary, 'config': manifest['config'], 'cache': cache_report,
            'controls_reused': False, 'replay_requested': replay, 'source_hashes_verified': len(hashes.cache),
            'audit_code_sha256': legacy.file_hash(__file__), 'legacy_audit_sha256': legacy.file_hash(BASE/'cross_user_meta/audit.py')}


def write_results(report, destination):
    summary = report['summary']
    lines = ['# 固定轻度时间增强：独立审计结果', '',
             f'审计状态：{report["status"]}；有效完成运行：{summary.get("complete_runs", 0)}/16。', '',
             'CONTROL 与 AUG 均在同一当前环境重新训练；历史模型仅作来源核对。模型种子为 2026。',
             '主指标是八个外层测试用户等权平均的干净输入准确率；宏 F1 对每位用户实际存在的类别求平均。', '',
             '| 干净输入 | 用户等权准确率 | 用户等权宏 F1 |',
             '|---|---:|---:|']
    for mode in MODES:
        means = summary.get('means', {}).get(mode)
        if means:
            lines.append(f'| {mode} | {100*means["accuracy"]:.3f}% | {100*means["macro_f1"]:.3f}% |')
    if summary.get('complete_runs', 0) != 16:
        lines.extend(['', '表中暂列已审计运行的均值；完整八用户配对结束前不能据此判断增强效果。'])
    lines.extend(['', '| 测试用户 | CONTROL 准确率 | AUG 准确率 | 差值 pp | CONTROL 宏 F1 | AUG 宏 F1 |',
                  '|---|---:|---:|---:|---:|---:|'])
    rows = {(x['mode'], x['fold']): x for x in report['runs']}
    for fold in range(1, 9):
        if all((m, fold) in rows for m in MODES):
            a, b = (rows[m, fold]['metrics'] for m in MODES)
            lines.append(f'| {fold} | {100*a["accuracy"]:.3f}% | {100*b["accuracy"]:.3f}% | {100*(b["accuracy"]-a["accuracy"]):+.3f} | {100*a["macro_f1"]:.3f}% | {100*b["macro_f1"]:.3f}% |')
    lines.extend(['', '| 测试用户 | CONTROL 选中 / 最终 epoch | AUG 选中 / 最终 epoch |',
                  '|---|---:|---:|'])
    for fold in range(1, 9):
        if any((m, fold) in rows for m in MODES):
            values = [f'{rows[m, fold]["best_epoch"]} / {rows[m, fold]["last_epoch"]}'
                      if (m, fold) in rows else '未完成' for m in MODES]
            lines.append(f'| {fold} | {values[0]} | {values[1]} |')
    lines.extend(['', '仅按干净验证集 CE 的首次最小值选择 epoch；epoch 0 是保留的随机初始化候选。'])
    lines.append('')
    for metric, label in [('accuracy', '准确率'), ('macro_f1', '宏 F1')]:
        if metric in summary.get('paired_effect', {}):
            x = summary['paired_effect'][metric]; lo, hi = x['bootstrap_95_percentile_pp']
            lines.append(f'干净输入{label}配对平均变化 {x["mean_pp"]:+.3f} pp；用户配对 bootstrap 95% 区间 [{lo:+.3f}, {hi:+.3f}] pp；{x["improved_users"]}/8 用户改善。')
    gate = summary.get('gate')
    if gate is None:
        lines.extend(['', '当前不能判定预设开发门槛；原因：'+summary.get('gate_reason', '审计尚未完整通过')+'。'])
    elif gate['passed']:
        lines.extend(['', '达到“平均至少 +2 pp 且至少 6/8 用户改善”的固定开发门槛。此结果仅是单种子信号；稳定性结论需要第二种子验证。'])
    else:
        lines.extend(['', '未达到“平均至少 +2 pp 且至少 6/8 用户改善”的固定开发门槛，停止这一固定设置的扩展调参。此结论仅限当前增强幅度、数据和跨用户协议。'])
    lines.extend(['', '压力测试在干净验证 CE 选定模型后进行，不参与选择模型。'])
    if report.get('replay_requested'):
        lines.extend(['', '| 固定原始信号扰动 | CONTROL 准确率 | AUG 准确率 | 差值 pp | CONTROL 宏 F1 | AUG 宏 F1 |',
                      '|---|---:|---:|---:|---:|---:|'])
        for condition, values in report.get('stress_summary', {}).items():
            delta = values.get('paired_effect', {}).get('accuracy')
            means = values.get('means', {})
            if delta and all(m in means for m in MODES):
                a, b = (means[m] for m in MODES)
                lines.append(f'| {condition} | {100*a["accuracy"]:.3f}% | {100*b["accuracy"]:.3f}% | {delta["mean_pp"]:+.3f} | {100*a["macro_f1"]:.3f}% | {100*b["macro_f1"]:.3f}% |')
        lines.append('')
        lines.append('逐样本干净、验证与 ±2 扰动 logits/预测位于各运行 audit_replay_predictions.npz；文件哈希记录在审计 JSON。')
    else:
        lines.append('尚未运行独立 Right 部署重放与压力测试，不能据此声称部署重放或扰动鲁棒性已验证。')
    lines.extend(['', '干净输入结果仅反映当前离散手势数据、固定跨用户划分和单种子下的分类表现。训练折彼此重叠且只有八个用户，bootstrap 区间是固定协议下的描述性不确定性，不能替代更多用户、种子或现场实验。',
                  '压力测试仅考察与训练增强同族的两种固定信号空间时间扭曲；未验证其对应真实手部运动，不能将扰动收益解释为自然手势速度变化的鲁棒性，也不能代替干净输入提升。',
                  '训练仅使用六名来源用户；完整用户验证与测试分别隔离，平滑矩阵来自训练用户内部 OOF。'])
    if report['errors']:
        lines.extend(['', '审计错误：', *[f'- {x["run"]}: {x["error"]}' for x in report['errors']]])
    Path(destination).write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE/'runs')
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(1, 9)))
    parser.add_argument('--cache', type=Path, default=HERE/'cache')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    require(len(set(args.folds)) == len(args.folds) and all(h in range(1, 9) for h in args.folds), 'invalid/duplicate folds')
    report = audit_all(args.out, args.folds, args.replay, args.cache)
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
