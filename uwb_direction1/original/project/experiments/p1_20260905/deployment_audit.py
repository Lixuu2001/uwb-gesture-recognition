"""Post-training class/view diagnostics and isolated batch-1 timing."""
import argparse
import gc
import json
import time
import numpy as np
from pipeline import BASE, RESULTS, dump, transform, digest

OUT = BASE / 'followup_results'


def views():
    rows = []
    for view in 'LTR':
        for arm in ('off', 'concat', 'kd'):
            tag = f'sv_{view}_{arm}'
            records = []
            cms = []
            for fold in range(1, 9):
                stem = RESULTS / f'{tag}_s2026_h{fold}'
                r = json.loads(stem.with_suffix('.json').read_text())
                with np.load(str(stem) + '_predictions.npz') as p:
                    cm = np.zeros((12, 12), dtype=int)
                    np.add.at(cm, (p['y'], p['pred']), 1)
                    np.testing.assert_array_equal(cm, r['confusion'])
                records.append(r); cms.append(cm)
            cm = np.sum(cms, axis=0)
            recall = cm.diagonal() / cm.sum(1)
            errors = cm.copy(); np.fill_diagonal(errors, 0)
            ix = np.argsort(errors.ravel())[-5:][::-1]
            rows.append({'tag': tag, 'accuracy_pct': float(np.mean([r['accuracy'] for r in records]) * 100),
                         'worst_subject_pct': min(r['accuracy'] for r in records) * 100,
                         'confusion': cm.tolist(), 'class_support': cm.sum(1).tolist(),
                         'pooled_class_recall_pct': (recall * 100).tolist(),
                         'largest_errors': [{'true': f'G{i//12+1}', 'pred': f'G{i%12+1}', 'count': int(errors.ravel()[i])} for i in ix]})
    dump(OUT / 'view_class_audit.json', {'scope': 'Test-label descriptive analysis only; never used for fitting weights or selecting classes.', 'rows': rows})
    lines = ['# 单雷达视角与类别边界', '', '所有数字为seed2026完整8折。类别召回率按样本汇总，整体accuracy按用户等权，二者权重不同。', '',
             '|配置|accuracy%|最差用户%|最大混淆计数（真类→预测）|', '|---|---:|---:|---|']
    for r in rows:
        errors = '；'.join(f"{x['true']}→{x['pred']}:{x['count']}" for x in r['largest_errors'][:3])
        lines.append(f"|{r['tag']}|{r['accuracy_pct']:.2f}|{r['worst_subject_pct']:.2f}|{errors}|")
    lines += ['', '## 分类别召回率（%）', '', '|配置|' + '|'.join(f'G{i}' for i in range(1, 13)) + '|', '|---|' + '---:|' * 12]
    for r in rows: lines.append('|' + r['tag'] + '|' + '|'.join(f'{x:.1f}' for x in r['pooled_class_recall_pct']) + '|')
    lines += ['', 'HV2的G11/G12和HV6的G4已按预定质量掩码剔除，类支持不均衡。',
              '最大混淆是事后描述，不代表物理不可观测性真值；不得用本表为测试用户调整教师权重。']
    (OUT / 'VIEW_CLASS_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('VIEW_AUDIT_COMPLETE', len(rows))


def benchmark():
    import torch
    from train import Classifier, setup
    setup(2026)
    cache = np.load(BASE.parents[1] / 'cache/X_clutterremoved.npy', mmap_mode='r')
    raw = np.asarray(cache[:1, 2:3], dtype='float32')
    rows = []
    for tag, mode in [('sv_R_off', 'off'), ('sv_R_concat', 'concat'), ('sv_R_concat_kd', 'concat')]:
        path = RESULTS / f'{tag}_s2026_h1.pt'
        model = Classifier([2], mode).cuda().eval()
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state']); del checkpoint
        x = torch.as_tensor(transform(raw, mode), dtype=torch.float16, device='cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(30): model(x)
            torch.cuda.synchronize()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(200):
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                start.record(); model(x); end.record(); end.synchronize()
                timings.append(start.elapsed_time(end))
            peak = torch.cuda.max_memory_allocated()
            end_to_end = []
            for _ in range(50):
                torch.cuda.synchronize(); tic = time.perf_counter()
                xx = torch.as_tensor(transform(raw, mode), dtype=torch.float16, device='cuda')
                z, _ = model(xx); _ = z.argmax(1).cpu().item()
                torch.cuda.synchronize(); end_to_end.append((time.perf_counter() - tic) * 1000)
        rows.append({'tag': tag, 'mode': mode, 'parameters': sum(p.numel() for p in model.parameters()),
                     'network_median_ms': float(np.median(timings)), 'network_p95_ms': float(np.quantile(timings, .95)),
                     'preprocess_transfer_network_readback_median_ms': float(np.median(end_to_end)),
                     'resident_allocated_MiB': resident / 2**20, 'peak_allocated_MiB': peak / 2**20,
                     'checkpoint_sha': digest(path)})
        del model, x, xx, z; gc.collect(); torch.cuda.empty_cache()
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'batch': 1,
              'precision': 'bf16 autocast; float32 weights; float16 input', 'warmup': 30,
              'network_repeats': 200, 'end_to_end_repeats': 50,
              'scope': 'Isolated local GPU, fold1 checkpoint, one complete CR clip. Excludes radar acquisition (4.5s), raw clutter removal, continuous decision delay. End-to-end begins from existing CR data.', 'rows': rows}
    dump(OUT / 'deployment_benchmark.json', result)
    lines = ['# 单雷达推理成本', '', f"硬件：{result['gpu']}；PyTorch {result['torch']}；batch=1，bf16 autocast，30次预热。", '',
             '|配置|参数量|网络中位ms|网络P95 ms|CR预处理→GPU→预测读回中位ms|CUDA峰值MiB|', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['tag']}|{r['parameters']:,}|{r['network_median_ms']:.3f}|{r['network_p95_ms']:.3f}|{r['preprocess_transfer_network_readback_median_ms']:.3f}|{r['peak_allocated_MiB']:.1f}|")
    lines += ['', 'KD在训练期增加教师，部署模型与concat完全同结构。两模型微小时延差异来自重复计时波动，不解释为KD加速。',
              '参数量变化相对幅度基线为约+65.6%；输入宽度从189增至378，不应称低成本不变。',
              '上述计时不包含4.5秒数据采集、原始杂波滤除和连续提交延迟；不能据此声称完整系统毫秒响应。']
    (OUT / 'DEPLOYMENT_BENCHMARK.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    ap = argparse.ArgumentParser(); ap.add_argument('--benchmark', action='store_true'); a = ap.parse_args()
    benchmark() if a.benchmark else views()
