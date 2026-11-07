"""Render only the frozen aggregate cohort result; no predictions, labels, fitting or selection."""
from pathlib import Path
import csv, hashlib, json, time
P = Path('/workspace/uwb-vrex-loso6-20261106')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert (P/'FINALIZE_ONCE.lock').exists()
    result = read(P/'SUMMARY.json'); inference = read(P/'INFERENCE_COST.json')
    audit = read(P/'ALL_SIX_FREEZE_COUNTER_COST_AUDIT.json')
    rows = sorted(result['rows'], key=lambda row: row['held'])
    assert [row['held'] for row in rows] == list(range(1, 7))
    assert result['finalizer_invocations'] == 1 and result['users7_8_labels_used'] is False
    assert inference['held_labels_read'] is False and inference['new_training_or_optimizer_updates'] is False
    output = P/'report_artifacts'; output.mkdir(exist_ok=True)
    files = []
    user_csv = output/'PER_USER.csv'
    fields = ['held','n','missing_classes','baseline_accuracy','baseline_present_class_macro_recall',
              'method_accuracy','method_present_class_macro_recall','paired_macro_delta_pp',
              'selected_epoch_validation_only','training_wall_seconds','replay_wall_seconds','output_audit_wall_seconds']
    with user_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row, cost in zip(rows, audit['folds']):
            b, m = row['baseline'], row['method']; assert row['held'] == cost['held']
            assert b['class_support'] == m['class_support'] and b['n'] == m['n']
            writer.writerow(dict(held=row['held'],n=m['n'],missing_classes=json.dumps(m['missing_classes']),
                baseline_accuracy=b['accuracy'],baseline_present_class_macro_recall=b['present_class_macro_recall'],
                method_accuracy=m['accuracy'],method_present_class_macro_recall=m['present_class_macro_recall'],
                paired_macro_delta_pp=100*(m['present_class_macro_recall']-b['present_class_macro_recall']),
                selected_epoch_validation_only=cost['selected_epoch_validation_only'],
                training_wall_seconds=cost['incremental_stage_seconds']['train']['wall'],
                replay_wall_seconds=cost['incremental_stage_seconds']['replay']['wall'],
                output_audit_wall_seconds=cost['incremental_stage_seconds']['output']['wall']))
    files.append(user_csv)
    class_csv = output/'PER_CLASS_SUPPORT_AND_RECALL.csv'
    with class_csv.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['held','class_0based','support','baseline_recall','method_recall'])
        for row in rows:
            for k, support in enumerate(row['method']['class_support']):
                b = row['baseline']['class_recall'][k]; m = row['method']['class_recall'][k]
                writer.writerow([row['held'],k,support,'null' if b is None else b,'null' if m is None else m])
    files.append(class_csv)
    inference_aggregate = {}
    for arm in ['baseline','method']:
        current = [r for r in inference['rows'] if r['arm'] == arm]
        n = sum(r['n'] for r in current); assert len(current) == 6 and n == 6900
        inference_aggregate[arm] = dict(n=n,wall_seconds=sum(r['wall_seconds'] for r in current),
            CPU_seconds=sum(r['CPU_seconds'] for r in current))
        inference_aggregate[arm]['wall_ms_per_sample'] = inference_aggregate[arm]['wall_seconds']*1000/n
        inference_aggregate[arm]['CPU_ms_per_sample'] = inference_aggregate[arm]['CPU_seconds']*1000/n
    costs = dict(main_phase_final=result['remaining_phase_cost'],per_fold=audit['folds'],
        training_wall_seconds=audit['train_wall_sum'],training_CPU_seconds=audit['train_CPU_sum'],
        replay_wall_seconds=audit['replay_wall_sum'],replay_CPU_seconds=audit['replay_CPU_sum'],
        output_audit_wall_seconds=audit['output_audit_wall_sum'],output_audit_CPU_seconds=audit['output_audit_CPU_sum'],
        paired_label_free_inference=inference_aggregate,inference_worker=inference,
        private_archive_build=read(P/'FULL_ARCHIVE_BUILD_COST.json'),private_raw_transfer=read(P/'FULL_TRANSFER_AUDIT.json'),
        whole_archive_allmember_verification=read(P/'FULL_PRIVATE_READBACK_COST.json'),prior_MMD_cost=result['prior_MMD6_cost'],
        do_not_add_phase_and_included_child_stage_costs=True,archive_build_child_CPU_not_measured=True,
        inference_is_one_fixed_measurement_not_statistical_speedup_evidence=True)
    cost_file = output/'COST_REPORT.json'; cost_file.write_text(json.dumps(costs,indent=2)+'\n'); files.append(cost_file)
    figure_note = ''
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams['svg.hashsalt'] = 'uwb-fixed-vrex-loso6-20261106'
        x = np.arange(6); b = np.array([r['baseline']['present_class_macro_recall'] for r in rows])*100
        m = np.array([r['method']['present_class_macro_recall'] for r in rows])*100
        fig, axes = plt.subplots(2,1,figsize=(7.2,5.7),gridspec_kw={'height_ratios':[2,1]},layout='constrained')
        axes[0].bar(x-.18,b,.36,label='GAP64',color='#7b8794'); axes[0].bar(x+.18,m,.36,label='VREx 10/500',color='#2864a0')
        axes[0].set_ylabel('Present-class macro recall (%)'); axes[0].set_ylim(0,100); axes[0].legend(frameon=False)
        axes[0].set_xticks(x,[str(i) for i in range(1,7)]); axes[0].set_xlabel('Held-out development subject')
        axes[0].set_title('Frozen six-subject LOSO; one seed; exploratory development')
        delta=m-b; axes[1].bar(x,delta,color=['#b84a48' if v<0 else '#2864a0' for v in delta]); axes[1].axhline(0,color='#444444',lw=.8)
        axes[1].set_ylabel('Paired delta (pp)'); axes[1].set_xticks(x,[str(i) for i in range(1,7)]); axes[1].set_xlabel('Held-out development subject')
        for axis in axes:
            axis.spines[['top','right']].set_visible(False); axis.grid(axis='y',alpha=.2); axis.set_axisbelow(True)
        image = output/'LOSO_COMPARISON.svg'; fig.savefig(image,metadata={'Date':None}); plt.close(fig); files.append(image)
        figure_note = '\n逐用户图：[LOSO_COMPARISON.svg](LOSO_COMPARISON.svg)。\n'
    except ImportError as error:
        figure_note = '\n绘图依赖不可用：'+str(error)+'；原始聚合指标和CSV完整保留。\n'
    b, m = result['baseline'], result['method']
    verdict = '通过预设花费继续门槛；先执行已登记联合匹配对照，尚不归因于variance。' if result['cohort_spending_gate_passed'] else '未通过预设花费继续门槛；关闭这一确切配置，不依据测试结果救参。'
    table = ['| 测试用户 | 样本数 | GAP accuracy | GAP present-macro recall | VREx accuracy | VREx present-macro recall | 配对差(pp) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        b0,m0=row['baseline'],row['method']
        table.append(f"| {row['held']} | {m0['n']} | {100*b0['accuracy']:.4f}% | {100*b0['present_class_macro_recall']:.4f}% | {100*m0['accuracy']:.4f}% | {100*m0['present_class_macro_recall']:.4f}% | {100*(m0['present_class_macro_recall']-b0['present_class_macro_recall']):+.4f} |")
    text = f'''# VREx 完整开发六人 LOSO：统一评分与成本

准确率提升证据：用户等权 present-class macro recall，GAP64 {100*b['equal_user_mean_macro_recall']:.4f}%，VREx {100*m['equal_user_mean_macro_recall']:.4f}%，配对平均差 {100*result['mean_paired_macro_recall_delta']:+.4f}pp；严格改善 {result['strict_user_wins']}/6。{verdict}

原创贡献证据：尚未验证。VREx是[Krueger等 ICML2021/PMLR139](https://proceedings.mlr.press/v139/krueger21a.html)的已发表方法，本试验迁移固定DomainBed默认10/500组件，使用native GAP617612、三雷达/RMS、B64、64轮、seed20261029。等域BCE及完整optimizerreset500相对原global-BCE GAP有声明的差异，正结果也必须经过预登记等域BCE+同reset/无variance联合对照后才能讨论variance归因，不能把迁移本身称原创。

{chr(10).join(table)}

accuracy是全部样本正确率；present-class macro recall是仅对真实support>0类别平均的召回率。用户2缺0-based类别10/11，用户6缺类别3，其他用户12类齐全；所有现有类别各100条，因此本数据逐用户两种指标恰好相同，定义仍不同。预测为缺失类别仍算错误。缺失类召回为null；完整每类support/两方法recall见[PER_CLASS_SUPPORT_AND_RECALL.csv](PER_CLASS_SUPPORT_AND_RECALL.csv)。逐用户完整数值/选定epoch/成本见[PER_USER.csv](PER_USER.csv)。

用户等权accuracy：GAP {100*b['equal_user_mean_accuracy']:.4f}%，VREx {100*m['equal_user_mean_accuracy']:.4f}%。按样本加权pooled accuracy分别为{100*b['pooled_sample_accuracy']:.4f}%/{100*m['pooled_sample_accuracy']:.4f}%，不与用户等权均值混称。最差用户present-macro分别为{100*b['worst_user_macro_recall']:.4f}%/{100*m['worst_user_macro_recall']:.4f}%；最差配对变化{100*result['worst_paired_macro_recall_delta']:+.4f}pp。

完整64轮生产和各自fresh64重放、所有原生审计/角色ID隔离/验证选模、全部76块550659116字节1980载荷私有raw/allmember SHA冻结及精确Git发布完成后，才进行一次固定无标签推断成本及一次完整cohort真值配对。所有模型只由各自验证next用户选择，单人测试分数没有决定其他折是否继续。原始模型、预测、日志、ID私有索引和全部正负结果保留。统计范围为一个seed、6名重复开发用户；用户7–8未用于新方法选择，历史全8人已有开发，不称独立新用户验证或完整八人确认。

生产总wall/CPU {audit['train_wall_sum']:.6f}/{audit['train_CPU_sum']:.6f}秒；独立重放{audit['replay_wall_sum']:.6f}/{audit['replay_CPU_sum']:.6f}秒；输出审计{audit['output_audit_wall_sum']:.6f}/{audit['output_audit_CPU_sum']:.6f}秒。主阶段STATE总wall/CPU {result['remaining_phase_cost']['wall']:.6f}/{result['remaining_phase_cost']['cpu']:.6f}秒，已包含子阶段，不能再相加。此前MMD全部wall/CPU30865.434969/30891.040083秒完整另计，旧40.6258%失败结论不改。

固定一次无标签推断：GAP {inference_aggregate['baseline']['wall_ms_per_sample']:.6f}ms/样本，VREx {inference_aggregate['method']['wall_ms_per_sample']:.6f}ms/样本（1线程/B64，各6900条；模型及BN不变、冻结logits逐位一致）。这是一次固定计时，不作为统计速度改进证据。完整分用户/CPU/模型SHA/归档构建与回读成本见[COST_REPORT.json](COST_REPORT.json)；构建child总CPU未独立测量，parentCPU不能当总值，传输收据跨度与其他只读工作重叠。

门槛预先固定为均值至少+0.5pp、至少4/6严格改善、最差配对不退超2pp；门槛不是任何正负证据是否存在的定义。原MMD、旧混合94.6290%不同预算及旧失败结论不追改。当前route：`{result['route']}`。研究目标尚未达成，保持现有自动任务启用/频率和云环境；后续继续文献支撑的具体实现与真实实验，不调用旧held1 once、不扫描测试集或修改本配置。
'''+figure_note
    report=output/'FINAL_REPORT.md'; report.write_text(text); files.append(report)
    receipt=dict(state='derived_report_from_single_frozen_cohort_summary_only',summary_sha256=sha(P/'SUMMARY.json'),
        inference_cost_sha256=sha(P/'INFERENCE_COST.json'),source_script_sha256=sha(__file__),
        files={str(f.relative_to(P)):sha(f) for f in files},raw_labels_or_predictions_opened=False,new_fits=False,
        gate_or_selection_changed=False,display_rounding_only=True)
    (P/'DERIVED_REPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(report=str(report),artifacts=len(files),gate_passed=result['cohort_spending_gate_passed'],
        baseline_mean=b['equal_user_mean_macro_recall'],method_mean=m['equal_user_mean_macro_recall'],
        delta_pp=100*result['mean_paired_macro_recall_delta'],wins=result['strict_user_wins'])))

if __name__ == '__main__': main()
