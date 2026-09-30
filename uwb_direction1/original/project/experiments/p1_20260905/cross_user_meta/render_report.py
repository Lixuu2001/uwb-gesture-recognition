"""Render a Chinese RESULTS.md exclusively from the final independent audit.

No models, checkpoints, raw test inputs, or training code are imported. A complete
claim requires 32 unique arms/users and a passed standalone Right replay for each.
"""
import argparse
import json
from pathlib import Path

ARMS = ('A0', 'A1', 'A2', 'A3')


def completion_status(audit):
    rows = [r for r in audit.get('runs', []) if r.get('arm') in ARMS]
    expected = {(h, a) for h in range(1,9) for a in ARMS}
    reasons = []
    if audit.get('status') != 'verified' or audit.get('errors') or audit.get('missing'):
        reasons.append('独立审计尚未全部通过')
    if len(rows) != 32 or {(r['fold'],r['arm']) for r in rows} != expected:
        reasons.append('缺少完整且不重复的四臂 × 八用户结果')
    if audit.get('summary', {}).get('complete_runs') != 32 or audit.get('summary', {}).get('gate') is None:
        reasons.append('正式实验门槛尚不可判定，或当前仍为 smoke 配置')
    if not audit.get('replay_requested') or any(not r.get('replay', {}).get('passed') for r in rows):
        reasons.append('独立 Right 部署重放尚未逐组通过')
    if any(not r.get('last_update_verified') or not r.get('last_parameter_updates') for r in rows):
        reasons.append('最后训练状态的实际参数更新尚未逐组验证')
    return reasons


def _percent(value):
    return f'{100*value:.3f}'


def render_report(audit):
    reasons = completion_status(audit)
    if reasons:
        return '# 多视角辅助跨人元训练：结果待完成\n\n状态：尚未完成，当前不能给出效果结论。\n\n' + \
            '\n'.join('- '+reason+'。' for reason in reasons) + '\n\n本文件未将部分结果或资源测试解释为正式效果。\n'
    summary = audit['summary']; config = audit.get('config') or {}
    by_key = {(r['fold'],r['arm']): r for r in audit['runs']}
    lines = [
        '# 多视角辅助跨人元训练：独立核验结果', '',
        '状态：四臂 × 八用户的效果结果已完成，共 32 组正式结果通过独立审计与单 Right 部署重放。', '',
        '所有结果均来自所选检查点保存的预测，准确率、macro-F1、验证交叉熵和用户配对差由审计程序独立重算。', '',
        '## 比较对象与训练机制', '',
        '| 组别 | support 监督 | query 的 Right 预测 | 比较目的 |',
        '|---|---|---|---|',
        '| A0 | Right | 原始参数 | 同预算普通单视角训练 |',
        '| A1 | Right | 一步可微虚拟更新后的参数 | 单视角跨人元训练 |',
        '| A2 | Right + 0.5 × 三视角辅助 | 原始参数 | 普通多视角辅助联合训练 |',
        '| A3 | Right + 0.5 × 三视角辅助 | 一步可微虚拟更新后的参数 | 多视角辅助跨人元训练 |', '',
        '模型保留原 Right 编码器、时空 token 展平和 2048→256→12 分类头；训练时增加共享编码器上的三视角辅助头。'
        '每个 episode 从六位训练用户中轮换一位作为 query，其余五位提供 support。'
        'A1/A3 对 support 梯度更新继续求导，包含精确二阶元梯度；正式外层优化器更新共享编码器及对应分类头。', '',
        'Left/Top 仅参与合法训练 support 的辅助监督。验证、测试和部署均只输入 Right；不在验证或测试用户上做梯度更新，'
        '因此这是训练期跨人元学习，不是推理时适配。', '',
        f'固定配置：模型种子 {config.get("seed", "未记录")}；每用户每 episode {config.get("per_user", "未记录")} 个样本；'
        f'虚拟步长 {config.get("alpha", "未记录")}；分支最多 {config.get("epochs", "未记录")} 轮，'
        f'至少 {config.get("min_epochs", "未记录")} 轮后执行 patience={config.get("patience", "未记录")}；'
        f'公共初始化最多 {config.get("warmup_epochs", "未记录")} 轮。', '',
        '## 六用户训练与整用户验证协议', '',
        '外层测试用户为 h，验证用户为 (h mod 8)+1，其余六位用户用于参数训练。'
        '公共初始化从随机权重训练，标签平滑矩阵仅使用该折六位训练用户重新拟合。'
        '四个分支从同一公共初始化的所选完整状态开始，并共享采样序列的公共前缀。'
        '所选检查点是验证交叉熵首次达到全程最低值的一轮，允许选择尚未继续更新的第 0 轮。', '',
        '过滤规则沿用已核验数据质量协议：用户 2 的 G11/G12 与用户 6 的 G4 不进入本轮评价。'
        '主 macro-F1 在各用户实际存在的真实类别上计算；固定 12 类 macro-F1 另列。'
        '总体结果按用户等权平均，不按样本数加权。该协议使用固定的一种内层验证用户映射，不是穷举嵌套交叉验证。', '',
        '## 用户等权总体结果', '',
        '| 组别 | 准确率 / % | observed-class macro-F1 / % | 固定 12 类 macro-F1 / % |',
        '|---|---:|---:|---:|']
    for arm in ARMS:
        mean = summary['means'][arm]
        values = [by_key[h,arm]['metrics'].get('macro_f1_fixed_classes') for h in range(1,9)]
        fixed = _percent(sum(values)/8) if all(v is not None for v in values) else '未记录'
        lines.append(f'| {arm} | {_percent(mean["accuracy"])} | {_percent(mean["macro_f1"])} | {fixed} |')
    lines += ['', '## 八位用户逐项结果', '',
              '| 测试用户 | A0 准确率 / F1 | A1 准确率 / F1 | A2 准确率 / F1 | A3 准确率 / F1 |',
              '|---|---:|---:|---:|---:|']
    for held in range(1,9):
        cells = []
        for arm in ARMS:
            row = by_key[held,arm]['metrics']
            cells.append(f'{_percent(row["accuracy"])} / {_percent(row["macro_f1"])}')
        lines.append(f'| {held} | '+' | '.join(cells)+' |')
    lines += ['', '表中准确率与 observed-class macro-F1 均以百分数表示。', '',
              '## 配对差与用户 bootstrap 区间', '',
              '| 比较 | 指标 | 平均差 / pp | 改善用户数 | 95% 用户 bootstrap 区间 / pp |',
              '|---|---|---:|---:|---:|']
    for key in ['A3-A1','A3-A0','A3-A2','interaction']:
        label = '交互 (A3−A1)−(A2−A0)' if key == 'interaction' else key.replace('-', '−')
        for metric, name in [('accuracy','准确率'),('macro_f1','macro-F1')]:
            row = summary['comparisons'][key][metric]
            low, high = row['bootstrap_95_percentile_pp']
            lines.append(f'| {label} | {name} | {row["mean_pp"]:+.3f} | {row["improved_users"]}/8 | [{low:+.3f}, {high:+.3f}] |')
    bootstrap = summary.get('bootstrap', {})
    lines += ['', f'bootstrap 以外层用户为成对重采样单位，共 {bootstrap.get("repetitions", "未记录")} 次，'
              f'随机种子 {bootstrap.get("seed", "未记录")}。同一重采样用户索引同时用于所有比较组。'
              '区间只描述这八位用户下的配对差不确定性，不代表更广泛人群或新场景的普遍保证。', '',
              '## 预设开发门槛', '']
    gate = summary['gate']; primary = summary['comparisons']['A3-A1']['accuracy']
    lines.append(f'判定：**{"达到" if gate["passed"] else "未达到"}预设开发门槛**。'
                 f'A3−A1 的用户等权准确率差为 {primary["mean_pp"]:+.3f} pp，'
                 f'{primary["improved_users"]}/8 用户改善。')
    lines += ['', '预设条件为：A3−A1 ≥ +2 pp、至少 6/8 用户严格改善，且 A3 平均准确率同时严格高于 A0 和 A2。'
              '门槛按点估计判定；bootstrap 区间同时报告，不替代原先的门槛。', '']
    if gate['passed']:
        lines.append('该结果支持追加独立初始化和同用户同类别的跨视角错配控制，以进一步区分可迁移视角信息与一般训练正则化。'
                     '当前结果本身不证明一种普遍有效的新机制。')
    else:
        lines.append('停止扩大当前固定配置。负结果仅约束本轮网络、样本预算、种子、验证用户映射与精确二阶目标；'
                     '不能推导多视角辅助或跨人元学习在所有设置下都无效。若只有 A1 相对 A0 改善，应归为单视角跨人训练收益。')
    lines += ['', '## 第 0 轮选择与实际参数更新', '',
              '| 组别 | 选中第 0 轮的用户数 | encoder 改变/总张量 | head 改变/总张量 | auxiliary 改变/总张量 |',
              '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        rows = [by_key[h,arm] for h in range(1,9)]
        cells = []
        for group in ('encoder','head','aux'):
            counts = sorted({(r['last_parameter_updates'][group]['changed'],
                              r['last_parameter_updates'][group]['total']) for r in rows})
            cells.append('、'.join(f'{changed}/{total}' for changed,total in counts))
        zeros = sum(r['best_epoch'] == 0 for r in rows)
        lines.append(f'| {arm} | {zeros}/8 | '+' | '.join(cells)+' |')
    verified = sum(by_key[h,a]['last_update_verified'] for h in range(1,9) for a in ARMS)
    lines += ['', f'最后参数更新验证：{verified}/32 组通过。表中改变数由最后训练状态 progress.state 与公共初始化的所选状态逐张量比较获得；'
              '若不同折的数量不同，列出各个实际值。各组 encoder 与 head 均发生更新；A0/A1 的辅助头保持不变，A2/A3 的辅助头发生更新。'
              '未参与读出的 view_emb 允许不变。', '',
              '选中第 0 轮表示继续训练未在该验证用户上获得严格更低的验证交叉熵，验证选择因此保留公共初始化权重；'
              '这不是训练没有执行。上面的最后状态更新证据与所选检查点是两个不同的审计对象。', '',
              '## 检查点选择与实际耗时', '',
              '| 用户折 | 公共初始化 | A0 | A1 | A2 | A3 |',
              '|---|---:|---:|---:|---:|---:|']
    for held in range(1,9):
        cells = []
        for arm in ('warmup',)+ARMS:
            row = by_key.get((held,arm))
            cells.append('未记录' if row is None else f'{row["best_epoch"]}/{row["last_epoch"]} · {row["training_seconds"]/60:.2f}')
        lines.append(f'| {held} | '+' | '.join(cells)+' |')
    total_seconds = sum(r.get('training_seconds', 0.) for r in audit['runs'])
    lines += ['', '每格为“所选轮 / 最后一轮 · 累计分钟”。轮次耗时含各轮训练与验证，不含初次验证、最终预测导出及审计重放。',
              f'上表记录的累计轮次耗时为 {total_seconds/3600:.2f} 小时。相同 epoch 不是相同算力预算；A1/A3 额外计算虚拟更新及二阶梯度。', '',
              '## 结论边界', '',
              '历史原 Right 结果 74.773% 使用此前协议；本轮改变了参数训练用户数量、整用户验证选择和训练预算。'
              '因此历史数字只能作为背景，不能与本轮结果直接做配对差，也不能把跨协议变化归因于方法贡献。', '',
              '本轮比较仅在固定配置内解释 A0–A3 的用户配对差。结论以这八位用户和固定验证映射为边界；'
              '推广到其他用户群体或部署场景需要新的独立验证。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='从最终 AUDIT_REPLAY.json 生成中文 RESULTS.md；不运行模型。')
    parser.add_argument('audit', type=Path, help='最终独立审计 JSON 路径')
    parser.add_argument('--out', type=Path, help='默认写到审计文件同目录 RESULTS.md')
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    path = args.out or args.audit.with_name('RESULTS.md')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(audit), encoding='utf-8')
    print(json.dumps({'report': str(path), 'complete_effectiveness': not completion_status(audit)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
