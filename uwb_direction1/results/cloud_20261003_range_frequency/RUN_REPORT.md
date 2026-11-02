# 距离高低频源内读出：未发现足够的互补收益

2026-10-03。协议和训练脚本在任何真实诊断标签评分前提交4a52e791e9d5c8e75c36eac8f5d9010bcb5183be。54个模型全部完成。只用用户3–8共7100条：每折五用户训练、一用户诊断；用户1/2不参加模型处理。不同于原六训练/一验证/一测试的八折协议，不与93.4417%或94.1843%直接相减。

固定DCT沿现有63距离单元分为0–20低频和21–62高频，保留90时间帧。只分析幅度，不分析相位。每路雷达分别做低频256维、高频256维、低频128+高频128混合固定Gaussian投影。所有线性读出均256输入+截距、12输出，alpha=1，无网格；均值/方差、用户类别权重和系数只拟合该折五个源训练用户。相同可拟合参数数目不代表各臂信息量或有效正则化完全等价。

| 雷达 | 低频 | 高频 | 混合 | 混合−低频 | 改善/持平/下降用户 |
|---|---:|---:|---:|---:|---:|
| Left |55.8939%|21.5707%|54.8030%|−1.0909pp|2/0/4|
| Top |55.2929%|22.2285%|55.8687%|+0.5758pp|3/1/2|
| Right |53.8510%|31.7323%|53.1793%|−0.6717pp|2/0/4|

只有Right通过“高频平均≥25%、至少四用户≥20%”信息门槛。三路均未通过混合平均比低频≥1pp、至少4用户改善且最差用户降幅≤1pp的互补门槛；不启动本配置八折/扩大截止频率、正则系数或投影种子。Top一用户浮点差约1e-16，精确分数审计认定持平，不能计作改善；这不改变任何继续门槛结果，原始SUMMARY数值保留。

高频具备一定线性可读出的跨源用户类别信息，但不足以支持当前混合配置。不能把门槛失败解释为高频只有噪声、压缩必然无损，或低频模型就是最终有效方法。投影和线性表达限制也可能掩盖非线性信息。这是探索性信息诊断，不是新的论文算法或独立新用户确认。

成本探针只取64源信号和合成标签，估计特征与54次拟合11.82秒，低于预设300秒；不含完整审计、压缩写盘及输入SHA检查。实际特征、拟合、存盘与内置审计18.49秒，输入完整性SHA检查在该计时之外。全部54模型另用sklearn Ridge独立拟合，logits与类别预测符合容差；重载权重重放、源ID/分割和162个保存产物SHA再验证通过。独立余弦矩阵检查16条真实特征最大差9.05e-14，不声称对全部7100条重算特征。

目录/workspace/uwb-range-frequency-20261003保留features.npz、54组model.npz/predictions.npz/split.json/result.json、协议/成本/汇总/审计；不把二进制数组进Git。备份回执另见DRIVE_BACKUP_RECEIPT.json（生成前不应假设上传已完成）。所有54模型、预测、ID、配置和必要源码打包，特征缓存可由已验证信号快速重建，不包含原始大数组。

文献阅读新增CVPR2020 HFC作者代码及作者2021博士论文第2章全文文字/表格/图注（印刷页5–16），完整来源/核验与适用限制见research/RANGE_FREQUENCY_REVIEW.md。不声称CVPR正式全文或雷达近邻论文已全部读完。旧压缩原理已存在，未建立原创性。

后续：结束这一固定线性频带配置。下一可考察的不同问题是现有CNN步长下采样对小距离位移的混叠敏感性；先核查抗混叠CNN近邻和原源码，再以源信号/冻结源模型诊断，不预设距离位移保持手势标签，不依据旧外层测试标签选择。该项尚未实现或启动。当前无活动训练，研究目标尚未达成。

复现到新的空目录：
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /workspace/uwb-cloud/uwb_direction1/tools/probe_range_frequency.py cost --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --out /workspace/frequency-cost-NEW.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /workspace/uwb-cloud/uwb_direction1/tools/probe_range_frequency.py run --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --out /workspace/frequency-NEW --cost /workspace/frequency-cost-NEW.json
OPENBLAS_NUM_THREADS=1 python /workspace/uwb-cloud/uwb_direction1/tools/audit_range_frequency.py --out /workspace/frequency-NEW --data /workspace/uwb-restored-20260930
```
