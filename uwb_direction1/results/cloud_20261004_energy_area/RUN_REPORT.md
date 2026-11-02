# 距离带能量时序面积：源内分类先导完成，联合门槛未过

2026-10-04。预注册743923c39fc3db8bb117b96046e5ae71b58a5e85先正常推送并回读，再源训练64例测成本、执行五臂固定读出；未更改参数、主臂或门槛。用户要求的有效原创贡献仍未达到。

## 结果

六个既有研发源用户3–8；每次四训练/下一源用户验证/一源用户最终诊断，保留7100个质量样本。每用户存在类macro再等权六人平均。不是原八折六训练/一验证/一测试协议，也不是独立用户确认。

| 方法 | 六源平均 |
|---|---:|
| cnn_readout | 93.063131% |
| base | 92.771465% |
| signed | 93.079545% |
| unsigned | 92.983586% |
| shuffled | 92.968434% |
| cnn | 92.708333% |
| tail_cnn | 92.722222% |

cnn是同一第24轮段平均检查点的原神经分类头；tail_cnn是已验证旧tail13–24模型平均结果。cnn_readout只在同一128维隐藏特征上重拟合平衡岭回归；其余臂同样读出。base加入7距离带均值/std/首末差，signed再加有向面积，unsigned去符号，shuffled仅对面积输入作固定内部帧置乱。

| signed对照 | 平均差pp | 正差用户 | 最差配对差pp | 预定通过 |
|---|---:|---:|---:|---|
| base | +0.308081 | 5/6 | +0.000000 | False |
| unsigned | +0.095960 | 2/6 | -0.166667 | False |
| shuffled | +0.111111 | 3/6 | -0.583333 | False |
| cnn_readout | +0.016414 | 4/6 | -0.666667 | True |
| tail_cnn | +0.357323 | 4/6 | -0.166667 | True |

唯一主候选signed为93.079545%，CNN普通读出已93.063131%；两者仅差+0.016414pp。相对base虽5/6改善，+0.308081pp不足+0.5pp；相对unsigned/shuffled分别+0.095960/+0.111111pp，不足+0.25pp且只有2/6、3/6正差。对旧tail+0.357323pp不能替代这些控制。联合门槛失败，停止本带数/级数/归一化/shuffle/正则/seed网格与据此八折扩展。不能推断所有路径签名无效，亦不作新的原创贡献主张。

## 实现与解释边界

原189距离单元固定7×27带RMS，逐片段/雷达最大值归一化；90帧形成7维能量路径。各雷达21维level2反对称面积，合计63维，不涉及跨雷达同步乘积，也不是手部3D轨迹。路径平移/折线细分不变是数学性质，完整CNN+均值/std管线不因此具有现实速度不变性。抽取无参数且逐片段，岭回归所有均值/尺度/系数仅四训练用户拟合，三lambda仅下一源用户验证选择，最终诊断用户未参与拟合/选择。

文献已读范围及固定hash见ENERGY_AREA_REVIEW/REVIEW_RECEIPT；包含作者2015/2017数学说明第1–3、5–7页和库文档，以及动作签名实现README。不是TOMS或CV动作论文全文复现。原项目跨雷达轨迹面积先例明确保留，不能声称首次使用面积。

## 执行与审计

CPU单线程、PyTorch2.14.0+cpu。源64例与合成254维解算成本估计220.805秒（2倍余量），预算1800秒；实际特征提取/90次岭拟合/压缩保存75.220秒，不含输入全量哈希和独立审计。没有神经网络重新训练。

合成Chen递推、方向反转、平移、折线细分、RF整体增益/符号及零输入通过。独立审计重建全部7100片段的原始RF距离带与Chen递推面积，最大差3.192e-15；30个所选读出另用scikit-learn重拟合，诊断预测最大差1.377e-13，90个候选验证预测在运行时独立核对。36个固定首部样本作函数式CNN隐藏/输出核对（每折每角色2例），未声称全部隐藏向量逐个独立前向。40个运行产物SHA、角色、验证选择、分数与判定通过；审计4.550秒。运行后再确认原始/prepared/标签/用户四输入及核心helper/protocol SHA不变、原源码快照不变。

## 产物与复现

结果路径 `/workspace/uwb-energy-area-source-20261004`；运行日志 `/workspace/uwb-energy-area-source-20261004.log`；成本和独立审计同前缀cost/audit。保存全部源ID、描述、每折7100例隐藏向量、验证所有候选预测、最终诊断预测、30组均值/尺度/截距/系数，以及原检查点SHA。私有备份85成员、62014153压缩字节、9个≤7MiB owner-only分块，全部认证回下载及块/重组包/成员SHA通过；含上述全部数据产物、30读出参数、六个旧CNN检查点及tail来源、代码/日志/数学PDF。回执DRIVE_BACKUP_RECEIPT.json。原始信号数组不在备份包或Git中；报告/CSV/最终监测另存Git。

```bash
python uwb_direction1/tools/run_energy_area.py --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --models /workspace/uwb-swad-source-runs-20261003 --protocol uwb_direction1/research/ENERGY_AREA_PROTOCOL.json --out /workspace/uwb-energy-area-cost-NEW --cost-only
python uwb_direction1/tools/run_energy_area.py --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --models /workspace/uwb-swad-source-runs-20261003 --protocol uwb_direction1/research/ENERGY_AREA_PROTOCOL.json --out /workspace/uwb-energy-area-source-NEW --cost /workspace/uwb-energy-area-cost-NEW/COST.json
python uwb_direction1/tools/audit_energy_area.py --runs /workspace/uwb-energy-area-source-NEW --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930
```

当前新读出实验结束，没有训练或下一实验进程。下一步需要独立的新雷达观测机制及先例核查，不能沿未过门槛结果重开参数网格，也不能用常规线性读出的工程改进冒充论文创新。自动接续保留，整体目标未达。
