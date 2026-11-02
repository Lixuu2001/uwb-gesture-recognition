# 端到端跨视角特征重建：预定协议

延续用户授权继续研究重建。上一轮冻结编码器的重建降低MSE但未改善分类；本轮检验允许编码器适应是否改变结论。

## 唯一主问题

同一外折三视角教师初始化的单视角编码器，训练时共同更新编码器、两路特征预测器和分类器，是否优于同条件单视角读出及同容量无重建损失？重建监督目标始终由冻结外折教师产生，避免teacher/target共同坍缩。

## 范围和停止规则

首轮主部署Left（此前幅度单视角最强，事前固定），8折LOSO；固定seed2026。三臂：fixed_mask、capacity、joint。joint必须对两对照都满足至少+2pp、至少6/8被试改善才扩展Right/Top并补随机种子。否则报告当前端到端方案的负结果，不以挑测试被试或类别救结果。学习率等不根据测试结果调节。

## 方法与公平性

- 复用原9300样本质量掩码、seed2026的train/val/test；教师ckpt和特征缓存哈希一致。
- 编码器从该外折teacher_concat初始化，所有臂相同。xmv=0，不输入其他两路。每样本forward只接受[B,1,90,189]。
- 使用可反向传播的时间/距离token池化得到992维；不能复用旧带no_grad的compact函数。
- 标准化参数来自教师的梯度训练样本，冻结；缺失目标是该教师固定992维特征。学生变化不会改变目标。
- 读出沿用Completion：单视角槽真实特征+另两槽预测或零值，头2976→256→12。capacity/joint全模型相同初始化和shuffle，只有joint额外0.1 MSE。
- fixed_mask与另外两臂共享编码器和分类头初值，但没有decoder，因此模型容量不同，必须同时报告capacity。
- 所有臂可更新编码器；Adam，encoder lr1e-5，head/decoder lr1e-3，wd1e-4，batch128/microbatch32，bf16；最多20epoch，12epoch后patience6。验证CE选checkpoint，纯测试标签不参与任何训练选择。
- 目标损失以固定外折教师特征为准；测试时只保存MSE/NMSE作为诊断。主指标逐用户accuracy差、macroF1、最差用户准确率；NMSE不能当作分类率。
- 本轮是三视角预训练基础上的端到端微调，不是从头训练、原始回波生成或三维轨迹重建；因此只在本轮对照间归因。

## 实施检查

- [x] tests/test_e2e_reconstruction.py：确认重建梯度到达编码器、target无梯度、拒绝三路输入、同初始化。
- [x] e2e_reconstruction.py：独立输出e2e_reconstruction_results，断点复跑，保存超参/哈希/索引/逐样本/验证历史。
- [x] 训练前一批验收；不得把smoke产生的短训练checkpoint当正式结果。
- [x] 跑齐Left三臂8折，汇总配对bootstrap（仅8用户）、前轮差异和是否满足扩展规则。

## 后续可靠性门控检查

端到端联合重建将NMSE降到0.915，但相对同条件fixed-mask下降0.52 pp，且错误统计为修正482条、引入524条。为检验是否是低质量补全被强行读出的原因，新增两个同结构模型：每个预测缺失视角乘一个由可见特征生成的sigmoid标量门控；门控为0时退化为固定零填充。`gate_capacity`只用CE，`gate_joint`用CE+0.1 MSE，二者8折同初始化。结果见`gated_reconstruction_results/RESULTS.md`；`gate_joint`为71.50%，低于`gate_capacity` 71.79%和端到端fixed-mask 71.96%，均未通过+2 pp/6-of-8闸门，故不扩展Top/Right或第二随机种子。

## 同期稳定性诊断（独立于端到端主比较）

在前轮冻结992维特征上补岭回归：每个视角X→两路Y，W=(XᵀX+nαI)⁻¹XᵀY，α∈{.001,.01,.1,1}仅按训练内验证缺失MSE选取。输入/目标标准化仅拟合train，预测接固定real3分类头。L/T/R各8折。该诊断对比前轮pure_recon的神经预测器，不能和新端到端头混比；没有神经encoder微调，也不声称创新。
