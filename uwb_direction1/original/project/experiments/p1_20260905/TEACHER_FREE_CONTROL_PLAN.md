# 教师必要性消融：固定协议

混淆结构候选已跑齐，structure74.51%、uniform74.61%，未超过预设控制，停止扩展该候选。新增8折仅用于判断uniform结果是否需要三视角教师；不增加新模块、不寻找更优权重。

## 唯一变化

复用Right concat、seed2026、原train/val/test、相同初始化/批顺序/优化器/CE/最多40epoch/patience8/验证CE选择。使用同一仅由Right训练用户折外概率得到的U矩阵。

已完成teacher_uniform的软目标为q_T=(p_T+p_T U)/2，p_T=softmax(teacher_logits/2)。

本消融软目标为q_Y=(e_y+e_y U)/2，e_y为当前梯度训练样本的one-hot标签。损失仍为CE+0.5*T² KL(q_Y||softmax(student_logits/T))，T=2。无三视角教师checkpoint/logits加载，U也不使用Left或Top。它不是标准单温度CE label smoothing，两项损失及温度均与既有对照匹配。

## 评估

主比较teacher_uniform−label_uniform，回答同平滑规则下三视角教师的增量；辅助比较label_uniform−concat CE及label_uniform−普通KD。按用户配对bootstrap，8折齐全后判断。不把差异区间包含零当作统计等效；若teacher_uniform没有明确优势，只报告未发现该实现需要教师的增量证据。

同时只读核算训练样本p_T与e_y的差异，以及q_T与q_Y的L1距离；它只解释训练目标相似性，不能预先代替训练效果结果。

## 执行

- [x] 验证label目标定义、U不反传、模型梯度有效；核查训练循环与原runner的采样/选择一致。29项测试通过，GPU梯度smoke通过。
- [x] label_uniform.py独立训练入口，仅改变软目标来源。旧train.py不改，记录来源哈希、U哈希、初始分类头和全部预测。
- [x] label_uniform_queue.py完成8折、两进程和自动汇总；每折与teacher_uniform的索引、参数量及初始分类头哈希一致，错误日志为空。
- [x] label_uniform为74.65%，teacher_uniform为74.61%，差值-0.04 pp、区间[-0.31,+0.22]、3/8用户改善；当前实现未发现三视角教师的额外收益。label_uniform相对concat CE为+2.24 pp、7/8用户改善。

由于label_uniform的收益也可能由常规平滑解释，后续只执行LABEL_SMOOTHING_CONTROL_PLAN.md中的全局强度匹配对照。重建和混淆结构保持不扩展，普通KD仍只作辅助基线。
