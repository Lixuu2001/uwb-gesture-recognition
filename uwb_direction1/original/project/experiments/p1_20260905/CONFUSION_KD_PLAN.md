# 单视角混淆结构约束蒸馏：可行性闸门

日期：2026-09-06。用户已授权继续规划和执行第一部分。本文固定后续实验，不用测试集调节目标、权重、类别或阈值。

## 事实与假设

Right幅度+相位上的普通KD两seed平均增量为+1.20 pp；可保留辅助效果，但仍低于既有+2 pp研发扩展阈值。去符号控制提示运动表征有用，符号单独增量未获充分支持。接下来检验：教师对三路可分的类别给出尖锐监督时，按训练用户中单路的混淆结构适度重分配软概率，能否优于普通KD？这是假设，不是已证实的新方法。

## 数据与估计隔离

保留原9300样本、8外层LOSO、seed2026和原train/val/test。**不得用已有学生的val_logits拟合训练损失目标**，否则原验证标签通过目标矩阵进入训练，已不再是公平的原验证协议。

采用只在原梯度训练索引上拟合的轻量代理：单路幅度和相位图各按10个慢时间块×21个距离块平均，拼成420维；每个外层折，在其7名训练用户中再次留一用户，用其余6人的原梯度训练样本拟合StandardScaler+LogisticRegression(C=1,max_iter=1000,tol=1e-4)，得到该用户梯度训练样本的折外概率。原外层val和test从不进入代理拟合或混淆估计。

按原类y聚合这些折外概率，得C[c,j]=mean p_proxy(j|x) for y=c。每行和为1，不用held-out测试混淆或类别选择。该矩阵只是固定低维代理的跨用户混淆统计，不称已标定的物理可观测性；它可能反映弱代理偏差，需由受控实验判断价值。

## 两个新臂及既有对照

- 普通KD：复用sv_R_concat_kd seed2026，q=p_teacher(T=2)。
- structure：q=(p_teacher + p_teacher C)/2。
- uniform：保留C每行对角元素，将该行其他概率均匀分到其余11类，得U；q=(p_teacher + p_teacher U)/2。

uniform控制保留每一源类的自类保留质量，但不保留类别间偏向。其熵不保证与structure完全相等，因此即使structure胜出仍不能排除所有分布形状效应。若值得扩展，再安排熵/谱匹配控制；本轮不自动增加第三个候选。

两新臂都复用原Right concat学生结构、同fold教师concat、同seed、同初始化和shuffle、相同CE+0.5*T²KL、T=2、Adam1e-4、batch128/microbatch32、最多40epoch、验证CE选择与patience8。C/U始终detach，不增加部署参数。只改变固定teacher概率的投影；原train.py不改，通过明确记录的wrapper替换KD函数。

## 执行清单

- [x] 验证已结束的A/B结果与文稿更新；完成独占GPU的成本测量和L/T/R类别审计。
- [x] tests/test_confusion_kd.py：分布合法、单位矩阵还原普通KD、teacher/matrix无梯度、uniform保对角、外层val/test和内层held-user排除。全部26项测试通过。
- [x] confusion_kd.py：420维代理缓存、7×8折外LogReg预测、C/U与索引/源SHA/警告保存、训练wrapper及smoke。56次LogReg拟合无收敛警告，GPU反向传播smoke通过。
- [x] confusion_kd_queue.py：已启动structure/uniform Right seed2026各8折的队列；两进程，失败自动停并写状态，保留已完成产物。启动不等于全8折已完成。
- [x] confusion_kd_summary.py：16次训练完整，索引/初始化/参数/教师哈希配对通过；structure74.51%，uniform74.61%，普通KD73.42%。structure相对uniform−0.10 pp、4/8改善，相对普通KD+1.09 pp，未通过预定扩展门槛，停止扩展混淆结构候选。

后续仅补教师必要性消融（TEACHER_FREE_CONTROL_PLAN.md），区分uniform结果中的教师与标签平滑贡献；这不是放宽门槛继续扩展structure。

通过扩展需要structure相对普通KD和uniform均平均至少+2 pp且至少6/8用户改善；同时列出相对CE。否则降级为当前候选的负结果，不改阈值、不挑测试类别或用户。此为探索性可行性筛选，不能把同一批用户的不断试验当独立确认。

## 文献与创新边界

[C²KD, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html)已有软标签错配与选择机制；[CVPR2020: Revisiting Knowledge Distillation via Label Smoothing Regularization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf)研究软目标的正则化作用；[WACVW2024: Knowledge-Distillation-Based Label Smoothing for Fine-Grained Open-Set Vehicle Recognition](https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Wolf_Knowledge-Distillation-Based_Label_Smoothing_for_Fine-Grained_Open-Set_Vehicle_Recognition_WACVW_2024_paper.pdf)已有类别混淆相关软标签。故概率投影/混淆平滑本身不宣称新颖，此轮只验证雷达单路训练用户折外混淆结构的实用增量。

本批实现和结果置于本目录及confusion_kd_results。原始论文和开题DOCX不改。
