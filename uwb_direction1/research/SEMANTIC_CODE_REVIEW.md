# 固定语义输出编码：源开发筛选

方向一仍未达标。前项RSC失败后，检查了原config.py标签、tests/test_attr_prior.py、2026-07-26属性原型预注册与2026-07-28裁决草案。4–7是对角滑动，不是先后动作。原项目已有属性方向读出/原型平行四边形/少样本注册，不能再称属性组合为原创；原三分类>70失败不改判。本项只问现有闭集12类的固定输出几何是否有帮助，不做未见类注册、不声称修复物理盲区，也不签署或修改历史裁决草案。

## 已核查资料与范围
scikit-learn提交a442e4bb39551feb7b0af4c00075e2cb91cf9b77，doc/modules/multiclass.rst第270–390行，blob94ef1a4c7b6a5be0494c100ee9ea77e1f3148ea0：核查OutputCodeClassifier完整说明与参考列表。其说明逐位二元学习、最近码字解码、随机码书，并明确分类错误相关可能妨碍纠错。
链接：https://github.com/scikit-learn/scikit-learn/blob/a442e4bb39551feb7b0af4c00075e2cb91cf9b77/doc/modules/multiclass.rst#L273
文献线索：Dietterich与Bakiri，Solving multiclass learning problems via error-correcting output codes，JAIR2，1995；James与Hastie，The error coding method and PICTs，JCGS7，1998。本轮读的是官方方法说明/书目，不是两篇论文全文，不能声称全文核查或论文精确复现。

## 机制、限制和负控制
把单轴/旋转/推/空动作的已知词表关系写成A，行归一化；码书C=[I,A]/sqrt(2)，相似动作码字更近。它可能借助相关类别的证据，也可能扩大混淆，因此必须有随机打乱A行的控制。等维正交码保持CC^T=I，应退化为普通onehot；它排除多输出维数本身的解释。
对固定训练标准化特征Z、样本权W、onehot标签Y和截距，B_C=(Z^TWZ+lambda I)^(-1)Z^TW(Y-b)C=B_Y C。最近码字在各行范数为1时等价argmax(P CC^T)。所以这里只改决策几何，无新增表示能力；把线性输出编码称新表示学习是不成立的。独立审计必须直接用C[y]重新拟合并显式计算距离，不调用这条代数捷径。

配置/门槛在任何本项真实拟合和诊断前提交远端。主要比较语义码对onehot至少+0.5pp、打乱码至少+0.25pp、旧tail至少+0.5pp，各至少4/6严格胜、任一用户下降不超过1pp。固定等权混合，无事后权重/seed/码长调参。通过也只得到开发证据，绝非创新或八折成功。失败封存，不把控制臂替换成主方法。
