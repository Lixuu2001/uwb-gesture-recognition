# 类别自适应平滑必要性消融

全局控制与自适应版本使用相同网络、初始化、训练循环和训练样本平均平滑质量。

|配置|折数|accuracy%|macro-F1%|
|---|---:|---:|---:|
|sv_R_label_uniform|8/8|74.65|73.68|
|sv_R_label_global|8/8|74.77|73.84|
|sv_R_concat|8/8|72.41|71.27|
|sv_R_concat_kd|8/8|73.42|72.36|

|候选|基线|Δ pp|95%用户bootstrap CI|改善用户|通过门槛|
|---|---|---:|---|---:|---:|
|sv_R_label_uniform|sv_R_label_global|-0.12|[-0.64,+0.43]|3/8|False|
|sv_R_label_global|sv_R_concat|+2.36|[+1.69,+2.99]|8/8|True|
|sv_R_label_uniform|sv_R_concat|+2.24|[+1.35,+3.09]|7/8|True|

主判据：adaptive相对global至少+2.00个百分点且至少6/8用户改善。
区间包含零不证明统计等效；失败时只报告未发现类别自适应的增量证据。