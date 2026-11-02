# 单视角混淆约束蒸馏：结果

|配置|完成折|accuracy%|macro-F1%|最差用户%|
|---|---:|---:|---:|---:|
|sv_R_project_structure|8/8|74.51|73.53|53.27|
|sv_R_project_uniform|8/8|74.61|73.59|53.82|
|sv_R_concat_kd|8/8|73.42|72.36|51.27|
|sv_R_concat|8/8|72.41|71.27|51.82|

|候选|对照|Δ pp|95%用户bootstrap CI|改善用户|扩展门槛|
|---|---|---:|---|---:|---|
|structure|sv_R_project_uniform|-0.10|[-0.76,+0.51]|4/8|False|
|structure|sv_R_concat_kd|+1.09|[+0.08,+2.14]|6/8|False|
|structure|sv_R_concat|+2.10|[+1.23,+2.98]|8/8|True|

需要同时胜过普通KD与uniform，平均至少+2 pp且至少6/8用户改善。
只有8名独立用户，所有区间是探索性统计；未齐全不下结论。混淆矩阵只从原梯度训练样本的内部留一用户预测估计。