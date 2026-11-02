# 第一部分续研：已完成结果与缺口

只对完整8折报告均值；未完成实验仅报告折数。

|方法|seed|完成折|accuracy%|macro-F1%|
|---|---:|---:|---:|---:|
|sv_R_concat|2026|8/8|72.41|71.27|
|sv_R_concat|2027|8/8|72.40|71.55|
|sv_R_concat_kd|2026|8/8|73.42|72.36|
|sv_R_concat_kd|2027|8/8|73.80|72.98|
|sv_R_duplicate|2026|8/8|67.86|66.80|
|sv_R_duplicate|2027|0/8|待完成|待完成|
|sv_R_off|2026|8/8|68.57|67.40|
|sv_R_off|2027|8/8|68.22|67.13|
|sv_R_phase_abs|2026|8/8|70.83|69.57|
|sv_R_phase_abs|2027|0/8|待完成|待完成|
|teacher_concat|2026|8/8|92.27|91.48|
|teacher_concat|2027|8/8|91.75|90.65|

|候选|基线|seed|增量pp|用户bootstrap 95%区间|改善用户|
|---|---|---:|---:|---|---:|
|sv_R_concat|sv_R_off|2026|+3.84|[+1.79,+6.31]|8/8|
|sv_R_concat|sv_R_off|2027|+4.18|[+2.13,+6.70]|8/8|
|sv_R_concat_kd|sv_R_concat|2026|+1.00|[+0.28,+1.70]|6/8|
|sv_R_concat_kd|sv_R_concat|2027|+1.40|[+0.54,+2.39]|8/8|
|sv_R_concat_kd|sv_R_off|2026|+4.85|[+2.63,+7.27]|7/8|
|sv_R_concat_kd|sv_R_off|2027|+5.58|[+3.53,+7.77]|8/8|
|sv_R_concat|sv_R_phase_abs|2026|+1.58|[-0.69,+3.72]|5/8|
|sv_R_phase_abs|sv_R_duplicate|2026|+2.97|[+1.12,+4.77]|7/8|

## 两seed按用户聚合（仍只有8名独立用户）

- sv_R_concat − sv_R_off：+4.01 pp；CI [+2.17,+6.38]；8/8用户的seed均值改善。
- sv_R_concat_kd − sv_R_concat：+1.20 pp；CI [+0.48,+1.97]；6/8用户的seed均值改善。
- sv_R_concat_kd − sv_R_off：+5.21 pp；CI [+3.33,+7.38]；8/8用户的seed均值改善。

## KD错误迁移账本

|candidate|baseline|seed|新增正确|损失正确|净正确数|教师机会|捕获机会|
|---|---|---:|---:|---:|---:|---:|---:|
|sv_R_concat_kd|sv_R_concat|2026|328|235|93|2002|310|
|sv_R_concat_kd|sv_R_concat|2027|401|273|128|1962|383|
|sv_R_concat_kd|sv_R_off|2026|1082|624|458|2346|1022|
|sv_R_concat_kd|sv_R_off|2027|1076|557|519|2340|1015|

concatKD−off包含相位和KD共同变化；只有concatKD−concat检验相位之上的KD增量。
按样本汇总的净正确数与按用户等权平均的pp有不同权重，不能直接互换。
Right是前轮多配置探索选出的视角；两seed使用相同测试用户，不构成独立外部验证。