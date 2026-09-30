# 跨用户折外三视角特权蒸馏

每个训练样本的软标签来自未见过该样本所属用户的代理；部署学生只读取Right幅度+相位。

|配置|折数|accuracy%|macro-F1%|
|---|---:|---:|---:|
|sv_R_oofshape_multi|8/8|71.90|70.89|
|sv_R_oofshape_single|8/8|72.14|71.10|
|sv_R_label_global|8/8|74.77|73.84|
|sv_R_label_uniform|8/8|74.65|73.68|
|sv_R_concat_kd|8/8|73.42|72.36|
|sv_R_concat|8/8|72.41|71.27|

|候选|基线|Δ pp|95%用户bootstrap CI|改善用户|单项门槛|
|---|---|---:|---|---:|---:|
|sv_R_oofshape_multi|sv_R_oofshape_single|-0.23|[-2.15,+1.67]|4/8|False|
|sv_R_oofshape_multi|sv_R_label_global|-2.87|[-3.83,-2.06]|0/8|False|
|sv_R_oofshape_multi|sv_R_label_uniform|-2.75|[-4.09,-1.58]|0/8|False|
|sv_R_oofshape_multi|sv_R_concat_kd|-1.51|[-3.07,-0.19]|2/8|False|
|sv_R_oofshape_multi|sv_R_concat|-0.51|[-1.76,+0.60]|3/8|False|

联合扩展门槛通过：False。
只有同时超过单视角OOF和全局平滑控制才归因于三视角折外信息。