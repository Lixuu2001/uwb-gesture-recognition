# 单距离单元位移敏感性：未触发抗混叠训练筛查

2026-10-03。协议与脚本在运行前提交6ebb2e4fa45760f1d3f9c58aae128bb38a36235e。复用源用户3–8的六个ERM模型，各四人训练、一人验证、一人诊断；固定权重与原选择不变。共7100条、42600个单雷达±1单元扰动，没有新训练，没有计算真实标签准确率。一个单元对应原3个距离bin，未换算物理距离。

| 雷达 | 全部扰动翻转率 | 边界损失≤1%时翻转率 | 子集对数 | 达到5%的用户 |
|---|---:|---:|---:|---:|
| Left |0.9912%|0.9912%|14200|0/6|
| Top |1.1427%|1.1044%|13995|0/6|
| Right |0.7740%|0.7680%|13778|0/6|

翻转率先各用户平均两个移位方向，再对六用户等权；筛选子集各用户按该用户保留扰动对平均。三路各用户均覆盖≥200对，但没有用户超过5%，均未过冻结筛查。结果不支持按这项诊断启动抗混叠训练或扩大位移网格；不等于混叠为零，也不排除抗混叠作为一般正则化有其他作用。预测相同不是预测正确，不与旧八折识别率比较。独立雷达位移也不保证符合真实3D运动或保持手势语义。

实际76.33秒，不含初始信号SHA与独立post审计；CPU探针估计82.49秒，低于300秒预算。六模型完整未移位logits逐位重放旧缓存；每用户首64样本六种位移用独立NumPy索引与函数式网络检查，最大logit差0。对全部42600对的边界能量/概率JS/翻转统计独立重算通过，JS最大误差2.47e-16。六个预测文件SHA及7100唯一ID通过，不冒称所有移位logits都做了第二网络前向。

原模型仍保留在/workspace/uwb-swad-source-runs-20261003/held*/selected.pt，本诊断AUDIT.json列出逐权重SHA；其私有恢复入口在results/cloud_20261003_swad/DRIVE_BACKUP_RECEIPT.json。新输出/workspace/uwb-shift-diagnostic-20261003/held*.npz含ID、原/位移logits、边界比例、JS，不含标签。本轮归档及认证重下载回执见DRIVE_BACKUP_RECEIPT.json（该文件生成前不可推定上传完成）。原始大数组与权重不进入Git。

文献：实际取得ICML2019作者camera-ready PDF并读主文§1–5正文/公式/表格/图注，以及BlurPool作者源码；APS CVPR2021仅README。详见research/ANTIALIAS_REVIEW.md。BlurPool、自适应多相采样及单轴低通均已有先例，不能组合后直接称创新；当前只完成敏感性诊断，未实现抗混叠新训练方案。

下一项先核查原实验是否已有动作起止/静止背景处理，再研究源用户运动事件占比和时段分布。需区别于已失败的DTW/时间扭曲和距离矩路线；不直接重跑这些配置，不预设裁剪保留类别语义。下一项未实现、未启动。当前无活动训练，研究目标未达成。

复现到新空路径：
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /workspace/uwb-cloud/uwb_direction1/tools/diagnose_range_shift.py cost --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --runs /workspace/uwb-swad-source-runs-20261003 --out /workspace/shift-cost-NEW.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /workspace/uwb-cloud/uwb_direction1/tools/diagnose_range_shift.py run --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --runs /workspace/uwb-swad-source-runs-20261003 --out /workspace/shift-NEW --cost /workspace/shift-cost-NEW.json
OPENBLAS_NUM_THREADS=1 python /workspace/uwb-cloud/uwb_direction1/tools/audit_range_shift.py --out /workspace/shift-NEW --prepared /workspace/uwb-circular-prepared-20260930
```
