# IRM100/500 complete native LOSO6 negative result: all6_result

## 2026-11-08：IRM100/500完整原生六人LOSO一次评分89.4240% vs GAP93.6672%，-4.2432pp，0胜/1平/5负；固定配置关闭，不救参

本次基于最新父c3f86e7b459f46d1d80ab7e821c659be1a95aade，全部25文件精确回读及75块/1995载荷完整私有raw-allmember先完成，再绑定FULL_PRIVATE_RELEASE；盲推断一次12组全部bitwise匹配冻结logits、参数buffer不变、held truth未读，随后原冻结finalize_cohort_once仅调用一次成功，原MMD旧单折once仍未调用。没有在单人测试分数出来后停止后续折，全部6折64生产+fresh64和完整native/roles/选模/预测/私有冻结均先于任何新held评分，配置/seed/epoch选择只由训练/验证确定。

|held|GAP accuracy|IRM accuracy|paired present-macro pp|
|---|---:|---:|---:|
|1|98.4167%|87.5833%|-10.8333|
|2|98.7000%|98.4000%|-0.3000|
|3|90.0000%|89.0833%|-0.9167|
|4|94.2500%|94.2500%|+0.0000|
|5|96.0000%|87.5000%|-8.5000|
|6|84.6364%|79.7273%|-4.9091|

用户等权mean accuracy及present-class macro：GAP93.6671717172%，IRM89.4239898990%，差-4.2431818182pp；0/6严格改善，held4持平，5/6下降。worst-user macro从84.6363636364%降至79.7272727273%；最差paired为held1 -10.8333333333pp。pool sample accuracy另列GAP93.6521739130%、IRM89.3043478261%，不能代替用户等权mean。用户2缺0-based类10/11、用户6缺类3，missing recall为null；每人现存每类100条，所以accuracy与presentmacro数值一致到报告精度，但定义不相同，完整support/12类recall与两个指标均见SUMMARY和FINAL_REPORT。

固定门槛mean>=+.5pp、>=4严格胜、最差paired>=-2pp全部未过。保留该确切IRM100/warm1/anneal500/native reset/617612/all3RMS/B64/64/seed20261029/globalBCE配置的真实负结果，路线close_exact_IRM_LOSO6_lambda100_warmup1_anneal500_nativeGAP64_seed20261029_no_rescue。原科学文件45项SHA857e06d91fd2da1ac1428a2716e1df541db8198245dc1d8041922f8d75ad8b6d不改，不按held分数改lambda/退火/学习率/seed/验证选模或输入，不重新评分，不激活只在正结果时规定的reset500/no-IRM归因对照。该失败不证明所有IRM实现无效，也不抹去原已发表方法的文献价值；没有准确率提升证据或原创贡献证据。本次是已发表IRM在本UWB原生协议的迁移实验，不是作者benchmark复现。

训练阶段含6折生产/独立重放/监督wall29509.953316993022/CPU29548.175221702秒（8.20h），生产27776更新、fresh replay再27776，各折stage累计差分成本见COST_BREAKDOWN，不把cumulative数相加多次。12组盲推断worker wall37.574744265992194/CPU37.949418691999995秒；完整子进程成本见INFERENCE_EXECUTION_COST（含载入检查与启动），原300秒wall/CPU、12GiB上限内。唯一scorer完整进程wall6.457474405004177/CPU6.866555秒，未把ready/hash预检查成本漏说成纯metric时间。全档75块548348323字节、1995载荷，SHA3436cbcc7bb738e30e5279acc9c998c3468a10954e66aa68ad9b08440494cf10，传输75成功/0失败/0补传，client传输窗口1494.5988554954529秒含工具等待/并行元数据工作，不能当纯网络kernel或与同时publication的墙钟再叠加。全档构建child CPU及完整remote/network CPU未测，不记0；完整raw-allmember与初始/中途/单折保险成本分别保留，非新模型fits。

旧MMD40.6258%、VREx92.8444%、Fishr93.5737%固定六折失败，以及旧混合94.6290%不同预算原门槛失败、LTR93.4417%、首次冻结cross-user92.7231%均保持原记录与成本，不追改。旧MMD wall30865.434969089023/CPU30891.040083422995（h1子成本不双加），VREx31157.163366941997/31189.589142517，Fishr29559.94331930898/29599.386025808秒仍另计。所有8人历史反复开发，本6人结果探索性，users7–8未用于当前方法选择，不称独立新用户验证。原789、仓库可见性、现有自动任务频率启用和云环境不变，当前没有训练进程。

研究目标仍未达到，继续下一项有文献依据、完整native规模的可运行方法。已核查旧GroupDRO是seed2026/GN小骨干/24轮/用户3–8/每组16循环采样/CE/10倍decay.01并有matched-reg对照；它不是完整617612/all3RMS/native混合B64/64/seed20261029的DomainBed默认组件，不能将旧缩小改写等价认定为完整默认实现已跑，也不能改旧失败结论。下一项优先核查并实现固定GroupDRO eta.01完整原生迁移：先独立NumPy/梯度/native优化器q状态/RNG/CPU预算及缺组规则审计，只有完整六折协议、全部源码/私有初态精确预登记后才训练；当前该新训练尚未启动。它首先作为有价值的已发表强基线迁移，不声称组合即创新，不用当前held分数选eta/正则/采样/seed。

