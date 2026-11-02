# QUANT成本窗口未完成已关闭；DrCIF500/CIT20完整实现及成本预登记

截至 2026-11-02T17:36:29.558862+00:00，先前QUANT完整depth6/div4/539217特征/200ExtraTrees/seed20261029/jobs4合成成本工作进程87746已结束，未重复启动。监督器在3600.0768854669295秒墙钟、13597.843748秒CPU后按预登记一小时窗口发送TERM，exit−15；峰值RSS10498296KiB，低于12GiB停止线，cgroupOOMdelta0。full4700fit未完成，没有COST_RESULT、完整模型、保存/恢复测时或12小时总成本估计。冻结科学代码、测试、日志原样保留；新增只读执行关闭检查确认原哈希/依赖及终止证据，结果为verified_full_QUANT200_onehour_synthetic_screen_incomplete_wall_closed。

这是完整配置的资源筛查未完成，不是准确率失败，也不能推断完整方法必定超过12小时。未接触真实UWB值/标签，不能声称QUANT真实训练已启动或已有新准确率。关闭确切一小时执行配置，不改特征、树数、深度、seed、精度或视角救援，不重跑旧窗口，不否定整个方法家族。现有Arsenal25×2000/jobs4的96.75%诊断/94.3%验证及完整私有恢复证明保持有效，确切准确率配置已关闭，不重新评分。

下一方法固定为已发表DrCIF完整论文规模500棵树与原生ContinuousIntervalTree（CIT）20阈值。依据Middlehurst等2021 HIVE-COTE2.0，MachineLearning110:3211–3243，DOI10.1007/s10994-021-06057-9。已核查主文3.2及实验结构、主源索引Table2配置；直接表格URL在工具中不可访问，未声称读完PDF或完成Java数值复现。最新四记录的有界精确DrCIF/CIFClassifier/CanonicalInterval检索无命中，不是穷尽历史证明，旧通用描述子/catch22模块也不等于完整随机区间森林。

使用原样aeon1.6.0固定commit b694b1c03c0a753cc844dcb9ee357d77ca6bda6b，分类器、CIT、baseforest、periodogram、randominterval、catch22共6个安装源码逐字节Gitblob核验通过，BSD3license及原文件本地保留。现代库默认200/None熵CART显式设为500/CIT20；每表示10-of29属性、区间数4+floor(sqrt(repLen)*sqrt(channels)/3)，最小3、最大表示长度一半；三表示raw/diff/periodogram。现代原生periodogram把90补零至128后取64bin，因此长度[90,89,64]，567通道请求区间[79,78,67]，去重前每树2240列。原生区间去重、各表示独立属性采样、500概率平均均保留；这不是原Java随机抽样/运行环境精确复现。额外z-normalization无，输入仍拟用既定有符号RMS。seed20261029、n_jobs4/threading、BLAS1/Numba1固定；线程共享完整表示以避开有限磁盘和多进程输入复制，不承诺4倍加速或所有Python/Numba释放GIL。

执行故障已实际发现并修复：第一次合成full500串行fit因原生Catch22的Numba无法统一float32/float64outlier_series类型而在完成前失败。未访问真实UWB。保留attempt01 wrapper/test/完整日志，改为进入原生fit/predict前把输入完整提升为float64；float32RMS每个值均可精确表示，未改原生算法源码，也未用任何准确率选择精度。修复后完整500树检查通过。float64引入的实际内存与时间将纳入新的正式成本，不用旧float32结果外推。

独立检查通过：差分和补零FFT对FP64参考、7种基本统计对NumPy参考；CIT含NaN的3分支样例的分布、熵增益与索引精确。合成[24,3,90]、12类上完整500树jobs1与threading4的全部区间、属性子集、树节点、叶概率精确一致；原生/分块预测和概率、pickle全部学习状态恢复精确一致；关闭推理跳过掩码、重新计算全部所选特征后，手工500成员概率组合精确一致。该检查计129910个所选列、195576个非有限特征值；原生把无定义/无穷统计转NaN交给CIT缺失分支，原样保留，不能把结果描述为所有原始特征均有限或擅自归零。这里并未独立重写22种catch22全部内核，也不是full567/4700成本或准确率证据。

正式成本目录 `/workspace/uwb-drcif500-full-cost-20261102`，11个冻结文件/2671个依赖哈希；COST_HASHES SHA=4e77d0fa486239f16241d3c846e414fe13e7629edd05a1d3d4a42bd2f3b15256。4项成本边界检查与缺失远端登记必须阻断检查通过。全部本次变更精确远端回读及分支头核验后，才允许一次full4700×567×90合成数据，原float32fixture进入原生时提升float64，完整500CIT20/[79,78,67]区间拟合；fullfit包含提升和原生变换。不是小N外推，不减少树/区间/属性/视角，不改CIT为更小CART。

预登记资源策略：一小时墙钟/CPU软14380硬14400秒/RSS12GiB/.2秒监测及cgroupOOM。保守3fullfit+2×2200×maxeval64/64+4maxsave+2maxload+180≤12h，CPU4倍≤48h，最大实测RSS+2GiB≤14GiB，8模型+64MiB≤剩余磁盘。全模型先经Protocol5计数字节/SHA的非存储sink，若8模型预算超出空间则明确记录存储门槛关闭，不写检查点、不冒称完整时延或准确率通过。成功路径三次eval64/完整保存重载，全部树/区间/属性学习状态逐树流式摘要与概率/预测精确；仅已记录SHA/大小且成功恢复的可弃合成副本可移除，不涉及真实UWB模型。

当前正式成本尚未启动，真实DrCIF训练也未启动。后续先检查LAUNCH/PID/STATE/TERM，避免重启；仅正常exit0且无任何停止/OOM才运行独立verify_result.py一次。资源门槛通过仍需另行冻结预登记真实runner、完整独立审计、私有保存和一次评分finalizer，不能提前访问真实训练数据。固定未来协议train用户3–6/4700、val用户2/1000、diag用户1/1200及canonicalIDs，不使用fit_predict/OOB作新选择。验证支持0..9、诊断0..11，固定参考.9866666666666667/+ .005/tol1e-12；完整数值审计及私有分块/全成员恢复后才一次评分；失败确切配置关闭无seed/树/区间/属性/基树/精度/角色救援，通过也只另行source6确认。

用户7–8标签不用于新选择。历史八用户反复开发均探索性，无独立验证/新八折提升/原创贡献证明。已发表方法可作为强基线迁移，本轮不会仅因尚无原创性而停止验证。目标未达，原自动接续保持启用。原789源码、用户README清理、仓库当前可见性和Drive私有性保持；已有数据路径有效未重下载，全部云端不依赖Windows。
