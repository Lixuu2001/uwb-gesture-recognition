# Published Fishr native full implementation: native_preregistered

## 2026-11-07：Fishr完整原生617612骨干/严格六人LOSO实现与合成审计通过；固定CPU预算通过，拟合尚未启动

最新远端父提交27c3b3efeeee3f0f47e42911ca0b7019cb2d651d的四份权威记录和LOSO纠正协议/报告已通过GitHub与同提交blob核对。MMD40.6258%和VREx92.8444%相对GAP93.6672%的固定配置失败结论、原模型/once/私有档案及成本全部保留；原VREx64529为僵尸已结束，没有重复训练、旧held1once或测试救参。

Fishr（Ramé等ICML2022/PMLR162，https://proceedings.mlr.press/v162/rame22a.html）完整native集成已实际完成，独立目录/workspace/uwb-fishr-native-loso6-20261107。固定作者DomainBed b93c22a默认lambda1000/EMA.95/anneal1500，完整native617612参数/三雷达signedRMS全部输入/B64/64轮/seed20261029/AdamWLookahead/clip2、原生全局mean12 BCE、全部4个MLP分类器参数18060梯度坐标；人口方差及作者固定1/(1-.95) detached历史EMA。完整优化器于update1501之前一次reset，保留EMA；每个原生混合sourcebatch仅一次骨干/分类器forward和combinedbackward。warmup0的反向图代数化为classification，仍计算更新EMA；缺组历史不变且不计当批距离，singleton人口方差0。BCE替作者CE及原生sampler/骨干/优化器迁移差异明确，非作者整个图像benchmark复现，已发表baseline不称原创。

严格六折loader和训练/验证truth切片、每折独立初始化、完整EMA/optimizer/Python/NumPy/Torch/order RNG/BN checkpoint、两原子slot/journal与未提交suffix回放、完整64生产+各自fresh64逐step/65epoch/首步/所选输出审计、ownedPID/boot/startticks资源监督、完整私有归档与只在全6之后的唯一统一评分和无标签推断程序已实现并冻结。沿用h测试/next验证/其他四人训练及train-val选模，全部折不受单个测试人分数控制。正结果才执行同全局BCE/同reset1500/no-variance匹配对照，再讨论归因；不拿VREx等域均值/reset500充当对照。

全6不同角色配置的生成数据实际核验均通过。完整native warmup模型/全部梯度/原生optimizer/BN/全部RNG逐位等于原生BCE；活跃Fishr与独立逐例autograd全MLP梯度和直接执行作者MovingAverage/l2 AST参考的完整模型梯度本次最大差0，penalty约1.23412888797e-8且梯度连通骨干；1501次真实小参数optimizer更新确认reset整个fast/slow一次且保留EMA，完整native随机dropout下一步/重置边界磁盘恢复逐位一致，tail20/单例/缺组、metric定义和自有进程identity停止核验通过。这些全为生成数据，不是识别准确率或完整作者协议复现证据。初次源码组装匹配了类内step导致IndentationError、初次检查误用作者注册表前缀key导致断言，均在任何UWB读数/拟合前修复，旧错误/执行日志/旧manifest留痕；作者无关regex SyntaxWarning保留。

唯一固定CPUprofile为同完整B64输入的两臂各9次真实生成更新，Fishr三warmup+六active、第四步设completed1500合成counterfixture（不是1500次真实大模型训练），没有多次测速挑最快或救预算grid。native约5.3122wall/5.3102CPU秒、Fishr4.5703/4.5696；native首步冷启动影响，短测比.8605不当速度提升证据，预算强制max(1,ratio)。预先固定1.25*31189.589142517*max(1,ratio)+600=39586.9864秒=10.9964小时，低于主phase wall/CPU各43200秒，RSS上限12GiB；profile实际9.9053wall/9.9025CPU秒，peak593152KiB，生成核验/失败开销与既有MMD/VREx成本另留不抹去。

当前fitting_preregistered完整code/config/seed发布阶段、native_worker_implemented=true；新生产训练0，PHASE_IDENTITY/STATE不存在，真实UWB样本/标签未读。仍需此实际Git精确预登记release、私有untrained initial/全部角色ID raw/allmember回读与exactprivate release，完成后直接启动唯一完整6fold phase而不是等待确认。不修改自动任务频率/启用/云环境，不创建重复任务。之后全部六折64+64/审计/私有回读冻结后一次统一评分，准确率/macro/classsupport/逐用户/mean/worst/paired/成本全报，固定门槛+.5pp/4胜/最差配对>=-2pp；用户7–8不用于选择，历史八人探索性非独立验证。

