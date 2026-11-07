# Fishr native LOSO6: mathematical core only

已实现并实际运行合成数值检查，尚无新的UWB训练、完整native worker或拟合预登记。

候选为Ramé、Dancette、Cord的Fishr（ICML2022/PMLR162），固定DomainBed提交b93c22a1cfc3b2428398272c1a116c8de1f4139e的默认lambda1000/EMA.95/anneal1500/前1500更新penalty0及一次optimizer reset。作者固定EMA除数1/(1-ema)保留，历史state detach；没有改成1/(1-ema**t)或源梯度范数校准。正式论文核查复用既有FISHR_PUBLISHED_REVIEW.md及其出版方PDF记录；本轮新核对源码，不冒称新完成整个作者benchmark复现。

fishr_core.py对native两层128→128→12/ReLU/dropout.2分类器一次实际Dropout前向，解析计算全部4个参数张量、18060个分类器参数的每例BCE梯度，保留二阶图可回传到特征；不是仅末线性层的旧缩小统计。当前只实例化分类器，目标完整GAP617612骨干尚未接入。native BCE mean12代替作者CE是明确迁移差异；原native全局分类BCE和强GAP基线不降低/重训。

CORE_CHECKS.json是实际合成结果：解析每例梯度与逐例autograd最大差4.163336e-17；Fishr penalty差0；对特征与全部分类器参数的二阶梯度最大差1.654361e-24。固定提交MovingAverage/l2函数经Git blob SHA独立核验并由AST直接执行，5次EMA输出逐位一致，当前样本梯度补偿和detached history、序列化继续、空组/单例/连通零、1500边界检查通过。所有输入和标签均为固定seed生成；optimizer_steps=0；约.040593秒wall/.039096秒CPU仅为数学检查体耗时，不是完整native性能预测或准确率证据。作者原misc.py未改，解析时保留一个无关regex转义SyntaxWarning，检查exit0。

历史SOURCE_FISHR_PROTOCOL为265836参数/24轮/seed2026/最后线性层CE梯度/时间偏差修正EMA/源梯度校准/无reset/固定tail选择，旧失败结论保持；不能等同于当前默认组件候选。只根据协议/实现范围辨别非等价，没有使用旧用户7–8的分数来选择新参数，也不把已发表方法包装原创。

迁移还明确声明native B64随机混合源组与作者分域loader不同：当前batch缺组时该组EMA不更新且不计当前penalty，单例人口方差为0。完整参数和这些差异写在FITTING_PROTOCOL_DRAFT.json；它是候选设计，不是可直接启动的预登记。接下来必须接入完整native GAP/角色ID隔离/64轮worker、完整EMA与RNG/optimizer checkpoint、独立fresh64及原生审计和owned-PID预算监督，测完整native CPU成本，冻结全部科学文件/配置/seed与完整六折协议并完成私有initial和精确Git release后，才能训练。训练/验证选择；不得held1单折筛选、用户7–8选择、测试救参或启动第二份旧任务。

完整6固定花费门槛提议维持mean+0.5pp/至少4严格胜/最差配对不退超2pp；拟合前仍须正式冻结。若未来有效，先做同全局BCE/同reset1500/no-variance匹配六折控制；VREX的等域均值/reset500控制不可当本候选匹配对照。所有历史全八人反复开发仍探索性，不称独立新用户验证。本核心通过只说明实现数值正确性，不能声称方法提高识别率；新六折训练没有启动，研究目标未达。
