# Published Fishr native full implementation: all6result

## 2026-11-07：已发表Fishr完整native六人LOSO统一评分完成：93.5737% vs GAP93.6672%，-0.0934pp，1/6胜；三项预设门槛均失败，关闭确切配置

以完整私有冻结发布139a73a427140c21b1d8d8b6550395778c7b56c8为父。全部六折64轮生产+各自独立fresh64/native/角色ID/验证选模/counter审计及77块561095162字节1984载荷私有owner-only raw/allmember回读、22文件精确远端发布均完成后，才绑定真实FULL_PRIVATE_RELEASE并执行既定一次无标签推断和唯一whole6 cohort scorer；没有单折进入门槛、重训、旧once或测试救参。无标签12组推断logits与冻结输出逐位一致、模型/BN未变，external wall36.0421315440326/CPU36.420448秒，maxRSS1772940KiB，300秒cap内。唯一FINALIZE_ONCE已消费，FINAL_RESULT_STATE绑定SUMMARY SHA d7e4a3778ff49160c0b577c284e5047938c44667c676b4866ab7ac2819f7313c；原科学STATE仍保留pre-score冻结快照，不代表尚待重新评分，应以FINAL_RESULT_STATE为终态。41个科学冻结文件与原789源快照不改，旧MMD/VREx/混合结论完整保留。

逐用户GAP→Fishr的accuracy与present-class macro本数据恰同：h1 98.4167→97.7500(-0.6667pp)，h2 98.7000→98.2000(-0.5000)，h3 90.0000→90.0000(持平)，h4 94.2500→91.5833(-2.6667)，h5 96.0000→95.0000(-1.0000)，h6 84.6364→88.9091(+4.2727)。用户等权mean 93.6671717172→93.5737373737%，配对mean -0.0934343434pp；1严格胜/1平/4负。最差用户得分84.6364→88.9091%是真实局部正证据，不能包装成整体准确率提高；最差配对-2.6666667pp。样本加权pooled accuracy93.6521739130→93.5072463768%，不混称用户等权mean。h2缺0-based10/11，h6缺3，缺类recall=null；所有现有类各100条，所以accuracy/presentmacro数值恰同但定义不同，预测缺失类仍算错。完整support/每类recall、逐人选定epoch与成本在报告CSV。

既定mean>=+.5pp、至少4严格胜、最差配对>=-2pp均失败；有理数独立汇总核查mean_delta=-37/39600、worst_delta=-2/75、wins1，全部重建一致，不重新打开truth/预测数组、不新增评分或拟合。聚合报告器生成5个真实产物（MD/逐人CSV/每类CSV/成本JSON/SVG），SUMMARY及全部正负结果均保留。关闭确切Fishr1000/EMA.95/anneal1500/nativeGAP617612/globalBCE/B64/64/seed20261029配置，不做系数/EMA/重置/epoch/seed救参，不因h6正结果单独改门槛。预登记同globalBCE/reset1500/no-variance匹配对照的激活条件未满足，故未启动，不能归因于variance或reset，也不据此启动外层八人确认。

生产wall/CPU14779.878592/14779.175423秒，独立重放14691.573768/14686.370003，输出审计66.136240/64.978842；main最终29559.943319/29599.386026秒已包含各子阶段不能重复加。完整档案构建/三次provider上传故障恢复/原队列known-ID guard drain/77原始回读与成员校验成本和凭据保留；传输span1688.204913秒与其他操作重叠，未测CPU为null不写0。一次固定推断GAP2.108361/Fishr2.086092ms每样本，不作统计速度改进证据。此前MMD30865.434969/30891.040083秒及40.6258%失败、VREx31157.163367/31189.589143秒及92.8444%失败完整另计。scorer内部wall实际保留，未测scorer/报告CPU明确不捏造。

准确率证据：当前完整同条件LOSO6未显示平均提高，h6及worst_user改善单独如实保留。原创贡献证据：尚未验证。Fishr为Ramé等ICML2022/PMLR162已发表方法，固定DomainBed默认组件迁移到globalBCE/native fullhead/all3 radar，明确迁移差异和共享错误边界；精确重放并非无bug证明。用户7–8未用于新方法选择，历史八人反复研发仍探索性，不称独立新用户验证。原GAP均值/旧混合94.6290不同预算和旧失败不追改。

当前无训练/下一方法启动，未改变自动频率/启用或云环境，不创建任务，不依赖Windows。研究目标尚未达到，保持原自动接续；下一项需核对历史和不同机制文献、完整可运行实现，冻结同LOSO6配置/seed/预算后实际验证，不能重开本配置或仅以不能立刻宣称原创为由停止实验循环。恢复继续使用FULL_COMPOSITE_RECOVERY_INDEX/持久私有Drive链，原数组/权重/实际样本ID不入Git，无认证URL。当前Fishr唯一once已消费，后续严禁重训/重新评分该配置。


完整结果：[FINAL_REPORT.md](report_artifacts/FINAL_REPORT.md)。
