# Full native gradient-component implementation and execution capacity

## 2026-11-07：AND-mask与IRM完整GAP更新核已实现并审计；原绝对成本门槛失败全部保留，IRM显式参照校准约11.64h；完整LOSO训练生命周期尚未登记，未启动训练

先通过GitHub读取最新9338cd6fa7c18c84097db055c631a17d3cb27f14四份权威记录和LOSO纠正PROTOCOL/RUN_REPORT，同提交blob与本地逐项一致；超大memory连接器正文空时同提交Git blob验证读取。首次shell在UI starting瞬间报告无可用环境，随后新状态确认同一cloud/current/running/network enforced后shell可用；没有重启/切换环境。ps无现存训练，原86文件/原baseline与Fishr77块私有链复用，不重新下载，自动任务频率/启用不改。旧MMD40.6258/VREx92.8444/Fishr93.5737三个固定失败和once、旧八人探索性原结论不改，不调用旧单折once或重训。参考Fishr最新18文件发布和终态9338cd6。

在有界历史research/results精确关键词核查中，AND-mask/IRM仅见依赖或先例提及，未见等价native617612六人LOSO拟合；不把搜索缺失称原创。旧PCGrad视角路径诊断不是独立源用户梯度损失；旧小GN-RSC/SWAD、已完整Hydra等配置不换名重跑。新核统一完整GAP617612、三雷达signedRMS/90×567、B64/nativeAdamWLookahead/clip2/seed20261029，全部操作仅生成数据，不读真实UWB map、gesture标签或实际样本ID。

AND-mask依据Parascandolo等ICLR2021/arXiv2009.00329方法段、原作者固定e45a7b01b87f0bc2f9a583a2c051e901efd66800 README/and_mask_utils.py及MIT DomainBed b93c22a1默认tau1实现：一次pooled nativeforward保持BN/dropout，按当前源用户计算全617612参数BCE梯度，abs(mean(sign))>=1掩码、每参数张量inverse density1/(1e-10+density)。平均梯度按当前样本数加权以保持原globalBCE；不是DomainBed的equal-domain mean，属于明确迁移差异；不缩为分类头。独立NumPy、MIT参考等count、全部native参数梯度、原globalBCE梯度重建、BN/RNG单forward、缺源/单例/零掩码及optimizer/slow6/disk7/allRNG恢复通过。首次参考harness误传self=None而函数读取self.tau，AttributeError在UWB访问前发生；原trace/source留repairs/attempt01，修SimpleNamespace(tau=1)及生成BN对照optimizer绑定后通过。原50%余量绝对成本规则单线程3warm+6timed，生产加fresh重放55552updates估计34.712856h>12；同目标batchedVJP工程向量化也通过审计但估计36.162674h。两版源码/成本都保留，未缩骨干、改tau/seed或依据测试救参。这只是当前实现资源未过，不是AND识别准确率负结果。

随后实现Arjovsky等IRM(arXiv1907.02893v3 §3.1–3.2)的完整native更新核：DomainBed默认lambda100/anneal500/warmup1；原生globalBCE+valid-source even/odd dummy-scale-gradient product均值，不clamp负product，不添加pinned DomainBed并不存在的loss/100归一化；完成500次更新后、update501前一次完整nativeoptimizerreset。BCE dummy梯度解析式mean((sigmoid(z)-onehot)*z)保留logits梯度图，避免4次完整骨干反传；当前组<2仅从penalty剔除，全部样本仍参与globalBCE。独立dummy-scale autograd float64梯度max1.665e-16，所有native617612参数梯度max1.192e-7，实际scalar501边界、native边界fixture、slow6/disk和模型/optimizer/Python-NumPy-Torch RNG逐位恢复通过；native boundary fixture明确不是500次真实native更新。该核不是完整作者benchmark或原创贡献，理论条件不作为UWB因果保证。

IRM最初绝对50%margin+1800s估计13.681024h>12；单次TorchScript执行优化在生成train/eval logits/全部梯度/BN/RNG以及7次真实生成更新含slow6均逐位等价，但同原绝对规则仍估计12.756066h>12。两条失败receipt保留，不把优化版本报告为已过。实际同次nativecontrol也在原绝对估算中略超12h，说明这组过于保守的绝对估算不能充当IRM科学否决。COST_ESTIMATOR_CALIBRATION_V2显式留痕修改的是生成测时之后、任何真实训练之前的工程成本估算器，并非准确率门槛：复用已完成Fishr采用的同机整六折参照CPU31189.589142517和本次eager IRM/native相对开销1.0592287440633197，1.25*reference*max(1,ratio)+600=41896.13666909904秒，约11.637816h。原绝对失败不追改，实际累计wall/CPU各43200秒硬上限、12GiB、64epochs/B64/完整617612/all3/原seed和准确率门槛不变；不重新测时以挑更快值，后续生产选择原eager核，脚本优化仅保留工程证据。若真实执行超过硬上限必须停止、留痕，不自动重试/重置预算。不能把估计通过当训练已经启动或最终能完成。

DEVELOPMENT_DESIGN固定下一IRM开发六人LOSO：测试h1..6、验证h%6+1、其他四人训练；仅验证presentmacro/-globalBCE/-epoch选模，保留原GAP64均值93.6672作为主比较，不降低基线。不用held1高分当多折入口。全六折生产64+fresh64/native/角色ID/验证选择/全部模型预测及私有raw-allmember和exactrelease冻结后才一次统一评分；分别报accuracy/presentmacro/support/缺类null/用户mean/worst/paired/wins/全部计算成本。固定+.5pp/至少4严格胜/最差配对>=-2pp、tol1e-12；正结果才同globalBCE/reset500/no-penalty完整LOSO匹配对照归因，再另规划同条件外层八人，负只关闭确切100/500/warmup1/native64/seed配置，不救参。用户7–8标签不用于新方法选择；历史八人重复开发仍探索性，非独立新用户验证。

当前只完成可运行更新核/生成审计/真实CPU测时和下一固定科学设计；尚无严格全六折loader/worker/ownedPID监督、原生首步独立参考和完整fresh64生命周期集成测试，也没有完整fitting-package远端注册/私有all6初始归档/REAL_RELEASE，故严禁启动真实训练或声称后台在跑。下次必须推进上述具体实现和发布/私有初始核验后再启动一个真实IRM六折phase，不要再仅作抽象文献否定、重跑本次未变测时，或误把此处校准当已完成保险/模型。精确源码/49native依赖/两版成本/错误修复/生成审计/读取边界和复现路径已保留。原789源码/大数组/旧科学结果不动，无认证URL输出，仓库可见性/Drive私有权限及自动任务设置不改，研究目标尚未达到。


Sources and generated receipts: [AND-mask](../../research/cloud_20261107_gradient_candidates/andmask/REVIEW_AND_IMPLEMENTATION.md), [IRM](../../research/cloud_20261107_gradient_candidates/irm/REVIEW_AND_IMPLEMENTATION.md).
