# Complete IRM native LOSO6 private modelset verified before held scoring: all6_private

## 2026-11-07：IRM六折全部模型/轨迹/预测/角色与75块完整私有原始回读完成；未评分，待精确释放后盲推断成本与唯一cohort评估

最新父97a33372007509242692180d4c955d3f4d53863b为本次提交唯一父，force=false、全部文件精确回读后才移动ref。原PID92152已正常结束，六折STATE完成并保留预评分快照，background_training=false；没有重启/补训任何已完成折、没有调用旧MMD once/读某一held分数或自动换方法。保持原789/可见性/自动任务频率启用/云环境。

原冻结617612 native GAP/三雷达全部signedRMS/B64/64轮/seed20261029/global BCE/IRM100,warm1,anneal500/native optimizer reset500不变，45科学项SHA857e06d91fd2da1ac1428a2716e1df541db8198245dc1d8041922f8d75ad8b6d；六折完整64生产与fresh64、逐步native/model/slow/optimizer/全部RNG/65轮记录和所选/最终输出精确一致，角色ID与6900唯一source IDs、h%6+1验证/另外四人训练严格分离。所选epochs46/62/5/62/47/6均只由训练/验证选择。独立首步梯度检查每折全617612参数通过；完整同实现重放不单独排除共同实现错误，未声称逐步都独立autograd核查。

完整模型集归档75块548348323字节/1995payload（tar1996成员含清单），archiveSHA3436cbcc7bb738e30e5279acc9c998c3468a10954e66aa68ad9b08440494cf10。全块upload→owner-only metadata→raw fetch(download_raw_file=true/include_base64=false)→立即物化sizeSHA，经最多3整条修正队列pipeline；全部块、整档和每个载荷原始回读验证通过。完整归档含all6生产/重放/native/选模/角色/无评分输出/源依赖和布局，原始大数组和runtime外置通过持久索引恢复，真实ID/权重/预测不进Git，不输出认证URL。FULL_COMPOSITE_RECOVERY_INDEX联接完整75块、初态/source原注册、原始86文件/163块与现有GAP基线链，恢复不重训/不降低baseline。完整解包/中断resume未在健康环境执行，不把raw-allmember字节证明冒称端到端恢复实测。

本次75块传输初始失败记录数0，原始记录[]；TRANSFER_ATTEMPT_RESULTS按attempted/verified/failed分别计数，不把失败计为verified。任何恢复必须记录已知ID复用或未知ID精确父目录/文件名搜索后一次明确补传，不能盲重传；修复回执另附。先前held4单折上传断连及计数修复仍原样保留，不修改历史记录。

全六训练/重放/监督阶段wall29509.953316993022/CPU29548.175221702秒（8.20h，未跨折重置12h硬上限），原各阶段细分成本保留。完整raw-allmember验证wall3.7077735899947584/CPU3.706438515秒；全档构建仅测operator wall和parent CPU、不含child构建CPU，child CPU/完整网络和远程运维CPU未测，明确不是0。前四折保险与本次全量保险覆盖重叠，不能当成新增模型拟合或把重叠墙钟双加。旧MMD30865.434969089023/30891.040083422995（h1子成本不重复相加）、VREx31157.163366941997/31189.589142517、Fishr29559.94331930898/29599.386025808墙钟/CPU秒继续另计。

当前所有held预测已冻结，gesture truth/新准确率尚未配对；原始y只作既定二进制SHA核验，不是标签读取/调参。本次完整私有元数据精确远端回读后绑定FULL_PRIVATE_RELEASE：commit、完整archiveSHA、frozenSHA和once scorerSHA；随后严格按原注册operator/inference_cost.py --run只做一次盲配对推断（300秒wall/CPU、12GiB），逐位核验全部12组baseline/method logits与参数buffer不变，再调用原finalize_cohort_once.py一次。不可先评分单折、不可调lambda/退火/seed/输入/epoch/折。统一报告逐人accuracy与presentmacro/support/缺类null、用户等权mean/worst/paired/wins、pool另列以及全成本。+.5pp/4严格胜/最差paired>=-2pp门槛固定且不等同于任何实测增益证据；正才同globalBCE/reset500/no-IRM六折对照分析归因/再规划outer8，负关闭确切100/500不救参。已发表方法迁移证据与原创证据分开；users7–8不用选择，历史八人探索性非独立新用户验证，尚无新准确率或原创结论。

