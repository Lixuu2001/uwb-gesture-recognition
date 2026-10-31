# 2026-10-31 — 固定H-Inception源用户确认代码与协议预登记，尚未启动新训练

以最新远端b1ae88cbfeed1feee27e86bc674c601eeb26576f及四记录为准。已完成H pilot完整独立审计/私有恢复证据；held1 99.416667%相对固定98.666667%高0.75pp过原门槛。现单独登记source1–6确认，复用held1，仅新训held2–6，不切到Disjoint/MLSTM，不重训已有pilot。

同一完整851434/601006模型、RMS90×567+作者channelznorm、100epoch/B64/seed20261029/原Adam与最低trainingloss选模固定。heldh/valh%6+1/其他四训练；五新折36000updates/2290000训练出现。producer和独立完整replay auditor均不索引本折诊断标签；五折双预测冻结且全审计通过后才统一评分。预登记对已审计LocalCNN和GAP各自要求mean>=+1pp、>=4/6strictwins、worst>=−2pp，失败关闭确切配置无救援扫参。不同预算/input管线比较非因果架构消融，用户7–8标签不用，全部source探索性，尚无原创/八折结论。

6项协议测试通过；原held1初始化所有model/Adam/BN/metrics/RNG/callbacks/modes完全一致；variable-count合成3+2更新恢复/native观察隔离/跨折拒绝/train+eval sampler覆盖通过。登记前静态review修正继承的held1-only eval计数guard、pilot marker字符串和stdlib queue命名冲突；无新UWB值/标签/fit、无科学设置变更。1465冻结依赖SHA；实际4CPU/16GiB；最多两个独立job supervisor、各一线程train或audit，CPU计费分开。train43100wall/43000soft43100hardCPU，audit14400wall/14390soft14400hardCPU，无自动重试。

当前尚未启动新训练。先将study/19文本文件与本四记录创建latest-parent unreferencedcommit，全部逐字节回读，再force=false/expected_sha推进；之后写准确marker，核查无重复进程/jobs/run01后只启动一次controller。运行receipt另存，不以预登记声称已启动。完成后完整新modelset私有备份全部成员回读再发最终均值，原始大数组/实际ID/model/runtime binaries不入Git。

代码/协议/恢复说明：results/cloud_20261031_hinception_sixfold/study/。目标未达成，原有自动接续保持，不新建重复任务。

# 2026-10-31 — H-Inception真实source-held1 99.416667%，对固定98.666667%高0.75pp过门槛；完整审计与7私有块/193成员回读完成

先读fcb8c1afdbf544f063b3ed735510f1df229fd91f最新四记录、核对云端实际状态。手动修复fullaudit57621已exit0/no timeout，9955.080367621966wall/9953.554849CPU；完整100epoch/7400update/470000ID、所有初始/首步/best/last/finalprediction model/Adam/BN/metrics/callbacks/modes/allRNG、nativebest/last、6900rawRMS+原znorm、2200probs/decisions与全部1447SHA均通过。原attempt01 exit1/9453秒failure、Python-only observation隔离code/patch和所有原assert保持均保留，不重训、不改阈值。

完整归档50350252字节/7owner-only块/192payload+MEMBERS=193成员已在私有folder12AGMwT99yXCTYBQocQaX901Br-IelmOf元数据先核→rawfetch(download_raw_file=true,include_base64=false)/max3/立即物化sizeSHA→重组SHA/每成员验证全部通过；无分享设置变动。archiveSHA4110292ce8f3c1093d5e7fdba5e2c928d7c2b2666a1724c8f2f349d69081e8c4，receiptSHAf7e1f46b98fe02782a954870986ce2a3b1e3b5a8c15f088f6fa6003e17de2771，auditSHA09088bca39ed0cd965a5aa0a5d669f73fc56cced99a1bc34a027976bd3975623。包括所有模型、准确ID/log/preds、原失败+修复，非wholeenvironment/jobresume；raw/maps/runtime binaries另按既有恢复索引，不入Git。完整downloadindex/receipt在results/cloud_20261031_hinception_pilot/complete_backup/。

完成private后才解析真实audited score：held1 presentclass macro99.4166667%、val2 98.8%，比原固定ref98.6666667%高0.75pp，原>=0.5pp gate true，route=known_baseline_source_confirmation。选epoch75来自严格最低trainingloss，非diag。**这是一个研发用户的探索性正结果，不是source6均值、旧八折提升、独立验证或原创贡献。** 完整width32/bottleneck32/depth6/17fixed/851434total601006trainable、100/B64/seed20261029/RMS+authorchannelznorm固定不改；input1→567/nativeKerasTorch/CPU100适配明确，不称1500TF/五ensemble复现。报告results/cloud_20261031_hinception_pilot/FINAL_REPORT.md。

下一走已授权的**单独源用户确认**，不切到Disjoint/MLSTM：原方法/seed/B/E/optimizer/selection/input全固定，source heldh/valh%6+1/其他四train，复用本held1只训练尚缺held2–6；先实现完整runner/auditor、冻结成本/角色/强基线/gate并remote逐字登记后才新UWB fits，全部5新diagpreds冻结后再aggregate评分，7–8标签不作选择。现成LocalCNN6mean93.594444%、GAP6mean93.667172%与相同roles/quality IDs可作固定强pipeline比较，但输入/预算/优化器不同，非architecture-matched ablation；不得把单用户99.4与旧八折93.44相减当提升。source确认与matched消融/后续冻结八折证据仍未完成。

当前无active训练/审计，旧controller状态是历史failure/完成待agent，FINALIZATION与private receipt代表本轮已收尾；下一先检查是否有新的sourceconfirmation进程，避免重复。准确率价值与原创贡献分开，all8历史开发仍exploratory；用户目标未达、既有接续保持，不新建/暂停。原789和真实数组/ID值/models/binaries不上传Git。以下保留历史。

# 2026-10-31 — Python RNG观测隔离修复已远端22文件精确回读，完整手动重审57620/57621已实际启动

d5c536eb95a2875b2f5cc3157b142dfd673c33f4以ad7ce0b最新父、force=false、22文件逐字回读后才创建AUDIT_REPAIR_REGISTRATION并仅一次启动driver/supervisor57620、worker57621。当前repaired_full100epoch_audit_running，完整原100epoch/7400update重放、所有原assert与finalcompare/tolerance/gate保持，不是训练restart；相同14400wall/14390soft/14400hardCPU，原失败9453秒另记，不静默扩预算/自动retry。原audit_supervisor01移存audit_attempt01，九项原failure/source/log SHA全部复核不变，canonical audit_supervisor01现在是手动attempt02。1423science+19训练产物及16repair冻结SHA不变；diag SUMMARY未解析，无新分数/提升/原创结论。

**下一先查57620/57621和/workspace/uwb-hinception-audit-repair-20261031/STATE.json、canonical TERM/COMPLETE_AUDIT，不再按旧POSTPROCESS_STATE的postprocess_incomplete误判当前进程。旧root失败状态/AUDIT_INCOMPLETE保留描述attempt01，不能改成成功。** repaired worker pass且TERM exit0后driver才build complete_backup_20261031，builder保留所有原检查并纳入原failure/repair code/hash/log/patch；driver没有Drive/Git或重训/retry能力。agent还须owner-private metadata→download_raw_file=true/include_base64=false/max3/即物化SHA/重组全成员完整私有模型集回读，首步保险不可替代；之后才发布audited分数/原gate，pass须source确认，fail只关确切配置，不diag救援。

runtime/registration/原失败保留凭据与接续说明results/cloud_20261031_hinception_audit_repair/RUNTIME_AND_CONTINUATION.md，修复代码研究目录同名research路径。本阶段不选下一Disjoint/MLSTM实际cost/config/fit；准确率价值/原创证据分开，all8历史探索性、用户7–8不作新选择，目标未达、既有小时接续保持。原789和真实arrays/ID值/models/binaries不改或入Git。以下保留历史。

# 2026-10-31 — H完整审计末步Python RNG观测副作用失败；定位与回归通过，手动全程重审修复先远端登记

先读ad7ce0b123eadae90c46bcf2c8c98c9144305db4最新四记录/云端能力及实际进程。原audit55822 exit1/no timeout，9453.333773313905wall/9452.627575CPU；旧controller54482已postprocess_incomplete，无运行中的训练/审计。trace到final_prediction_state的精确比较，前面100epoch/7400update/2200prediction路径已到达但**不能当成完整审计pass/分数结论**。训练仍100epochs/7400updates完成、所有19产物/1423冻结SHA不变，不重训，不当方法准确率负结果，未解析diag SUMMARY。

只读checkpoint诊断明确：saved last→final除bestweights外optimizer/metrics/所有RNG/callbacks/modes/identity/cursor均一致；mirror保存native模型无状态改变。审计额外keras.models.load_model检查best/last时仅消耗Python random，model/Adam/metrics及其他RNG不动，这个构造步骤不存在producer预测路径。恢复inspection RNG再置bestweights，甚至不读真实输入/labels或realforward，就与保存final全状态exact。原故障是checker观测副作用，非数据模型/预测拟合问题。

修复只在native模型检查外包Python RNG隔离，之后全before/after snapshot仍必须exact；不恢复/忽略其他状态，不放松任何tolerance/原gate/原assert/finalcompare。合成原生reload回归pass且注入optimizer改动仍失败；4driver/staticguard测试pass，原所有assert和compare调用AST保留。原audit_complete.py/ops/producer/协议完全不改。新repaired auditor+helper/driver/backupbuilder code研究目录research/cloud_20261031_hinception_audit_repair/，报告results/cloud_20261031_hinception_audit_repair/REPAIR_AND_RERUN.md。

本提交先登记16修复文件+manifest逐字远端回读，再marker与**一次手动完整重审**（非自动retry），相同14400wall/14390soft/14400hardCPU。driver把原audit_supervisor01保留为audit_attempt01再新建canonical supervisor；原failurelog/TERM/POSTPROCESS状态都保留，不伪改exitcode。新repair/STATE才是当前接续状态。重新跑全部100epoch7400update，不从未保存的partial推断完整pass；额外审计计算明确记账，不是新增训练。若成功新canonical COMPLETE_AUDIT记录repaired auditor/helper及原failedTERM；成功TERMs后builder保留所有原备份检查并加firstfailure/repair完整provenance到complete localarchive。driver无Drive/Git能力，agent还必须owner-private metadata→rawfetch download_raw_file=true/include_base64=false/max3/即物化SHA/重组全成员完整模型集回读，再发布分数按原gate。

**本提交时修复全审未启动**，下一先看新STATE/PIDs/TERMs，已启动则绝不重复；失败保留trace手动修复不autoloop。准确率/原创结论均无、下一Disjoint/MLSTM实际config/cost/fit仍禁止在本阶段越过未关闭前置；pass须source确认，fail关确切CPU100不diag救援。all8历史探索性、用户7–8不作新选择，目标未达、既有小时接续保持。原789、真实arrays/ID值/模型/二进制不入本次Git。以下保留历史。

# 2026-10-31 — H-Inception完整100epoch/7400update训练exit0，47万ID及19产物SHA通过；原全程数值审计正在执行

先读197dce032e1a7968f2e1fbcd87e0bf24ff890b8e最新四记录和云端进程。原训练已完成100epoch/7400update，supervisor TERM exit0/no timeout，9616.520260029007wall/9614.596172childCPU秒。producer两套predictions均冻结后已按原协议计分（本轮不解析SUMMARY），7–8未用。原controller54482已仅一次自动启动audit supervisor55820/worker55822，当前active，完整audit/complete private尚未完成。**background_training_running现在false、background_audit_running true；不是新训练，也未重复restart。**

新增只读产物核验pass：1423science/source/runtime/prior冻结SHA不变、19runfile SHA（SUMMARY只字节hash不解析）、best/last/native checkpoint与两套unscored predictions哈希相符；100轮×4700unique/7400batch/470000ID出现与dedicated sampler逐项一致、末批28；原strictmintrainingloss选epoch75符合100日志。核验器无model/prediction/raw/y数组打开或realforwardfit，**不是全数值重放/准确率审计pass，也不据此独立证明pred-before-label时序**。首attempt在最后进程关系检查遇到云端procfs缺少/proc/PID/task/PID/children；仅换ps --ppid、保留父子/命令/唯一worker断言后attempt02通过，原源/log/ENGINEERING_REPAIR保留。现有训练与审计科学代码不动、不重启。code/evidence research/cloud_20261031_hinception_completion_check/；report results/cloud_20261031_hinception_completion/TRAINING_COMPLETION_REPORT.md。

并推进下一可运行工具而非抽象否定：完整Disjoint-CNN默认2L synthetic profiler/纯cost算式，复用已pin source/nativeweighted engine，无B/seed/E默认，保留source mini=int(min470,requestedB)、classweight train/val unweighted、2587468total/2586956trainable。保守timeformula计入每epoch best+last两次fullstate、nativebest最坏每epoch，fixedordered预算和峰RSSgate。4算式/非法输入/orderedresourceclose/先拒绝测试通过；实际guard在当前missing fullaudit时先拒绝，未读policy、未import native model/计时fit、未创建profile。**没有选下一实际costpolicy/config/seed，也没有正式profile/新UWB数组标签读取或拟合；只是CLI代码，后续还须冻结policy与bounded supervisor。** code research/cloud_20261031_disjoint_cost_tools/。前source+engine50textrefs再核，隔离runtime/789/source不改。

下一先查现有audit/controller/PIDs/TERMs/POSTPROCESS；完成7400update fullnumeric audit并exit0后，核验全hash、owner-private metadata→rawfetch(download_raw_file=true,include_base64=false)/即物化SHA/max3并发/重组全成员，保存完整模型集，再发布真实audited分数按原gate。现有controller仅本地audit/archive，无Drive/Git能力，必须由agent接续，首步保险不可替代fullbackup。pass须单独source确认；fail关确切CPU100多变量迁移，不diag救援、不否定未跑1500/原TF/五ensemble。目标未达，准确率与原创贡献分开、all8历史探索性、7–8不作新方法选择；既有小时自动接续保持，不新建/暂停。以下保留历史。

# 2026-10-31 — 完整H-Inception多变量输入迁移CPU100已实际云端训练；102Adam/17fixed/64ID首步与2私有块保险通过

251695c6d7dc41149e508a15fbf15310b471263a的82新增/更新文件全部远端逐字读回且expectedparent/force=false推进后才创建marker并仅一次启动supervisor54454/worker54456。协议SHA8a97c1f8a1901004d81f03c03846cee15e9cedcd7bf1424d9ea62e974c3ed7fd，1423science/source/dependency/runtime/prior冻结SHA再核不变。完整32filters/bottleneck32/depth6/residual2/17regularfixedConv的H-Inception，明确原作者输入1→567/nativeKeras多变量迁移、单模型非五ensemble/原TF2.4.1/UCR复现；851434total/601006trainable/50trainabletensors/102Adam，100/B64/seed20261029/1CPU，全90×567RMS+原channelznorm，train3–6/val2/diag1。观察已完成4epoch/296更新，每轮4700全用/74批末28。两套最终预测未冻结、held1标签未读、7–8未用，**尚无最终诊断准确率、提升或原创结论**。

首步只读验证pass：全seed初始化一致，66模型状态张量变化/17固定卷积完全不变，102Adam变量/iteration1/全部有限；model/Adam/BN/metrics/callbacks/modulemodes/allRNG恢复逐项一致、64实际dedicatedSamplerID、1423SHA通过。firstSHA7763060e953621fad91ec3f6b6710a555b5323f821746651124238bc0e63cd03，verifierSHA90c7e2a2b7a706ffa13e761d03d497e5523e7e86626291ea20a769882e6a1b81。核验器无raw值/gesturelabel/新增realforwardfit；producer实际训练，不把核验器false字段误读为没有训练。

first保险11310946字节/2owner-only块/143payload+manifest144，archiveSHA b58ba7ec6c7905ee51058f6b81ea7747d7cc840da52cb8ce0f29d4335a2560ad，receiptSHA8d88132ff8f2c2e13a09f6de1da6362e7c9e9799e413e9ade087f03fc49f9aec。folder/chunk metadata、authenticatedrawfetch(download_raw_file=true,include_base64=false)即物化size/SHA/max2并发、重组全143成员通过。索引results/cloud_20261031_hinception_pilot/first_backup/DRIVE_BACKUP_RECEIPT.json。仅initial/first完整states、ID/role/evalpartitions、code/dependencyrefs，不含live/latest/best/futurecomplete模型/raw/maps/wholeenvironment/jobresume。最终完整模型集仍须审计后再备份，不能用首步保险代替。

controller54482已实际启动waiting_existing_training，注册冻结的code/HASH确认。只等待现有训练成功TERM→一次100epoch/7400update/470000ID/全部state与nativebest,last/6900RMS+原znorm（独立float64preset容差）/2200probs完整重放→本地complete_backup_20261031，无trainrestart/retry/resume/Drive/Git能力。**下一先查现有54454/54456/54482、run01STATE/ckpt/TERMs/POSTPROCESS，不重复启动或下载，运行中不选下一实际方法配置/cost。** 审计与成功TERMs/allartifactSHA通过后，agent按owner-private/max3/rawfetchbase64false/即物化SHA/全成员回读保存完整模型集，再按原registeredgate发布真实分数；失败留trace先修，不当科学负结果。

保持100/B64/seed20261029/输入适配/原strict最低trainingloss tiesfirst选模/ReduceLRloss/noearlystop不变，不held1/7–8label救援。ref.9866666666666667+.005/1e-12 gate，pass仅另source确认，fail关确切CPU100多变量迁移，不否定未跑1500/原TF/单变量/五ensemble。最终predstate是bestweights+lastoptimizer的推理审计状态，best.pt另含其自己的优化器；不宣称自动wholejob/midepochresume。准确率证据与originality分开，all8历史开发仍exploratory非独立test/旧八折；目标未达、既有小时接续继续，不新建/暂停。OS-CNN95.1667%/−3.5pp完整审计与10私有块收尾、LITEMVnativeTorch资源关闭（非准确率失败）已在前提交保存，不等价重跑。以下保留历史。

# 2026-10-31 — OS-CNN真实准确率95.1667%/−3.5pp完整审计及私有模型收尾；LITEMV资源关闭；完整H-Inception多变量迁移CPU100待远端预登记

先读bf6ee81a3bb249154e7011df995e92e1e58d42f0最新四记录；同一持久云端恢复可执行，无重复下载/训练/审计。OS-CNN原53525完整29400更新/470000ID出现/100epoch/714periodicevalbatch及mask副作用、所有初始/首步/末步/finalprediction model/Adam/BN/scheduler/allRNG、6900RMS与2200logits/preds全部精确重放通过。held1macro95.1666667%、val95.3%，固定ref98.6666667%低3.5pp，原gatefalse，关确切CPU100/B16/seed20261029，不按诊断救援，不否定未跑2000/500。训练TERM exit0/4903.09715075104wall/4902.700678CPU，auditTERM exit0/4900.682930332026wall/4900.316126CPU；81产物SHA/58science冻结再核，auditSHA cc2e00874b91c5b64ae02bd6131c75cffe12c6252a5403d0e40fb6aa8834ae27。

OS complete67336510字节/10owner-only块/93payload+manifest94，metadata→authenticatedrawfetch(download_raw_file=true,include_base64=false)即物化size/SHA/max3pipeline→整包/全部成员通过；包SHA d76caf191750bf19e7cc8198c979c01c53451ff39422794327ca3c92717f20fa，receiptSHA51520f7b5cefe4c3e8706c0cdd3d13e93fd8290995b20e202c26bfba4dd2c27a。索引results/cloud_20261031_oscnn_pilot/complete_backup/DRIVE_BACKUP_RECEIPT.json，FINAL_REPORT同目录根。完整protocolmodelset不依赖首步4块，不含raw/maps/whole runtime/jobresume；原进程已完成，不能据旧pid字段重启。

关闭验证后才固定LITEMV前瞻cost policy：原完整32/3层/17depthwise固定filters、621680trainable/889624total/B64/seed20261029/RMS+原channelznorm，1500→100预算43200/1CPU/14GiB，不batch/width/seed救援。单次synthetic8train/4eval/3nativekeras与3fullsnapshot计时exit0：61.004834954976104wall/60.952504CPU，maxtrain5.5936s/maxeval1.7112s/RSS2958280KiB。保守1500=1330044.5983s、100=88897.3594s均超限，**未选可运行轮数/未读新UWB值或标签/未LITEMV真实fit；这是nativeTorch资源关闭，不是准确率失败或原TF无效。** 1366冻结SHA未变，research/litemv_cost_tools_20261031及litemv_cost_20261031/RESOURCE_REPORT.md保存工具/策略/原始计时与成功TERM；synthetic模型留本地不Git。临时未注册LITEMV入口草稿移除，无实际job。

转向结构不同的已发表H-Inception：CF-4-TSC cf4d0e55ee11b7393ae3d317cdf5f22dcc51ba3e，modelblobbb8c08bf24cd3b3dcdfd90f079403900ab4df1eb，7原UTF8源Gitblob/SHA及GPL-3.0-only保留。原789精确词0命中，旧远端只prior-art提及，不是等价fullH-Inception训练证明。2022BigData DOI10.1109/BIGDATA55660.2022.10020496作者10页PDF已读abstract/intro/III.A–D/IV.A/IV.B开头，非全文results/figures。**原作者univariate输入1明确扩为567（一处literal），26tf.keras绑定nativeKeras、一处.hdf5→.keras；非原TF2.4.1/UCR或五模型H-InceptionTime复现。** 32filters/bottleneck32/depth6/residual2/k40,20,10/17固定regularConv按原代码sumchannels保留，非LITEMV的9639featureconcat缩小改写。实际601006trainable/851434total/50trainabletensors/102Adam。固定filters数学/不训练/3更新不变、NumPy sumchannel/evenSAME、恢复next2和native.keras全部通过；3epoch nativefit对显式batch/指标/所选ckpt/callback/allRNG epoch恢复/251plateau/stricttrainloss/tiesfirst通过。sourcepreparation research/hinception_source_probe_20261031/。

随后一项Hcost固定1500→100/B64/seed20261029/1CPU/RMS+channelznorm/相同资源门槛，单次23.338583308039233wall/23.316883CPU成功，RSS815392KiB，保守1500=310817.49064172595s失败/100=20890.104619972175s通过，按先验规则选CPU100；1384SHA不变。无真实UWB值/labels/fit用于cost，未按accuracy选择预算。research/hinception_cost_20261031/COST_REPORT.md。

**当前真实H-Inception入口/完整auditor/controller已实现测试冻结，但尚未启动、尚无真实UWB值/labels/fit；须此提交所有文件远端逐字读回后marker才能一次启动。** 工作区/workspace/uwb-hinception-pilot-20261031，协议SHA8a97c1f8a1901004d81f03c03846cee15e9cedcd7bf1424d9ea62e974c3ed7fd、1423science/dependency/source/runtime/prior冻结。full32/depth6/input567/12softmax/100/B64/seed20261029，source3–6 train4700/val2 1000/diag1 1200，74批末28/7400steps/470000appearances。原nativeAdam .001/eps1e-7/ReduceLR(loss,.5,patience50,min_delta1e-4,minlr1e-4)/strict最低trainingloss tiesfirst选模，无earlystop/诊断选模。全90×567RMS后原znorm AST逐sample/channel，固定12taxonomy不独立valOHE；全ID/batch/metrics/LR、initial/first/best/rollinglast/finalpredictionstate与nativebest/last.keras保留，双预测及finalpredictionstate先冻结再读held1labels。最终predstate是bestweights+lastoptimizer的推理审计状态，非best优化轨迹resume；best.pt有其完整优化器。

6项guard/native epoch/状态恢复测试及3controller守卫、独立NumPy含zeroSTD预处理检查通过。首个guard测试因坏前序状态fixture先触发cost字段KeyError；只重排拒绝顺序，attempt01源码/trace及repairJSON保留，复验6pass，无真实fit/科学配置更改。原库NumPy2 copykw deprecationwarning保留，不是训练失败。完整auditor重建100轮7400更新/全部实际ID、nativeckpt与所有states/RNG/modes/6900rawRMS+znorm（含独立float64容差2e-5/2e-6）/2200probs；训练43200/audit14400独立限时。controller只能等此一次训练→一次完整audit→本地complete包，不能restart/Drive/Git。源码协议research/hinception_pilot_20261031/；样本ID/arrays/maps/模型/wheel二进制不Git。

下一先查新工作区marker/LAUNCH_RECEIPT/run01/现存pid，防本轮远端后已启动而重复起。若此提交确读回而未start，才marker+一次launch；首step完整状态只读验证+owner-private保险+controller实际启动后另commit真实状态。若仍未注册，不能声称后台训练已跑。全程不用7–8labels，held1原ref+.005/1e-12 gate不变，pass另source确认/fail关确切CPU100多变量迁移不救援。all8历史开发仍exploratory非独立test/旧八折。accuracy与originality分开，目标未达、既有小时接续继续，不新建/暂停。以下保留历史。

# 2026-10-31 — 完整作者LITEMV通用训练/状态/ID组件及守卫实测通过；OS-CNN既有完整审计继续

本轮先读取5d564bcc849a0931b681458dbf25c7bad11e2bdc最新四记录及云端runtime/network，实际同一持久环境可执行。OS-CNNcontroller53415/audit53524/53525仍计算，无完成TERMs/COMPLETE_AUDIT；58科学冻结SHA未变。没有重复download/train/audit/controller、未读取producer最终分数，未作accuracy/gate结论或声称complete私有模型已备份。用户无需Windows/手工搬数据。新工作是可运行LITEMV工程组件，不是再次抽象否定文献。

/workspace/uwb-litemv-engine-20261031/source_engine.py复用已固定MSD-IRIMAS/LITE e62f64393104be8eda415b4d974987028ca68754原5architecturemethods/显式nativeKeras绑定，GPL-3.0-only，不缩小、不TFstub、非原TF执行复现。shape/epochs/batch/seed/identity均需调用者明确传入，无真实默认配置。完整90×567/12类合成fixture B4/末2/seed19/3epoch6updates与一次原生Keras fit全部batch/epoch指标、model/Adam/BN/metrics/callbackstate/所选checkpoint逐项相同；observer实际合成ID/顺序与无observer的所有状态RNG一致；epoch1完整保存恢复后2轮含modulemode/callbacks/allRNG逐项相同。错误identity/settings在状态变更前拒绝，checkpoint声称steps须等于Adamiteration。固定12taxonomy、prediction接口不读label、原znormalisation函数独立NumPy含0variance核对通过。未断言nativefitDataAdapter全局RNG等价，未实现自动wholejob/midepochresume。

另8种重叠ID/非法order/类别/shape/nonfinite输入在任何更新前拒绝，全部model/Adam/BN/metrics/callback/mode/RNG不变；8项readiness测试通过。真实readiness于2026-10-31T04:54:49.919989+00:00按预期拒绝“前OS-CNN完整audit未完”，**未打开diagnostic SUMMARY，未选择下一实际配置/正式CPUcost/真实UWB值、标签或fit**。未来守卫还要求前29400step/100epoch/sourcefinal_last完整audit/两个成功TERM/standalone私有complete包认证回读及archive/member/auditSHA一致后才能读原gate；若原gatepass须另source确认，不能换候选。source/runtime manifests及1302既有isolated依赖字节重核，不全局安装。report/checks/logs均在research/litemv_engine_20261031/；不是完整UWB runner/auditor或实验预登记。

补查固定filters原始先例：Ismail-Fawaz等IEEE BigData2022，DOI10.1109/BIGDATA55660.2022.10020496，作者hosted10页PDF读abstract/introduction/III.A–D/IV.A及IV.B开头，非完整results/figure阅读、PDF未本地物化SHA。CF-4-TSC作者库仅识别，未fetch/实现。它支持固定trend/peak filters出处，非UWB提升/原创证据，也不是另选H-Inception实验；详见PRIOR_ART_ADDENDUM.md/PRIOR_ART_ACCESS.json。已发表方法仍可作为有价值准确率基线继续真实验证。

下一优先查现有OS-CNN进程/完整audit/TERMs，完成后核全artifactSHA与owner-private/max3/rawfetchbase64false/即物化SHA/全成员重组的完整模型包，再发布冻结gate。fail仅关确切CPU100/B16/seed20261029、不诊断救援，pass只另source确认；关闭验证后才前瞻固定LITEMV成本规则/输入适配/一配置seed及完整runner/auditor、远端预登记读回后actualfit。作者构造2轮、论文1500轮与合成3轮不同，不能混作同配置负结果。用户7–8label不用，all8历史开发使后续仍exploratory。准确率与originality目标仍未完成，既有小时接续继续，不新建/暂停。以下保留历史。

# 2026-10-31 — OS-CNN100真实训练正常完成、既有完整审计在跑；完整作者LITEMV实现与合成重放通过

现有OS-CNN训练已exit0/no timeout，100轮29400更新470000训练出现；supervisor wall4903.09715075104秒/childCPU4902.700678秒。两套最终预测冻结后producer读取held1标签，7–8未用。controller53415既有自动启动audit53524/53525，观察2026-10-31T04:42:55.022726+00:00，58冻结项全部SHA不变。**没有重复下载/train/audit/controller；完整审计尚未完成，未读/发布producer最终分数，未作gate或提升结论，complete私有模型包未完成。** first-update4私有块保险仍有效但不是最终模型集。记录results/cloud_20261031_oscnn_pilot/{AUDIT_RUNNING_OBSERVATION.json,TRAINING_COMPLETE_AUDIT_RUNNING.md}。

等待时完成可运行的不同候选LITEMV准备：MSD-IRIMAS/LITE e62f64393104be8eda415b4d974987028ca68754，modelblobef4aafa16864dfc7f95c7b312459e68573a86961，10原UTF8源文件Gitblob/SHA核验保留GPL-3.0-only。原789及最新4记录精确LITEMV等词0命中，非完整语义等价或原创证明，LightTS不同。DOI10.1007/s41060-024-00708-5出版方abstract/3.2/3.3.1–3.3.3/4.1–4.2及预印本开头已读，非26页/figure全文。已发表先例可测accuracy，不能因为非原创停止循环。

完整author5architecturemethods/21个tf.keras显式绑定现有nativeKeras3.11.3TorchCPU，无TFstub/手写缩小替代；90×567/12类实际621680trainable/889624含fixedBN/18trainabletensors/38Adamvars。17组handcrafted depthwisefilters逐项解析相同、nontrainable且3次更新不变；model/Adam/BN/metrics/allRNG恢复后2步逐项一致及native.keras模型/优化器重载；独立NumPy dilationseparable/evenSAME通过。完整nativefit对显式批循环3epoch/6updates(合成B4末2)权重/Adam/BN/指标/每batchepoch日志/所选checkpoint一致；保存恢复后2轮含callbacks/allRNG一致；251常loss plateau独立规则、stricttrainlossmin/tiesfirst/不由val改选、固定12类缺val类均通过。firstfixture因CallbackList复制logs后ReduceLR写learning_rate导致原字典少key失败；只修记录postcallbackRecorder，不删比较key；attempt01源/trace和02pass都留存，非科学失败。现有1302runtime文件及10wheelSHA复验，无全局安装/覆盖；不是完整环境/jobresume。

作者当前构造默认2轮与论文1500不同；不把2轮当fullpaper负结果。多变量fixedconcat9639features/BN9735，不能引用univariate9814参数或假设便宜。原loader逐样本逐channel z-normalization；未来若RMS适配须前瞻说明。原fit独立valOHE/原main测试派生类数不复制，固定12tasktaxonomy；原ensemble5另属LITEMVTime，未实现。完整code/logs/readscope/repair/manifests见research/litemv_source_probe_20261031/REVIEW_AND_IMPLEMENTATION.md。**仅合成B2/B4/seed7工程fixtures，未选真实LITEMV配置/成本策略/seed/epoch，未跑正式cost或真实UWBfit，无LITEMV后台job/八折/准确率结果。** 原TF执行/RNG未声称复现，nativefitDataAdapter全局RNG与显式loop未声称相同；后者自己恢复逐位已证。

下一先查现有53524/53525/53415、TERMs/COMPLETE_AUDIT/POSTPROCESS，勿重复启动。成功后核全replay/artifactSHA，owner-private/max3/rawfetch(download_raw_file=true,include_base64=false)/即物化SHA/全成员重组完成模型包，再发布原gate；工程故障先留trace修复，不作科学负结果。OS source_final_last100/B16/seed20261029不改；fail仅关确切CPU100，不能据held1/7–8调参，pass另注册source确认。**既有OS全审计/私有收尾gate之前，不选择/实测下一真实LITEMV配置。** goal仍未达，accuracy与originality分开、all8历史开发仍exploratory非独立test/旧八折。既有小时自动接续继续；不新建/暂停，用户无需本地操作。以下保留历史。

## 2026-10-08T14:31+08:00 预检键名修正记录

两次执行器入口分别因特征路径和 manifest 键名拼写不匹配终止；调用栈均停在读取标签/样本/特征之前，未发生真实数据拟合，也未看诊断分数。已按 JSON 实际字段逐项检查全部12个入口字段，更正并重新固定代码 SHA。此前成本记录绑定旧协议 SHA，将不复用。当前协议 SHA256 `821a16048eb018fe1e8aa9d4d1db4c2f31954127dfb3f1ac707613642c2b0010`。准备在该版本上重新计时。

## 2026-10-08T14:30+08:00 预注册入口校验修正（真实数据读取前）

首次启动在协议路径键名校验处立即终止，尚未读取标签、特征、诊断预测或执行任何拟合。已更正 runner 的特征路径键映射并将 runner/测试 SHA 写入预注册协议；合成梯度及 sklearn 对照再次通过。成本预检文件绑定旧协议 SHA，故将重跑成本筛查后才允许拟合。协议 SHA256 `3907ce4b27209e405ef3e17b47f0994e027cab55cb2e0ee960774f9ab23884f6`。

## 2026-10-08T14:27:37+08:00 冻结读出损失函数八折基线预注册

复查原项目已包含多项逻辑回归，但未找到与当前 full8 同一冻结 hidden/checkpoint、用户折及读出选择规则下，平衡岭与多项逻辑回归的完整 2×2 对照。此为已知读出基线补齐，不是新方法或原创贡献。固定 128 维 epoch 选定冻结特征、八用户 leave-one-user-out、当前质量排除、每折六源训练/下一源验证/一源诊断。比较平方损失与多项交叉熵，分别使用均衡用户类权重与均匀记录权重。三档 lambda 仅以验证用户 macro、概率 one-hot MSE、较大 lambda 选择；全部选择/预测落盘后才评分诊断用户。成本筛查通过后才启动八折。无论结果如何，不调格点或扩模型；本结果不得称新贡献。协议 research/READOUT_LOSS_FACTORIAL_PROTOCOL.json，SHA256 `7d0b44284191073b5157d16cf6c127d418b1ce7d725de61affd95f6a71418569`。合成梯度和独立 sklearn 多项逻辑回归对照通过；尚未读真实数据。

## 2026-10-07T13:02:33.989885+00:00 时间散射六源筛选未达门槛

私有备份已逐份下载复验：3块、19393766字节、77成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

ridge 93.063131%、first 93.434343%、second 93.398990%、shuffled 93.544192%、standalone 85.439394%。二阶比岭+.335859pp，但比一阶−.035354pp、打乱−.145202pp，联合门槛失败；固定配置封存，不扩八折、不调参或提升控制。独立7100片段/28400臂输出、72含截距岭拟合、41产物SHA通过，全部argmax相同。补核SSP2023同数据WST直接先例，全文协议未核实，不称复现或创新、不比较论文95%。报告results/cloud_20261007_temporal_scattering/RUN_REPORT.md；私有备份待验证。无新编码器训练或后台队列。目标未达，下一项尚未注册。以下历史。

## 2026-10-07T12:44:39.115946+00:00 已知时间小波散射基线筛选预注册

补核Al-Jumaily/Khushaba SSP2023八用户十二类UWB的WST先例，摘要95%但完整划分/参数未核，不直接比八折或称论文复现/创新。固定Kymatio0.3.0 J3/Q1/T8、9距带、三时间段；主臂二阶追加冻结128特征，对照原岭、一阶、仅二阶时间位置打乱，另报独立散射基线。对岭+.5pp、两控制+.25pp，各4/6胜、最差降≤1pp。失败不扩八折/调参/提升控制。独立时域卷积/抽取合成检查通过，先推协议再计时和真实拟合；无训练或本项结果。协议research/TEMPORAL_SCATTERING_PROTOCOL.json。以下历史。

## 2026-10-07T11:01:45.688341+00:00 时间自相似六源筛选完成并未达门槛

私有备份已逐份下载复验：2块、14243831字节、72成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

ridge93.063131%、ordered93.242424%、lag_only93.155303%、shuffled93.199495%。主臂比岭+.179293pp，但对lag_only仅+.087121pp、打乱仅+.042929pp，联合门槛失败。固定块数/归一化/seed/读出封存，不扩八折、不调参或提升控制。独立7100片段/21300描述子臂、54含截距岭拟合、41产物SHA通过，全部argmax相同。已知雷达SSM先例不称创新。报告results/cloud_20261007_temporal_ssm/RUN_REPORT.md；私有备份已完成，见同目录DRIVE_BACKUP_RECEIPT.json。无新编码器训练或后台队列。目标未达、接续保留；下一项尚未注册。以下历史。

## 2026-10-07T10:58:25.017957+00:00 时间自相似固定筛选预注册

已核查He等2014雷达SSM直接先例、Junejo2008摘要，不称创新或原论文复现。513维有序包络Gram追加冻结128特征；比较原岭、仅lag均值、逐样本时间块打乱。四源训练/下一源验证/一源诊断，固定3lambda；全部预测保存后统一评分。对岭+.5pp、两消融+.25pp，各4/6胜、最差降≤1pp。失败不扩八折/调块数归一化seed或提升控制。合成独立矩阵检查通过；先远端冻结再合成计时和真实拟合。无本项真实结果、无后台训练。协议research/TEMPORAL_SSM_PROTOCOL.json。以下历史。

## 2026-10-07T10:24:02.481796+00:00 训练源选择性子空间过滤分类筛选完成

私有备份已逐份下载复验：13块、77951207字节、100成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

ridge93.063131%、selective93.027778%、all_views92.536616%、random_selected92.957071%。联合门槛passed=False。固定过滤/秩/选择阈值/seed/读出配置封存，不扩八折、不调参/提升控制。独立重算127800完整前向、54岭拟合、90内外特征分解，63产物SHA通过，所有argmax一致。机制能量通过不等于分类有效；冻结表示限制需保留。无编码器训练或下一队列。报告results/cloud_20261007_source_eigencancel/RUN_REPORT.md，私有回执已完成，见同目录DRIVE_BACKUP_RECEIPT.json。目标未达、接续保留。原项目FIRST_PART_REPORT.md:118及proposal_v2/31调研方案:151已引用MV-GLFNet2026及划分差异，本轮只补核原文，不能包装为新发现或新路线。下一项尚未预注册，须先完整查重原项目/失败清单和直接文献，再选不同观测机理假设；不把6/2划分论文分数直接比八折。补充读文记录见research/SUBSPACE_PRIOR_ART_UPDATE_20261007.md。无新实验队列。以下历史。

## 分类过滤已实际启动；先检查进程和检查点

协议f786ffd，SHA80cbe22a67ac43ad5b576caa4f53c1bc2f234613983f83bf3d78cd89f5e71d99。CPU16样本0.4977s，保守估计2718s通过3600s门槛。PID14905正在冻结推理/读出拟合，无新编码器训练。结果/workspace/uwb-source-eigencancel-runs-20261007，日志/workspace/uwb-source-eigencancel-source-20261007.log；续接先查进程，不能重复启动。空子空间机制43成员私有备份两块已下载SHA验证，见results/cloud_20261007_empty_subspace/DRIVE_BACKUP_RECEIPT.json。尚无分类结论。以下历史。

## 2026-10-07T10:01:30.236194+00:00 空手势子空间机制通过；分类过滤预注册

私有备份已逐份下载复验：2块、10261113字节、43成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

机制诊断左/右6/6通过、顶部0/6；左空保留11.1948%、活动98.3816%，右11.7299%/79.0905%。18独立矩与特征分解、4260直接残差检查通过；不是精度提升。另注册SOURCE_EIGENCANCEL_PROTOCOL：每分类折仅四训练用户内部决定过滤雷达，禁止把全六源左/右结果直接用于各折选择。比较原岭/选择性过滤/全部过滤/同秩随机过滤；主臂对岭+.5pp、其余+.25pp，各4/6胜、最差降≤1pp。失败不扩八折、不调参/提升控制。编码器冻结，真实分类尚未运行；目标未达。机制产物本地保存、私有备份待办。以下历史。

## 2026-10-07T09:48:36.942360+00:00 训练侧空手势子空间可分离性诊断预注册

仅外层fold1训练用户3–8，内部五源空手势拟合/一源能量诊断；用户1/2不参与。固定90%能量选秩，容量≤32，空保留≤20%、活动≥50%、差≥30pp且比同秩随机差多≥20pp；至少一雷达4/6源用户通过才允许另行分类研究。该能量代理不是分类精度或目标真值，不称RPCA/STAP复现或创新。失败不扫秩/阈值/seed，不扩八折。协议research/EMPTY_SUBSPACE_PROTOCOL.json；尚无真实诊断或训练进程。以下历史。

## 2026-10-07T09:16:54.029257+00:00 原RF时间倒序标签增强六源筛选完成

私有备份已逐份下载复验：7块、37622351字节、87成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

ridge93.063131%、mapped_mix92.404040%、naive_mix92.132576%、unmapped_mix88.766414%。联合门槛passed=False。固定映射/资格/混合/读出配置封存，不扩八折、不调参或提升控制。独立重建23200个训练倒序样本出现、46400次函数式前向与54岭拟合，51产物SHA通过，全部argmax一致。无新编码器训练或后台队列。原始相位须重算，不应直接翻转缓存；已知Retro-Actions先例不称创新。报告results/cloud_20261007_temporal_reversal/RUN_REPORT.md；私有回执同目录。目标未达、接续保留。以下历史。

## 2026-10-07T09:05:22.987559+00:00 原RF时间倒序与标签转换筛选预注册

核查Retro-Actions2019先例，非原创。仅训练可逆完整类别对做增强，排除push/empty及用户内缺失配对；主臂重算原RF倒序后的相位和归一化，对照原岭、直接翻转缓存特征、错误保持标签。冻结编码器，验证用户选固定三个lambda。主臂对岭+.5pp、两控制+.25pp，各4/6胜、最差降≤1pp；失败不扩八折/调参/提升控制。独立相位推导合成检查通过，尚无本项真实结果；先推送协议再训练样本计时。协议research/TEMPORAL_REVERSAL_PROTOCOL.json。以下历史。

## 2026-10-07T08:11:28.765802+00:00 训练空手势频谱预白化筛选完成并失败

私有备份已逐份下载复验：7块、12916333字节、95成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

ridge93.063131%、raw_blocks93.092172%、empty_whiten92.940657%、shuffled92.891414%。固定PSD/谱底/FFT/分块/seed配置未过联合门槛，封存不扩八折或调参；不提升控制臂。完整复数FFT/周期图独立复算127800个描述子样本出现及54岭拟合，全部argmax一致；63产物SHA、模板训练ID、参数选择、评分门槛核验通过。空手势不等于纯噪声，已知预白化不称创新。报告results/cloud_20261007_empty_whiten/RUN_REPORT.md及同目录私有备份回执。无编码器训练/后台队列，目标未达，接续保留。以下历史。

## 2026-10-07T07:44:06.339642+00:00 固定Huber残差影响筛选完成并失败

私有备份已逐份下载复验：7块、6842683字节、80成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

预注册850e167d98c91502ba95be3d61259b6b74038cb8。六源Huber92.934343%、ridge93.063131%、one_step92.902778%、shuffled93.022727%；主臂对ridge−.128788pp/2胜，对其余控制也未过门槛。固定阈值/尺度/lambda/seed配置封存，不扩八折或调参。独立IRLS重拟合72个输出、线性方程复核144个控制输出；全部argmax相同，最大目标差8.51e−14、分数差2.74e−6，51产物SHA及ID/权重/门槛核验通过。残差裁剪有激活，但不能据此认定标签错或提出创新。真实运行16.30秒，无编码器训练/后台队列。报告results/cloud_20261007_huber_readout/RUN_REPORT.md，私有备份以同目录回执为准。目标未达，接续保留。以下为历史。

## 2026-10-07T06:05:43.850653+00:00 源用户删除读出筛选失败并封存

私有备份已逐份下载复验：7块、17970544字节、79成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

预注册f9ec25f54c0be7b221dc80ecab0c7894fa343aa9。六源主臂domain_min93.049242%，ridge93.063131%，domain_mean93.174242%，row_min93.049242%，row_mean93.061869%；主臂比ridge低.013889pp，仅1/6胜，全部比较门槛失败。不扩八折、不提升平均控制臂、不搜索权重/分位数/seed/lambda。48个只用训练标签的读出重拟合、96个独立预测系统及51产物SHA/样本ID/门槛核验通过；编码器未重训，删除用户仍曾参与编码器训练，不称未见用户置信区间或创新。官方pasting文档核查完成，原论文全文未读。报告results/cloud_20261007_delete_user/RUN_REPORT.md及同目录私有Drive回执。无后台训练，目标未达，接续保留。以下为历史记录。

## 2026-10-07T04:59:28.136708+00:00 类条件协方差筛选完成，固定QDA未过门槛
私有备份7包/7块/33,511,688字节/84成员已通过owner-only元数据、认证回下载及全部SHA校验；本项收尾，无运行队列。


六源QDA92.424242%、ridge93.063131%、pooled92.998737%、diagonal92.275253%、permuted71.165404%。主臂比ridge低.638889pp/仅2胜，最差−5pp；对pooled和diagonal也未过门槛。封存固定协方差/归一化/jitter/seed，不扩八折。72协方差重建、576高斯分数系统/SciPy、45产物SHA及ID/门槛核验通过。一次核对遗漏旧onehot分数−.5常数已修复，参数/预测不变，失败日志保留；无新编码器训练。经典判别分析不称创新，原论文全文未读，依据为已核查官方数学文档/源码。报告results/cloud_20261007_class_scatter/RUN_REPORT.md，备份以同目录回执为准。无后台队列，目标未达，接续保留。以下为历史。

## 2026-10-07T04:16:10.174593+00:00 雷达左右交换固定推理实验失败，完成结果收尾
私有备份3块/21,727,798字节/92成员，owner-only元数据、认证回下载与块/整包/成员SHA全通过；本项已收尾，无训练队列。


主equivariant92.352273%低于原岭93.063131%（−.710859pp，仅2/6胜，最差−1.416667pp）；invariant76.327020%、mapped_swap84.839646%。联合门槛失败，不扩八折、不扫权重/映射/seed；只封存固定推理投影，不否定全部等变训练。14,200次样本出现的原输入与真实交换输入函数式复核隐藏值/分数逐位一致，39产物SHA及ID/选模/门槛通过。10月4日预测已完成，10月7日恢复收尾，无重复训练；报告results/cloud_20261004_radar_exchange/RUN_REPORT.md，备份状态以同目录回执为准。已知群平均与标签映射不称原创；硬件不对称与模型缺乏此归纳偏置尚无法区分。八折强岭94.897727%仍保留。无后台训练或下一队列，目标未达，自动接续保留。以下为历史。

## 2026-10-04T21:57:38.701003+00:00 查询图六源实验完成，互近邻主方法未过门槛
私有备份1块/6,150,518字节/97成员元数据、认证回下载及SHA全验证；无训练队列。

固定mu1/k10：mutual94.388889%、ridge93.063131%、union94.962121%、permuted79.810606%。mutual对ridge六折均提高(+1.325758pp)，但比简单union低0.573232pp且仅1/6胜，联合门槛失败。关闭该固定图k/mu/seed/归一化搜索，不改主方法、不扩八折。批次无标签邻近关系有用的初步证据不等于互惠创新成立；它需整批查询，不是单条实时识别，也不是新外部测试。PMLR LaplacianShot正文1–9页/作者代码已核查，凸图平滑适配非精确复现或原创。36稠密系统/365838非零图条目、全部预测/ID/残差/概率/SHA独立核验通过。报告results/cloud_20261004_query_graph/RUN_REPORT.md，私有回执同目录。无训练或下一队列；目标未达、自动接续保留，旧789源码未变。以下为历史。

## 2026-10-04T21:34:42.352622+00:00 固定语义输出编码完成六源筛选及八折控制扩展
本轮新增可复用强对照：相同八折冻结检查点的onehot岭读出94.897727%，较原CNN94.184280%高0.713447pp；语义码95.085795%只再增加0.188068pp（五胜三降）。不能把相对原CNN的全部增益归给语义。后续同协议新候选须保留该强读出对照，不能仅与原CNN比较；不继续当前码权/维数/seed/lambda搜索。
私有备份10包/19块/79,338,880字节/185成员认证回下载及SHA全通过，无训练进程。


六源semantic93.628788%对onehot93.063131%，全部门槛通过；保持码书/参数不变后完成八折。八折semantic95.085795%，onehot94.897727%，permuted94.794318%，旧CNN94.184280%；八折门槛失败，封存本固定语义码/权重/seed/lambda网格，不将源开发正结果当成最终成功。168次独立直接码目标拟合、全八折74400隐藏向量函数式复核、预测/选择/权重/SHA已通过。它只改变输出几何，不是新表示或已验证原创；八用户均历史使用过，不称独立测试。报告results/cloud_20261004_semantic_code_eightfold/RUN_REPORT.md，私有备份回执同目录。无新编码器训练或运行队列，789原源码未变，自动接续保留。以下为历史。

## 2026-10-04T20:41:09.958279+00:00 RSC 全部完成：主方法失败，最终私有备份验证通过

12新轨迹288轮及24新旧轨迹独立功能审计通过，714产物SHA、ERM/tail全部验证/诊断logits逐位一致。主tail：RSC91.992424%，随机93.218434%，旧普通92.722222%，SAM93.113636%，CNN岭93.063131%。RSC对普通−0.729798pp、随机−1.226010pp、SAM−1.121212pp，均0/6严格胜；CNN−1.070707pp、1/6胜。全部固定联合门槛未过。停止这组比例/梯度/批选择/seed/window网格及八折扩展，不把随机或次选模换主结果。此结论不否定全部RSC变体。

27逻辑包/64私有owner-only块/390810250压缩字节/468成员记录认证回下载、块/包/全部成员SHA通过，包含372个不重复训练产物。总索引results/cloud_20261004_source_rsc/DRIVE_BACKUP_RECEIPT.json组合PARTIAL01(held3/4)、PARTIAL02(held5/6/7)、FINAL_BATCH(held8与最终来源)。ANALYSIS、SUMMARY、AUDIT、PER_USER、SOURCE_HISTORY、FINAL_MONITOR及RUN_REPORT同目录。源数据/prepared/代码/协议/标签/用户SHA核验未变；原789源码不动，不重复迁移。

作者博士论文4.4.2方法/理论条件、4.4.3消融与4.4.4开头已读，PDF与SHA留私有最终包；不是Springer全文或整本论文。既有RSC/随机屏蔽先例、理论适用范围、当前DomainBed向量适配差异见research/SOURCE_RSC_REVIEW.md及THESIS_REVIEW_RECEIPT。源六用户反复研发，不可视作独立确认或与旧八折直接比较。

原训练4719已完成(Z)，没有下一训练启动；所有本轮工作已收尾，但有效原创目标未达，保留自动接续。下次先读最新记录与进程，寻找有新雷达观测/任务结构依据且排除既有失败路线的机制，先核查文献和源可行性；不要重开RSC/SAM/Fishr/DRO或描述符失败网格。以下保留历史记录。

## 2026-10-04T20:28:33.343669+00:00 RSC 累计十条完整轨迹已审计和私有备份，最后一条在训

原PID4719存活(R)，累计完成11/12条，目前random24/held8第17轮；未重启。新增held5/6/7两臂六条共144轮功能审计通过，ERM/tail全部验证与诊断logits逐位一致，180产物SHA+6resultSHA、source顺序/角色/选模及全部掩码计数/seed核验，段均值重建tail满足原固定浮点容差。结合PARTIAL01，十条完整轨迹共240轮已有审计。

新增214成员、13逻辑包、31分块、189961222压缩字节已私有owner-only认证回下载及块/整包/全部成员SHA核验。回执results/cloud_20261004_source_rsc/PARTIAL02_DRIVE_RECEIPT.json、PARTIAL02_AUDIT.json。一次附件transport closed，精确名称查询无结果后仅重传该块并认证校验成功；不影响训练、未重复启动。PARTIAL01的前四条152成员/21块仍有效，最终两条尚待全体结论及私有备份。

补读作者HaohanWang博士论文RSC章节4.4.2–4.4.3及4.4.4开头（印刷59–69页和70开头），正式会议DOI10.1007/978-3-030-58536-5_8由作者发表记录核实。论文文本说明固定特征/有限假设类/二分类理论条件及屏蔽增loss/小步长假设，不能直接保证当前UWB有效；batch选法、空间/通道和优化器配置与DomainBed适配有差异。已记录research/SOURCE_RSC_THESIS_REVIEW_RECEIPT.json，不声称读完112页学位论文或Springer全文，不改变冻结科学配置。PDF留云端待加入最终私有包。

下一步继续原队列，不根据局部结果调参；全部12新轨迹及24对照功能审计完成后按固定联合门槛判定。当前无整体效果/原创结论，目标未达，自动接续保留。先核验实际进程和检查点，不能把历史后台true当新观测。以下保留历史。

## 2026-10-04T19:12:06.400251+00:00 RSC 前四条完整轨迹审计与私有备份通过，原队列继续

PID4719仍是真实存活进程(R)，没有重新启动。原队列目前完成4/12条，当前rsc24/held5第17轮(step1275)。本轮冻结并核验rsc24、random24各held3/4四条完整轨迹，共96训练轮；其ERM/tail全部验证和诊断logits独立函数式推理逐位一致，120产物SHA加四个resultSHA、全部source顺序/角色/选模/每批掩码计数与种子记录通过；段均值独立重建tail最大绝对差6.3876311e-06，在预先代码固定atol2e-6/rtol2e-5范围内，不能把浮点段重建说成逐位一致。训练代码和科学协议SHA未变。

四条完整状态96段权重、优化器/cosine/RNG/best/tail、样本ID、全部预测、源码/配置/审计及日志截取共152成员、9逻辑包、21块、126332490压缩字节，全部私有owner-only上传、认证raw回下载、块/整包/成员SHA通过。索引results/cloud_20261004_source_rsc/PARTIAL01_DRIVE_RECEIPT.json，审计PARTIAL01_AUDIT.json，进程观测PARTIAL01_MONITOR.json。这不是12条最终备份；第2轮启动包仍是历史证据，恢复优先采用对应轨迹最新完整包。全量原始数据/prepared/旧基线仍引用既有私有包，不重复搬运。

继续既有/workspace/uwb-rsc-source-runs-20261004队列，日志/workspace/uwb-rsc-source-train-20261004.log，LAUNCH.json原命令。后续先查活进程与检查点，禁止重复启动。待12新轨迹/24轨迹功能审计与SUMMARY全完成再按原联合门槛判定；不根据局部诊断改屏蔽比例、选模、seed或门槛。没有报告整体识别提升或原创结论，尚不启动八折。目标未达，自动接续保留；不依赖用户Windows，789原源码不变。以下为历史。

## 2026-10-04T18:26:15.949170+00:00 RSC 两臂真实云端训练已启动，首个恢复备份验证完成

冻结提交21915bdf0203984f5ded80fbf40dcca08e18609a在真实成本/候选训练前推送，13文件远端逐字回读相同。真实普通CNN首epoch损失/验证CE/样本顺序/段平均权重与旧基线逐位相同。源批成本含75%余量3.110022h，通过预定10h门槛。云端训练PID4719实际存活，当前rsc24/held3第7轮(step518)，完成0/12新轨迹；不是完整候选结果。运行根/workspace/uwb-rsc-source-runs-20261004，日志/workspace/uwb-rsc-source-train-20261004.log；先查活进程及检查点，禁止重复启动。原始科学配置/门槛未改。

第2轮/step148完整快照7,261,158字节28文件，私有owner-only Drive元数据、认证raw回下载、块/整包/全部成员SHA均通过，见results/cloud_20261004_source_rsc/START_DRIVE_RECEIPT.json。START_AUDIT独立函数式重算该快照全部验证logits/CE逐位一致；保存完整AdamW/cosine/RNG/best/tail/精确source顺序及两段权重，未评价候选诊断输入。这里只是启动备份，12条全部轨迹及最终结果尚未备份。恢复步骤RESTORE.md，精确命令LAUNCH.json。

固定RSC24与random24各六源开发折，共288新epoch；主tail13–24，相对普通tail≥+0.5pp、random≥+0.25pp且不低于既有SAMtail/强CNN岭读出，各≥4/6胜且单人不降超1pp。队列完成自动审计12新+6旧普通+6旧SAM的ERM/tail全部验证/诊断logits、角色/选择/顺序/SHA；新臂另核所有batch屏蔽记录及逐epoch段均值重建tail。强CNN读出只用前次独立重拟合证据和冻结SHA，不冒称本轮重拟合。完成后还须私有最终全部状态/日志/代码备份及认证校验；没有配置上传后台watcher。若进程中断，用同命令从完整epoch恢复，lock防重入；环境销毁不保证后台持续。

RSC是既有方法，本改编不是精确论文复现，作者课程/空间选择与DomainBed固定向量规则差异已记录；未读论文全文，不能称原创。即使源门槛通过，也仍需八折强对照/消融/独立种子及原创研究贡献证据。当前有效原创目标未达到，保留自动接续。数据/789原源码快照不变，无需用户Windows。以下为历史记录。

## 2026-10-04T18:21:13.322263+00:00 源端 RSC 固定对照已实现，先预注册再真实计时

下一机制是训练时表示自挑战，已知ECCV2020 RSC，非原创。核查作者README/源码与固定提交DomainBed RSC规则；未取得论文全文。辅助正确类logit梯度屏蔽最高1/3特征，按概率下降选择1/3批样本；辅助无dropout，最终CE保留旧dropout，三路共享CNN1728维。random24在自身状态保持选择规则和逐行屏蔽数量，只随机坐标；不同轨迹选择不保证相同。严格分位数相等项规则、与作者课程/空间配置差异详见research/SOURCE_RSC_REVIEW.md。

六源3–8四训/一验证/一诊断，12新轨迹288轮。固定24轮/seed2026/batch64/CPU1线程/AdamW .001和decay .001，主tail13–24，次最佳验证CE不能替换。相对普通tail至少+0.5pp、random+0.25pp、不低于旧SAMtail和强CNN读出；每项≥4/6胜、任一用户不降超1pp。失败关闭本组掩码/比例/seed/window网格，不据此开八折。已知方法即使通过也不直接满足原创目标。

合成NumPy分位数差分、随机控制逐行数量、辅助RNG隔离、普通AdamW和完整runner两臂断点恢复(模型/AdamW/cosine/tail/best/RNG/预测)逐位一致。当前没有真实RSC计时或训练进程、无候选识别结果。下一步正常推送并逐字回读预注册，再真实源批计时和普通基线第一轮精确复现；10h含75%余量成本门槛通过才启动。运行计划/workspace/uwb-rsc-source-runs-20261004，日志/workspace/uwb-rsc-source-train-20261004.log。先检查进程/检查点防重复。原789源码与数据不动，目标未达、自动接续保留。

## 2026-10-04T17:28:23.104803+00:00 局部斜向结构可行性通过，但方向直方图识别门槛失败

先预注册dc3def9defa2d757239cf02589e70976bfe0f4aa并远端回读，再对固定568源用户片段诊断：原RF解析包络、3距离bin均值、最大值归一化开方、2D Sobel及sigma2结构张量，固定126采样格点；纹理trace上四分位、coherence≥.6、obliqueness≥.3。原始时序Left/Top/Right可用比例28.671480%/30.684777%/32.253196%，时间打乱4.214844%/3.369426%/3.504253%；各6/6用户满足≥20%原始、≥10pp增益，图像可行性18/18通过。全部568原片段/429408张量位置独立FFT/二维相关/特征分解重算，最大差6.939e-17。该结果不是分类正确率或真实手部速度。

随后独立预注册ead63d58813c328950378ccc084f89772c79f704并远端回读，才执行六源开发折7100条识别：已冻结源第24轮CNN128维+三视角2时间×3距离cell×9方向bin162维，四臂均290维；4用户训练/下一用户验证/1诊断，只训练拟合标准化和均衡岭，固定3lambda验证选择。主coherent筛选，all不筛选，sign_null逐点随机方向符号，cell_null随机cell位置；24所选模型/72拟合，未重训CNN。

主93.289141%，all93.364899%，sign_null93.049242%，cell_null93.339646%，强CNN读出93.063131%，旧tail92.722222%。主对all−0.075758pp(2/6胜)，sign_null+0.239899pp(4/6，仍不足+0.25)，cell_null−0.050505pp(3/6)，强CNN+0.226010pp(3/6，不足+0.5/4胜)，tail+0.566919pp(4/6)。联合门槛失败：不能只报超过tail，不能改换all为主臂包装成功。停止这组bin/cell/scale/筛选阈值/seed/正则网格，不据此开八折。可用斜向纹理存在，并未证明此次筛选或位置机制有效。

源训练成本估计含2倍余量80.523秒<1800秒，实测40.503秒。识别独立审计99.940秒：全部7100原RF片段重算2683800张量位置、4600800描述值，最大描述差3.497e-15；72sklearn重拟合/24所选模型/验证选择/诊断预测/源角色/ID/比较器来源/34产物SHA/指标门槛全部通过，预测最大差1.168e-13。旧CNN隐藏向量本轮只SHA及来源核验，未重新前向；其既有检查点审计仍保留。原数据/prepared/标签/用户及冻结核心代码/协议运行后SHA不变。

102成员133841531字节、19个私有owner-only块认证回下载及块/包/成员SHA通过，含诊断tensor/perm/ID、全7100tensor/描述、24读出系数、全部候选验证与最终预测、源码/配置/日志/审计、六个旧CNN隐藏向量和比较器来源。原RF/prepared和完整CNN权重引用既有私有备份，不入Git。报告/回执results/cloud_20261004_orientation/；真实云端/workspace/uwb-orientation-{screen,readout}-20261004。

文献核查为scikit-image HOG解释源码与v0.25.2 structure_tensor定义/实现，引用Dalal–Triggs2005与Lowe2004；未取得/通读原论文。HOG/结构张量/coherence筛选为明确先例，有限仓库检索不证明原创。图像坐标斜率不是已校准速度/3D轨迹，多径/幅度变化同样能形成结构；时间打乱也改变频谱及纹理覆盖，非纯物理因果对照。源六用户多次研发过，非独立外测。

当前无训练/诊断进程、无下一实验启动；有效原创目标仍未达，自动接续保留。下次先读最新远端与现存进程，需不同机制和先例证据，不重开本轮或此前已失败网格。789原源码快照保持不变，不依赖用户Windows或再次搬运数据。以下保留历史记录。

## 2026-10-04T15:20:44.233381+00:00 距离带能量有向面积源内对照完成：联合门槛失败

预注册743923c39fc3db8bb117b96046e5ae71b58a5e85先正常推送回读，再成本220.805秒（2倍余量）通过1800秒预算。源用户3–8共7100质量片段，六轮四训练/一验证/一诊断；旧第24轮CNN不重训，五臂90次岭拟合、30所选读出。主signed93.079545%，普通CNN读出93.063131%，base92.771465%，unsigned92.983586%，shuffled92.968434%，旧tail92.722222%。signed对base+0.308081pp（5/6胜），对unsigned+0.095960pp（2/6），对shuffled+0.111111pp（3/6），未达+0.5/+0.25/+0.25pp联合门槛；对普通读出仅+0.016414pp。停止这组带数/级数/归一化/shuffle/正则/seed网格及据此八折扩展，不能移除强控制或换主臂包装成功。

实际提取/拟合/保存75.220秒，独立审计4.550秒：全7100原RF距离带与独立Chen递推，最大差3.192e-15；30所选读出独立sklearn重拟合、36固定原数据CNN函数式抽查、40产物SHA/角色/选择/指标/门槛通过。90候选验证预测运行时独立sklearn核对。原数据/prepared/标签/用户和核心代码/协议运行后SHA不变。全部描述/隐藏向量/ID/候选及最终预测/30读出系数/六个旧CNN检查点和tail来源/代码/日志/作者数学PDF共85成员，62014153压缩字节9个私有owner-only块认证回下载及块/包/成员SHA通过。回执results/cloud_20261004_energy_area/DRIVE_BACKUP_RECEIPT.json，报告同目录RUN_REPORT.md。

文献确证level2面积/路径签名为既有数学，原项目也已有跨雷达轨迹面积；本轮只试单雷达内部距离带能量时序，不宣称物理轨迹或整体速度不变。读到作者Reizenstein2015/2017数学说明PDF第1–3、5–7页、库文档和动作签名实现README；不是TOMS或动作论文全文。详见research/ENERGY_AREA_REVIEW.md及receipt。

当前无训练/诊断进程，无下一实验启动；目标仍未达，自动接续保留。下一步须新的雷达观测机制和先例排重；不重复当前探针或已失败局部匹配/Fishr/SAM/DRO网格。普通冻结CNN岭读出的工程改善不等于原创贡献。原789源码及原始大数组不动，无需用户本地迁移。以下保留历史状态。

## 2026-10-04T14:50:15.931014+00:00 局部回波对应源诊断完成：门槛未过，无识别训练

预注册59a999ee7eeaf65a46ef11642ddf20ff42e561e3先推送回读后执行。只处理外层fold1训练用户3–8的568条分层源片段；每条件332280对局部patch。三路相邻可靠比例均值11.221%/13.240%/11.811%，远帧4.834%/7.811%/6.750%，差6.387/5.429/5.062pp；非零可靠位移8.277%/10.754%/9.493%。三路均0/6满足相邻≥20%、比远帧≥10pp、非零≥5%的规则（须2路各4/6），故停止此窗口/位移/置信阈值网格与据此启动识别分支。不是分类准确率，也不否定全部光流方法。高相关常伴随多解，不能把匹配位移当真实手部轨迹。

提取与压缩共13.281秒，成本预算600秒；独立审计4.263秒，29536个原信号FFT/标量相关核对最大误差1.984e-13，全代价体决策/18用户视角统计/源角色与SHA通过。初次审计因NPZ循环重复解压停止，修为一次缓存后完成；只改审计I/O，真实提取/科学配置未重跑或改动。完整代价体、源ID、位移/掩码、代码/协议/日志/审计及作者幻灯片19成员，143247631字节、20个≤7MiB私有owner-only块认证回下载和块/包/成员SHA全部通过。回执results/cloud_20261004_local_echo/DRIVE_BACKUP_RECEIPT.json；原数据及789源码快照不变。

文献核查是PWC-Net作者README、UnFlow作者README/指定losses.py段及作者AAAI2018报告34页文本/公式/表格，不冒称论文主文全文；详见research/LOCAL_ECHO_REVIEW.md。相关代价体和前后向一致性有既有先例，非原创。原始全局质心/展宽试验与此次局部匹配不同，但当前可行性仍不足。

当前无训练或诊断进程；用户目标未达，自动接续保留。下次先检查最新状态，不重复此源诊断或重开Fishr/SAM/DRO/适配等失败参数网格；需要新的雷达观测依据、先例排重和前瞻协议才启动下一实验。云端数据仍在，不要要求用户本地迁移。以下保留历史状态。

## 2026-10-04T13:12:57.455143+00:00 源梯度方差固定对照全部完成；联合门槛未过

18新轨迹432轮+6旧基线共24轨迹独立审计通过，714产物SHA、所有ERM/tail验证与诊断logits逐位一致。六源主tail：旧普通92.722222%、匹配ERM92.702020%、真实方差92.869949%、类内域置换92.771465%。联合门槛未过；停止这组 lambda/EMA/warmup/seed/tail/normalizer/coordinate 网格及八折扩展。不能用次选模或移除对照改变结论，也不能把源方差机制4/6通过当作分类增益。该结果不证明所有 Fishr 设置或域泛化方法无效。

全部38逻辑包/92私有owner-only块/565203039压缩字节/595成员认证回下载及块/包/成员SHA通过，含全部432段与18完整优化器/RNG/EMA状态；索引results/cloud_20261004_source_fishr/DRIVE_BACKUP_RECEIPT.json。最终SUMMARY、AUDIT、ANALYSIS、PER_USER、SOURCE_HISTORY和FINAL_MONITOR同目录。旧恢复证据另保留在PARTIAL01–03；原数据/prepared/789源码快照不变，不重复迁移。训练598和归档607已经完成；没有下一训练进程。自动接续仍开启，目标未达。

正式Fishr主文1–9页、DomainBed附录19–24页已通过出版方GitHub PDF读到，SHA与blob记录在research/FISHR_PUBLISHED_REVIEW_RECEIPT.json。主表测试域选模不适用用户约束；作者固定EMA除数有明确目的，本版本不同且不是精确复现。未事后改变冻结实现或门槛。Close this fixed gradient-variance intervention. Next investigate a distinct radar-signal representation mechanism against original experiments and published prior art, then pre-register a train-only diagnostic and matched controls if justified. Do not reopen failed Fishr, SAM, temporal-scale, DRO or adaptation grids. No next training is currently launched.

以下保留历史执行记录；历史“运行中/主文未读”不代表当前状态。

# 云端迁移、原数据审计与首次先导

日期：2026-09-30。迁移完成；研究目标尚未完成。没有需要用户补交的数据。

## 恢复证据

- 私有 Drive 的163个分块均通过认证接口实际下载到 `/workspace/uwb-transfer/chunks/`。下载最大并发3，压缩字节总数2,006,404,772，失败0，名称、大小和SHA-256逐项吻合。
- `transfer.py verify` 成功，`restore` 写入新空目录 `/workspace/uwb-restored-20260930`。86个原文件共2,141,981,100字节，全部原文件SHA-256通过；审计脚本又独立重算一遍。
- `TRANSFER_MANIFEST.json` SHA-256：`10516baf0b49a84e6a39169ef447fffcdcb29defa2a6cc79e301dc938ddceb4d`。
- `SOURCE_MANIFEST.json` SHA-256：`8f19d50aacdcc2cfe3457e9dc4785a0cf440ef76c5777a751806b5575a9f2884`。GitHub原快照共789文件；其中28个当前所需核心文件已从不可变提交`c735e9454ba3d5168d931eb0158f2bb0fcf12808`物化，并逐个验证原字节，位于 `/workspace/uwb-source-c735e94/project/`。
- 没有更改Drive分享权限。原始数组和权重未提交到GitHub；研究分支所在仓库已确认仍为private。

机器记录：[恢复凭据](TRANSFER_RECEIPT.json)、[RESTORE_VERIFIED](RESTORE_VERIFIED.json)、[完整审计](AUDIT.json)。私有持久下载索引与原迁移清单位于 `../../migration/20260930/`，不包含短期认证链接。

## 数据与旧预测重算

X为float32，形状`[9593,3,90,189]`，全量有限值检查通过。原标签/用户数组与prepared标签/用户数组逐项一致。原质量掩码排除HV2类别10/11的200条、HV6类别3的93条，剩9300条。检测到100对完整三视角记录的精确重复，均不属于保留样本之间的重复。

八折均核验：测试用户H、验证用户H%8+1、其余六用户训练；保存的split、预测中的ID/标签/用户/角色相符；缓存文件SHA与旧结果的SHA一致；验证logits逐元素重放一致。以下是每用户现存类别macro accuracy，再对八人平均，非重新训练结果。

| 旧缓存预测 | 八用户平均 | 最差用户 |
|---|---:|---:|
| 原始Right模型 | 71.2913% | 51.3636% |
| 子集训练模型R | 71.6765% | 51.5455% |
| 子集训练模型TR | 90.5833% | 67.0000% |
| 子集训练模型LTR | 93.4417% | 75.0000% |

LTR各用户依次：96.25%、98.70%、91.25%、96.75%、98.1667%、75.00%、96.4167%、95.00%。历史八折总值93.4417%已重算吻合。

## 首次CPU先导：类内跨用户差异约束

在看到本次新方法测试结果前保存[协议](../../research/FROZEN_PROBE_PROTOCOL.json)。输入为各折原模型已保存的真实三路256维特征，训练统计量只读取六个训练用户。普通读出按类和用户平衡权重作岭回归；候选额外惩罚同一类别不同源用户特征中心在分类输出上的差异。超参数仅由该折独立验证用户选择。此为线性机制探针，不能称为新颖算法或端到端网络训练。

| 方法 | 八用户平均 | 最差用户 |
|---|---:|---:|
| 原LTR融合头 | 93.4417% | 75.0000% |
| 七子集概率等权平均 | 92.8788% | 73.3636% |
| 普通平衡岭回归融合 | 92.8254% | 75.6364% |
| 类内跨用户差异惩罚 | 92.7231% | 75.8182% |

候选相对普通岭回归为−0.1023个百分点，仅2/8用户改善；相对原LTR为−0.7186个百分点。未达运行前规定的继续门槛，停止扩大这一冻结特征约束方案。完整配置、每折验证选择、混淆矩阵和配对差在[FROZEN_PROBE_RESULTS.json](FROZEN_PROBE_RESULTS.json)。每折模型系数和样本级预测保存在 `/workspace/uwb-frozen-probe-20260930/`，可用脚本在数秒内重新生成。

没有将测试用户用于拟合、选择超参数或填补质量排除的缺失类别。八用户已有多轮研发历史，因此所有新方法比较仍属于探索性结果。缺少迁移前模型权重，本轮只重算历史缓存，不声称完整训练重现。

## 复现命令

```bash
python /workspace/uwb-cloud/uwb_direction1/tools/transfer.py verify --package /workspace/uwb-transfer
python /workspace/uwb-cloud/uwb_direction1/tools/transfer.py restore --package /workspace/uwb-transfer --out /workspace/uwb-restored-NEW
python /workspace/uwb-cloud/uwb_direction1/tools/audit_restored.py --data /workspace/uwb-restored-20260930 --package /workspace/uwb-transfer --out /workspace/uwb-audit-NEW
python /workspace/uwb-cloud/uwb_direction1/tools/frozen_probe.py --data /workspace/uwb-restored-20260930 --audit /workspace/uwb-audit-NEW/AUDIT.json --out /workspace/uwb-frozen-probe-NEW
```

所有输出须为独立空目录，不能覆盖原结果。环境：Python3.12.14、NumPy2.3.5、SciPy1.17.0；线性计算限制3线程。迁移及协议检查共14项通过。没有GPU，当前无后台训练任务。

## 研究接续

这次完成了数据接管和一项可否决的真实先导，尚未实现用户要求的“有效且有研究贡献的方法”。下一阶段应检验端到端的共同/视角特有动作表征，保留CE三路、普通SupCon和跨用户正样本约束等对照；不可把本次线性负结果推断为一切跨用户表征学习无效。先补齐相近雷达工作和共同/特有分解文献的全文核查，再冻结实现与计算预算。现阶段读到的QMF、SupCon、MixStyle是作者仓库文本/源码，不能冒称已完成论文全文调研；仅增加普通融合或已知损失也不能单独主张创新。
