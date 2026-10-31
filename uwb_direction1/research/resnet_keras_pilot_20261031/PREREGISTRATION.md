# 完整作者网络 ResNet：固定 native Keras CPU100 源内准确率实验

目标是检验一个已发表强基线迁移能否提高同source诊断准确率，和原创贡献证据分开。作者hfawaz/dl-4-tsc commit e0233efd886df8c6ca18e6f1b545d15aaf423627（GPL3），原build_model AST只改artifact扩展名；使用真实Keras3.11.3/Torch CPU，非原TF1.14/TF2训练复现。完整64/128/128、8/5/3kernels、11Conv/12BN/3Add/GAP/12softmax和bias/第三块BNshortcut保留，834124总参数含BN非训练量，48训练张量/98Adam变量。

固定seed20261029、B64、100完整轮、1CPU线程/deterministic，每轮4700全记录/74批/末28，7400更新/470000样本出现。旧准备COST_POLICY先于计时；原1500轮估103363.775629秒>43200，预定义100轮估7011.481360秒通过。本实验不延长/换seed/缩宽/换批量救援，未来100轮负结果不否定1500轮配置的准确率。

native Adam.001/betas.9,.999/epsilon1e-7/无weightdecay或clip，CE/softmax。原ReduceLROnPlateau(monitor=loss,factor=.5,patience50,min_delta1e-4,min_lr=.0001)，按严格更小训练loss选best、ties留前；无earlystop，验证user2只监视不训练/选best。保留原native best_model.keras、last_model.keras与initial/first/best/last全权重/Adam/BN/metrics/callback/allRNG/identity/sampler快照。显式native train_on_batch累积指标是原fit的加权训练loss；记录每步累计loss/accuracy和tracker差分batchloss，不冒称累计loss为单批loss。

固定source6900记录，train用户3–6=4700、validation用户2=1000、held1诊断=1200；完整90×567有符号real record/view RMS，无resize/augmentation。IDs/maps与raw数组SHA和前Transformer完结/完整私有回执固定。12类使用既有任务taxonomy；不执行原main.py合并train+test标签拟合onehot/确定类别的路径。读取labels时仅先索引训练及user2，held1两个最终预测都冻结/哈希后才索引；7–8标签不用。

训练专用TorchGenerator(seed+epoch)确定4700记录次序；eval仅counts的RandomState(seed+101val/+102diag)固定B64分批（末40/48），保存并重放实际ID。显式loader/globalRNG与原nativefit数据适配器不同，已经声明；70合成记录三轮对照证明原生fit和显式nativebatch循环的全部模型/Adam/BN/指标/选中checkpoint一致，但不声称nativefit数据适配器RNG相同。新的真实入口合成三轮完整重放、callback/metrics/全部RNG立即恢复及下一轮逐位也通过。

预登记代码/协议/准备材料必须先逐字远端读回，再创建REMOTE_PREREGISTRATION及仅一次launch.py --run。缺marker/hash错误、现有run/supervisor或前任务未成功关闭均拒绝。训练独立43100wall/43000–43100CPU；完整逐步独立重放审计14400秒。异常保留STATE/trace/TERM，不自动retry/resume，不把工程失败当科学负结果。首次真实更新核模型/98Adam变量/64确切IDs/全部状态恢复后立即保存私有首步保险。

auditor从固定seed重建全部7400更新及470000实际ID、100轮native训练/验证指标和训练loss checkpoint/ReduceLR、initial/first/best/last全模型/Adam/BN/metrics/callback/RNG，核两native.keras和6900RMS生成式；最终2200 probabilities/predictions逐位。macro仅数值归约容差1e-12，不放松状态/输出位精度。controller仅等待现有训练成功，审计一次并生成本地完整备份，无训练重启、Drive上传或Git能力。agent后续按私有metadata→rawfetch(base64=false)→立即物化SHA→全成员恢复验证再发布。

冻结双最终val/held1概率/ID/pred后，单次held1macro gate：相同source reference .9866666666666667，提升>=.005（1e-12归约容差）才允许另注册source确认；否则关闭该确切CPU100配置，转下一文献方法，不扫诊断或7–8label。当前未真实fit/无诊断分数。八用户历史开发导致探索性，非独立测试/旧八折/原创贡献验证；目标未达，既有小时自动接续保持。文献只读相关primary方法/实验段，非全9/44页或图读，详见上一ResNet准备记录。

Git只包含小代码/配置/哈希/结果/记忆，原789快照不改；原数组/maps/模型/wheels/.so/实际样本ID值不入Git。运行依赖由1302文件与10wheelSHA/基底环境固定，私有保险/完整模型集不冒称整运行环境或整作业resume。
