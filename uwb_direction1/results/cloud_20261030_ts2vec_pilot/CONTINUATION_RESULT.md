# 2026-10-30 — TS2Vec完整默认真实实验失败，600步独立重放与私有备份完成；下一项TS‑TCC已执行合成原作者更新

TS2Vec600完整constructor输出320/hidden64/depth10实际11块/673472参数/batch16/seed20261029真实实验完整结束，原ten-C训练内5foldRBF-SVM选C=10，零收敛/拟合警告。train用户3–6=4700、val2=1000、held1=1200，full567×90/RMS；诊断65.50%、验证64.90%，冻结最强同held1参考98.666667%，差−33.166667pp。预设+.5pp门槛失败，close_exact_configuration；不加种子/迭代/C搜索，不扩六折。此为探索性已发表方法迁移结果，不是旧LTR八折提升/独立验证或原创贡献。两套预测先冻结再读val/held标签；未用7–8标签。预登记095171d72ebb156a101ea42bc1ca4af621a07763，协议SHAb8e9946ee351eeee735bfffcbb50539e2b4d8fb9bc0cc86614dcc8db6ad183a1，41项科学冻结文件未变。

训练exit0/no timeout，230.482428wall/230.383109childCPU。原首步只读证明48组参数/AdamW/平均/16个ID已通过。完整独立审计63项哈希、全部600更新/9600样本出现、initial/first/final raw+average+optimizer/所有RNG逐位，6900条RMS和全部6900特征逐位、原训练内SVM重拟合支持/dual、2200预测/decision完全一致。审计attempt01仅最后macro严格相等因求和顺序失败：独立.6549999999999999 vs producer.655，差−1.1102230246251565e−16。原审计源码、208.910755wall/208.868365CPU exit1和trace原样保留。单独repair01仅允许macro比较1e−12并记录两值，实际守卫接受1ULP/拒绝1e−7；全部逐位科学检查保留，重跑完整审计exit0，221.393807wall/221.330060CPU。producer/模型/训练从未重启。AUDIT_INCOMPLETE.json只描述原attempt01，最终以AUDIT_COMPLETION_RECEIPT.json/COMPLETE_AUDIT.json pass为准。

私有完整恢复包33451378字节/5块/87payload+manifest=88成员，SHA45a5175865dea6626c3f1daa8937fa8a1ed432f0aac5a014024d251f6e72c626，回执SHA1af36577d6d4f875ef48de04ff08fa9808ff3e67492db031c7037ff9d38b16fb。全部owner-only/目录/大小/MIME核验，authenticatedrawfetch/base64=false即物化校验，重组及所有成员SHA通过。含完整模型/优化器/RNG、SVM/训练CV ID、全部训练批ID、特征预测、源码许可与两次审计。索引 results/cloud_20261030_ts2vec_pilot/complete_backup/DRIVE_BACKUP_RECEIPT.json。原数组/源maps不入包/Git，仍由原迁移+RMS恢复。last.pt每50步滚动覆盖，单独initial/first和最终last持久保留；不声称各中间版本或自动整作业resume。旧POST_LAUNCH_AUDIT_HASHES是修复前支持代码快照，当前修复源码由REPAIRED_AUDIT_HASHES及完整回执builderSHA绑定；原41科学项未动。

下一候选TS‑TCC(Eldele等，IJCAI2021:2352–2359，DOI10.24963/ijcai.2021/324)作者commit15bdfd6eced6aeb6c638839828effe1b4b2ea391已固定并核验10个blob。已完整读README、model/TC/attention/loss/trainer/loader/augmentation/HARconfig/main；论文主文/图未读，不能称全文复现。原作者fullHAR默认batch128/40epochs/internal32–64–128/TC100/depth4/heads4/sixsteps保持；只作567输入/12类/13特征长度适配，实际一次合成原作者SSL model_train更新通过（encoder247500+TC313096参数，9+65优化器状态，有限loss13.717924）。新author_adapter.py、合成检查/恢复源码helper及哈希已实现；没有真实UWB值/标签/拟合或launcher。

明确下一实现事项：NumPy2原ragged分段置乱报错；仅分析用index-list兼容后，原ret[i]=pat[0,warp]把首通道广播到所有通道，4/4合成样本复现。原loader最小维度启发式也会调换90time/567channels。须核对论文多通道增强含义、固定透明兼容/axis适配、完成原步对照/完整RNG恢复及40轮完整pipeline成本，然后远端登记唯一source协议再真实跑；不能静默修正或因有先例停止实验。现有网络无通用网页读取工具，shell api.github.com受代理403，已用GitHub连接器取得源码，没有绕过网络策略；paper正文访问尚待解决，可继续上述实现。根LICENSE返回404，暂只发布本方adapter/检查与哈希/获取helper，不复制作者源码入Git。原源码与依赖留本地，原789快照未动。

当前无活动训练进程；TS2Vec已完整结束/审计/私有保存，TS‑TCC只完成合成准备。先检查真实进程和状态，禁止重启TS2Vec或TSLANet/重复下载。全部八用户历史反复开发，研究准确率提升/原创目标均未达，自动接续保持启用，继续论文→具体实现→真实固定实验。

