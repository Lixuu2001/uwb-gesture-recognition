# 新容器恢复、编码器来源审计与三雷达核交互筛查

2026-09-30。研究目标尚未完成。已获得可复算的有限机制信号，尚无可靠超过强基线的改善，也没有已证明的新颖方法。

## 环境与恢复

安全检出 `codex/uwb-direction1-cloud-20260930`，初始远端提交 `3e07ad7`；原工作树干净，未覆盖修改。checkout没有`.agents/skills`目录。开始时没有训练进程；CPU可用5核、约17 GiB内存，无GPU、无torch。当前全部实验已结束，无后台训练进程。

本容器原来没有数据，使用已有私有Drive索引和认证连接器重新下载；最大并发3。163块、2,006,404,772压缩字节全部校验通过，失败0；86原文件、2,141,981,100字节恢复到`/workspace/uwb-restored-20260930`并再次独立SHA-256核验。无需用户电脑开机或重新上传。完整789份源码现已在checkout逐一验证，不再是仅28份核心文件。见`SOURCE_AUDIT.json`、`RESTORE_VERIFIED.json`及`AUDIT.json`。

质量筛选仍为9300条；历史LTR八折缓存93.4416667%、最差用户75%再次吻合，非重新训练结果。

## 划分与编码器来源

固定外层测试h、验证h%8+1、其余六人训练。应称带独立验证用户的外层留一用户协议，不是七训练用户LOSO。

八折均核验：baseline identity和multiview identity中的fold/分割哈希相符；融合初始化明确对应本折Right权重哈希，源用户集合排除验证和测试；特征元数据、预测缓存和原result哈希相符；选择轮次确为最早最小验证CE。来源源码在训练时严格检查本折初始化和只从train采样，导出在eval模式下进行。证据见`ENCODER_PROVENANCE.json`及`tools/audit_encoder_provenance.py`。

**原始权重未迁移，因此这是记录、源码和缓存一致性审计，不是权重重新前向验证。八用户长期用于路线选择，所有结果是开发证据，不能称未见最终验证。**

基线是同折Right初始化的共享编码器，每路256维，L/T/R固定槽位拼接+3维可见掩码，771→256→12融合头；七种真实非空雷达子集均匀采样训练，按七子集平均验证CE选择。它本来已有非线性头。

## 冻结假设与对照

此次比较固定三路特征的加法核与联合乘积RBF核，假设联合响应可能比逐路可加相似度更能利用互补信息。它不是共享/专属分解、用户中心惩罚或选路重跑，也不是新算法。

`KERNEL_SCREEN_PROTOCOL.json`及实现先于真实结果提交于`bc9a152`，随后`74ae3ff`补充原子日志和文献依据。每折仅源六人拟合标准化、距离中位数和384个类/用户分层地标；两模型均384维Nyström读出、相同五值正则网格，仅验证用户选择。预设验证门槛：相对加法核和旧LTR平均各至少+0.5pp，至少6折优于加法核。通过后按预先安排补地标种子2027；两个种子模型锁定后一次性评估测试，没有根据测试改配置。

核和Nyström是既有方法。本次文献正文核查进一步确认MV-GLFNet已有三视角早融合、局部/全局分支、门控和双线性互动，其六训练/两测试及9600条协议不能直接比较本项目。见`research/KERNEL_SCREEN_RATIONALE.md`。

## 真实结果

主指标始终为用户内现存类别macro accuracy，再八用户等权平均。

| 阶段/地标种子 | 原LTR | 加法核 | 联合核 |
|---|---:|---:|---:|
| 验证/2026 | 93.1241% | 93.1657% | 94.3568% |
| 验证/2027 | 93.1241% | 93.1388% | 94.4400% |
| 外层测试/2026 | 93.4417% | 92.3688% | 93.8080% |
| 外层测试/2027 | 93.4417% | 92.4028% | 93.7712% |

两个验证门槛都通过。外层测试联合核相对加法核分别+1.4392pp（8/8改善）和+1.3684pp（7/8改善，1折相同），说明该固定特征与读出比较中存在交互信号。

但对**强基线原LTR**仅+0.3663pp和+0.3295pp，两次均4/8用户改善；描述性用户bootstrap 95%区间分别[-0.3883,+1.2792]、[-0.4413,+1.2737]pp，均跨零。最差用户为74.3636%、74.6364%，低于原75%。因此不宣布可靠改善，不替换原基线，不扩大当前核参数网格。

2026联合核逐测试用户：99.25%、99.60%、91.00%、95.75%、98.75%、74.3636%、95.75%、96.00%。完整逐类混淆、差值、超参数选择见两个`VALIDATION_SEED*.json`和`TEST_RESULTS.json`。

**两次种子只改变地标抽样；共享的是同一套旧seed2026编码器。它们不是两个端到端训练种子，不能将16折当成16个独立用户。** 用户bootstrap也仅为重叠训练折上的描述性区间，不是独立外部确认。

## 计算、检查点与核验

18项迁移/协议/核数值测试通过；包括PSD、缺失类别均衡、推理不改变源统计及XOR对照验证核可表达交互。独立指标与选择审计146项检查通过，96份本地二进制产物共78,418,221字节有哈希清单。预测计算同时用两种矩阵乘法顺序重放一致。见`TESTS.txt`和`INDEPENDENT_AUDIT.json`。

CPU BLAS限制3线程，2026八折拟合及保存23.727秒，2027为23.044秒；2026完整主进程墙钟24.108秒、user CPU 53.256秒、sys 0.608秒。没有声称测得神经网络重训成本或GPU性能。初次`/usr/bin/time -v`启动因该程序未安装返回127，训练未开始；改用bash的time后成功。该启动错误未产生候选结果。

模型/地标/验证分数保存在`/workspace/uwb-kernel-screen-20260930`及`...-s2027`，测试分数在`/workspace/uwb-kernel-test-20260930`；进度JSON原子写入，完整折可用`--resume`恢复。大二进制不进入Git；若环境消失，认证恢复原86文件后可以按下列命令确定性重建。代码、协议、完整JSON结果和哈希清单进入私有研究分支。

## 实际命令

工作目录`/workspace/git-local-demo`；所有正式输出均为新目录。

```bash
python uwb_direction1/tools/transfer.py verify --package /workspace/uwb-transfer
python uwb_direction1/tools/transfer.py restore --package /workspace/uwb-transfer --out /workspace/uwb-restored-20260930
python uwb_direction1/tools/audit_restored.py --data /workspace/uwb-restored-20260930 --package /workspace/uwb-transfer --out /workspace/uwb-audit-kernel-20260930
python uwb_direction1/tools/audit_encoder_provenance.py --data /workspace/uwb-restored-20260930 --out uwb_direction1/results/cloud_20260930_kernel/ENCODER_PROVENANCE.json
python -m unittest discover -s uwb_direction1/tests -v
python -u uwb_direction1/tools/kernel_screen.py --data /workspace/uwb-restored-20260930 --audit /workspace/uwb-audit-kernel-20260930/AUDIT.json --out /workspace/uwb-kernel-screen-20260930
python -u uwb_direction1/tools/kernel_screen.py --data /workspace/uwb-restored-20260930 --audit /workspace/uwb-audit-kernel-20260930/AUDIT.json --protocol uwb_direction1/research/KERNEL_SCREEN_REPEAT_PROTOCOL.json --out /workspace/uwb-kernel-screen-20260930-s2027
python uwb_direction1/tools/evaluate_kernel_screen.py --data /workspace/uwb-restored-20260930 --runs /workspace/uwb-kernel-screen-20260930 /workspace/uwb-kernel-screen-20260930-s2027 --out /workspace/uwb-kernel-test-20260930
python uwb_direction1/tools/audit_kernel_predictions.py --test /workspace/uwb-kernel-test-20260930 --runs /workspace/uwb-kernel-screen-20260930 /workspace/uwb-kernel-screen-20260930-s2027 --out uwb_direction1/results/cloud_20260930_kernel/INDEPENDENT_AUDIT.json
```

## 后续范围与阻碍

没有认证/数据阻碍。无GPU、torch及旧.pt权重，旧训练入口硬编码CUDA和Windows路径；因此完整端到端重训还需CPU路径适配及实际成本测试，不能直接宣称已启动。不得新增付费算力。

新机制应结合原始三雷达信号及错误诊断；可考虑保留时间残差与雷达专属响应，但先看旧`temporal_style_diagnostic/REPORT.md`及`mild_time_augmentation/COMPLETION.md`：三视角等权DTW没有跨人稳定性优势，温和时间增强仅单Right+0.9222pp且区间跨零。不要重复这些已完成设置，不把峰值称为手部轨迹真值，也不把本次弱于可靠门槛的联合核直接包装成论文贡献。
