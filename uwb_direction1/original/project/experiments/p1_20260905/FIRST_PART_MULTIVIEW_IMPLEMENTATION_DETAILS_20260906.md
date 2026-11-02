# 三视角训练、单雷达部署：四套候选方法的详细实现与实验计划

> 执行说明：后续获准实施时，按 superpowers:executing-plans 逐项推进。本文是设计说明，下面的新接口、实验名称和伪代码尚未成为仓库可执行训练入口。用户本轮只要求扩写Markdown，不运行新实验。
>
> Goal：明确额外两路雷达如何参与训练、单雷达实际接受什么监督，以及怎样证明准确率和跨用户泛化的收益来自三视角信息。
>
> Architecture：统一复用现有单视角幅相编码器和分类头，分别研究关系监督A、运动过程监督B、可预测教师分量C，以及预训练D。A/B优先，C/D按条件启用；先单模块归因，再考虑组合。
>
> Tech Stack：现有Python、PyTorch、NumPy、scikit-learn，RTX4060 Laptop 8GB，质量审计后的UWB-Gestures数据。

日期：2026-09-06。本文是[研究方向与文献总稿](FIRST_PART_MULTIVIEW_IDEAS_20260906.md)的实现细则。冲突时，具体张量、损失与实验协议以本文为准；研究假设仍待验证。

---

## 0. 推荐主线、执行边界与阅读方式

建议第一部分暂定题目：**多视角结构监督下的单雷达跨用户手势识别方法**。

要检验的科学问题是：三路观测能否提供“单路能学会、标签没有直接告诉它”的监督，使单路识别对未见用户更稳定。不能将教师高准确率本身视为学生可达到的上限，也不能把辅助预测误差下降视作分类成功。

推荐按以下顺序读：

1. 第1—4节：所有方案共用的输入、代码接口和训练边界。
2. 第5节：A，先做跨用户关系监督。
3. 第6节：B，与A独立开展的运动过程监督。
4. 第7—8节：C/D备选完整做法。
5. 第9—12节：组合、超参数选择、实验矩阵和验收。
6. 第13—15节：未来文件接口、实现任务、论文归因与文献。

为避免混淆，全文区分：
- **已核实**：当前代码/结果存在。
- **规定**：后续实验需要执行的一致性规则。
- **建议起点**：尚未验证的工程参数。
- **可选扩展**：最小方法成立后再做，不属于首批必做工作。

本文不承诺四套一定有正增量，也不把成熟方法改名为已证明的新方法。

## 1. 数据到底是什么、每一折怎样组织

### 1.1 数据对象与单位

一个样本是一次分段动作的三雷达记录。原始缓存行号作为全局sample_id，不能把Left的一次动作、Top的另一次同类动作误认为真实配对。

|对象|形状|含义|
|---|---|---|
|单次三视角RF输入|3×90×189|L/T/R，慢时间，距离采样|
|整个缓存|N_cache×3×90×189|物理数组行数以实际npy为准，不写死9300|
|质量保留索引|9300|有效参与训练/验证/测试的行|
|动作标签|N_cache|原始0—11|
|用户标签|N_cache|原始1—8|
|幅度图A|3×90×189|norm(abs(X))|
|相位运动图M|3×90×189|幅度加权逐帧相位增量后标准化|
|单视角联合输入|1×90×378|沿最后一维拼接[A_v,M_v]|

重要：378是189维幅度特征加189维相位运动特征，不是距离分辨率翻倍。距离分支会看到378个输入位置，但其后半部分不表示额外物理距离。

已核实质量掩码：
- 排除HV2的G11/G12标签冲突。
- 排除HV6的G4无法核实配对样本。
- 所有实验使用同一掩码。
- HV2测试实际支持10类，HV6支持11类，其余6名用户支持12类。
- 输出类别数始终12；各折macro-F1按测试真实出现类别计算，假阳性仍按该类别的标准定义计入。

不再重新筛样本，不根据新模型表现修改掩码。

### 1.2 现有文件入口

根目录：D:/uwb_gestures/experiments/p1_20260905/

|文件|本次已核实的作用|
|---|---|
|pipeline.py|quality_mask、split、norm、phase_features|
|prepared/off.npy|逐视角幅度标准化输入|
|prepared/only.npy|逐视角相位运动标准化输入|
|prepared/y.npy、subj.npy|与上述缓存同全局行号|
|train.py|Classifier、load_gpu、kd_loss、普通训练|
|vendor/uwb_osr/model.py|SharedDualPathEncoder、ClosedSetHead|
|reconstruction.py|compact特征提取；每视角992维|
|label_global.py、label_uniform.py|现有全局软目标基线的精确定义|
|followup_analysis.py|预测加载、split索引与指标核验|
|results_clean/|现有模型/预测，不能覆盖|

新方案放在独立子目录mv_transfer_v2下，不能伪装成旧tag的继续结果。

### 1.3 外层、内层和模型选择

外层留出用户h：
- train_ids和val_ids来自其他7人。
- test_ids只来自h。
- 原split函数按每个训练用户的每类约10%抽验证。
- 原val用于早停与checkpoint选择，不参与梯度训练。
- test不参与预训练、归一化拟合、目标生成器拟合、关系库、超参数或阈值选择。

“当前训练样本的三视角”可以生成自己的辅助目标；这就是特权训练。测试时只有单视角。

内部留用户诊断从train_ids中再分fit_ids与probe_ids：
- fit_ids：6个训练用户的梯度训练行。
- probe_ids：剩下1个源用户的梯度训练行。
- 外层val/test不加入。
- 有拟合参数的scaler、ridge等只在fit_ids拟合。

**关键限制：**复用已有外层教师时，该教师已经见过上述probe用户的其他/这些训练记录。此时内部划分只能证明新回归器或学生的分组隔离，不能声称教师严格OOF。

若用内部“未见用户表现”选择完整教师—学生方法的超参，教师也必须只在内部6用户训练；否则该验证有特权监督生成器见过probe用户的偏差。控制成本的首选是冻结本文工程起点，不用这种有偏内层分数调方法。确需调参时，另建严格内部教师，仅用于预先选定的少量内层验证；最终外层再用7用户教师。

### 1.4 分离随机种子

新实验建议：
- split_seed=2026固定。
- model_seed=2026/2027。
- sampler_seed固定并保存。
- auxiliary_seed单独控制新增head初始化/随机投影/mask。

这样新实验两个model_seed仅改变训练随机性，不改split。旧2026/2027结果同时改变了划分与初始化，不能直接充当此新协议的双种子对照。

若完全沿用旧“一个seed控制全部”的方式，则所有新臂及其基线都重新匹配，报告名称写清楚。不得混用2026教师与不同train_ids的2027学生。

## 2. 输入变换、张量接口与模型初始化

### 2.1 相位运动图的精确定义

对X_v(T,R)，沿距离维做解析信号变换得到Z：
1. Z←analytic(X_v)。
2. Z←Z−mean_t(Z)。
3. D_t=Z_t×conj(Z_{t−1})。
4. M_t=arg(D_t)×sqrt(abs(D_t))。
5. 首帧复制首个相位增量。
6. 对每个样本、每个视角的90×189矩阵独立均值/标准差归一化。

不是对整个数据集fit标准化，因此现有逐片段norm不使用其他样本统计；但它使用完整动作片段，不是因果在线处理。

不要从已经abs_std的幅度图计算相位，也不要把M称为已校准m/s径向速度。质量代理/运动强度应从未逐样本标准化的解析量生成，否则真实强弱可能被抹掉。

### 2.2 原模型的真实输出

当前dim=32，pool=flatten：

|模型|t_tokens|s_tokens|分类头输入|
|---|---|---|---|
|单路幅度|B×1×90×32|B×1×189×32|B×8928|
|单路幅相|B×1×90×32|B×1×378×32|B×14976|
|三路幅度concat教师|B×3×90×32|B×3×189×32|B×26784|

ClosedSetHead为输入维→2048→256→12，包含GELU及dropout0.1。

**容易实现错的一点：**encoder返回的第一个f加了view_emb；但当前实验Classifier丢弃这个f，而是从t_tokens与s_tokens重新flatten拼接。后续适配器必须保持这一行为才能严格复现旧分类输出，不能直接改为使用第一个f。

现有实验也关闭了跨视角注意力。只要新增方法不需要，就保持关闭。三视角身份通过固定拼接槽位保留，而不是宣称当前分类器已经使用了显式视角嵌入。

### 2.3 建议增加的纯接口适配函数

以下是未来实现时可复制的核心函数；本轮只写在文档中：

~~~python
import torch

def classification_features(model, x):
    # x: B,V,90,M_in
    _, temporal, spatial = model.encoder(x)
    per_view = torch.cat(
        [temporal.flatten(2), spatial.flatten(2)], dim=-1)
    merged = (per_view.flatten(1) if model.readout == "concat"
              else per_view.mean(1))
    logits = model.head(merged)
    return logits, merged, temporal, spatial

def student_aux_features(temporal, spatial):
    # Returns B,64 for one view; uses both paths.
    assert temporal.shape[1] == spatial.shape[1] == 1
    return torch.cat(
        [temporal[:, 0].mean(1), spatial[:, 0].mean(1)], dim=-1)

def teacher_compact(temporal, spatial):
    # Exactly follows existing compact definition for amplitude teacher.
    b, v, t, d = temporal.shape
    assert (t, spatial.shape[2], d) == (90, 189, 32)
    a = temporal.reshape(b, v, 10, 9, d).mean(3).flatten(2)
    c = spatial.reshape(b, v, 21, 9, d).mean(3).flatten(2)
    return torch.cat([a, c], dim=-1)  # B,V,992
~~~

验收：在eval模式同一输入上，适配器logits应与原model(x)[0]一致。因为dropout，不能拿两次train forward直接要求数值完全一致。

### 2.4 初始化与部署限制

先构造原Classifier并保存全state/hash，再用独立随机种子构造辅助head。基线与候选加载相同encoder/head初值，新增head不能改变主网络初值。

A/B/C部署只导出原classifier state，不保留训练目标缓存与其他视角读取逻辑。辅助头移除后eval logits应不变。

D微调阶段明确“编码器已多视角预训练”，不能称从未利用多视角的基线。

## 3. 基础损失：不要把74.77%的对照写错

### 3.1 CE对照

L_CE=CrossEntropy(logits, y)，硬标签，temperature=1。

### 3.2 现有全局软目标对照

它不是直接调用CrossEntropy(label_smoothing=epsilon)的标准单项实现。

当前全局矩阵U每行：
- 对角d_bar。
- 非对角(1−d_bar)/11。
- d_bar由当前train_ids的Right内部OOF类别统计构造，按训练类别数量加权。
- U只来自Right，不使用三视角教师。
- 目标q=0.5×(onehot(y)+onehot(y)@U)。

于是m=q_y=(1+d_bar)/2，各非真类(1−m)/11。

实际基础损失：
\[
L_{\rm base}=CE(y,z)+0.5\times 2^2
KL(q\Vert softmax(z/2)).
\]

~~~python
import torch.nn.functional as F

def classification_loss(logits, labels, true_mass):
    z = logits.float()
    y = F.one_hot(labels, num_classes=12).float()
    m = torch.as_tensor(true_mass, device=z.device, dtype=z.dtype)
    q = m*y + (1-m)*(1-y)/11
    soft = F.kl_div(F.log_softmax(z/2, dim=-1), q,
                    reduction="batchmean") * 4
    return F.cross_entropy(z, labels) + 0.5*soft
~~~

true_mass要从匹配split的global_matrix读取，不填一个猜测的epsilon。若采用其他split，重新生成Right内部统计。

建议先在新训练器复现此基线，再加A/B。若改了采样器/早停，74.77%仅作历史参照，以新匹配基线为准。

### 3.3 统一优化设置

建议起点：
- Adam lr=1e-4，weight_decay=1e-4。
- 最大40epoch，epoch≥12后允许patience=8。
- 按val硬标签CE选best checkpoint。
- effective batch=128，microbatch=32。
- autocast使用现有bf16，关系softmax/KL/目标统计用float32。
- teacher.eval()，requires_grad_(False)，no_grad。
- scheduler先不新增；若新增cosine，所有基线相同。
- 不因候选验证差就单独延长epoch。
- aux系数前5epoch线性爬升到固定值；所有方法使用相同规则。

每项损失先按样本/有效位置求平均，再乘系数；禁止一个损失按sum、另一个按mean却照用相同lambda。

## 4. 所有方法共用的“信息来源账本”

每次实验保存一个provenance.json，至少记录：

|字段|说明|
|---|---|
|student_input_views|例如[Right]|
|teacher_training_views|例如[L,T,R]或[Right]|
|teacher_target_views|该辅助目标实际读取的视角|
|teacher_training_ids_hash|老师见过哪些样本|
|student_train_ids_hash|学生梯度训练样本|
|validation_ids_hash、test_ids_hash|明确隔离|
|teacher_checkpoint_sha256|目标来源|
|target_transform_fit_ids_hash|scaler/PCA等fit集合|
|label_statistics_source|global smoothing如何生成|
|split_seed/model_seed/sampler_seed|分开记录|
|initial_classifier_hash|排除主干/分类头初始化差异|
|sampler_manifest_hash|训练曝光顺序和数量|
|code_snapshot_sha256|新脚本及vendor版本|
|aux_settings|损失系数、维度、温度、mask等|

对照要区分：
- **target-single**：三视角训练教师仅取Right分支目标。它仍间接利用了三视角训练。
- **teacher-single**：教师训练和产生目标都只使用Right。
- **teacher-multi**：教师训练与目标都使用多视角。

只有teacher-single才是严格无额外训练视角的教师对照。target-single用于判断“额外目标视角”的贡献，二者不能合并。

同类错配只在train_ids内执行。每个epoch以固定seed生成无固定点置换，尽可能在同类别、同源用户内换实例，保持用户/类别分布。原始三路教师应先真实配对提取目标，再错配目标；不要把不同实例的L/T/R组合喂给教师制造异常输入。

## 5. A：跨用户关系融合——具体到一个训练step

### 5.1 最小版本定义

教师：现有三路幅度concat网络；学生：Right幅相。

教师compact为B×3×992；各视角先做训练折拟合的通道标准化（std下限0.05），再L2归一化。对三路顺序拼接并再次L2归一化，得到B×2976。

首版不加可训练教师投影，不改教师分类头，不使用最终softmax概率。教师保持现有性能只是参考，新的关系目标还需单独验证。

学生辅助输入为时间/距离token均值拼接，B×64；投影MLP为64→256→128，GELU，中间可LayerNorm，最终L2归一化。这里与总稿中泛化写法不同，已把具体输入定为64维；不能把14976直接误当64。

原14976维分类特征不替换，不对整个主干强制跨视角一致。

### 5.2 batch怎样采样

每个关系microbatch选择4类、4名用户，每个类×用户取2条：
- 4×4×2=32。
- 4个microbatch累积一个128样本优化step。
- 4类从0—11循环均衡抽取。
- 4名用户从同时支持所选4类的训练用户中抽取；HV2/HV6缺类通过候选用户交集处理。
- 每个类—用户池少于2条时重新采样组合；若处于低标签实验可允许有放回，但必须显式记录。
- 同一个microbatch不放入重复sample_id。
- 无足够共同用户时本轮改抽类别集合；达到100次失败报错，不静默少采。
- 预生成每个epoch的索引数组，所有A相关基线共享。
- steps_per_epoch=ceil(len(train_ids)/128)，采样是均衡采样而非“每样本必定一次”；报告各类/用户曝光次数。

新采样器会改变基线训练分布，必须重跑A-GS基线。

不将邻接关系跨microbatch计算，称“32样本关系组”。如果以后使用128全batch关系，需要保存带梯度特征或用正确的梯度缓存方法，不能认为普通梯度累积自动扩大对比负样本集。

### 5.3 关系对象与损失

有效邻居掩码：
M_ij=1[user_i≠user_j]。排除同人，包括自身。
正对掩码：
P_ij=M_ij×1[y_i=y_j]。
32样本结构保证每个anchor至少有6个跨用户同类正对。

教师邻接：
R_T(i,j)=softmax_j(cos(t_i,t_j)/tau_T)，仅在M=1上归一化。
学生邻接同理，初始tau_T=tau_S=0.2。

关系损失对所有跨用户邻居分布做KL。首版不用“只保留教师自信样本”的筛选，以便先评估关系本身。

监督对比：
每个anchor对所有正对的log概率求平均，再在anchor间平均。它只用y/user，没有三视角信息，是必须单列的对照。

~~~python
def cross_user_relation_losses(s, teacher, labels, subjects, tau=0.2):
    # s and teacher: B,D; teacher target is detached.
    s = F.normalize(s.float(), dim=-1)
    t = F.normalize(teacher.detach().float(), dim=-1)
    allowed = subjects[:, None] != subjects[None, :]
    positive = allowed & (labels[:, None] == labels[None, :])
    if not allowed.any(1).all() or not positive.any(1).all():
        raise ValueError("batch lacks cross-user neighbours or positives")

    sl = (s @ s.T / tau).masked_fill(~allowed, -1e4)
    tl = (t @ t.T / tau).masked_fill(~allowed, -1e4)
    log_s = F.log_softmax(sl, dim=1)
    log_t = F.log_softmax(tl, dim=1)
    prob_t = log_t.exp()

    # Use finite masked logits; avoid 0 * (-inf) NaNs.
    rel = (prob_t * (log_t-log_s) * allowed).sum(1).mean()
    con = -((log_s * positive).sum(1) / positive.sum(1)).mean()
    return rel, con
~~~

A总损失：
L=L_base+ramp(epoch)×(0.1L_con+0.1L_rel)。
lambda先固定，不同时开普通logit KD。若加入KD，必须作为额外消融。

### 5.4 完整训练step伪代码

~~~text
读取当前fold的原始主网络初值；所有A臂加载同一份
读取teacher_train_features和训练索引，检查provenance
for epoch:
    ramp = min(epoch / 5, 1)
    for four_groups in saved_epoch_sampler:
        optimizer.zero_grad()
        for ids32 in four_groups:
            只加载Right幅相输入，得到logits,tokens
            s = student_projector(aux_mean_tokens)
            loss = base_classification_loss(logits, y)
            若为A-SupCon或A-Relation：
                使用同一cross-user掩码计算SupCon
            若为A-Relation：
                按sample_id读取指定来源的teacher目标
                计算teacher/student邻接KL
            (loss / 4).backward()
        optimizer.step()
    仅用val的Right输入计算CE
    保存当前val CE最优学生，不保存val最优教师关系
训练结束：重新加载best，只用Right测试一次
~~~

所有监督必须作用到encoder，不只训练投影头。单step检查encoder temporal/spatial至少一处梯度非零。teacher参数梯度应全部None。

### 5.5 教师关系预诊断

在源训练数据内输出：
- 跨用户近邻的类别纯度。
- 相同类别内部邻接熵、特征方差和样本距离分布。
- R_T与“同类均匀、异类极小”的label-only关系距离。
- 三视角与Right目标的邻接差异。
- 在2次温和观测噪声下关系排序的稳定性。
- teacher-single与teacher-multi的同类内部差异。

不要把teacher已经训练过样本的近邻纯度当未见用户泛化。目的只是发现塌成类别中心、全部相似或被用户支配的退化。

若三视角关系几乎仅是label-only矩阵，优先记录失败，不继续大范围搜索温度。这里没有预定数值阈值能证明“值得训练”；应在正式外测前冻结选用的表征层与温度。

### 5.6 A的实验臂

|tag|输入|新增监督|目的|
|---|---|---|---|
|A-GS|Right幅相|只有L_base|新采样器下基线|
|A-SC|Right幅相|跨用户SupCon|标签与用户信息贡献|
|A-TS|Right幅相|SC+三视角教师Right分支关系|target-single控制|
|A-US|Right幅相|SC+纯Right教师关系|teacher-single主控制|
|A-MV|Right幅相|SC+三视角融合关系|主候选|
|A-SHUF|Right幅相|SC+同类实例错配三视角关系|细粒度对应必要性|
|A-RKD|Right幅相|普通RKD距离损失|成熟关系方法控制|

首批至少A-GS/A-SC/A-US/A-MV。A-TS用于解释训练共享与目标融合的区别。所有臂的辅助head、teacher compact维度差异要在归一化/损失尺度上控制；分类部署参数不变。

可选共享/私有分支仅在A最小版成立后尝试。若增加两个128维分支，必须让分类head参数匹配，不直接与原14976分类head比较后归因。

## 6. B：运动过程伪监督——从RF到15段目标

### 6.1 首先解决时间一致性

不能因为三视角数组行号一致，就假设三个第t帧是同一物理时刻。

建议审计输出：
- 公开采集程序是否说明触发方式、帧率、雷达间时间戳。
- 仅在train_ids上计算每对视角非负运动包络的归一化互相关。
- lag搜索先限制[-6,+6]帧，逐样本记录最优lag、峰值与第二峰差。
- 按视角对报告lag直方图、不同类/用户分布。
- 多峰或动作相关lag变化明显，则互相关不足以证明同步。
- 只存在稳定偏移且有采集证据支持时，可以固定每视角整数shift对齐到部署视角时轴；shift只在train上确定。
- 禁止每个测试样本读取额外视角重新对齐。
- shift填充区标为invalid，不循环roll回序列另一端。

首版K=15、每段6帧，是工程尺度，不是经真值验证的动作阶段。若连这一尺度都不能支持时间对应，就降为整段统计辅助，不执行时间监督版本。不得把任意DTW拟合良好当同步成立。

### 6.2 每路动态证据

从ClutterRemoved RF解析信号Z计算：
D=Z_t×conj(Z_{t−1})，
phi=angle(D)，
w=sqrt(abs(D))。

两个独立候选强度：
- 相位运动E_phi(t)=sum_r w(t,r)×abs(phi(t,r))。
- 幅度变化E_amp(t)=sum_r abs(abs(Z_t)-abs(Z_{t−1}))。

首版选择E_phi；E_amp为控制，不同时拼进目标以免无法归因。

每视角尺度s_v=训练有效帧E_phi的95百分位，epsilon下限。scaled_E=clip(E/s_v,0,5)。clip上限5为数值保护起点，不称消除了异常人体运动。它保留片段内强弱变化。

quality版本：
- C_v(t)=abs(sum_r w×exp(j phi))/(sum_r w+epsilon)。
- q_v(t)=0.25+0.75C_v(t)。
- 基础版q=1。
C反映圆相位集中程度，不是置信度标签；高C也可能不是目标手势，低C也可能是正常复杂运动。

不能由q高直接跳过噪声核验，更不能把q当“可观测概率”。

### 6.3 生成15段监督目标

训练三视角融合：
E_F(t)=sum_v q_v(t)E'_v(t)/sum_v q_v(t)。

self控制仅用目标视角，同样的scale、分段与归一化。
multi-max控制逐帧取max，不与主均值融合混用。

对每个6帧段求平均E_k，生成：
a_k=(E_k+epsilon)/(sum_j E_j+15epsilon)。
有效段掩码处理shift无效区；全片总能量接近数值零的样本辅助valid=False。
对全无效/零能量样本不强造均匀“动作目标”，只计算L_base。
若需要按低能量剔除辅助目标，阈值仅由train分布确定并写入配置；不得删掉其分类样本。

累计目标c_k=sum_{j≤k}a_j。
因此c是a的CDF，**不是第二种独立标注**；同时用二者损失是在不同误差度量上约束同一过程，必须通过消融验证是否必要。

未来核心实现示例：

~~~python
def fuse_process_targets(energy, quality, frame_valid, eps=1e-8):
    # All B,V,90; energy is nonnegative, already scaled using train only.
    b, v, t = energy.shape
    assert t == 90 and quality.shape == frame_valid.shape == energy.shape
    q = quality.float() * frame_valid.float()
    denominator = q.sum(1)
    frame_ok = denominator > eps
    fused = (energy.float()*q).sum(1) / denominator.clamp_min(eps)

    cnt = frame_ok.reshape(b, 15, 6).sum(-1)
    summed = (fused*frame_ok).reshape(b, 15, 6).sum(-1)
    segment_energy = summed / cnt.clamp_min(1)
    seg_ok = cnt > 0
    total = segment_energy.sum(-1)
    valid = (total > eps) & seg_ok.any(-1)

    smoothed = (segment_energy + eps) * seg_ok
    activity = smoothed / smoothed.sum(-1, keepdim=True).clamp_min(eps)
    activity = torch.where(valid[:, None], activity, torch.zeros_like(activity))
    progress = activity.cumsum(-1)
    return activity.detach(), progress.detach(), seg_ok, valid
~~~

对于有不完整时间段的样本，目标/预测都在同一seg_ok上归一化。学习过程中不允许预测给invalid段分配概率。首版应报告invalid比例和各用户分布，避免有效样本选择改变辅助训练人群。

### 6.4 单雷达过程head

学生t_tokens：B×1×90×32。
先按每6帧平均得到B×15×32，然后：
Linear(32,64)→GELU→Linear(64,1)，输出B×15的activity_logits。
只使用当前完整离线片段，不声称causal；原时间Transformer已可访问整个片段。

预测hat a=softmax(activity_logits)；
hat c=cumsum(hat a)。
只有一个head，减少重复任务和额外参数。

~~~python
def process_loss(logits15, activity, progress, seg_ok, sample_ok):
    if not sample_ok.any():
        return logits15.sum() * 0, logits15.sum() * 0
    z = logits15[sample_ok].float()
    mask = seg_ok[sample_ok]
    log_p = F.log_softmax(z.masked_fill(~mask, -1e4), dim=-1)
    p = log_p.exp()
    a = activity[sample_ok].float()
    c = progress[sample_ok].float()

    # Sample-mean KL; no division by number of bins a second time.
    kl = (a * (a.clamp_min(1e-8).log()-log_p)).sum(-1).mean()
    # CDF distance penalizes probability mass at the wrong time.
    cdf = F.smooth_l1_loss(p.cumsum(-1), c, reduction="none").mean(-1).mean()
    return kl, cdf
~~~

L_B=L_base+ramp×(0.1L_activity+0.05L_CDF)。
先Ablation只开activity，再比较加CDF；不能把两个lambda一起调出结果后省略分解。

### 6.5 B与“分类注意力”的关系

第一版仅辅助监督，导出时丢弃过程head；这样不改原分类参数。

第二版若希望运动权重直接影响识别，可将hat a对时间token加权形成一个32维摘要，作为残差辅助输入。这会改变分类器，需要：
- 同结构无监督过程head控制。
- 单视角过程监督控制。
- 匹配参数的普通attention pooling控制。
- 不允许测试时使用真实三视角a_F做pooling。

训练使用a_F作强制注意、测试改用hat a，会产生teacher forcing落差。首版避免这种不一致。

### 6.6 时间增强的正确位置

首版不添加时间warp，先验证目标价值。

扩展版：
1. 在RF时间轴上对同一次动作三路应用同一温和单调时间变换。
2. 重新计算解析相位增量与归一化，或从统一变换后的解析序列生成全部派生量。
3. 学生输入与teacher过程目标使用同一时间坐标。
4. 不随意裁掉手势起止或改变标签。
5. 不默认使用时间反转、距离镜像、随机视角交换。

仅在预计算phase图上插值是表征扰动，不等同物理一致的速度变化。若采用，必须如实标明。

### 6.7 B的实验臂

|tag|目标|作用|
|---|---|---|
|B-GS|无过程目标|同训练器基线|
|B-SELF|Right相位强度过程|无额外视角控制|
|B-MV|三视角等权相位过程|主候选|
|B-Q|三视角质量加权过程|质量设计必要性|
|B-AMP|三视角幅度变化过程|相位机制控制|
|B-CENTROID|原质心回归aux|排除旧aux换名|
|B-SHUF|同类实例间过程错配|实例对应是否必要|
|B-TIME|固定置换时间段，重新计算CDF|时间顺序是否必要|
|B-CDF|B-MV只用activity，不加CDF|第二误差项是否必要|

对于B-TIME，先置换a，再重新cumsum，不能把a和c独立乱序制造数学不一致的目标。置换种子固定，不能根据某类的真实语义选择“最难排列”。

注意：三视角强度目标不一定区分向左/向右。方向信息仍由原单视角分类输入承担，过程监督的目的主要是动作组织与噪声关注，而非恢复不可观测运动方向。

## 7. C：可预测教师分量——如何筛选、如何避免偷看

### 7.1 固定特征坐标，不先做跨折无法对应的PCA

教师时间token分6段，每段15帧，得到6×32；距离token全局均值32。
每视角224维，三路拼接672维。
固定同一外层teacher与池化定义，使7个源用户内部回归器评估的是同一672坐标。

若每内折先独立PCA，不能将各自第k主成分当同一方向平均R²。这是选择固定池化坐标作为首版的原因。

Right输入phi_R固定为420维：幅度90×189按10×21格、每格9×9均值得到210维；相位同样210，拼接。

不要使用三视角训练编码器的Right特征作为唯一“无特权预测输入”控制；那样在参数上已经接受多视角训练。固定原始幅相池化能更清晰地区分信息来源。

### 7.2 内层回归算法

每个外层h：
1. 获取train_ids；把外层val/test完全排除。
2. 对7个源用户u循环：
   - fit=train_ids中user!=u；
   - probe=train_ids中user==u；
   - 只在fit上拟合phi和teacher目标的标准化；
   - Ridge(alpha=10.0)多输出回归，首版固定alpha；
   - 对probe预测，逆标准化回固定teacher坐标；
   - 保存实际目标、预测、源用户及fit索引。
3. 得到每条外层训练样本恰一次折外预测。
4. 对每个目标坐标计算跨用户预测误差与类均值控制。

Ridge只筛选，不在测试时运行；学生实际预测器仍为神经head。轻量筛选的负结果不能证明不存在非线性可预测性。

### 7.3 相对类均值的增量预测

对内层fit，计算每类teacher目标均值mu_c；不要用probe标签拟合mu_c。

构造残差：
r_teacher=z_T−mu_y，
r_phi=phi_R−mean(phi_R|y,fit)。

在fit残差上另外训练ridge，probe按fit类均值中心化后评估。标签仅用于训练期诊断/目标筛选，不作为部署输入。

每用户、每坐标定义：
skill(u,k)=1−SSE_predict(u,k)/(SSE_classmean(u,k)+epsilon)。

SSE_classmean用probe目标与fit类中心的距离，不是probe自身均值。避免把容易预测类别误判成学会类内结构。

候选筛选起点：
- 至少5/7源用户skill>0。
- 平均skill>0。
- 分母方差不能接近数值零。
- 按mean_skill−0.5std_skill排序。
- 选前32维；不足32则选实际通过数量并做匹配维度控制；少于8则停止该首版。
这些是可审计工程阈值，不是可观测性定理，不能在外测后修改成正结果门槛。

### 7.4 任务信息与来源局限

可预测目标可能只反映用户习惯或背景。因此在源用户内部，用选中目标训练简单分类探针并比较随机同维、PCA同维。目标类别信息弱则不扩大训练。

固定外层teacher本身已见过source用户，因此“OOF回归”不能声称teacher没见过该用户。上述得分用于分量筛选，不是泛化性能结论。若对teacher也要求严格OOF，需要独立内部教师并统一潜在坐标，成本与方法复杂度显著增加，本计划首版不做。

### 7.5 标准化、权重与学生head

选中坐标后，用全部外层train_ids拟合选中目标的mu/std，保存。
w_k=max(mean_skill_k,0)，归一化到平均1，再clip到[0.25,4]，重新归一化。避免因为维度更多或整体权重更大带来损失尺度变化。

学生aux输入B×64；MLP64→128→d，d为筛选维度。目标t=(z_selected−mu)/std。
Loss逐维Huber后按w加权平均，再按样本平均：

\[
L_C=L_{\rm base}+ramp\times0.1
E_i\left[\frac{\sum_k w_k\,Huber(\hat t_{ik},t_{ik})}{\sum_kw_k}\right].
\]

推理删除此head。原分类器不读取真实y、teacher目标、phi_R回归结果或剩余两路。

### 7.6 C的完整对照

|tag|目标|
|---|---|
|C-FULL|672维全部teacher特征，损失按维平均|
|C-RAND|随机选相同d个坐标，多随机种子|
|C-PCA|train拟合PCA到d，输出白化尺度匹配|
|C-PRED|筛选d维，均匀权重|
|C-WEIGHT|筛选d维，加上述权重|
|C-CLASS|只用train类中心作为目标|
|C-US|仅Right训练教师执行同样筛选|
|C-SHUF|同类实例错配筛选后目标|

C-FULL比其他臂多输出参数，必须报告；真正筛选机制主要看C-PRED vs C-RAND/C-PCA/C-CLASS，均匹配d和head大小。

若C-PRED只比C-FULL好、与C-RAND相同，最多说明降维有益；若与C-CLASS相同，不能说迁移了实例细节。只有辅助拟合指标变好而分类无改善，则停止。

## 8. D：跨视角掩码潜在预训练——给出可实现的双路径版本

### 8.1 先解决教师目标宽度与学生宽度不同

teacher输入B×3×90×189；student B×1×90×378。二者距离token数量不同，不能直接逐token MSE。

首版只预测教师粗时间token：
- teacher每6帧时间token平均，得到B×3×15×32。
- 同一时间段按L/T/R拼接，得到B×15×96。
- 按训练数据fit每视角32通道mu/std。
- 单视角目标控制重复Right32通道3次形成96维，保持输出head维度一致。
- 严格teacher-single控制还要换成Right-only训练teacher。

不通过教师训练一个新的可变融合网络；目标分支固定，规避潜在目标坍塌和额外归因问题。

### 8.2 掩码究竟放在哪里

直接在RF上置零某段，会同时改变所有距离token；直接使用已缓存全片spatial tokens又可能通过它们暴露被mask的时间值。因此必须明确student真的看不到哪部分。

首版采用“预处理完成后的双通道矩阵遮挡”，数据处理步骤：
1. 先按原方案计算完整单视角A/M。
2. 随机选15个时间段中的5段，两个输入块同时置零这些行。
3. 额外可选3个9-bin连续距离块；幅度的r与相位的189+r成对置零。
4. 对遮挡后的输入不重新逐样本标准化。
5. student的temporal与spatial两分支都从同一个遮挡输入重新编码，不能缓存未遮挡分支。
6. 只预测被遮挡时间段的teacher时间潜在目标。

这属于**派生表示上的掩码**。由于M涉及相邻帧且full-clip clutter removal使用全片信息，它不能被称为原始RF信息严格不可见。若进一步研究严格局部可见性，应先在RF上mask、处理相位跨边界依赖并重新计算预处理，作为独立更严格控制；两种mask协议不可混称。

默认mask是5/15=33.3%而非恰好30%，总稿的30%只是量级起点。距离mask首版关闭，使比较聚焦时间。

### 8.3 预测器与损失

student遮挡输入得到90×32时间token，按6帧平均成15×32。
预测器：
- 输入15×32。
- 两层小Transformer，dim32、heads4、ffn128、dropout0.1。
- 位置编码为固定15位置。
- Linear32→96得到每段目标。
- 不将teacher输入喂给student predictor。
- loss只对被mask段求均值，使用标准化目标Huber。

L_D_pre=mean_masked Huber(pred,target)。
目标teacher no_grad并固定，不需要EMA更新。预测器被丢弃后没有三视角重建路径。

**如果student使用原分类监督训练过的teacher目标，则这是有监督特权预训练/蒸馏，不是纯自监督。**仅仅预训练loss里没出现y，不能消除teacher先前使用y的事实。

### 8.4 训练与微调

- 预训练40epoch固定预算，lr1e-4；相同train_ids、mask种子、步数。
- 不在外层test上选择预训练epoch。
- 预训练时原分类head不参与优化；保存encoder与predictor。
- 微调丢弃predictor，将encoder装入学生，分类head使用统一新初始化。
- 微调最多40epoch、val CE早停。
- 预训练总成本与微调成本都报告，不能只计算最后40epoch。

低标签设置：
- 若teacher已用100%源标签训练，只给student10%标签，必须写“学生低标签、教师全标签”，不能声称总标注预算减少90%。
- 若要研究系统整体10%标签，teacher也只能用同一10%标签；其余源样本可用于无标签预训练，所有对照权限一致。
- 若要研究纯自监督多视角预训练，应另建不使用动作标签的目标学习框架，并检验坍塌；不属于固定教师首版。

### 8.5 D的实验臂与停止条件

|tag|预训练|
|---|---|
|D-NONE|从头分类，相同微调|
|D-LONG|单视角额外训练步数控制|
|D-US|仅Right训练teacher、Right目标|
|D-TS|多视角训练teacher，仅Right目标|
|D-MV|多视角teacher、三路时间目标|
|D-SHUF|同类跨实例目标错配|
|D-SELFREC|单视角掩码自身表征重建控制|

D-MV至少超过D-US/D-TS中相应更强控制才能讨论额外目标视角贡献。只有33%遮挡下辅助误差下降、分类无提升，停止。
“额外预训练40epoch”本身不是方法贡献，必须与计算预算控制比较。

## 9. A+B组合如何实施而不变成损失堆叠

前提：A-MV和B-MV分别超过各自单视角控制。

组合共享同一Right编码器，保留两个小head：
- A head读取双路径均值64维。
- B head读取15×32时间token。
- 分类器读取原14976维特征。

总损失：
L=L_base+ramp×(0.1L_con+0.1L_rel+0.1L_activity+0.05L_CDF)。

这组lambda只是组合起点，不能推断单独最佳系数相加仍最优。先用固定值测一次，然后仅在严格源用户验证允许的小预算内调整整体aux倍率{0.5,1}，不要每项开大网格。

必须跑：
- base；
- A；
- B；
- A+B；
- A+self-process；
- single-relation+B。
最后两臂用于判断组合中的每一项是否确实需要三视角。

诊断梯度：
可记录各aux对encoder的梯度范数和与CE的余弦，相同模型状态下计算。负余弦只是优化冲突诊断，不自动触发根据测试表现调整权重。若引入梯度手术，则作为新的独立算法及对照，不悄悄塞入最终方法。

## 10. 训练配置、命名和文件布局

### 10.1 首轮配置示例

以下是计划配置，不是现成可运行CLI。

~~~yaml
protocol:
  dataset: uwb_gestures_quality9300
  class_outputs: 12
  held_subjects: [1, 2, 3, 4, 5, 6, 7, 8]
  split_seed: 2026
  model_seeds: [2026, 2027]
  deployment_view: Right
  test_access: single_view_only

student:
  input: amplitude_phase_concat
  shape: [1, 90, 378]
  dim: 32
  depth: 4
  heads: 4
  pool: flatten
  cross_view_attention: false
  preserve_original_classifier: true

optimization:
  optimizer: Adam
  lr: 0.0001
  weight_decay: 0.0001
  effective_batch: 128
  microbatch: 32
  max_epochs: 40
  min_stop_epoch: 12
  patience: 8
  selection: validation_hard_ce
  auxiliary_ramp_epochs: 5

classification:
  hard_ce: true
  global_soft_target: true
  temperature: 2
  soft_weight: 0.5
  true_mass_source: matched_fold_global_matrix

method_a:
  projector: [64, 256, 128]
  relation_group: 32
  classes_per_group: 4
  users_per_group: 4
  instances_per_class_user: 2
  tau_teacher: 0.2
  tau_student: 0.2
  lambda_contrast: 0.1
  lambda_relation: 0.1

method_b:
  segments: 15
  frames_per_segment: 6
  process_head: [32, 64, 1]
  quality_mode_initial: equal
  lambda_activity: 0.1
  lambda_cdf: 0.05

method_c:
  teacher_candidate_dimensions: 672
  radar_probe_dimensions: 420
  ridge_alpha: 10.0
  selected_dimensions: 32
  minimum_eligible_dimensions: 8
  positive_source_users_required: 5
  lambda_prediction: 0.1

method_d:
  pretrain_epochs: 40
  masked_time_segments: 5
  total_time_segments: 15
  range_mask_initial: false
  predictor_dim: 32
  predictor_depth: 2
  target_dimensions: 96
  teacher_update: frozen
~~~

配置不应默认同时启用A/B/C/D。每次run必须有一个明确method_tag，互斥检查后才执行。

### 10.2 新目录建议

以下路径都是未来拟创建，本文不创建Python脚本：

D:/uwb_gestures/experiments/p1_20260905/mv_transfer_v2/

|拟建文件|唯一职责|
|---|---|
|protocol.py|manifest、split校验、fold访问边界|
|model_adapter.py|原分类输出适配与aux heads|
|sampling.py|可复放跨用户关系组|
|teacher_targets.py|冻结教师特征提取与来源记录|
|relation.py|A的mask、SupCon、关系KL|
|motion_targets.py|B的RF强度、时间掩码、目标融合|
|process.py|B的预测头/损失|
|predictability.py|C的OOF ridge、残差指标、坐标筛选|
|masked_pretrain.py|D的mask、predictor、预训练|
|train_student.py|统一A/B/C与base优化/early stop|
|run_queue.py|计划任务队列、失败状态、断点|
|audit.py|缓存权限、梯度、推理输入与指标|
|summarize.py|用户配对统计与结果表|
|configs/|锁定配置|
|tests/|边界测试|
|runs/|新模型与预测，不覆盖results_clean|

共同模块只实现一次。不要为每个lambda复制train.py，以免不同实验悄悄使用不同早停和采样。

### 10.3 每次run保存什么

目录名示例：A-MV_split2026_model2026_h1。
必须保存：
- config.json：全部解析后的参数。
- provenance.json：第4节账本。
- split.npz：train/val/test绝对sample_id。
- sampling.npz：每epoch的关系batch或原始permutation。
- metrics.jsonl：每epoch分项loss、train/val指标。
- best.pt：实际被选中的checkpoint。
- predictions.npz：test ids/y/logits/pred。
- validation_predictions.npz：最终best在val预测。
- deployment_state.pt：只含允许部署模块。
- status.json：pending/running/complete/failed及错误原因。

不能用“模型文件存在”判断任务完成。只有best、预测、完整指标、审计标记一致才记complete。若代码/缓存hash变化，不静默复用旧结果。

## 11. 超参、统计与泛化声明

### 11.1 先冻结，再看外层结果

建议先执行默认值，不进行大规模超参搜索。

确需搜索：
- 先确定两个源用户内部验证任务，选择只基于其余源训练用户。
- teacher/所有fit模块必须相应重训或明确该验证不是严格未见用户。
- 只允许一次改动一项：lambda候选{0.03,0.1,0.3}、温度候选{0.1,0.2}等。
- 所有方法搜索预算相同。
- 记录全部候选，不能只记录获胜者。
- 外层结果一旦看过，再修改方案的版本属于新探索，不能当预注册确认。

已经多次探索同一8名用户，新增seed不构成外部独立确认。设计严格只能减少新增偏差，不能抹掉历史探索。

### 11.2 主指标与配对效应

每折acc=correct/n_h，主结果8折等权均值；同时给macro-F1。

对同fold/seed的候选和baseline：
delta_h=acc_candidate_h−acc_baseline_h。
两个seed先在每个h平均delta，再对8个用户重采样bootstrap。不能把16个seed×用户当16名独立用户。

建议报告：
- mean_delta_pp；
- 95%用户bootstrap CI，10000次，固定统计seed；
- positive_users/8；
- 每用户全部accuracy；
- 最差用户值，作为描述指标；
- 新增正确、原正确变错、净变化；
- 部署参数、时延、峰值显存。

加入质量掩码后不同折类别不完整，需要在表下注明，不宣称每个用户都是完整12类测试。

### 11.3 研发门槛与科学结论分开

继续沿用：
- 相对最强匹配单视角控制平均至少+2pp；
- 至少6/8用户改善；
- 第二model_seed复验方向一致；
- 不明显牺牲macro-F1/少数用户；
- 输入权限与来源审计通过。

+2pp是研发投入门槛，不是显著性定义。CI是否跨零单独报告。若+1.2pp稳定，不称“无效”，可保留小幅辅助；但不能自动升级为主创新。

不同来源的收益应拆开：
- representation_gain：幅相vs幅度。
- privileged_gain：多视角版本vs同输入单视角监督版本。
- regularization_gain：base soft-target vs CE。
- compute_gain：更多步数/预训练带来的收益。
主论文要展示privileged_gain，而不是将这些全加起来称三视角贡献。

### 11.4 扰动与跨用户不是同一结论

可做测试时合成扰动：
- 附加幅度噪声。
- 局部距离bin遮挡。
- 短时帧丢失。
- 轻微预处理变化。

每种扰动定义作用在RF还是派生图，固定强度，并给所有方法相同样本扰动。测试结果仅说明合成鲁棒性，不等于真实新房间/新安装位置泛化。

若目标仍是“跨用户”，主证据必须来自LOSO，不用随机划分或扰动分数代替。

## 12. 最小实验矩阵与预算顺序

### 12.1 先做机制诊断，不启动四套全量矩阵

阶段P0：
- 模型适配数值一致性。
- source-only目标缓存。
- A关系退化诊断。
- B时间一致性与零能量处理。
- C轻量可预测性诊断。
- D暂不预训练。

这些不是新增模型效果结果。

阶段P1：
- A-GS/A-SC/A-US/A-MV，固定Right、model_seed2026、8折。
- B-GS/B-SELF/B-MV，固定Right、model_seed2026、8折。
若A-GS/B-GS采样器不同，不能合并为同一基线；二者训练预算各自匹配。

阶段P2：
- 只给有效候选补错配、target-single、机制消融和第二seed。
- A/B有独立收益才做组合。
- 不同时打开C/D来“挑一个最好”。

阶段P3：
- 主候选扩到L/T。
- 固定部署视角主张与任意视角主张分开。
- 是否另做低标签，取决于D或A的具体研究问题，不为增加表格强加实验。

### 12.2 算力估算方式

不在文档承诺“几小时一定完成”。先跑一个完整源内试验记录：
- teacher目标提取秒数；
- 每训练epoch秒数；
- best epoch和实际停止epoch；
- eval时间；
- RAM/GPU峰值。

总预算估计=sum(计划折数×每折实测中位耗时)+缓存/审计时间，保留20%异常缓冲。共享teacher缓存只计一次。不要将本机不同同时运行任务的时间混作算法加速证据。

8GB显存首版推荐一次一个训练进程；32 microbatch×4梯度累积。需要并发时先实测显存余量，不把旧能并行两任务直接推广到加入预测器的模型。

## 13. 实现任务与验收用例

以下是未来实现的工作清单，不是本轮已完成的代码。每项先写能暴露错误的边界用例，再写最小实现。

### Task 1：协议与模型适配

- [ ] 从pipeline.split生成manifest，校验9300有效样本在8个test折恰好覆盖一次。
- [ ] eval时原Classifier与classification_features在相同输入上logits一致。
- [ ] 检查teacher/student token形状，确认378为特征拼接。
- [ ] 各臂主网络初始hash一致。
- [ ] 随机打乱test的隐藏L/T数组，student输出应逐元素不变。
- [ ] 仅teacher_multi读取L/T，student推理函数参数不能接收full_triplet。

### Task 2：A最小闭环

- [ ] 采样器输出32组结构且同组无重复sample_id。
- [ ] 每anchor有跨用户正对，缺类用户不会被错误采入。
- [ ] 全部同一用户输入必须报错，而不是除零产生nan。
- [ ] 相同关系目标与student表示时KL接近0。
- [ ] 修改teacher的同用户邻接不应改变跨用户关系loss。
- [ ] 允许不同teacher/student特征维度，关系矩阵仍为B×B。
- [ ] teacher梯度None，student encoder梯度非零。
- [ ] teacher-single和target-single的provenance不同。
- [ ] 完成source内单epoch smoke后才准入8折队列。

### Task 3：B目标与损失

- [ ] 能量只在第3段时a主要集中第3段，CDF单调且终点1。
- [ ] 纯零输入valid=False，辅助loss有限且为0，不删分类样本。
- [ ] 含invalid frame的段按有效帧数平均，不当作真实0运动。
- [ ] 三路相同输入时mean融合等于self。
- [ ] 正负相位能量取绝对值后不互相抵消。
- [ ] 改变val/test数据不应改变训练scale/shift/targets。
- [ ] B-TIME由打乱a重新生成CDF，保证数学一致。
- [ ] 删除过程head后分类logits不变。
- [ ] 配对/时间审计不过时拒绝进入time-supervision队列。

### Task 4：C筛选器

- [ ] 合成目标含可预测维与独立噪声维，预测skill能区分。
- [ ] 只有类别中心信息时，残差预测不能虚报稳定细节。
- [ ] 每probe用户不进入该ridge/scaler fit。
- [ ] 全局y=0—11保留，缺类fit要显式处理，不错位索引。
- [ ] 选中坐标/标准化由train拟合，test改动不影响结果。
- [ ] 无足够维度通过时返回明确停止状态。
- [ ] PCA对照不跨内折混平均不同主成分编号。

### Task 5：D预训练

- [ ] 同一时间mask同时作用到幅度/相位及两编码分支输入。
- [ ] teacher目标是15×96，student输出相同维度。
- [ ] 非mask位置不进入预训练误差。
- [ ] teacher固定，无梯度；student encoder有效梯度。
- [ ] 导出微调classifier不携带teacher/predictor。
- [ ] 所有预训练ID属于source train。
- [ ] 低标签配置记录teacher实际用过的label_ids，防止标注预算错报。

### Task 6：结果审计

- [ ] 从预测argmax重算acc和macro-F1，与metrics一致。
- [ ] 比较臂train/val/test、init、采样manifest一致或差异明确。
- [ ] 成功计数必须8折齐全，不用2折早期排名下结论。
- [ ] bootstrap以用户为单位，两seed先按用户聚合。
- [ ] 所有失败/中止任务留日志，summary不读取残缺checkpoint。
- [ ] 训练目标误差与分类指标引用同一模型，防止旧重建表格错配复发。

不在本轮提供可直接执行的run命令，因为这些Python入口尚未实现。文档中的函数是核心参考片段，正式代码还需集成、边界测试和日志后才可运行。

## 14. 如何把结果写成论文，而不是把方法名称堆在一起

### 14.1 研究问题—机制—证据一一对应

|要写的论点|必须出现的证据|
|---|---|
|三视角训练对单雷达有帮助|teacher-multi超过teacher-single/无教师控制|
|跨用户关系有效|A-MV超过A-SC及对应关系控制|
|传递类内结构|正确实例关系优于同类错配，不仅优于随机标签|
|过程信息有效|B-MV超过B-SELF，时间顺序控制有区别|
|质量权重有效|B-Q超过B-MV，并审查低质量样本分布|
|可预测分量筛选有效|C-PRED超过同维随机/PCA/类中心|
|预训练而非预算有效|D-MV超过同预算D-US/D-LONG|
|部署仍为单雷达|输入接口、导出模型和隐藏视角扰动测试|
|泛化改善|完整LOSO用户配对结果，不能仅随机split|

如果某个机制对照失败，论文可以保留总体方法效果，但相应机制解释要降级。不要因整体涨点就宣称每个模块都有效。

### 14.2 可能的章节结构

1. 单雷达跨用户识别问题与多视角训练条件。
2. 观测与标签层次：幅相输入、动作实例、动作过程。
3. 三视角监督构建：按最终成立的A或B展开。
4. 单雷达学习与训练/部署流程。
5. 统一LOSO协议与数据质量审计。
6. 主结果、单视角信息控制、机制消融。
7. 失败模式与迁移边界。
8. 与开放集识别接口。

若A/B组合不比单项更好，章节只保留更强的一项作为方法；另一项放探索/消融，不硬凑两个创新模块。

### 14.3 与第二、三部分的边界

第一部分全部是已知类分段识别；第二部分按已知/未知类别重新训练，第一部分看过的12类模型不能直接作为“从未见过未知类”的编码器。

第三部分部署连续流，需要重新处理因果归一化、窗口状态与提交规则；当前B的CDF基于整段总运动量，不能当实时已知的完成度。离线过程监督可以作为预训练知识，但在线推理必须只从可用前缀计算。

FMCW的接收天线通道与三台分开放置UWB雷达不是同一视角定义。跨数据集验证先建立信号/标签协议，不能将二者concat训练后混称同一9300协议。

## 15. 文献依据、借鉴范围与当前设计新增部分

本次没有宣称新的论文检索结果；沿用总稿已核对的原始论文。这里将每套实现与来源再对应：

- A：样本关系迁移借鉴[RKD，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)；多视角正对借鉴[WACV 2023多视角动作对比](https://openaccess.thecvf.com/content/WACV2023/papers/Shah_Multi-View_Action_Recognition_Using_Contrastive_Learning_WACV_2023_paper.pdf)。本项目的跨用户采样、雷达分视角目标与对应控制是待验证适配。
- 保留本视角信息的依据：[FOCAL](https://proceedings.neurips.cc/paper_files/paper/2023/hash/93e98ddf39a9beb0a97fbbe56a986c80-Abstract-Conference.html)与[FactorCL](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html)。首版没有实现其完整共享/私有分解，不能用其方法名给我们的简单projection贴标签。
- B：融合产生监督借鉴[RODNet](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_RODNet_Radar_Object_Detection_Using_Cross-Modal_Supervision_WACV_2021_paper.pdf)；时间对应思想借鉴[TCC](https://openaccess.thecvf.com/content_CVPR_2019/html/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.html)。本文15段运动强度/CDF目标是拟议的雷达过程伪监督，不是上述论文提供的真值。
- C：选择迁移对象的动机参考[RadarDistill](https://cvpr.thecvf.com/virtual/2024/poster/30749)、[C²KD](https://openaccess.thecvf.com/content/CVPR2024/papers/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.pdf)、[ICML2025信息瓶颈对齐](https://proceedings.mlr.press/v267/almudevar25a.html)。本计划的ridge残差skill筛选不是这些论文原算法。
- D：多模态预训练后单模态迁移参考[MultiMAE](https://multimae.epfl.ch/)；潜在特征预测参考[V-JEPA](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)。固定有监督teacher首版不可称纯自监督JEPA。
- 数据背景：[UWB-Gestures原始论文](https://www.nature.com/articles/s41597-021-00876-0)。本项目的9300样本是本地审计协议，不是原论文宣称的数据总量。

当前最值得验证的组合是“A的跨用户实例关系＋B的片段内运动过程”。它是否足以形成硕士论文第一部分的方法贡献，要看teacher-single与错配等关键对照，不能仅靠命名判断。

