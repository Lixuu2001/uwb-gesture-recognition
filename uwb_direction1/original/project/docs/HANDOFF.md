# HANDOFF — MV-OSR 项目当前状态(新会话从这里开始)

> 这份文档是冷启动入口:新开一个会话,读完这份就能直接接着干,不必回溯长上下文。
> 最后更新:2026-06-20(实验② S4 缺视角合成 5-case LOSO 已定稿 → §29;两条腿均落地,下一步=纯写作)。
> 配套文档:[RESEARCH_PLAN.md](RESEARCH_PLAN.md)(两篇论文路线)、
> [RESULTS.md](RESULTS.md)(全部实验编号 §1–§27,数字唯一真源)、
> [FUSION_DESIGN.md](FUSION_DESIGN.md)(融合层级分析)、
> [paper/mvosr_paper.tex](paper/mvosr_paper.tex)(论文一草稿)。

---

## 0. 一句话现状

融合大方向已查到底:**架构级融合证伪(F0–F6),转目标函数级(生成式 B2)**。
**论文一两条腿都已落地,转入收尾/写作阶段**(GPU 空闲):
- 拒识腿(§28):e_recon standalone 与 inspector 打平,但**融合 = 88.39,比 inspector +3.42 全赢**(正交互补)。
- 缺视角腿(§29):虚拟视角合成可行,对公平基线 R1 **≈打平**(+0.19),**卖点=统一性 + 选择性合成**(助Left损Top)。
下一步 = 把 §27/§28/§29 写进 `paper/mvosr_paper.tex`(纯写作,非新实验);之后 S1 已绿可启动论文二。

---

## 1. 正在跑的实验

**无 —— GPU 空闲**。S4 LOSO 诚实版已 5/5 跑完并定稿(2026-06-20,详见 RESULTS §29.3)。

### ✅ S4 缺视角合成 5-case LOSO 定稿(诚实基线版,RESULTS §29.3）
配置 `--synth-view --view-dropout 0.5 --loso`(Stage-1 用 R1 训练,故 synth_drop_v=公平基线)。
5-case 均值 synth vs R1-drop(OSCR):

| 缺视角 | synth | R1-drop | Δ |
|---|---|---|---|
| Left | 79.20 | 77.11 | **+2.09**(合成有用)|
| Top | 74.75 | 77.18 | **−2.43**(合成加噪)|
| Right | 78.12 | 77.21 | +0.91(≈平)|
| 三视角平均 | | | **+0.19(≈打平)** |

- **结论**:对上公平基线 R1,合成**≈R1**(不是早筛"+26"那种碾压,那是稻草人 §29.2b)。
  **真卖点=统一性**(同一 B2 脊柱不需专门 dropout 训练就追平 R1)+ **选择性合成**(丢重要
  视角 Left 合成 +2、丢冗余视角 Top 合成 −2,按视角重要性/重建误差门控)。
- synth_real3 83.74≈inspector 83.83;drop 绝对值 ~77 是跨人 LOSO 诚实尺度(比 R1 §19.5 随机切分低~14)。

### ⚠️ S4 早筛绿灯但基线需修正(2026-06-19,RESULTS §29.1/29.2b)
- 合成替换三视角只掉 1.4–2.1 分(均<5),几乎追平三真视角上界 95.95 → 合成质量够用,这条成立。
- **但 "合成救回 26 分" 是对稻草人**(naive drop)。对上公平基线 R1(§19.2,vd0.5 的 2 视角 ≈93.5):
  S4 synth(93.86/94.17/94.58)**≈ R1**,顶多 +0.5,不是碾压。
- **S4 真正卖点 = 统一性**(同一 B2 脊柱不需专门 dropout 训练就追平 R1),不是性能更高。别写"+26 分"。

### ✅ 实验① e_recon @ LOSO 已定稿 + λ 融合(2026-06-17/06-19,详见 RESULTS §28)

全 5-case LOSO(OSCR %):

| | inspector | e_recon | msp | **e_recon ⊕ inspector(λ=1.0)** |
|---|---|---|---|---|
| 均值±std | 84.97 ± 2.68 | 84.06 ± 4.06 | 74.22 ± 3.92 | **88.39 ± 2.56** |

- **standalone 打平**:e_recon ≈ inspector(84.06 vs 84.97,在 std 内),原"e_recon 更抗跨人"硬卖点不成立。
- **但融合是真增益(主结论)**:e_recon ⊕ inspector(λ=1.0)= **88.39,比 inspector +3.42 且 5-case 全赢**
  → 两头正交互补(与 E4/E5 的冗余相反),这是拒识腿的**真卖点**。最优 λ≈1.0(0.3 几乎等同),见 §28.6。
- B2 init 对 inspector 跨人≈无用(84.97 vs S3 无 B2 84.34,噪声内)。
- 工程坑:case-3 首跑中途挂掉(6/8 折),`s_erecon_loso_c3redo.bat` 重跑补全(§28.4)。

---

## 2. 论文一进度清单

| 部件 | 状态 | 出处 |
|---|---|---|
| 主结果:三视角 vs 单视角 OSCR | ✅ 95.14±0.81(随机切分) | RESULTS §2-3,论文 §4.3 |
| 融合策略系统比较(F0–F6,架构级证伪) | ✅ | RESULTS §20-23,论文 §4.4 |
| 缺视角鲁棒(R1 view-dropout) | ✅ 随机切分强阳性 | RESULTS §19,论文 §4.5 |
| 主协议跨人基线(S3) | ✅ **84.34±2.80**(三视角 LOSO) | RESULTS §27 |
| 单视角忠实复现 + 因素分解 | ✅ 86.74±1.29 | RESULTS,论文 §4.2 |
| 属性组合泛化(S1,垂直涌现/水平失败) | ✅ | RESULTS §24-26,论文 §4.7 |
| **e_recon @ LOSO + λ 融合**(实验①) | ✅ 定稿:融合 88.39±2.56,比 inspector +3.42 全赢(正交互补) | RESULTS §28 / §28.6 |
| **虚拟视角合成 S4**(实验②) | ✅ 5-case LOSO 定稿:合成≈R1(统一性卖点),助Left损Top | RESULTS §29 |
| e_recon 随机切分早筛 | ✅ AUROC 95.48 / OSCR 95.01 | RESULTS §18 |

论文一草稿已编译(`paper/mvosr_paper.tex`,16 页,0 错误)。图 1 = 主架构 PNG;
新增 fig6 = 融合变体三面板图;融合表已规范化(同 case-0、分随机/LOSO 协议)。

---

## 3. 研究路线:两篇论文(详见 RESEARCH_PLAN.md)

- **论文一(扎实,进行中)**:跨视角生成式模型 = 统一拒识(e_recon)+ 缺视角补全,
  主协议 LOSO。不需要 JEPA。脊柱价值靠实验①②证明(都和"修水平组合"无关)。
- **论文二(冒险,待论文一立住)**:从"拒识未知"到"描述未知"——组合语义(CZSL,
  柱二 S1 已 GREEN:垂直组合 73.5% 且多视角涌现;水平轴失败是 limitation)+ 少样本
  注册 + conformal。需 z(t) 潜在轨迹 + JEPA。
- **论文三(保险)**:注册 + conformal,不依赖生成模型,随时能独立成篇。

**关键已定结论(别重走)**:
- 架构级融合(CrossViewFusion、MBT 瓶颈)对开集无增益、LOSO 下更差 → 死路,别再试。
- 水平轴组合失败是传感模态内在性质(目标函数和输入符号都救不了,S2/S2b 已排除)→
  别再为它砸 JEPA,scope 成 limitation。
- 多视角增益本质 = 三个半独立 inspector 的距离平均(E4),不是分歧 var,不是融合。

---

## 4. 下一步 = 论文一写作任务卡(新会话冷启动从这里干)

**所有实验已跑完,这一步是纯写作:把三块新结果写进 [paper/mvosr_paper.tex](paper/mvosr_paper.tex)。不需要再跑 GPU。**

要写进去的三块(数字以 RESULTS.md 为唯一真源):

1. **跨人主协议 S3(RESULTS §27)**:三视角无融合 LOSO 5-case = **84.34 ± 2.80**;
   随机切分 95.14 → 跨人代价 −10.8。作为论文一的诚实主结果 + 回应 session 泄漏质疑。

2. **拒识腿 = e_recon ⊕ inspector 融合(RESULTS §28 / §28.6)**:
   - standalone:e_recon(84.06)与 inspector(84.97)跨人下**打平**(原"更抗跨人"卖点不成立,别写)。
   - **融合(λ≈1)= 88.39 ± 2.56,比 inspector +3.42,5-case 全赢** → 两信号正交互补(与 E4/E5 冗余相反)。这是拒识腿的卖点。

3. **缺视角腿 = 虚拟视角合成 S4(RESULTS §29 / §29.3 / §29.3b)**:
   - ⚠️ **别写"合成救回 +26 分"**——那是对稻草人(没训缺视角的病态模型,§29.2b)。
   - 公平基线是 **R1(训练时 view-dropout)**;诚实版 LOSO 下合成 **≈ R1**(三视角平均 Δ=+0.19)。
   - **卖点 = 统一性**(同一 B2 脊柱不需专门 dropout 训练就追平 R1)**+ 选择性合成**(丢重要视角 Left 合成 +2.1、丢冗余视角 Top 合成 −2.4,按视角重要性/重建误差门控)。

**写作风格**:遵守 memory「论文写作风格要求」(严谨不像AI、少括号破折号、别标榜"忠实复现"、
配置细节只放消融、不用内部代号 E4/F1/S4 等)。论文里 §29 用诚实基线,**不准出现"+26"**。

**相关代码/产物**:`uwb_osr/synth.py`(S4 合成,`--synth-view`);融合在 `pretrain.py:recon_osr_heads`
(`--lambda-recon`);论文当前 16 页已编译,图1=主架构、fig6=融合变体。

**(可选,论文一投出后)论文二**:S1 已绿,z(t) 潜在轨迹 + JEPA + 组合泛化,见 §3 与 RESEARCH_PLAN。

---

## 5. 工程约定(怎么跑实验,务必遵守)

- **环境**:`D:\D_conda\envs\pytorch\python.exe`(torch 2.7.1+cu126,单卡 RTX 4060 8GB)。
- **长跑必须脱离会话**:写 `.bat` 用 `Start-Process -WindowStyle Hidden` 启动,
  外面包 `tools\run_awake.py`(防 Windows 睡眠 + 防会话杀进程)。
  曾有一次 GPU 跑被外部杀掉,从此一律脱离会话跑。
  模板见 `s3_run.bat` / `s_erecon_loso.bat`。
- **GPU 跑**:命令里加 `--device cuda`(冒烟用 `--device cpu` 可与 GPU 跑并行)。
- **先冒烟**:真跑前先 `--smoke --device cpu` 验证管线不崩(合成数据,数字无意义)。
- **每个实验存档**:跑完把结果写进 RESULTS.md 新一节(仿现有 §N 格式:动机/表/判决/
  日志),数字以 RESULTS.md 为唯一真源。
- **进度监控**:别轮询 GPU 抢资源;用 `until [ -f .../X.done ]; do sleep 120; done`
  挂后台监视器等完成标记。
- **LOSO 成本**:每折 ~75–100min(case-0,8 折);加 B2 预训练(mae 100ep)每折 +~30min。
  5-case LOSO ≈ 全天量级,按 case 拆开跑(各自日志 + done 标记)。

## 6. 关键代码地图

| 文件 | 作用 |
|---|---|
| `main.py` | 入口,argparse 覆盖配置,跑 5-case / LOSO,出汇总 |
| `config.py` | 全局 dataclass + 5 个 OSR case 定义 |
| `uwb_osr/model.py` | 编码器 + (已证伪的)融合模块 + 检察器 + 顶层网络 |
| `uwb_osr/inspector.py` | deep inspector 开集头(主),逐视角距离聚合 |
| `uwb_osr/pretrain.py` | B2 跨视角掩码重建 + e_recon(实验①②的核心) |
| `uwb_osr/engine.py` | Stage-1 训练 + view-dropout(R1) |
| `probe_attributes.py` | S1/S2 属性组合泛化探针(--encoder closed|b2) |
| `paper/mvosr_paper.tex` | 论文一草稿 |
