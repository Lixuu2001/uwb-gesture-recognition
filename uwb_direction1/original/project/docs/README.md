# 多视角 UWB 开集手势识别 (MV-OSR)

在论文 *UWB Impulse Radar-Based Open-Set Gesture Recognition (Eom et al., IEEE IoTJ 2025)*
的单视角方法基础上,利用 **UWB-Gestures (Figshare 12652592)** 数据集里同步采集的
**三个雷达视角 (Left / Top / Right)**,把开集识别 (OSR) 做强。

核心思路:**闭集已饱和 (~97%)**,真问题是**拒识没学过的手势**。
主开集头是 per-class 的 **deep inspector**(每个手势、每个视角学一个原型,测试时按
"样本离原型的距离"判已知/未知);三视角的 per-view 距离一平均,把 OSCR 从单视角 ~83
拉到 **95.14**。在这之上加一条**跨视角生成式脊柱**:遮住一个视角、用另两个重建它,
重建误差 `e_recon` 当额外拒识信号(与 inspector 融合后再 +3.4 分),同一个解码器还能在
某个雷达失效时把它合成回来。

> 早期版本曾以"每视角证据 (Evidential) + Dempster-Shafer 融合"为主路,后经消融证明
> **多视角下不如 inspector**(三视角证据 81 vs inspector 92.6,RESULTS §6),现仅作对照
> 变体保留(`--osr-head evidential`)。当前主路 = **deep inspector**。

---

## 1. 环境

已在 `D:\D_conda\envs\pytorch` (Python 3.9, torch 2.7.1+cu126, 单卡 RTX 4060 8GB)。

```bat
set PY=D:\D_conda\envs\pytorch\python.exe
```

## 2. 数据

- 路径:`D:\uwb_gestures\12652592`,主结果用 **ClutterRemoved** 版本
  (单视角忠实复现用 **Raw**,见 §6;多视角下 Raw≈ClutterRemoved,RESULTS §D1)。
- 文件名规整、文件夹命名不一致 → 代码用 **glob + 正则解析文件名** 定位,不拼路径。
- 每 CSV `(9000,189)` → `reshape(100,90,189)`,第 i 个样本 = `data[i*90:(i+1)*90]`。
- 三视角按 **样本下标配对**(软同步,同一次手势的三个投影)→
  多视角单元 `X (N, 3, 90, 189)`,N≈9593(个别文件 99 样本)。
- **逐视角标准化**(三雷达振幅量级差 ~10x,必须归一);默认 `abs_std`(先取 |x| 再
  逐样本逐视角 z-score),匹配 Eom 的"基于幅度的识别"。

首次运行先建缓存(读 288 个 CSV):

```bat
%PY% main.py --build-cache
```

## 3. 跑实验

```bat
%PY% main.py --smoke                          :: 合成数据冒烟,验证管线(无需真数据)
%PY% main.py --pool flatten                   :: 主结果:三视角 + inspector(95.14 随机切分)
%PY% main.py --single-view --pool flatten     :: Eom 单视角忠实复现(86.74)
%PY% main.py --pool flatten --loso            :: 跨人留一交叉验证(84.34,更诚实)
%PY% main.py --pool flatten --mae-pretrain --lambda-recon 1.0  :: 拒识腿:e_recon ⊕ inspector
%PY% main.py --pool flatten --mae-pretrain --synth-view --view-dropout 0.5  :: 缺视角腿:虚拟视角合成
%PY% main.py --no-fusion                      :: 关掉跨视角注意力(融合已证伪,见 §8)
%PY% main.py --osr-head evidential            :: 证据 + DS 融合对照变体(弱于 inspector)
%PY% main.py --case 0 --epochs 30             :: 只跑 case 1,自定义 epoch
```

> 注意:`config.py` 默认 `pool=mean`(轻量,64 维)。**论文主结果用 `--pool flatten`**
> (8928 维),per-view inspector 的距离几何更靠它。

每次运行**同时输出多套 OSR 指标**:主头(`inspector`)与基线(`msp`,朴素 softmax 阈值),
便于直接对比。指标:Accuracy / Macro-F1 / AUROC / OSCR(5 case 均值±标准差)。

## 4. 代码结构

```
config.py              全局配置 (dataclass) + Eom Table I 的 5 个 OSR test case
main.py                入口:argparse 覆盖配置,跑 5 case / LOSO,出汇总表
uwb_osr/
  data.py              CSV glob/解析、缓存、abs_std 归一化、OSR/LOSO 切分、合成数据
  model.py             共享双路编码器 + 跨视角融合(已证伪) + 闭集头 + 证据头/DS(对照) + 顶层网络
  inspector.py    ★    主开集头:deep inspector(per-class 原型+距离),多视角异常分聚合
  pretrain.py     ★    B2 跨视角掩码重建 + e_recon 拒识信号
  synth.py        ★    S4 虚拟视角合成(解码器当生成器,补回失效雷达)
  engine.py            Stage-1 训练 + 视角 dropout (R1) + 跨视角自蒸馏 (KD)
  losses.py            交叉熵 + EDL/TMC 证据损失(对照) + 监督对比
  metrics.py           纯 numpy:AUROC / OSCR / Macro-F1 / 最佳阈值扫描
probe_attributes.py    S1/S2 属性组合泛化探针(论文二素材)
tools/run_awake.py     长跑防睡眠 + 防会话杀进程
```

## 5. 方法 → 代码 → 消融 对照

| 蓝图模块 | 代码位置 | 开关 | 对应消融 |
|---|---|---|---|
| Siamese 双路编码器 | `model.SharedDualPathEncoder` | 始终 | vs 单视角 (`--single-view`) |
| **deep inspector 开集头(主)** | `inspector.py` | `osr_head=inspector`(默认) | vs MSP(同表自动对比) |
| 跨视角掩码重建 + e_recon | `pretrain.py` | `--mae-pretrain` | standalone / 融合 (`--lambda-recon`) |
| 虚拟视角合成 | `synth.py` | `--synth-view` | vs 真三视角 / 纯缺视角 |
| 视角 dropout 鲁棒 | `engine._sample_kept_views` | `--view-dropout` | `--ablate-views` 读降级矩阵 |
| 证据 + DS 融合(对照) | `model.EvidentialHead`/`ds_combine` | `--osr-head evidential` | 多视角下弱于 inspector |
| 跨视角注意力融合(已证伪) | `model.CrossViewFusion` | `--no-fusion` 关闭 | F0–F6 负结果链 |

## 6. 实验阶梯(论文主线)

| 编号 | 命令 | 作用 |
|---|---|---|
| 单视角复现 | `--single-view --pool flatten --data-type Raw` | 忠实复现 Eom 单视角 **86.74 ± 1.29** |
| 主结果 | `--pool flatten` | **三视角 + inspector,OSCR 95.14 ± 0.81**(随机切分) |
| 跨人诚实基线 | `--pool flatten --loso` | LOSO **84.34 ± 2.80**(随机切分有 session 泄漏,−10.8) |
| 拒识腿 | `--pool flatten --mae-pretrain --lambda-recon 1.0` | e_recon ⊕ inspector,LOSO **88.39 ± 2.56**(比 inspector +3.42,正交互补) |
| 缺视角腿 | `--mae-pretrain --synth-view --view-dropout 0.5` | 虚拟视角合成,对公平基线 R1 **≈打平**(卖点=统一性 + 选择性合成) |

## 7. 调参与注意事项

- **主结果用 flatten(8928 维)不是 mean** —— 对 per-view inspector 的距离几何关键。
- **inspector Stage-2 batch=256**(默认),不要照搬论文的 16:在 flatten 8928 维下小 batch
  原型更新过噪、会过拟合(RESULTS §5,−11 分)。
- 缺视角鲁棒:共享权重 + DS 支持任意视角子集 forward,改 `cfg.views` 或加 `--ablate-views`
  即可测缺视角降级矩阵。
- 证据头(EDL,对照变体)若启用,需完整 epoch 预算(≈150)+ KL 退火才会反超 MSP;
  低 epoch 下 MSP 暂时领先属正常(EDL 早期被 KL 正则压低证据)。
- 资源:单卡即可;真数据 5 case × 150 epoch 约数小时;LOSO 每折 75–100 min,按 case 拆跑。

## 8. 关键结论与已知偏差(相对 Eom 原文)

- **主协议建议 LOSO**:随机切分(95.14)有 session 泄漏,跨人诚实代价约 −11 分(→84.34)。
- **架构级融合是死路**:`CrossViewFusion` / MBT 瓶颈融合对开集无增益、LOSO 下更差
  (F0–F6 负结果链,见 `FUSION_DESIGN.md`),默认 `--no-fusion` 行为之外不作主线。
- **多视角增益来自距离均值,不是分歧方差**:E4 已证视角间 disagreement(var)不提供独立
  信号,增益本质 = per-view inspector 距离的降噪 + 几何互补(RESULTS §E4)。
- **S4 合成对公平基线 R1 ≈打平**(非"+26",那是对没训缺视角的稻草人);真卖点 = 同一条
  生成式脊柱不需专门训练就追平 R1 + 可按视角重要性选择性合成。
- 特征码默认 `mean`(64 维,轻量),主结果切 `flatten`(8928 维,贴 Eom 的 c=D(T+M))。

> 完整实验编号与数字以 `RESULTS.md` 为唯一真源;当前状态看 `HANDOFF.md`;
> 融合层级分析看 `FUSION_DESIGN.md`;两篇论文路线看 `RESEARCH_PLAN.md`。
