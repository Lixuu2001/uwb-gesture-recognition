# Fishr 完整开发六人 LOSO：统一评分与成本

准确率提升证据：用户等权 present-class macro recall，GAP64 93.6672%，Fishr 93.5737%，配对平均差 -0.0934pp；严格改善 1/6。未通过预设花费继续门槛；关闭这一确切配置，不依据测试结果救参。

原创贡献证据：尚未验证。Fishr是[Ramé、Dancette、Cord，ICML2022/PMLR162](https://proceedings.mlr.press/v162/rame22a.html)的已发表方法，本试验迁移固定DomainBed默认1000/.95/1500组件，使用native GAP617612、三雷达/RMS、B64、64轮、seed20261029。分类项保持原native全局BCE；全部18060个MLP分类器参数的逐例梯度人口方差、作者固定分母EMA.95、warmup0/1500及一次完整nativeoptimizerreset1500均已声明。相对GAP的差异还包含reset，正结果仍须经过同globalBCE/同reset1500/无variance匹配对照后讨论variance归因；BCE、混合source sampler与骨干是迁移设定，不声称作者完整图像benchmark复现或原创。

| 测试用户 | 样本数 | GAP accuracy | GAP present-macro recall | Fishr accuracy | Fishr present-macro recall | 配对差(pp) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1200 | 98.4167% | 98.4167% | 97.7500% | 97.7500% | -0.6667 |
| 2 | 1000 | 98.7000% | 98.7000% | 98.2000% | 98.2000% | -0.5000 |
| 3 | 1200 | 90.0000% | 90.0000% | 90.0000% | 90.0000% | +0.0000 |
| 4 | 1200 | 94.2500% | 94.2500% | 91.5833% | 91.5833% | -2.6667 |
| 5 | 1200 | 96.0000% | 96.0000% | 95.0000% | 95.0000% | -1.0000 |
| 6 | 1100 | 84.6364% | 84.6364% | 88.9091% | 88.9091% | +4.2727 |

accuracy是全部样本正确率；present-class macro recall是仅对真实support>0类别平均的召回率。用户2缺0-based类别10/11，用户6缺类别3，其他用户12类齐全；所有现有类别各100条，因此本数据逐用户两种指标恰好相同，定义仍不同。预测为缺失类别仍算错误。缺失类召回为null；完整每类support/两方法recall见[PER_CLASS_SUPPORT_AND_RECALL.csv](PER_CLASS_SUPPORT_AND_RECALL.csv)。逐用户完整数值/选定epoch/成本见[PER_USER.csv](PER_USER.csv)。

用户等权accuracy：GAP 93.6672%，Fishr 93.5737%。按样本加权pooled accuracy分别为93.6522%/93.5072%，不与用户等权均值混称。最差用户present-macro分别为84.6364%/88.9091%；最差配对变化-2.6667pp。

完整64轮生产和各自fresh64重放、所有原生审计/角色ID隔离/验证选模、全部77块561095162字节1984载荷私有raw/allmember SHA冻结及精确Git发布完成后，才进行一次固定无标签推断成本及一次完整cohort真值配对。所有模型只由各自验证next用户选择，单人测试分数没有决定其他折是否继续。原始模型、预测、日志、ID私有索引和全部正负结果保留。统计范围为一个seed、6名重复开发用户；用户7–8未用于新方法选择，历史全8人已有开发，不称独立新用户验证或完整八人确认。

生产总wall/CPU 14779.878592/14779.175423秒；独立重放14691.573768/14686.370003秒；输出审计66.136240/64.978842秒。主阶段STATE总wall/CPU 29559.943319/29599.386026秒，已包含子阶段，不能再相加。此前MMD全部wall/CPU30865.434969/30891.040083秒完整另计，旧40.6258%失败结论不改；此前VREx全部wall/CPU31157.163367/31189.589143秒另计，92.8444%原固定配置失败结论保留。

固定一次无标签推断：GAP 2.108361ms/样本，Fishr 2.086092ms/样本（1线程/B64，各6900条；模型及BN不变、冻结logits逐位一致）。这是一次固定计时，不作为统计速度改进证据。完整分用户/CPU/模型SHA/归档构建与回读成本见[COST_REPORT.json](COST_REPORT.json)；构建child总CPU未独立测量，parentCPU不能当总值，传输收据跨度与其他只读工作重叠。

门槛预先固定为均值至少+0.5pp、至少4/6严格改善、最差配对不退超2pp；门槛不是任何正负证据是否存在的定义。原MMD、旧混合94.6290%不同预算及旧失败结论不追改。当前route：`close_exact_Fishr_LOSO6_lambda1000_ema0p95_anneal1500_nativeGAP64_seed20261029_no_rescue`。原创贡献尚未验证；先按已冻结route处理本结果，保持现有自动任务启用/频率和云环境；后续继续文献支撑的具体实现与真实实验，不调用旧held1 once、不扫描测试集或修改本配置。

逐用户图：[LOSO_COMPARISON.svg](LOSO_COMPARISON.svg)。
