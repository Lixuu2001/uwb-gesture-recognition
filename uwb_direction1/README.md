# UWB 手势方向一：云端接续入口

最新：[融合头四臂八折与CPU可行性报告](results/cloud_20260930_heads/RUN_REPORT.md)。32次训练完成，最佳新读出93.3625%未过强基线门槛；CPU版PyTorch已安装，原始网络真实源数据反向探针通过。另保留[核方法筛查](results/cloud_20260930_kernel/RUN_REPORT.md)。当前无后台训练，研究目标尚未完成。

2026-09-30（北京时间）。用户要求在 `uwb_gestures` 上继续寻找能实现方向一的新方法，允许一部或三部雷达，不再将平均两路的预算门槛作为必要目标。**私有 Drive 的163个分块已全部下载，86个原文件已逐字节校验恢复；八折旧三路基线重算93.4417%。** 首项冻结特征先导已完成，但未超过基线，尚无新方法成功或论文创新性的结论。

- [本次恢复、审计和实验结果](results/cloud_20260930/RUN_REPORT.md)
- [持久私有下载索引](migration/20260930/CLOUD_DOWNLOAD_INDEX.json)：保存文件ID和持久链接，不保存短期认证URL。

- [本地迁移步骤](MIGRATION.zh-CN.md)：在 Windows 上导出原项目并上传代码。
- [项目记忆](memory/PROJECT_MEMORY.md)：目标、失败证据、协议与原文件位置。
- [论文线索与下一轮实验](research/NEXT_EXPERIMENTS.md)：已核查作者仓库、待阅读全文、可验证的候选机制。
- [运行状态](STATUS.json)：区分已完成迁移工具和未执行实验。

迁移工具使用 Python 标准库，Python 3.9+；审计与CPU先导另用 NumPy、SciPy、threadpoolctl。`transfer.py pack` 只读取原项目，代码快照和数组分块写到另一个目录；`restore` 在云端恢复并核对原文件 SHA256。原始数组、权重和中间训练恢复文件不进入普通 Git。

目录 `original/` 是用户上传的源码与旧文档快照，`SOURCE_MANIFEST.json` 记录本地来源提交 `01c9785ee47b8de784fc0efb53c307498edf5b7f`。它不包含原始数组。`memory/recovered/` 是较早从历史聊天文件创建记录恢复的旧文档，注明来源，未在本次重新运行其所述实验。

云端工具检查：

```bash
python -m unittest discover -s uwb_direction1/tests -v
```

数据到达后先验证来源、原始 9593 条记录与质量筛选后的 9300 条记录的对应关系，恢复旧预处理与基线，再冻结新候选方案。软件检查通过、合成数据往返通过都不能算手势方法有效。
