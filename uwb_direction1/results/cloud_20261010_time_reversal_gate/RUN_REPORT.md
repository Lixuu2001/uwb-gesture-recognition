# 时间反转/逆手势增强先例门控（2026-10-10）

## 结论

关闭“将慢时间序列反转并把样本重标为相反方向手势，作为通用增强/独立新方法”的候选；本轮没有读取项目标签或数组值、没有拟合/训练，也没有产生识别分数。该决定不等于所有更具体的结构化等变模型都已被穷尽；任何后续模型仍须重新审查其明确机制和贡献。

## 直接先例

- Chen 等，RaGeoSense，Scientific Reports 2025，全文明确列出点云手势增强：additive noise、sequence inversion、temporal shifting、temporal warping，并说明序列反转用于学习手势对称性。即使输入形式与这里的 IR-UWB 时距图不同，“反转手势序列来扩充/学习动作对称性”也已是雷达手势先例。[论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12046016/)；[DOI 10.1038/s41598-025-00065-8](https://doi.org/10.1038/s41598-025-00065-8)。
- Price 等，Retro-Actions，ICCV Workshops 2019，研究时间反转/水平翻转等变换，并显式研究变换如何交换类别标签，为时间反转对应相反动作标签提供视频动作识别先例。[作者机构记录](https://research-information.bris.ac.uk/en/publications/d22db0a8-9c0a-46a5-be93-f7db1d89d976)；[CVF 论文页与 PDF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/MDALC/Price_Retro-Actions_Learning_Close_by_Time-Reversing_Open_Videos_ICCVW_2019_paper.pdf)。本轮只核读了机构摘要/元数据，CVF PDF 抓取返回403；没有声称已通读全文。
- UWB-Gestures 公开类定义包含方向成对的 swipes 以及 clockwise/anticlockwise 类；该事实也见该数据集描述及 2026 年同数据集论文图示。[Scientific Data 数据集论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8041886/)；[同数据集多视角论文](https://doi.org/10.3389/fnins.2026.1864448)。

## 对本数据的边界

按已公开类别名称可写出候选逆映射：左右、上下及对角 swipe 配对，顺/逆时针配对。inward-push 没有公开的 outward-pull 对类，因此全体类别并非封闭的时间反转群作用。更关键的是，类别名称只说明目标动作语义；当前 90-row clutter-removed 缓存及其采集记录没有提供逐帧手部轨迹真值，无法验证逐样本时间反转后的雷达响应是否等价于真实反向动作。故不能仅凭名称把反转样本当作真实独立动作，也不能在缺少语义闭合/物理验证时把常规增强包装为原创。

## 执行记录与下一步

只做了公开论文与现有门控记录的文献核查。云端 `/workspace/uwb-restored-20260930` 仍在位；进程检查只发现三个 defunct Python 条目，没有活跃训练。没有 GPU/采集硬件。本轮没有启动训练，也没有恢复/重下载数据。

没有新的模型候选通过“方法上区别于先例、与现有缓存输入可检验、协议可预注册”三项门槛。下一步继续只审查一个明确的新机制；若仍无法通过，应等待独立采集的带同步动作真值/校准数据，而非重复已有模型网格。
