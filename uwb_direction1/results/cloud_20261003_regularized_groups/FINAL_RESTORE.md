# 最终私有备份恢复

当前云端原结果若仍存在且哈希匹配，不要恢复或重新训练。本次18轨迹已完成、门槛失败；恢复只用于复核和延续研究，不扩大失败参数网格。

以DRIVE_BACKUP_RECEIPT.json为持久索引。先核验每个Drive对象的大小、owner-only、shared=false和parent_folder_id；用fetch(download_raw_file=true, include_base64=false)获取认证文件引用后立即download_file物化，最多3并发，不输出认证URL。先比对每个分块bytes及SHA-256；相同且已验证文件可跳过，冲突文件保留。按每个逻辑archive的part顺序拼接到独立临时文件，再检查整包bytes及sha256。

所有包都含CONTENTS_SHA256.json。恢复到全新空目录，拒绝绝对路径、..、链接、目录之外路径和重名覆盖；仅写regular-file成员，再逐个核对嵌入清单的长度和SHA-256。37逻辑包共997内容成员（不计各包自身清单），包含972训练产物及25元数据。若目标非空，换新目录；不能覆盖现有结果。

归档布局：每条轨迹在<restore>/<arm>/held<h>，公共训练元数据在<restore>/runs/，代码在<restore>/tools/，研究记录在<restore>/research/，原启动/成本/测试凭据在<restore>/provenance/。若要重建训练器所期望的原布局，可将三个arm目录复制或移动到<restore>/runs/之下；目标arm目录必须尚不存在。tools与research相邻关系保持，运行结果目录则传<restore>/runs。不得仅从selected.pt恢复优化器；每轨迹progress.pt才包含完整模型/AdamW/余弦调度/RNG/组风险/最佳及尾平均状态。

此备份不含原始信号/prepared数组、旧六源tail对照或原789源码。原数据另依migration/20260930持久索引与SHA凭据；旧六源基线依cloud_20261003_swad/DRIVE_BACKUP_RECEIPT.json，加cloud_20261003_view_sharing/BASELINE_STATE_DRIVE_RECEIPT.json。原环境仍齐全时不用重新下载。当前代码身份记录torch2.14.0+cpu及NumPy2.3.5，路径或环境改变不能忽略identity校验来续训；完整结果的独立审计可在明确记录迁移后的路径配置后执行，不改变实验数值。

验证对应原执行命令：

    python tools/regularized_group_source.py audit --protocol research/REGULARIZED_GROUP_PROTOCOL.json --cost provenance/COST.json --out <restored-original-layout>

协议内data/prepared/baseline为原绝对路径；该命令只有路径和哈希依赖吻合时才适用。跨环境重映射需要单独记录，不修改原协议或冒充同一训练身份。训练已完整完成，无需再运行train命令。
