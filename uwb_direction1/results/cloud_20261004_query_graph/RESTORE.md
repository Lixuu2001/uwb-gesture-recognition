# 私有恢复

读取本目录DRIVE_BACKUP_RECEIPT.json，用持久file_id/url先查shared=false、owner-only及大小，再fetch download_raw_file=true/include_base64=false，立即物化，最多3并发，不输出认证URL。按part_index校验并重组，核验整包SHA及CONTENTS_SHA256.json全部成员，只解至独立空目录。

runs/包含六折全部图CSR、预测、评估标签、split/selection/solver记录、汇总/审计。tools/和research/保存独立实现及冻结配置；literature/保存正式论文PDF和抽取正文；cost/和logs/保存实际计时/运行日志。源onehot读出/checkpoint、128维冻结特征沿用prior_backup/两个已验证回执：semantic code源读出与energy-area源特征，勿换成八折模型。把QUERY_GRAPH_PROTOCOL.json的两个reference路径映射到对应恢复位置（保留其内容SHA及原协议存档），随后重建或单独审计；重新运行必须新输出目录。

完整原信号仍按既有迁移清单恢复，本包不重复原始大数组。本轮没有训练优化器状态，因为未训练参数；沿用的读出参数和CNN检查点可在前述回执恢复。归档报告生成在上传回执之前，以最终Git报告/回执为准。
