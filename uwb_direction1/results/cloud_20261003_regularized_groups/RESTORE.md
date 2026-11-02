# 云端队列接续

实验目录：`/workspace/uwb-regularized-group-runs-20261003`；日志：`/workspace/uwb-regularized-group-train-20261003.log`。启动PID以LAUNCH_VERIFIED.json为观察记录，不保证以后仍活跃。先检查进程、queue_status.json和各臂held目录的progress.json；活跃时禁止重复启动或改训练代码/协议。

冻结协议提交9011739140cb0686fd1f259f7e2a8dcbfcfdb78f。18条轨迹（源用户3–8×uniform_base/uniform_reg/dro_reg），每条24轮；每轮保存progress.pt（model/AdamW/scheduler/RNG/q/best/tail/history）、segment_XX.pt、updates_XX.npz（逐步ID、组均值CE、q）。原始RF不写Git。代码和帮助函数/数据/成本身份必须一致。

如果进程退出，先读日志和queue_status错误，不覆盖现场。相同环境按原命令恢复，脚本取得排他锁并从最后完整epoch恢复，已完成且哈希一致的轨迹跳过。中途epoch未存盘的部分确定性重做，不允许把半轮进度冒称恢复。

```bash
python /workspace/uwb-cloud/uwb_direction1/tools/regularized_group_source.py train --protocol /workspace/uwb-cloud/uwb_direction1/research/REGULARIZED_GROUP_PROTOCOL.json --cost /workspace/uwb-regularized-group-cost-20261003.json --out /workspace/uwb-regularized-group-runs-20261003
```

完成后脚本自动独立函数式重放新18轨迹ERM/tail全部验证和诊断预测、重建采样与全部q更新、核验选模/权重/产物SHA，再写AUDIT.json和SUMMARY.json。若该审计失败，保留模型/错误并修复纯审计问题；不能因此改训练超参或评分协议。`audit`子命令可只重做审计，不训练。

最终需再保存全部模型/optimizer/RNG/q/源ID/预测/日志到owner-only私有Drive，并认证重下载整包及成员SHA；当前START_DRIVE_RECEIPT仅启动阶段快照，不等于18条完整训练已备份。大包可按≤8MiB精确分块、最多3并发。最终结果以最新远端父提交force=false写回，只改本研究文件，不覆盖789原始快照。

主候选固定dro_reg验证CE所选端点，对两种匹配普通训练和旧强tail均需满足预设门槛。固定tail仅次要分析，不可事后换主结果。失败停止本次decay/eta/seed/窗口网格；通过也不能直接称创新。完整结果前不要根据中途诊断分数调参。用户目标仍未完成，自动接续保持。

启动包恢复：把progress.pt、split.json、所有segment_XX.pt和updates_XX.npz放入新恢复根目录uniform_base/held3/；identity.json与protocol.json放入恢复根目录，COST.json复制到命令的--cost文件。包内包含快照所见全部已完成轮次，快照后的工作未备份。先核成员SHA、数据与版本身份，再使用同一train命令和恢复根目录继续；不覆盖仍在运行的原目录。
