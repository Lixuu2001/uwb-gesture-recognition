# 云端接续与私有恢复

当前进程PID16830，目录`/workspace/uwb-sharing-source-runs-20261003`，日志`/workspace/uwb-sharing-source-runs-20261003.log`。先查`queue_status.json`和实际进程；活动时不得重复启动。完整12轨迹结束自动做18轨迹ERM/tail的独立推理审计，生成AUDIT.json、SUMMARY.json。未完成审计不能报最终有效性。

当前环境原地恢复命令（仅确认旧进程不在运行时执行）：

```bash
python -u /workspace/uwb-cloud/uwb_direction1/tools/view_sharing_source.py train \
  --data /workspace/uwb-restored-20260930 \
  --prepared /workspace/uwb-circular-prepared-20260930 \
  --baseline /workspace/uwb-swad-source-runs-20261003 \
  --out /workspace/uwb-sharing-source-runs-20261003 \
  --cost /workspace/uwb-sharing-cost-20261003/COST.json
```

脚本持有队列文件锁，并核对源码、协议、准备数据、环境版本和六条基线结果哈希。每轮保存完整model/optimizer/scheduler/RNG、best和tail，恢复会从下一轮继续；已完成结果只在身份和文件SHA一致时跳过。不能清空文件夹或改配置绕过不一致。每轮内中断只能回到上一个完整epoch；确定性样本顺序和RNG保证完整重算该轮。

如果原云端目录不在：不要再次要求用户上传。原始86文件按既有私有163块索引恢复，prepared按旧审计脚本重建并核原SHA。源库中Git代码使用本次预注册版本fd4d754b，训练依赖不得漂移。PyTorch2.14.0+cpu，numpy2.3.5，Python3.12；当前Python路径/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python，torch路径/home/agent/.local/lib/python3.12/site-packages/torch。

1. 旧共享基线先依`results/cloud_20261003_swad/RESTORE.md`恢复七包，最终结构是NEW_SWAD/runs/{identity.json,protocol.json,held3/...}。再取本目录BASELINE_STATE_DRIVE_RECEIPT.json的三个补充包，将六个heldN/progress.pt恢复到NEW_SWAD/runs/heldN/progress.pt。补充包是原16,088,729字节状态文件的精确字节复制，哈希与原result.json一致，包含optimizer、RNG及平均队列。仅原七包不含这些状态，无法满足本实验基线复用审计。
2. 本目录START_DRIVE_RECEIPT.json定位新实验早期启动证据包。`tools/`和`research/`包含匹配代码/协议；`cost/COST.json`是身份匹配的成本回执；`runs/`保存新队列元数据及held3 untied128第3轮的完整可恢复状态和第1–3轮段平均。此包只保证早期恢复；晚于第3轮的计算可能需重跑，**不是完整12轨迹最终备份**。
3. 在新的独立空目录恢复，不能覆盖已有实验结果。每包先元数据核private/owner、大小，再fetch(download_raw_file=true,include_base64=false)、立即download_file物化、核整包SHA和CONTENTS_SHA256.json的所有内容。最多3并发，不打印认证URL。tarfile用filter='data'；对三个旧基线补充包只将清单指定的heldN/progress.pt写入目标，清单另存到各包独立子目录，避免三个同名清单互相覆盖。若目标文件已存在仅在SHA一致时跳过，否则保留冲突并使用新目录。
4. 用新路径替换命令的--baseline/--out/--cost/代码目录；身份未绑定绝对目录，数据及代码哈希必须完全相同。START包中的旧LAUNCH命令只是历史，不要误指向另一个环境的旧路径。当前环境一切已在磁盘，不需要执行恢复。

最终12轨迹结束还须补存所有新权重、完整进度状态、段权重、日志和逐样本预测，以<32MiB独立私有包上传并认证回下载核验。配置、摘要、文献边界和小日志可进Git；原始数组和大权重不上Git。当前文献与创新边界见VIEW_GRADIENT_REVIEW.md，不能将解除共享本身称为原创方法。
