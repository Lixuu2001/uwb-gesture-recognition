# 图二 v6 高保真可编辑重建 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成与参考效果图结构和视觉密度高度一致、且除真实 CIR 外全部可编辑的图二 v6 PPTX。

**Architecture:** 继续使用 Artifact Tool 的 JavaScript RPC 构建器，在同一页中按参考图坐标重建三栏。复用真实 CIR 资源和基础图元函数，重写三种融合面板，随后由本机 PowerPoint 做实际渲染、溢出和对象检查。

**Tech Stack:** JavaScript ES modules、Artifact Tool、PowerPoint COM、PowerShell 验证。

---

### Task 1: 锁定参考比例和视觉语法

**Files:**
- Modify: `paper1/.codex_tmp/fig2_editable_pptx/build_fig2_editable.mjs`

- [ ] 将画布改为 `1749 × 915`，按参考图设置标题、三栏位置、外框粗细和紧凑间距。
- [ ] 设置主流程、局部连接、跨视角虚线和回写虚线四级线宽。
- [ ] 将输出改为 `paper1/figs/fig2_three_fusion_strategies_cn_editable_v6.pptx`。

### Task 2: 重建 (a) 与 (b)

**Files:**
- Modify: `paper1/.codex_tmp/fig2_editable_pptx/build_fig2_editable.mjs`

- [ ] 在 (a) 中补齐 `t=1…T`、`r=1…R`、距离分数与三路曲线汇聚。
- [ ] 在 (b) 中补齐索引 `t`、注意力矩阵内部实/虚线、线型图例、三路输入和三路输出。
- [ ] 保证距离 token 旁路独立且不穿过注意力框文字。

### Task 3: 高保真重建 (c)

**Files:**
- Modify: `paper1/.codex_tmp/fig2_editable_pptx/build_fig2_editable.mjs`

- [ ] 将六组输入 token 分别接入四个瓶颈 token，并保留彩色路径。
- [ ] 添加平均节点、紫色虚线回写和后段六组更新 token。
- [ ] 添加右侧双路径说明框与两种小型路径示意。

### Task 4: 构建和验证

**Files:**
- Create: `paper1/figs/fig2_three_fusion_strategies_cn_editable_v6.pptx`

- [ ] 使用 bundled Node 运行构建脚本，预期退出码为 0。
- [ ] 使用 PowerPoint 打开并导出 1749×915 PNG，预期 1 页且可往返保存。
- [ ] 检查所有有文本对象的 BoundWidth/BoundHeight，预期溢出数为 0。
- [ ] 检查图片对象数为 9，并核对 L/T/R 媒体哈希各精确匹配 3 次。
- [ ] 全尺寸检查参考图与最终渲染，修正不一致的路径、标签和留白。
