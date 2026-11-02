# 论文一图二三栏重排 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留原 PPTX 的前提下，输出按已确认 C 方案重排、线条加粗、圆角统一且重复内容明显减少的可编辑图二新版本。

**Architecture:** 使用 JavaScript ES module 与 `@oai/artifact-tool` 构建单页 PPTX，嵌入现有三张真实 CIR，并用原生 PowerPoint 形状重建三栏流程。用独立的 Node 结构测试先约束对象命名、重复文本、图片数量、线宽和面板几何，再通过整页渲染与溢出检查完成视觉验收。

**Tech Stack:** JavaScript ES modules、`@oai/artifact-tool`、PowerPoint PPTX、bundled presentation render/test helpers。

---

### Task 1: 冻结结构验收条件

**Files:**
- Create: `paper1/.codex_tmp/fig2_layout_r4/test_fig2_layout.mjs`

- [ ] 写入一个只读取 PPTX OOXML 的 Node 测试，检查：单页、9 张图片、三个等尺寸圆角面板、每栏仅一个编码器标签、删除 `t=1…T`/`r=1…R`/自连接图例、三条结论文案、主线最小宽度与输出文件名。
- [ ] 对 `fig2_three_fusion_strategies_cn_editable_v6_faithful_r3.pptx` 运行测试，预期至少因重复编码器文本和旧 token 索引而失败。

### Task 2: 构建重排版 PPTX

**Files:**
- Create: `paper1/.codex_tmp/fig2_layout_r4/build_fig2_layout_r4.mjs`
- Create: `paper1/.codex_tmp/fig2_layout_r4/source-notes.txt`
- Create: `paper1/figs/fig2_three_fusion_strategies_cn_editable_v6_faithful_r4.pptx`

- [ ] 使用 `@oai/artifact-tool` 创建与源文件相同画布比例的单页演示文稿。
- [ ] 先创建全部流程线，再创建三栏面板、真实 CIR、编码器、特征/融合节点和文字，使连线位于节点后方。
- [ ] 实现三栏统一纵向基线与 L/T/R 三条固定视角轴。
- [ ] 实现 (a) 三路独立到 `d_L/d_T/d_R` 再直角汇合到 `mean + β var`。
- [ ] 实现 (b) 编码后 `z_L/z_T/z_R` 进入简化跨视角自注意力，再进入后续检查/判决。
- [ ] 实现 (c) 编码器前半段、时/距紧凑块、少量瓶颈 token、平均回写虚线、编码器后半段和检查/判决。
- [ ] 为幻灯片写入来源备注，并导出新 PPTX，拒绝覆盖源文件。

### Task 3: 结构与视觉验证

**Files:**
- Generate: `paper1/.codex_tmp/fig2_layout_r4/rendered/slide-1.png`
- Generate: `paper1/.codex_tmp/fig2_layout_r4/qa-ledger.txt`

- [ ] 对新 PPTX 运行 Task 1 的结构测试，预期全部通过。
- [ ] 使用 bundled renderer 渲染整页 PNG，并用 `slides_test.py` 检查画布越界。
- [ ] 全尺寸检查三栏对齐、文本换行、线条穿框、箭头方向和 CIR 清晰度；将发现记录到 QA ledger。
- [ ] 若发现问题，修改构建器并重新执行构建、结构测试、渲染和视觉检查，直到无未解决问题。

### Task 4: 最终交付核验

**Files:**
- Verify: `paper1/figs/fig2_three_fusion_strategies_cn_editable_v6_faithful_r4.pptx`

- [ ] 重新运行结构测试、渲染和溢出测试，保存最新输出证据。
- [ ] 核对源 PPTX 的大小与修改时间未发生变化。
- [ ] 只交付最终 PPTX，不提交临时构建文件或用户工作区中的其他修改。
