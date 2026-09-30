# Figure 2 立体跨视角注意力权重模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一张可直接评估并可继续调整的白底立体跨视角多头注意力权重效果图，同时交付高分辨率 PNG 与内嵌 SVG 的 HTML 源文件。

**Architecture:** 使用单一 HTML 文件承载固定比例的 SVG 场景。SVG 按输入特征带、四层权重矩阵、残差路径和三路输出分组；独立检查脚本验证关键语义与结构，浏览器无界面渲染脚本输出最终 PNG。所有权重强弱仅为视觉示意，不写入数值。

**Tech Stack:** HTML5, SVG, CSS, JavaScript ES modules, bundled Node.js, headless Microsoft Edge or Chromium.

---

## File Structure

- Create: `D:/uwb_gestures/paper1/.codex_tmp/attention_weight_3d/check_attention_module.mjs` — 静态结构与语义检查。
- Create: `D:/uwb_gestures/paper1/.codex_tmp/attention_weight_3d/render_attention_module.mjs` — 无界面浏览器渲染与 PNG 导出。
- Create: `D:/uwb_gestures/paper1/.codex_tmp/attention_weight_3d/visual-qa.txt` — 全尺寸视觉检查记录。
- Create: `C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_可调.html` — 最终可调整 HTML/SVG。
- Create: `C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_效果图.png` — 最终高分辨率效果图。

### Task 1: 固定可验证的图形结构

**Files:**
- Create: `D:/uwb_gestures/paper1/.codex_tmp/attention_weight_3d/check_attention_module.mjs`
- Test: `C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_可调.html`

- [ ] **Step 1: 写入初始结构检查脚本**

```js
import fs from "node:fs";

const htmlPath = "C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_可调.html";
const html = fs.readFileSync(htmlPath, "utf8");
const required = [
  'id="attention-weight-module"',
  'data-role="input-view"',
  'data-role="attention-head"',
  'data-role="output-view"',
  'data-role="residual-path"',
  'A_t^(h)',
  'z_L^t', 'z_T^t', 'z_R^t',
  'z̃_L^t', 'z̃_T^t', 'z̃_R^t',
  '跨视角多头注意力',
];
for (const token of required) {
  if (!html.includes(token)) throw new Error(`Missing required token: ${token}`);
}
const headCount = (html.match(/data-role="attention-head"/g) || []).length;
const inputCount = (html.match(/data-role="input-view"/g) || []).length;
const outputCount = (html.match(/data-role="output-view"/g) || []).length;
if (headCount !== 4) throw new Error(`Expected 4 heads, found ${headCount}`);
if (inputCount !== 3) throw new Error(`Expected 3 inputs, found ${inputCount}`);
if (outputCount !== 3) throw new Error(`Expected 3 outputs, found ${outputCount}`);
for (const forbidden of ["真实权重", "距离路径参与", "8928维全局注意力", "自连接图例", "互连接图例"]) {
  if (html.includes(forbidden)) throw new Error(`Forbidden claim: ${forbidden}`);
}
console.log("PASS: attention module structure and labels are valid");
```

- [ ] **Step 2: 运行检查并确认在 HTML 尚未生成时失败**

Run:

```powershell
& $env:RUNTIME_NODE 'D:\uwb_gestures\paper1\.codex_tmp\attention_weight_3d\check_attention_module.mjs'
```

Expected: FAIL，提示目标 HTML 不存在。

### Task 2: 绘制最终 HTML/SVG

**Files:**
- Create: `C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_可调.html`

- [ ] **Step 1: 建立固定 16:9 导出画布**

HTML 使用 `1600 × 900` 的 `.export-frame`，内部 SVG 的 `viewBox="0 0 1600 900"`。页面背景为 `#EEF2F5`，导出画布背景为纯白；模块在画布内水平居中，上下保留不少于 90 px 空白。

- [ ] **Step 2: 绘制三路输入特征带**

为 L/T/R 创建三个 `data-role="input-view"` 分组。每组包含圆形视角标记、半透明层叠特征片和标签 `z_L^t`、`z_T^t`、`z_R^t`；颜色分别使用 `#F2C45E`、`#55B4DD`、`#E98A9C`。

- [ ] **Step 3: 绘制四层 3×3 权重矩阵**

创建四个 `data-role="attention-head"` 分组，按 `(22 px, -18 px)` 的方向错位叠放。每层均有 9 个透视单元格，统一采用青绿色；透明度序列只用于形成视觉起伏，例如：

```js
const illustrativeWeights = [
  [0.82, 0.24, 0.51, 0.31, 0.91, 0.43, 0.63, 0.35, 0.76],
  [0.66, 0.38, 0.72, 0.48, 0.84, 0.29, 0.57, 0.69, 0.45],
  [0.74, 0.53, 0.27, 0.36, 0.79, 0.61, 0.42, 0.32, 0.88],
  [0.89, 0.28, 0.46, 0.52, 0.86, 0.37, 0.58, 0.41, 0.81],
];
```

这些值不得显示为文字。格子轻微抬高，但不得形成统计柱图。矩阵行列只以 L/T/R 小圆点标识，最前层显示 `h = 1`，后层以省略号和 `h = 4` 说明多头结构。

- [ ] **Step 4: 添加公式、残差路径与输出**

权重张量上方显示 `A_t^(h) = softmax(Q_t^(h) K_t^(h)T / √d_h)`。创建一个 `data-role="residual-path"` 的细灰蓝弧线绕过矩阵，输出端加入 `weighted sum + residual`。右侧创建三个 `data-role="output-view"` 分组，显示 `z̃_L^t`、`z̃_T^t`、`z̃_R^t`。

- [ ] **Step 5: 加入克制的论文图标注**

模块下方只显示标题“跨视角多头注意力”和副标题“每个对齐位置 t 沿视角维完成重加权”。右下角以 15 px 灰字标明“weights shown schematically”，防止示意权重被误解为真实数据。

- [ ] **Step 6: 运行静态结构检查**

Run:

```powershell
& $env:RUNTIME_NODE 'D:\uwb_gestures\paper1\.codex_tmp\attention_weight_3d\check_attention_module.mjs'
```

Expected: `PASS: attention module structure and labels are valid`。

### Task 3: 导出高分辨率 PNG

**Files:**
- Create: `D:/uwb_gestures/paper1/.codex_tmp/attention_weight_3d/render_attention_module.mjs`
- Create: `C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_效果图.png`

- [ ] **Step 1: 写浏览器渲染脚本**

```js
import { chromium } from "playwright";
import { pathToFileURL } from "node:url";

const html = "C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_可调.html";
const png = "C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_效果图.png";
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1800, height: 1100 }, deviceScaleFactor: 2 });
await page.goto(pathToFileURL(html).href, { waitUntil: "networkidle" });
await page.locator(".export-frame").screenshot({ path: png });
await browser.close();
console.log(png);
```

- [ ] **Step 2: 执行导出**

Run:

```powershell
& $env:RUNTIME_NODE 'D:\uwb_gestures\paper1\.codex_tmp\attention_weight_3d\render_attention_module.mjs'
```

Expected: 生成 PNG，像素尺寸为 `3200 × 1800`，文件大于 150 KB。

### Task 4: 全尺寸视觉 QA 与修正

**Files:**
- Modify: `C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_可调.html`
- Create: `D:/uwb_gestures/paper1/.codex_tmp/attention_weight_3d/visual-qa.txt`

- [ ] **Step 1: 打开 PNG 全尺寸检查**

检查：三路输入输出是否一一对应；四层矩阵是否分离清楚；最前层 3×3 单元格是否可辨；浅透视是否仍像论文图；是否出现柱状统计图错觉；公式、标题和示意声明是否清晰；阴影是否过重。

- [ ] **Step 2: 记录并修正所有可见问题**

`visual-qa.txt` 使用以下固定格式：

```text
Canvas: PASS/FAIL
Input-output mapping: PASS/FAIL
Four-head stack: PASS/FAIL
3x3 matrix readability: PASS/FAIL
No statistical-chart ambiguity: PASS/FAIL
Typography at full size: PASS/FAIL
Typography at 50% scale: PASS/FAIL
Shadow and perspective restraint: PASS/FAIL
Corrections applied: <semicolon-separated list or none>
```

- [ ] **Step 3: 重新运行结构检查和 PNG 导出**

Run:

```powershell
& $env:RUNTIME_NODE 'D:\uwb_gestures\paper1\.codex_tmp\attention_weight_3d\check_attention_module.mjs'
& $env:RUNTIME_NODE 'D:\uwb_gestures\paper1\.codex_tmp\attention_weight_3d\render_attention_module.mjs'
```

Expected: 静态检查 PASS；最终 PNG 覆盖旧预览并保持 `3200 × 1800`。

### Task 5: 最终文件验证

**Files:**
- Verify: `C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_可调.html`
- Verify: `C:/Users/19252/Desktop/Figure2_立体跨视角注意力权重_效果图.png`

- [ ] **Step 1: 校验文件尺寸与 PNG 元数据**

Run:

```powershell
Get-Item -LiteralPath 'C:\Users\19252\Desktop\Figure2_立体跨视角注意力权重_可调.html','C:\Users\19252\Desktop\Figure2_立体跨视角注意力权重_效果图.png' | Select-Object FullName,Length
```

Expected: 两个文件均存在且非空；PNG 大于 150 KB。

- [ ] **Step 2: 将最终 PNG 展示给用户**

最终回复中直接嵌入绝对路径 PNG，并提供 HTML 文件链接。此阶段不修改原始 `fig2_three_fusion.pptx`。
