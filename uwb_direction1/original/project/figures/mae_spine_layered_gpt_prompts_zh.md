# 分层 MAE 生成式脊柱图 GPT image 提示词

目标：生成类似 UWB impulse radar 论文中 encoder / unknown detection 那种“过程具体到每一层”的方法图，而不是概念总览图。所有图均只服务于 **跨视角 MAE 预训练与生成式脊柱**，不画旧的三视角闭集分类 + inspector 主流程。

## 图 1：跨视角 MAE 预训练层级图

**用途**：解释 Stage-0 MAE 训练时，CIR 如何被切成时间/距离 patch，如何进入共享双路径编码器，如何由 decoder 重建被遮蔽视角，并用 MSE 更新 `enc₀+dec₀`。

**GPT image prompt**

Create a detailed Chinese scientific method diagram, in the style of an IEEE/Nature Machine Intelligence paper architecture figure: black outline arrows, pale color blocks, clean white background, exact layer-by-layer process, no decorative shadows, no 3D.

Title at top: “跨视角 MAE 预训练：从可见视角重建被遮蔽视角”

Overall layout: landscape 16:9, left-to-right pipeline. Draw it like a technical architecture diagram, not a marketing infographic. Use three horizontal stages separated by subtle dashed vertical dividers: “输入与遮蔽”, “共享编码器”, “解码与损失”.

Left input area:
- Draw three stacked UWB CIR matrices as small heatmap rectangles, labeled exactly:
  - “左视角 x_L”
  - “上视角 x_T（被遮蔽）”
  - “右视角 x_R”
- The middle `x_T` matrix must be visibly masked with a red cross or semi-transparent red overlay.
- Add small shape note below the stack: “x ∈ [B, 3, 90, 189]”.
- Draw a rounded block “随机选择遮蔽视角 v*” and a small note “例：v*=T”.
- Split into two branches:
  - upper branch: “可见视角 x_{-v*} = {x_L, x_R}”
  - lower branch: “目标视角 x_{v*} = x_T”

Patch/tokenization area:
- For the visible views, draw two parallel patching paths exactly like the reference encoder figure:
  1. “沿时间轴切分 CIR → 90 个时间 patch”
  2. “沿距离轴切分 CIR → 189 个距离 patch”
- Each path goes to a peach rounded rectangle:
  - “Patch + Position Embedding”
  - below it label “+ 视角嵌入 view embedding”
- Then each goes into a blue Transformer stack:
  - “时间 Transformer Encoder”
  - “距离 Transformer Encoder”
- On the right side of the encoder draw a zoom-in inset showing one Transformer component:
  inside the component stack vertically:
  “Norm → MSA → Add → Norm → MLP → Add”
  with a bracket label “× L 层，共享权重 enc₀”

Memory merge:
- Merge the two visible-view token streams into a memory block:
  “可见视角 memory M”
  “M ∈ [B, 2T, D]”
- Make clear this memory comes only from `x_L` and `x_R`, not from masked `x_T`.

Decoder area:
- Draw a row of 90 small query tokens at the top labeled:
  “90 个可学习 mask query Q ∈ [B,90,D]”
- Query tokens enter a purple module:
  “掩码视角解码器 dec₀”
- Inside or zoom-in the decoder, show:
  “Cross-Attention: Query attends to memory M”
  then “FFN / MLP”
  then “Linear head”
- Decoder output block:
  “重建上视角 x̂_T ∈ [B,90,189]”

Loss area:
- Draw a red dashed arrow from the lower target branch “目标视角 x_T” to the loss block.
- Draw a yellow diamond or rounded block:
  “MSE 重建损失”
  formula: “L_MAE = ||x̂_T − x_T||²”
- Add a backward dashed arrow from loss back to encoder and decoder labeled:
  “反向传播更新 enc₀ + dec₀”

Color palette:
- input heatmaps: teal/viridis-like radar matrices
- mask/target/loss cue: pale red #FDE8E5 with red outline
- patch embedding: pale peach #F6D2AE
- encoder: pale blue #DDEBF7
- decoder: pale purple #EFE5F7
- memory/token blocks: light grey-blue #EEF3F7
- loss: pale yellow #FFF2CC

Strict constraints:
- Chinese labels must be large and readable.
- Do not draw closed-set classifier, known gesture classifier, inspector bank, accept/reject decision, or prototype distance.
- Do not draw people, lab instruments, or decorative icons.
- The diagram must explicitly show CIR split into temporal patches and spatial/range patches, Patch+Position Embedding, Transformer Encoder internals, decoder cross-attention, and MSE loss.

## 图 2：训练后生成式脊柱部署图

**用途**：解释 `enc₀+dec₀` 训练完后不是丢掉 decoder，而是冻结成生成式脊柱，分别服务于 `e_recon` 和虚拟视角合成。

**GPT image prompt**

Create a detailed Chinese scientific method diagram in paper style, with black outline arrows and pale color blocks. The figure should explain the deployed generative spine after MAE pretraining. White background, technical architecture look, not abstract infographic.

Title: “训练后保留 dec₀：生成式脊柱的两个读出”

Overall layout: landscape 16:9. Put a large central vertical spine module in teal:
“冻结的生成式脊柱”
“enc₀ + dec₀”
“mask-one-view generator”
Inside the spine show two submodules stacked:
1. “共享双路径编码器 enc₀”
2. “掩码视角解码器 dec₀”
Add a lock icon or small text “冻结参数，不再更新”.

Left branch: reconstruction-error open-set evidence
- Input block on far left: “测试样本 x = [x_L, x_T, x_R]”
- Under it draw three small substeps:
  1. “遮蔽 L，用 T,R 重建 L”
  2. “遮蔽 T，用 L,R 重建 T”
  3. “遮蔽 R，用 L,T 重建 R”
- These three substeps enter the central spine.
- The spine outputs three reconstruction errors:
  “MSE_L”, “MSE_T”, “MSE_R”
- Aggregate block:
  “e_recon = mean(MSE_L, MSE_T, MSE_R)”
- Then two interpretation boxes:
  “低 e_recon：三视角自洽，像已知手势”
  “高 e_recon：三视角不自洽，可能未知”
- Use red/pink for this branch because it is open-set evidence.

Right branch: virtual-view synthesis under sensor failure
- Input block on far right: “传感器失效：缺失 x_T，只剩 x_L, x_R”
- Arrow into the same central spine.
- Output block: “合成虚拟视角 x̃_T = dec₀(enc₀(x_L,x_R))”
- Then show a reconstructed three-view stack:
  “[x_L, x̃_T, x_R]”
- Add note:
  “补回完整输入，供下游识别或鲁棒部署使用”
- Use green for this branch.

Bottom optional branch:
- Dashed grey-purple branch from spine:
  “可选未来：潜在运动轨迹 z(t) / JEPA”
- Note:
  “不是论文一主依赖”

Bottom summary:
“同一条生成式脊柱：既提供开集拒识证据，也提供缺视角补全能力。”

Strict constraints:
- Do not draw closed-set classifier, inspector bank, prototype distance, accept/reject boxes.
- Make every stage concrete: three masking/reconstruction passes for e_recon, and one missing-view synthesis pass.
- Chinese labels must be readable, variable names accurate.

## 图 3：Decoder cross-attention 与 e_recon 细节图

**用途**：把“decoder 怎么从可见视角生成被遮蔽视角”画细，类似论文右侧把 Transformer component 展开那样。

**GPT image prompt**

Create a technical Chinese layer-level diagram explaining the masked-view decoder and reconstruction error computation. Style: IEEE/Nature paper architecture figure, white background, black arrows, pale module colors, very concrete step-by-step layers.

Title: “掩码视角解码器 dec₀ 与 e_recon 计算细节”

Layout: landscape, two panels side by side.

Left panel title: “a  dec₀ 如何生成被遮蔽视角”
Show:
1. Top left: visible-view memory block:
   “memory M = enc₀(x_L, x_R)”
   “180 个可见 token = 2 × 90”
2. Bottom left: query token row:
   “90 个 mask query：q₁ … q₉₀”
   “每个 query 对应目标视角的一帧”
3. Middle: decoder layer stack, show 2 layers or L_d layers:
   Each layer should contain vertical sub-blocks:
   “Self-Attention over queries”
   “Cross-Attention: Q 查询 memory M”
   “Add & Norm”
   “MLP / FFN”
   “Add & Norm”
4. Right: linear projection head:
   “Linear head: D → 189 range bins”
5. Output matrix:
   “x̂_T ∈ [90,189]”
   draw a small predicted CIR matrix.

Add callout arrows:
- “第 t 个 query 生成第 t 帧”
- “cross-attention 从可见视角 memory 中选择相关证据”

Right panel title: “b  从重建误差到 e_recon”
Show three rows:
1. “mask L:  x̂_L vs x_L → MSE_L”
2. “mask T:  x̂_T vs x_T → MSE_T”
3. “mask R:  x̂_R vs x_R → MSE_R”
Then merge to:
“e_recon = (MSE_L + MSE_T + MSE_R) / 3”
Then interpretation:
“已知手势：低误差”
“未知手势：高误差”

Use colors:
- memory blue-grey
- query purple
- decoder purple
- output blue
- error red/yellow

Strict constraints:
- Do not include classifier, inspector, prototype, or accept/reject pipeline.
- Must show decoder internal layers, query-to-frame relation, cross-attention to memory, and three-view averaged e_recon.
- Chinese labels must be readable and not crowded.
