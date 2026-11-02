# GPT image prompts for MV-OSR manuscript figures

Use these prompts with GPT image as raster design drafts.  They intentionally
repeat layout and anti-overlap rules because generated text and dense labels are
the highest-risk parts.

## Global visual contract

- Scientific manuscript figure, clean Nature Machine Intelligence style.
- White background, no gradient background, no 3D, no decorative shadows.
- Low-saturation palette: baseline grey-blue #8290B0, proposed method teal #079A87, final/key result navy #2F557F, reference/harm red #D94A3A, neutral grey #9AA3AF, pale grid #E6E8EC.
- Use large readable Chinese labels in a clean sans-serif style similar to Microsoft YaHei / SimHei; English and numbers in Arial-like sans-serif.
- Strict alignment, generous margins, no overlapping text, no labels inside crowded data regions.
- Put legends outside plotting regions whenever possible.
- Use exact numeric values from the prompt; do not invent data, categories, or extra annotations.

## Fig. 2 prompt - single-view reproduction waterfall

Create a polished academic waterfall chart for a Chinese MV-OSR paper.

Canvas: wide 16:9 scientific chart, white background.
Title text: "单视角复现因素分解"
Y axis: "case-4 OSCR (%)", range 70 to 91.5.
X labels, exactly six groups:
1. "基线\n(杂波对消)"
2. "+ 式(12)\nmixup 负样本"
3. "+ 检查器\nbatch=16"
4. "+ 忠实\nD_key=32"
5. "+ 原始回波\n(Raw)输入"
6. "复现结果"

Data:
- start bar: 73.18, grey-blue #8290B0, label "73.18"
- floating step: +2.20 to 75.38, teal #079A87, label "+2.20"
- floating step: +0.01 to 75.39, grey #9AA3AF, label "+0.01", small note "伪因素"
- floating step: +2.46 to 77.84, teal #079A87, label "+2.46"
- floating step: +10.51 to 88.35, teal #079A87, label "+10.51", small note "主因"
- final bar: 88.35, navy #2F557F, label "88.35"
- red dashed horizontal reference line at 87.98, label outside plot: "论文 case-4: 87.98"

Design: thin pale horizontal gridlines, dotted connectors between waterfall bars, no overlapping text. Leave enough vertical room above labels. No legend is needed.

## Fig. 3 prompt - main per-case OSCR

Create a polished grouped bar chart for a Chinese MV-OSR paper.

Canvas: wide scientific chart, white background.
Y axis: "OSCR (%)", range 60 to 101.
X groups: case 1, case 2, case 3, case 4, case 5, "5-case\n均值".
Each group has two bars: single-view grey-blue #B7C2D8 and three-view navy #2F557F.
Legend at top center, outside the plotting area: "单视角(Left)" and "三视角(L+T+R)".

Data:
- case 1: single 87.81, three-view 96.19, gain label "+8.4"
- case 2: single 84.53, three-view 94.80, gain label "+10.3"
- case 3: single 87.39, three-view 94.72, gain label "+7.3"
- case 4: single 87.18, three-view 94.04, gain label "+6.9"
- case 5: single 86.81, three-view 95.95, gain label "+9.1"
- mean group: single 86.74 ± 1.29, three-view 95.14 ± 0.81, include error bars, put value labels "86.74" and "95.14" inside the bars in white
- red dashed horizontal reference line at 88.42 with label at far right outside dense bars: "Eom 2025 单视角\n88.42"

Design: thin pale gridlines, clean axis, no overlapping labels, keep reference label away from mean bars.

## Fig. 4 prompt - fusion strategy schematic

Create a clean three-panel academic schematic for a Chinese MV-OSR paper.  This is a diagram, not a chart.

Canvas: wide 16:9, white background. Three horizontal panels stacked vertically, equal height, equal width, aligned to the same left and right margins. Rounded pale-grey panel border. No overlap.

Global title: "跨视角融合策略对比"
Small subtitle at top right: "相同输入与检查器；仅交换层级不同"

Panel (a) title: "(a) 决策级聚合：本文采用"
Flow: three input lanes L, T, R -> shared encoder "共享编码器" -> per-view features f_L, f_T, f_R -> pale dashed box "无特征交换" -> per-view inspectors η^(L), η^(T), η^(R) -> teal score box "⊕\n异常分数 s" -> "拒识\n判决".
Bottom note in panel (a): "跨视角信息只在分数层汇合".

Panel (b) title: "(b) 特征级融合 F1：编码后交换"
Flow: L,T,R -> shared encoder -> f_L,f_T,f_R -> red highlighted box "跨视角\n注意力" -> "融合特征\nf̃_v" -> red "检查器" -> grey "⊕\n异常分数 s" -> "拒识\n判决".
Bottom note: "交换提前到特征层，破坏逐视角独立性".

Panel (c) title: "(c) 瓶颈中层融合 F2/MBT：编码中交换"
Flow: L,T,R -> "编码器\n前段" -> amber highlighted "瓶颈\ntoken" -> "编码器\n后段" -> f_L,f_T,f_R -> dashed "无特征交换" -> amber "检查器" -> grey "⊕\n异常分数 s" -> "拒识\n判决".
Bottom note: "少量共享 token 作窄带交换".

Design: all modules same size within each class, all arrows aligned horizontally on three lanes, panel titles outside arrows, no text touching boxes or arrows. Use teal only for adopted decision-level aggregation, red for feature-level exchange, amber for bottleneck exchange, grey for neutral. Do not put any performance numbers, OSCR values, ΔOSCR values, case labels, or LOSO labels in this architecture figure; those belong in the table and caption text.

## Fig. 5 prompt - missing-view robustness dumbbell plot

Create a polished dumbbell plot for a Chinese MV-OSR paper. This figure must emphasize exact OSCR positions, not decorative arrows.

Canvas: wide scientific chart, white background.
X axis: "OSCR (%, 5-case 均值；误差棒为标准差)", range 33 to 103.
Y rows from top to bottom:
1. "全视角 L+T+R"
2. "上失效(L+R)"
3. "右失效(L+T)"
4. "左失效(T+R)"
5. "仅剩左 L"
6. "仅剩上 T"
7. "仅剩右 R"

For each row, draw one straight light-grey horizontal connector segment from the baseline OSCR point to the view-dropout OSCR point. Do not draw arrowheads anywhere. Baseline point is grey-blue #8290B0 with horizontal error bar. View-dropout point is teal #079A87 with horizontal error bar. The connector endpoints must coincide with the two point centers.
Data:
- 95.33±0.79 -> 94.88±0.94, label "-0.45 (噪声内)"
- 86.22±2.63 -> 90.81±2.28, label "+4.6"
- 80.21±2.62 -> 91.97±0.93, label "+11.8"
- 71.40±9.75 -> 92.35±1.07, label "+20.9"
- 54.71±6.46 -> 83.61±3.21, label "+28.9"
- 53.37±7.11 -> 72.12±3.94, label "+18.8"
- 43.35±7.51 -> 81.30±1.36, label "+37.9"

Add red dashed vertical reference line at x=86.74, label above it: "单视角直训 86.74".
Legend at top, outside plotting area: "朴素三视角训练(基线)" and "随机丢视角训练(p=0.5)".
Add pale horizontal separators between the 3-view, 2-view, and 1-view row groups, but do not write right-side group labels. Do not write "3 路", "2 路", or "1 路" anywhere.
Put all delta labels in one vertically aligned right-side column at the same x position, outside the plotted points/error bars. The labels must appear top to bottom as: "-0.45 (噪声内)", "+4.6", "+11.8", "+20.9", "+28.9", "+18.8", "+37.9".
No overlapping text, no labels inside point/error-bar clusters. Keep the right-side delta column aligned and readable.

## Fig. 6 prompt - ablation overview

Create a two-panel academic figure for a Chinese MV-OSR paper.

Canvas: wide scientific figure, white background, two panels side by side with enough spacing.

Panel (a) title: "(a) 归一化是首因，池化是次因"
Panel (a): grouped bar chart. Y axis "OSCR (%)", range 60 to 103.
X groups:
1. "mean\nper-view\nstd†"
2. "flatten\nper-view\nstd"
3. "mean\nabs-std"
4. "flatten\nabs-std★"
Each group has two bars: "单视角" grey-blue #8290B0 and "三视角" teal #079A87.
Data:
- 69.71 and 92.56, both hatched because only case-0
- 72.42±4.73 and 90.63±2.26
- 83.34±3.34 and 93.63±2.05
- 82.90±4.83 and 95.14±0.81
Label values above bars. Footnote below panel: "† 仅 case-0；其余为 5-case 均值 ± std".

Panel (b) title: "(b) 架构级融合：零增益或有害"
Panel (b): horizontal bar / forest plot. X axis "ΔOSCR (相对无融合基线, %)", range -2.2 to 1.0.
Draw pale grey vertical band from -0.81 to +0.81 and note above it: "灰带：主结果 ±1 std". Draw vertical x=0 line.
Rows from top to bottom:
- "F0 移除跨视角注意力": +0.05, grey, protocol "随机·5-case"
- "F2 瓶颈中层融合(最优)": -0.03, grey, protocol "随机·case-0"
- "E5 物理先验偏置": -0.04, grey, protocol "随机·5-case"
- "E4 分歧分量 β=1": -0.57, red, protocol "随机·5-case"
- "F1 融合特征喂检查器": -0.69, red, protocol "随机·case-0"
- "F6 瓶颈融合 @ LOSO": -1.79, red, protocol "LOSO·case-0"
Put numeric labels at bar ends and protocol labels outside the right edge, not over bars.
No overlapping title, axis label, footnote, or protocol text.
