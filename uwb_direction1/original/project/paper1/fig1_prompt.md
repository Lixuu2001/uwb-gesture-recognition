# 架构图(图 1)绘图提示词

> 用途:把下面的提示词喂给任何画图工具(GPT/Claude 画 SVG·draw.io/Mermaid AI·
> PPT AI·Figma AI 等),复现 `figs/fig1_architecture.tex` 同款 MV-OSR 架构图。
> 中文版在前,English version at the end(多数工具英文效果更稳)。
> 本仓库的参考实现是 TikZ(`figs/fig1_architecture.tex`),编译产物
> `figs/fig1_architecture.pdf` 可作为目标对照。

---

## 中文提示词(直接复制)

请画一张学术论文用的系统架构图(矢量风格、白底、横向布局,宽高比约 3:1),
主题是"多视角 UWB 雷达开集手势识别(MV-OSR)"。严格按下面的网格布局、模块清单、
箭头清单和配色执行,不要自行增删模块或改变箭头指向。

### 总体布局(从左到右 6 列,中部 3 条平行泳道)

- 画布分 6 列:① 输入 ② 归一化 ③ 共享编码器 ④ 特征码 ⑤ 开集检察(下)/
  闭集分类(上) ⑥ 判决输出。
- 中部有 3 条水平泳道(从上到下):L 泳道、T 泳道、R 泳道,等间距;
  列①②③④⑤ 的逐视角模块都精确对齐在这 3 条泳道上。
- 阶段一(闭集分类)整体放在右上方第 5–6 列上方一行;阶段二(开集检察)
  占据右半部的 3 条泳道。

### 模块清单(共 16 个圆角矩形 + 2 个虚线分组框)

列①(输入,3 个深蓝描边浅蓝填充框,泳道对齐):
1. 「左雷达 L\nCIR 90×189」 2. 「上雷达 T\nCIR 90×189」 3. 「右雷达 R\nCIR 90×189」
   - 三个框外套一个灰色虚线圆角框,框下方注释小字:
     「训练期随机丢视角:每批次以 p_drop=0.5 随机保留 1–2 路」

列②(1 个青绿描边浅绿填充高框,纵向横跨 3 条泳道):
4. 「振幅归一化\n|x| 后逐样本、逐视角除以标准差(abs_std)」

列③(1 个浅蓝大容器框,纵向横跨 3 条泳道,标题「共享双路径编码器」,内部上下两个白底小框 + 底部小字):
5. 内框 A:「时间路径 Transformer(90 个时间 token × 4 层)」
6. 内框 B:「距离路径 Transformer(189 个距离 token × 4 层)」
   底部小字:「权重共享(三视角同一套参数)+ 视角嵌入;两路 token 展平拼接」

列④(3 个灰蓝小框,泳道对齐):
7. 「f_L(8928 维)」 8. 「f_T(8928 维)」 9. 「f_R(8928 维)」

列⑤下(阶段二,3 个红描边浅红填充框,泳道对齐,外套红色虚线分组框):
10. 「检察器 η_k̂^(L):d_L=‖l_L−φ_k̂^(L)‖」
11. 「检察器 η_k̂^(T):d_T=‖l_T−φ_k̂^(T)‖」
12. 「检察器 η_k̂^(R):d_R=‖l_R−φ_k̂^(R)‖」
13. 其右侧 1 个红框(对齐中间泳道):「多视角聚合(异常分数)\n s = d̄/σ̄ + β·var(d_v)/σ̄²」
    红色虚线分组框下方标签:「阶段二:逐类 × 逐视角深度检察(开集判决)」

列⑤上(阶段一,3 个灰紫描边框排成一行,外套灰紫虚线分组框,框上方标签「阶段一:闭集分类」):
14. 「特征聚合」 15. 「闭集分类器\n(交叉熵训练)」 16. 「预测类 k̂」(此框用橙色描边、浅橙填充,表示关键中间结果)

列⑥(2 个输出框,上下排列):
17. 「s≤τ:接受 → 输出已知类 k̂」(绿描边浅绿填充)
18. 「s>τ:拒识 → 未知手势」(红描边浅红填充)

### 箭头清单(必须严格执行,全部直角折线、箭头实心三角)

主流(每条泳道 4 段水平箭头,L/T/R 三条泳道各一组):
- 输入框 → 归一化框(进入框左边、对应泳道高度)
- 归一化框 → 编码器容器框(同泳道高度)
- 编码器容器框 → 对应 f 框
- f 框 → 对应检察器框

阶段一支路(电路母线画法):
- 在 f 框与检察器框之间画一条竖直母线,在 3 条泳道与母线交点画实心圆点
  (表示三路都接入),母线向上延伸到阶段一行高,再水平接入「特征聚合」左侧
- 特征聚合 → 闭集分类器 → 预测类 k̂(两段水平箭头)

关键虚线箭头(最容易画错,务必准确):
- 从「预测类 k̂」底部出发,向下→向左→向下,垂直插入红色虚线分组框
  (三个检察器的外框)的顶边中点;箭头中段标注小字
  「为预测类 k̂ 选取对应检察器组(每视角一个)」;此箭头用橙色虚线。

阶段二汇聚:
- 三个检察器框右侧 → 「多视角聚合」框左侧(上下两条先右行一小段再直角折到
  中间泳道汇入,中间一条直行)
- 「多视角聚合」→ 右侧分叉为两条直角折线,分别指向「接受」「拒识」两框

### 配色(NPG 科研风,低饱和浅填充 + 同色系描边)

- 深蓝 #3C5488(输入描边)/ 填充 #ECEFF6
- 青绿 #00A087(归一化、接受框)/ 填充 #E6F4F1
- 浅蓝 #4DBBD5(编码器)/ 填充 #EBF6FA
- 灰蓝 #5C6B8A(特征码)/ 填充 #F0F2F7
- 灰紫 #8491B4(阶段一)/ 填充 #F1F2F8
- 橙 #E18727(预测类 k̂、选择检察器的虚线)/ 填充 #FDF3E7
- 红 #E64B35(阶段二、拒识)/ 填充 #FDEEEB
- 箭头中性灰 60% 黑;正文黑 85%;字体无衬线(中文黑体系,西文 Arial/Helvetica)

### 底部小注(图内左下,小一号灰字)

「注:架构中不含跨视角特征交换——消融表明其无增益甚至有害(正文 §3.5);
跨视角信息仅在决策级聚合。V=1 时方差项消失,退化为 Eom 等人的单视角方法。」

### 必须避免的错误

1. 三条泳道上的模块(输入/特征码/检察器)必须像素级水平对齐,箭头水平不歪斜;
2. 「预测类 k̂ → 检察器组」的虚线必须指向三个检察器的整体分组框,
   不能指向某一个检察器、更不能指向聚合框;
3. 母线三个交点要画实心圆点,否则读者会以为线只是交叉而非连接;
4. 阶段一在上、阶段二在泳道层,两个虚线分组框不得互相重叠;
5. 编码器是"一个"容器(权重共享),不要画成三份;
6. 不要添加任何"跨视角注意力/融合"模块(本架构刻意没有)。

---

## English prompt (for tools that work better in English)

Draw a publication-quality system architecture diagram (vector style, white
background, horizontal flow, ~3:1 aspect) for "Multi-View UWB Radar Open-Set
Gesture Recognition (MV-OSR)". Follow the grid, module list, arrow list and
palette EXACTLY; do not add, remove, or re-route anything.

Layout: 6 columns left→right — (1) inputs, (2) normalization, (3) shared
encoder, (4) per-view feature codes, (5) open-set inspection (lower, on three
lanes) plus closed-set branch (upper row), (6) decisions. Three horizontal
swim-lanes (L/T/R) run through columns 1–5; all per-view blocks sit exactly on
these lanes.

Modules:
- Col 1: three rounded boxes "Left radar L / CIR 90×189", "Top radar T / ...",
  "Right radar R / ..." (deep-blue stroke #3C5488, fill #ECEFF6), wrapped in a
  gray dashed group box captioned below: "Training-time view dropout: keep a
  random subset of 1–2 views per batch with p_drop = 0.5".
- Col 2: one tall box spanning all lanes: "Amplitude normalization — |x| then
  per-sample, per-view std division (abs_std)" (teal #00A087, fill #E6F4F1).
- Col 3: one large container spanning lanes titled "Shared dual-path
  Transformer encoder" (light-blue #4DBBD5, fill #EBF6FA) containing two white
  inner boxes: "Temporal path Transformer (90 time tokens × 4 layers)" and
  "Range path Transformer (189 range tokens × 4 layers)", footer note:
  "Weights shared across views + view embedding; flatten & concat both token
  streams".
- Col 4: three small boxes "f_L (8928-d)", "f_T", "f_R" (gray-blue #5C6B8A,
  fill #F0F2F7), lane-aligned.
- Col 5 lower (Stage 2): three boxes "Inspector η_k̂^(v): d_v = ‖l_v −
  φ_k̂^(v)‖" for v∈{L,T,R} (red #E64B35, fill #FDEEEB), lane-aligned, wrapped
  in a red dashed group box labeled below "Stage 2: per-class × per-view deep
  inspection (open-set decision)"; to their right one red box on the middle
  lane: "Multi-view aggregation (anomaly score) — s = d̄/σ̄ + β·var(d_v)/σ̄²".
- Col 5 upper (Stage 1): a row of three boxes "Feature aggregation" → "Closed-
  set classifier (cross-entropy)" → "Predicted class k̂" (gray-purple #8491B4,
  fill #F1F2F8; the k̂ box uses orange #E18727, fill #FDF3E7), wrapped in a
  dashed group box labeled above "Stage 1: closed-set classification".
- Col 6: two terminal boxes: "s ≤ τ: accept → output known class k̂" (teal) and
  "s > τ: reject → unknown gesture" (red).

Arrows (solid triangular heads, orthogonal routing, neutral gray):
- Per lane: input → normalization → encoder container → f_v → inspector_v
  (four straight horizontal segments per lane, entering boxes at lane height).
- Stage-1 tap: a vertical bus line just right of the f-column with filled
  junction dots where it crosses each of the three lanes, rising to the
  Stage-1 row and entering "Feature aggregation"; then aggregation →
  classifier → k̂.
- KEY dashed arrow (most error-prone): from the BOTTOM of "Predicted class k̂",
  route down → left → down into the TOP EDGE of the red dashed GROUP box that
  encloses the three inspectors (NOT into any single inspector, NOT into the
  aggregation box). Orange dashed; midpoint label: "select the inspector set
  of predicted class k̂ (one per view)".
- Stage-2 merge: the three inspectors → left edge of the aggregation box
  (outer two route with one right-angle bend to the middle lane); aggregation
  → fork into the two terminal boxes.
- Bottom-left footnote (small gray): "Note: no cross-view feature exchange
  anywhere — ablations showed zero gain or harm (Sec. 3.5); cross-view
  information is aggregated only at decision level. With V = 1 the variance
  term vanishes and the architecture reduces to Eom et al.'s single-view
  method."

Avoid: misaligned lanes; the dashed k̂-arrow pointing at a single inspector or
at the aggregation box; missing junction dots on the bus; overlapping group
boxes; drawing three encoders instead of one shared container; adding any
fusion/attention module.
