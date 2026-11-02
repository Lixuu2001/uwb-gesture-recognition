# 图2三种融合策略：术语核查与顶会配图参考库

本目录面向论文图2的重新设计。当前只整理方法定义、项目实现与原论文配图，不修改现有 PPTX。

本地材料共包括：

- `papers/`：23 篇论文 PDF；
- `figure_pages/`：34 张包含相关架构图的完整论文页；
- `focused_figures/`：27 张聚焦裁剪图；
- `gallery.html`：按方法分组的可视化浏览页。

## 一、先给结论：图2中建议采用的正式名称

| 当前概念 | 是否准确 | 中文期刊中建议的正式名称 | 图内短标签 |
|---|---|---|---|
| 权重共享编码器 | 基本准确，但容易被误解成“三个编码器模块” | **权重共享的逐视角双路径 Transformer 编码器** | 共享编码器 $E_\theta$ |
| Siamese 编码器 | 只可作类比，不建议作为主名称 | **Siamese-like / tied-weight encoder（孪生式权重共享编码）** | 不建议写 Siamese |
| 跨视角注意力 | 名称可用，但项目实现需进一步说清 | **编码后沿视角维的多头自注意力** | 视角维自注意力 |
| 特征级融合 | 正确，但太宽泛 | **编码后特征级融合（视角维自注意力）** | 特征级融合 |
| MBT | 不能单独当作通用模块名 | **MBT 式瓶颈中层融合**，首次出现写“多模态瓶颈 Transformer（MBT）式” | 瓶颈中层融合 |
| 共享瓶颈 token | 基本准确，但“共享”含义需要图示 | **跨视角融合瓶颈 token**；正文说明“逐视角更新后取平均并回写” | 瓶颈 tokens |
| 决策级融合 | 正确，但本项目融合的是距离证据 | **决策级距离聚合**或**决策级异常证据聚合** | 距离聚合 |

图2三列最稳妥的标题建议为：

1. **决策级距离聚合（本文采用）**
2. **编码后特征级融合（视角维自注意力）**
3. **MBT 式瓶颈中层融合**

这样比“无融合 / 跨视角注意力 / 共享瓶颈”更准确，因为三列比较的是**跨视角信息第一次发生交换的位置**，而不是三种完全不同的主干网络。

## 二、结合本项目代码的准确解释

核查依据：`D:\uwb_gestures\uwb_osr\model.py` 中的 `SharedDualPathEncoder`、`CrossViewFusion`、`BottleneckFusionLayer` 和 `MultiViewOSRNet.forward`。

### 1. 权重共享的逐视角编码器

三个视角并不是各自拥有一套独立参数。代码先把输入从 `(B,V,T,M)` 展平成 `(B×V,T,M)`，再统一送入同一个时间路径和距离路径编码器。因此可写为：

$$
f_v=E_\theta(X_v)+e_v,\qquad v\in\{L,T,R\}.
$$

- $E_\theta$：三个视角共用的同一个编码函数和同一组参数 $\theta$；
- $e_v$：区分左、上、右雷达身份的可学习视角嵌入；
- “共享权重”不等于“输出相同”：输入和视角嵌入不同，输出仍然不同。

通俗理解：像同一位检验员用同一套规则，分别检查三台雷达的数据；不是三位各有一套标准的检验员。

“Siamese”来自成对或多分支网络共享参数的经典画法。你的编码器在结构上是 **Siamese-like**，但任务不是孪生网络的配对相似度学习，因此中文期刊正文宜写“权重共享的逐视角编码器”，不要把“Siamese 网络”作为方法名。

### 2. 编码后特征级融合：实际是“沿视角维的自注意力”

代码 `CrossViewFusion` 在每一个对齐的慢时间位置 $t$，把三个视角 token 排成长度为 $V$ 的序列：

$$
Z_t=[z_{L,t};z_{T,t};z_{R,t}],\qquad
\widetilde Z_t=\operatorname{MHA}(Z_t,Z_t,Z_t)+Z_t.
$$

因此这里的 $Q,K,V$ 都来自同一个三视角序列，数学上属于**沿视角维的多头自注意力**。它确实完成了跨视角信息交换，所以“跨视角注意力”作为功能名称没有错；但它不是 CrossViT/MulT 那种“一个分支作 Query，另一个分支单独作 Key/Value”的定向 cross-attention。

还要注意两个实现细节：

- 当前没有屏蔽对角线，每个视角既能关注其他视角，也能关注自己；
- 该模块只作用于时间路径 token，距离路径保持编码后独立。只有在 `inspector_feats="fused"` 时，交互后的时间特征才送入逐视角检查器。

所以论文中原先“各视角以其余视角为键值”的表述与当前代码并不完全一致。若保持代码不变，建议改成：

> 在每个对齐的慢时间位置，将三路视角 token 组成视角序列并施加一层多头自注意力，使各视角表示在保留自身残差的同时聚合其他视角信息。

### 3. MBT 式瓶颈中层融合

代码把编码器最后 $K$ 个块替换为瓶颈融合块。每层中，各视角只与同一组少量瓶颈 token 拼接计算：

$$
[H_v^{l+1},B_v^{l+1}]
=\operatorname{Transformer}([H_v^l;B^l]),
\qquad
B^{l+1}=\frac{1}{V}\sum_v B_v^{l+1}.
$$

不同视角的完整 token 不直接两两相连；逐视角产生的瓶颈副本在层末取平均，平均后的 $B^{l+1}$ 再进入下一融合层。因此，少量瓶颈 token 是唯一跨视角通道。

通俗理解：三个视角不直接互发整份材料，只把摘要写到一块小白板上；白板汇总后，再供三方下一轮读取。

这与 Nagrani 等人提出的 MBT 核心思想一致，但你的对象是三个同质雷达视角，而原论文对象是音频与视频模态。因此应写“**MBT 式**”或“**受 MBT 启发的**瓶颈中层融合”，不要暗示完全复现了原始 MBT 架构。

### 4. 决策级距离聚合

本文主方法保持三路特征和逐视角检查器相互独立，直到每个视角都输出原型距离 $d_v$ 后才聚合：

$$
s(X)=\frac{\operatorname{mean}_v(d_v)}{\bar\sigma}
+\beta\frac{\operatorname{var}_v(d_v)}{\bar\sigma^2}.
$$

这里合并的是异常分数/距离证据，而不是特征向量或 token，所以“决策级距离聚合”比宽泛的“晚期融合”更精确。

通俗理解：三位检验员先各自独立打分，最后只汇总分数；不会在检查过程中互相修改观察结果。

## 三、容易混淆的几个术语

| 术语 | $Q$ 来自哪里 | $K,V$ 来自哪里 | 是否符合本项目 F1 |
|---|---|---|---|
| Self-attention | 同一序列 | 同一序列 | **符合**：序列维度是视角维 |
| Cross-attention | 目标流 | 另一个源流 | 不完全符合 |
| Co-attention | 两个流互相各做一次 cross-attention | 对方流 | 不符合当前实现；ViLBERT/LXMERT 更典型 |
| All-pairs feature exchange | 每个视角 | 其他全部视角，通常显式排除自身 | 当前实现没有排除自身 |
| Bottleneck attention | 视角 token 与少量 latent/bottleneck token | 同一拼接序列内 | **符合 F2 的核心机制** |

Perceiver 也使用 latent bottleneck，但它让 latent query 从大规模输入中读取信息，主要解决输入规模问题；MBT 则让瓶颈 token 成为多个独立模态/视角流之间的受限通信通道。两者不能画成同一种方法。

## 四、论文与配图索引

### A. 权重共享与多视角编码器

| 编号 | 论文与本地材料 | 重点图 | 适合借鉴 | 不要误用 |
|---|---|---|---|---|
| 02 | [Koch et al., Siamese Neural Networks for One-shot Image Recognition](papers/02_Koch2015_Siamese_OneShot.pdf) | Fig. 3；[裁剪图](focused_figures/02_Koch2015_Siamese_OneShot_p03_figure.png) | 两个分支完全同构、共享权重的经典视觉语法 | 你的任务不是配对相似度学习，不要把整列叫 Siamese Network |
| 03 | [Chen et al., SimCLR, ICML 2020](papers/03_SimCLR_ICML2020.pdf) | Fig. 2；[裁剪图](focused_figures/03_SimCLR_ICML2020_p02_figure.png) | 用同一个 $f(\cdot)$ 表示同一编码器处理多个 view | SimCLR 的 view 是数据增强，不是三台物理雷达 |
| 04 | [Su et al., MVCNN, ICCV 2015](papers/04_MVCNN_ICCV2015.pdf) | Fig. 1；[裁剪图](focused_figures/04_MVCNN_ICCV2015_p02_figure.png) | 多个观测、重复的共享 CNN、明确的首次汇合点 | MVCNN 在 view pooling 后形成单一描述符，不保留逐视角检查器 |
| 05 | [Feng et al., GVCNN, CVPR 2018](papers/05_GVCNN_CVPR2018.pdf) | Fig. 1；[裁剪图](focused_figures/05_GVCNN_CVPR2018_p02_figure.png) | 输入小图、分组层级、汇聚模块的层次感 | 分组依据与本项目雷达位置无关 |
| 06 | [Wei et al., View-GCN, CVPR 2020](papers/06_ViewGCN_CVPR2020.pdf) | Fig. 2；[裁剪图](focused_figures/06_ViewGCN_CVPR2020_p03_figure.png) | 视角节点、关系边、分层汇聚的精细图形语言 | 你的三视角没有显式图卷积 |
| 07 | [Hamdi et al., MVTN, ICCV 2021](papers/07_MVTN_ICCV2021.pdf) | Fig. 2；[裁剪图](focused_figures/07_MVTN_ICCV2021_p03_figure.png) | 用真实/渲染观测图提升流程图观感 | 论文重点是学习相机视点，不是融合位置 |
| 09 | [Zhao et al., Surrounding Cameras to BEV, CVPRW 2022](papers/09_SurroundingCameras_BEV_CVPRW2022.pdf) | Fig. 2；[裁剪图](focused_figures/09_SurroundingCameras_BEV_CVPRW2022_p04_figure.png) | 多相机输入 + shared CNN + Transformer 的整洁分区 | 相机到 BEV 是坐标查询，不等于普通多视角融合 |

### B. Cross-attention、co-attention 与跨视角交互

| 编号 | 论文与本地材料 | 重点图 | 适合借鉴 | 不要误用 |
|---|---|---|---|---|
| 10 | [Vaswani et al., Attention Is All You Need, NeurIPS 2017](papers/10_AttentionIsAllYouNeed_NeurIPS2017.pdf) | Fig. 1/2；[裁剪图](focused_figures/10_AttentionIsAllYouNeed_NeurIPS2017_p03_figure.png) | Add & Norm、残差回路、粗细线条的标准画法 | 原图是编码器—解码器，不是三视角结构 |
| 11 | [Lu et al., ViLBERT, NeurIPS 2019](papers/11_ViLBERT_NeurIPS2019.pdf) | Fig. 1/2；[总览](focused_figures/11_ViLBERT_NeurIPS2019_p02_figure.png)、[co-attention](focused_figures/11_ViLBERT_NeurIPS2019_p03_figure.png) | 双流、虚线 K/V 交换、残差主路与交换支路的区分 | “co-attention”意味着双向交换，不等于你的视角维 self-attention |
| 12 | [Tan & Bansal, LXMERT, EMNLP 2019](papers/12_LXMERT_EMNLP2019.pdf) | Fig. 1；[裁剪图](focused_figures/12_LXMERT_EMNLP2019_p03_figure.png) | 单流编码器与跨模态编码器分区，模块层级清楚 | 模态专属编码器并不共享权重 |
| 13 | [Tsai et al., MulT, ACL 2019](papers/13_MulT_ACL2019.pdf) | Fig. 2/3；[总览](focused_figures/13_MulT_ACL2019_p03_figure.png)、[Q/K/V细节](focused_figures/13_MulT_ACL2019_p04_figure.png) | 明确标注“谁作 Q、谁作 K/V”的最佳参考之一 | 你的 F1 不是这种定向 pairwise cross-attention |
| 14 | [Chen et al., CrossViT, ICCV 2021](papers/14_CrossViT_ICCV2021.pdf) | Fig. 2–4；[整体结构](focused_figures/14_CrossViT_ICCV2021_p03_figure.png)、[融合分类](focused_figures/14_CrossViT_ICCV2021_p04_figure.png)、[细节](focused_figures/14_CrossViT_ICCV2021_p05_figure.png) | 用四个小示意并列比较融合方式，重点非常集中 | CrossViT 用一支 CLS token 读取另一支 patch token，与当前 F1 不同 |
| 15 | [Zhou & Krähenbühl, Cross-View Transformers, CVPR 2022](papers/15_CrossViewTransformers_CVPR2022.pdf) | Fig. 2；[裁剪图](focused_figures/15_CrossViewTransformers_CVPR2022_p03_figure.png) | query、相机特征、几何射线和输出地图的精致组织 | 这里的“cross-view”是相机视角到地图视角，不是相机之间全连接 |
| 17 | [Bai et al., TransFusion, CVPR 2022](papers/17_TransFusion_CVPR2022.pdf) | Fig. 2；[裁剪图](focused_figures/17_TransFusion_CVPR2022_p03_figure.png) | 两阶段 query 传递、主流程与辅助信息的线宽分级 | 这是 LiDAR-camera 目标检测，语义与本文不同 |
| 23 | [Huang et al., Multi-View Transformer for 3D Visual Grounding, CVPR 2022](papers/23_MultiViewTransformer_3DGrounding_CVPR2022.pdf) | Fig. 3；[裁剪图](focused_figures/23_MultiViewTransformer_3DGrounding_CVPR2022_p04_figure.png) | 多视角 token 堆叠与 Transformer 层次 | “multi-view”来自旋转生成的空间视点，不等于多雷达同步观测 |

### C. MBT 与瓶颈 token

| 编号 | 论文与本地材料 | 重点图 | 适合借鉴 | 不要误用 |
|---|---|---|---|---|
| 19 | [Nagrani et al., Attention Bottlenecks for Multimodal Fusion, NeurIPS 2021](papers/19_MBT_NeurIPS2021.pdf) | Fig. 1/2；[融合层级比较](focused_figures/19_MBT_NeurIPS2021_p02_figure.png)、[MBT结构](focused_figures/19_MBT_NeurIPS2021_p03_figure.png) | **与你 F2 最直接的理论和配图来源**；颜色区分 token，虚线框表示融合层，瓶颈 token 用少量特殊节点 | 原论文是音频—视频；你的方法是三视角同质输入且瓶颈副本取平均 |
| 20 | [Jaegle et al., Perceiver, ICML 2021](papers/20_Perceiver_ICML2021.pdf) | Fig. 1；[裁剪图](focused_figures/20_Perceiver_ICML2021_p02_figure.png) | 大输入到小 latent array 的“宽—窄”视觉隐喻 | latent bottleneck 主要压缩输入，不是多流之间的唯一通信通道 |
| 24 | [Shvetsova et al., Everything at Once, CVPR 2022](papers/24_EverythingAtOnce_CVPR2022.pdf) | Fig. 2；[裁剪图](focused_figures/24_EverythingAtOnce_CVPR2022_p04_figure.png) | 共享 Transformer 与多种模态组合的视觉编码 | 其共享机制和 MBT 瓶颈机制不同 |

### D. 其他融合方式与反例

| 编号 | 论文与本地材料 | 重点图 | 适合借鉴 | 不要误用 |
|---|---|---|---|---|
| 08 | [Chen et al., Transformer-based Multi-view 3D Reconstruction, ICCV 2021](papers/08_MultiView3DReconstructionTransformer_ICCV2021.pdf) | Fig. 1；[裁剪图](focused_figures/08_MultiView3DReconstructionTransformer_ICCV2021_p03_figure.png) | 2D-view encoder、3D-volume decoder 与 attention inset 的多层级组织 | 输出目标是体素重建 |
| 16 | [Wang et al., Multimodal Token Fusion for Vision Transformers, CVPR 2022](papers/16_TokenFusion_CVPR2022.pdf) | Fig. 1/2；[同质模态](focused_figures/16_TokenFusion_CVPR2022_p04_figure.png)、[异质模态](focused_figures/16_TokenFusion_CVPR2022_p05_figure.png) | token 级替换、可靠性分数和虚线投影的精细表达 | 它是动态替换低信息 token，不是普通 cross-attention 或 MBT |
| 18 | [Li et al., DeepFusion, CVPR 2022](papers/18_DeepFusion_CVPR2022.pdf) | Fig. 1–3；见 `figure_pages/` | 真实传感器图 + 特征对齐 inset，适合学习“配图增强观感” | 核心问题是异构增广与几何对齐 |
| 21 | [Ngiam et al., Multimodal Deep Learning, ICML 2011](papers/21_MultimodalDeepLearning_ICML2011.pdf) | Fig. 3；[裁剪图](focused_figures/21_MultimodalDeepLearning_ICML2011_p03_figure.png) | 早期/联合表示学习的经典分层结构 | RBM/自编码器年代的模块不能直接套到 Transformer 图上 |
| 22 | [Baltrušaitis et al., Multimodal Machine Learning: A Survey and Taxonomy, TPAMI 2018](papers/22_MultimodalMachineLearning_Survey_TPAMI2018.pdf) | 分类与定义；见 `figure_pages/` | 用于核对 representation、alignment、fusion 等术语 | 是综述，不是可直接复现的具体融合模块 |

## 五、最适合下一版图2的视觉组合

不建议照抄某一张论文图，而应组合三套成熟视觉语言：

1. **总体对比骨架参考 MBT Fig. 1**：三列只比较信息交换发生的深度；同色 token、灰色层级轴、虚线局部作用域。
2. **共享编码器参考 MVCNN Fig. 1**：三张真实 CIR 小图进入同一个标有 $E_\theta$ 的共享模块，通过重复的小型同构分支和“shared $\theta$”括号明确参数共享。
3. **特征级细节参考 Transformer / ViLBERT**：主残差流用较粗实线，注意力权重或 K/V 信息流用较细虚线；但依据项目代码画成三视角 token 的视角维自注意力，而不是照搬 ViLBERT 的双向 K/V 交换。
4. **瓶颈细节参考 MBT Fig. 2**：三个视角 token 流保持分离，仅用 3–4 个高饱和小圆 token 连接；“逐视角更新 → 平均 → 回写”用一个小型聚合节点表现。
5. **删除后半段冗余**：每列在“逐视角输出 / 距离证据”处收束，分类头、检查器内部 MLP、最终阈值判断不展开。图2只回答“什么时候交换、交换什么、交换带宽多大”。

### 推荐的三列信息密度

- 左列：真实 CIR 三视角小图 → 共享 $E_\theta$ → 三个独立特征/距离 → 一个小型距离聚合符号。
- 中列：真实 CIR → 共享 $E_\theta$ → 对齐时间 token 的视角维自注意力 inset → 三个耦合后的逐视角特征。
- 右列：真实 CIR → 编码器前段 → 少量瓶颈 token 交换 inset → 编码器后段 → 三个逐视角特征。

三列共同保留的内容采用低饱和灰蓝色；唯一变化的交换位置采用一种统一强调色。不要给三种策略各换一套主色，否则读者会误以为除交换层级外还有其他架构差异。

## 六、原始在线来源

- [MVCNN — ICCV 2015, CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.html)
- [View-GCN — CVPR 2020, CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_View-GCN_View-Based_Graph_Convolutional_Network_for_3D_Shape_Analysis_CVPR_2020_paper.html)
- [CrossViT — ICCV 2021, CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_CrossViT_Cross-Attention_Multi-Scale_Vision_Transformer_for_Image_Classification_ICCV_2021_paper.html)
- [Cross-View Transformers — CVPR 2022, CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Cross-View_Transformers_for_Real-Time_Map-View_Semantic_Segmentation_CVPR_2022_paper.html)
- [TokenFusion — CVPR 2022, CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Multimodal_Token_Fusion_for_Vision_Transformers_CVPR_2022_paper.html)
- [TransFusion — CVPR 2022, CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Bai_TransFusion_Robust_LiDAR-Camera_Fusion_for_3D_Object_Detection_With_Transformers_CVPR_2022_paper.html)
- [MBT — NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/76ba9f564ebbc35b1014ac498fafadd0-Abstract.html)
- [Perceiver — ICML 2021, PMLR](https://proceedings.mlr.press/v139/jaegle21a.html)
- [MulT — ACL 2019](https://aclanthology.org/P19-1656/)
- [LXMERT — EMNLP 2019](https://aclanthology.org/D19-1514/)
- [ViLBERT — NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html)
- [SimCLR — ICML 2020, PMLR](https://proceedings.mlr.press/v119/chen20j.html)

> 版权说明：本目录中的论文与截图仅作为论文阅读、方法核查和内部配图设计参考。投稿图应根据本项目的真实结构重新绘制，不直接拼接或复用原论文图形素材。
