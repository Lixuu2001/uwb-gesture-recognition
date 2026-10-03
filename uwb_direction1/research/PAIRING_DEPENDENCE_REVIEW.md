# 下一问题：片段级配对依赖，而非继续解除共享

2026-10-03。状态：完成新近邻/旧修复代码核查和配对工具；模型诊断尚未实现、预注册或运行，没有新分类训练或结果。共享参数直接干预已失败，按冻结规则停止该假设的宽度、种子和八折扩展。

## 尚可检查的问题与不可推断之处

三雷达输入按动作重复索引配对，但现有入口没有逐帧时间戳/同步证据。下一问题只能表述为“模型是否依赖片段级实例对应或跨雷达一致的用户风格”，不能称已掌握真实帧同步或三维几何。

考虑只在每个模型**实际训练用户**内比较：原配对、同类别同用户的不同实例重配、同类别不同用户重配。所有映射必须在完整源集合上生成，而不是microbatch内；逐路保持源样本边缘分布，并保存源/目标ID、用户、类别和固定点率。另有三路一致重排的恒等分布控制，用于发现索引或实现错误。禁止在诊断/测试用户上用标签构造这些配对。

这样最多衡量模型对构造干预的敏感性；合成组合不一定物理可实现，改变logits或训练CE并不等于发现跨用户泛化原因。是否值得训练，须在成本和测量定义核查之后另行冻结规则；不能看敏感性结果再改门槛。也不能因同类标签保留就自动宣称增强合理。

旧B-SHUF存在microbatch内打乱导致68.42%呈现未改变的问题，旧机制结论已撤回，修复的全集合无固定点版本未重训，见TEMPORAL_EXTENT_REVIEW.md所列原始报告。必须进一步读原修复代码以判断可复用的映射验证；不把该修复或撤回的旧结论当成新实验。旧过程监督、DTW和MixStyle失败记录仍保留。

## 新核查的近邻与边界

1. **Geodesic Multi-Modal Mixup for Robust Fine-Tuning**，作者仓库标注NeurIPS2023：[changdaeoh/multimodal-mixup](https://github.com/changdaeoh/multimodal-mixup)。本次实际读README、完整loss.py及main.py的sph_inter/do_train部分（约109–228行），未读论文全文，也未运行作者实验。
   - README blob `9076a44ffa6051e5d3c3731ebc99a4ddf9cd3bc2`：CLIP/OpenCLIP图文微调；Flickr30k、MSCOCO、SIMAT。
   - loss.py blob `16537669a32b1c903641f9115e92ad92fd389fea`：配对索引的双向对比CE，以及单位矩阵和反序矩阵构造的软目标。
   - main.py blob `6df3ad6b5c75e4f7333c3557d678baaf6ebe456d`：球面插值单模态特征、共同混合两模态特征，或图文混合构造非对角负样本。保持正配对对角和替换负样本的机制，并不是同类不同用户的雷达原片段重配。
   - 多模态混合、跨实例增强和对比目标已经存在，不能把“换样本/混视角”当作原创。作者的预训练图文共嵌入空间和检索目标也不等于当前三部UWB雷达分类设置。没有复制作者实现。
2. [RH-Lin/m3ixup](https://github.com/RH-Lin/m3ixup)：只读README（blob `ae9e510814aab8816a3374501c8f68fabeaf15b5`）和完整递归文件列表。README自称“Adapt and Explore: Multimodal Mixup for Representation Learning”的官方实现，但本次树只见README、图示、数据下载脚本与依赖文件，没有训练代码；README注释还留有GMC引用。不能据此声称已复现算法、确定发表信息或阅读该论文全文。
3. [HongyuZhu-s/MS-Mix](https://github.com/HongyuZhu-s/MS-Mix)：只读README（blob `3bfa09aca7997a3ed6c24b88e101abaa4ee58ac0`），题为Dual-Decoupling Augmentation for Multimodal Sentiment Analysis，陈述随机混合可能导致语义不一致。未核查正文、模型源码或发表状态，只作为需要进一步核查的线索，不用它的性能或机制为UWB有效性背书。

公开仓库精确blob已记录；未获得主文全文，不能称本领域综述完成。下一步优先复核原无固定点修复和这一诊断的可识别性，必要时否决这个候选；目前没有把它命名成新方法，也没有宣称论文贡献。用户目标仍未达成。

## 原修复代码实读与新的工具验证

实读原不可变快照的mv_transfer_v2/relation.py中global_within_class_derangement/shuffle_within_class，blob39e9cf45770188d3eeebda41efc3c0706038e290；train_student.py中mismatch映射与两个辅助目标使用处，blob3ddb314b7e8c9756ca6a3e27a138e06914976be0。旧B-SHUF改变的是教师关系/过程目标ID，分类输入不随之替换；下一问题考虑三路分类输入的片段配对，不能把两者实验含义混为一谈。原独立复核报告blob24cddf4d95dae81dc1a4c89159bbc917d475d21e。

发现旧全局修复的一个罕见兜底缺陷：64次尝试后若只剩一个固定点，它把该位置改成邻位已有目标，却没有交换回去，可能失去双射。例如强制候选[0,2,3,1]，实际旧函数返回[2,2,3,1]，changed_fraction=1，但只有3个不同目标。已用仅包含该原函数的AST和强制随机流复现，见OLD_FALLBACK_COUNTEREXAMPLE.json；**没有证据说明历史真实训练触发过这个罕见分支**，不得据此否定其他原结果。789文件原快照未修改。

新增独立tools/source_pairing.py：同(class,user)组内用Sattolo单循环，单例明确保留并报告；跨用户按类随机排列用户块，再按最大块长度循环移位。后者在最大用户占比超过一半时拒绝，因为无法同时实现跨用户且边缘分布不变的双射。两构造都不是均匀采样全部合法置换，也不是新算法；不能把changed_fraction=1代替目标唯一性检查。

合成组合检验覆盖2–40样本非单例组的780个种子案例、2/3/4用户组多种大小下的可行与不可行案例，另外检验池外标签/用户改动不影响映射、单例处理和非法ID池拒绝。精确数量和源码SHA见results/cloud_20261003_pairing_preparation。未执行任何真实模型敏感性诊断或新训练；下一接续仍需先冻结样本、模型时点、读数、对照、成本与推进规则，再决定是否值得运行。
