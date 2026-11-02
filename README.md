# UWB 手势识别

基于 UWB 雷达的手势识别研究项目，包含原始项目源码、开放集识别与多视角融合相关实验，以及云端迁移和研究记录。

## 项目入口

- [原始项目](uwb_direction1/original/project/)：保留上传的源码与研究文档快照。
- [训练与评估入口](uwb_direction1/original/project/main.py)、[配置](uwb_direction1/original/project/config.py)和[模型代码](uwb_direction1/original/project/uwb_osr/)。
- [云端研究说明](uwb_direction1/README.md)：迁移说明、实验记录和相关入口。
- [研究状态](uwb_direction1/STATUS.json)与[实验结果目录](uwb_direction1/results/)：具体实验结论以对应记录为准。
- [源码来源清单](uwb_direction1/original/SOURCE_MANIFEST.json)：记录原始文件来源。

## 数据与运行

原始雷达数组、模型权重及部分运行产物未纳入普通 Git，运行实验需要按项目配置准备相应数据与环境。迁移操作见[迁移说明](uwb_direction1/MIGRATION.zh-CN.md)。

迁移工具的检查命令（不等同于识别模型效果验证）：

```bash
python -m unittest discover -s uwb_direction1/tests -v
```

原项目保留在现有目录中，以维持源码、实验记录及相对路径之间的对应关系。
