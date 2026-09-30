# 开题报告目录说明

按国内工科硕士开题报告通行体例组织，共十节。两个版本内容一致：

- `开题报告.tex` / `开题报告.pdf`：LaTeX 版本，ctexart + xelatex，14 页。编译命令
  `xelatex 开题报告.tex` 跑两遍。技术路线图用 TikZ 绘制在正文内，无外部图片依赖。
- `开题报告.md`：Markdown 版本，便于快速改稿或粘贴到 Word 模板。技术路线图为 mermaid 源码。

三部分核心内容的位置：

- 主要研究内容：第三节 3.2，四项研究内容 + 3.3 关键问题
- 技术路线：第四节 4.1 总体路线图与 4.2 分层方法设计
- 论文目录大纲：第六节，三级提纲

## 内容与数据出处对照

| 开题报告位置 | 来源 |
|---|---|
| 一、二节（背景、现状） | `thesis/chapters/ch1_intro.tex` |
| 3.2 内容一、4.2 决策层 | `paper1/mvosr_paper_cn.tex`、`thesis/chapters/ch3_mvosr.tex` |
| 3.2 内容二、4.2 注册与校准层 | `paper3/fsgr_paper.tex`、`docs/RESULTS.md` 注册线各节 |
| 3.2 内容三、4.2 部署评估层 | `paper4/owda_paper.tex`、`docs/RESULTS.md` 开放世界线各节 |
| 3.2 内容四（探索性） | `docs/superpowers/specs/2026-07-26-attribute-prior-registration-preregistration.md` |
| 六、论文目录大纲 | `thesis/main.toc`，按开题体例扩为三级并预留 4.3.4 |
| 十、参考文献 | `thesis/chapters/references.tex` |

## 使用说明

- LaTeX 版的一级标题为「一、二、三」，二级为 1.1，引用用 `\upcite` 上标，与中文期刊体例一致；
  学校若提供开题报告模板，替换 `\ctexset` 与页面设置即可，正文不必改动。
- Markdown 版的 mermaid 图与 LaTeX 版的 TikZ 图内容相同；另有已排版的 `thesis/figs/fig_roadmap.pdf`
  可用于答辩幻灯片。
- 参考文献编号与 `thesis/chapters/references.tex` 的条目顺序不同，学位论文定稿时以论文内的编号为准。
- 表头信息、进度安排的具体月份需按学院日历与学校模板填写。
