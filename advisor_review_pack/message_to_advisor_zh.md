# 发导师预审消息草稿

老师您好，我整理了一版博士初期第一篇小论文的阶段性稿件，想请您帮忙判断研究定位和下一步是否需要补实验。

论文主题是低资源 regression ADMET 预测中的 descriptor-guided multi-to-uni transfer。核心问题是：训练阶段利用 RDKit descriptor / descriptor-access RF teacher，但推理阶段保持 ECFP4-only student。

目前主要结果是：

- 固定 `lambda_distill = 1.0` 的 Descriptor-Teacher AdapterFusion 在 11/15 个 regression dataset-ratio setting 中优于原始 AdapterFusion。
- 平均 RMSE delta 为 `-0.0656`。
- AdapterFusion 在 15/15 个 regression setting 中优于简单 DescPred。
- descriptor-access RF 仍然更强，所以本文不主张 SOTA，也不主张超过 descriptor-access model。
- classification 结果目前只作为 supplementary reference，不作为主结论。

附件中我放了：

1. `manuscript_submission.pdf`：英文论文初稿。
2. `one_page_summary_zh.md`：中文一页摘要。
3. `submission_checklist.md`：结果一致性和投稿风险检查。
4. `figures/` 和 `tables/`：主要图表。

想请您重点反馈：

1. regression-only 作为第一篇小论文主线是否可以接受？
2. 当前更适合定位为 method paper，还是 controlled empirical study？
3. 是否必须在投稿前补 Chemprop / pretrained encoder baseline？
4. 短期投稿目标更适合 workshop、BIBM-style conference，还是 journal short communication？

谢谢老师。
