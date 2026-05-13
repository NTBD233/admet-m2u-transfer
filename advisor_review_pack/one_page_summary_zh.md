# 博士初期第一篇论文导师审阅摘要

## 题目

**Descriptor-Teacher Guided Multi-to-Uni Transfer for Low-Resource Regression ADMET Prediction**

暂定中文理解：

**面向低资源回归 ADMET 预测的 descriptor-teacher 引导 multi-to-uni 迁移方法**

## 研究问题

低资源 ADMET 预测中，标签数据有限，但 RDKit descriptor 等分子知识便宜、稳定且有预测价值。直接把 descriptor 拼接到模型输入中通常有效，但推理阶段就依赖 descriptor。

本文研究一个更受控的问题：

> 能否在训练阶段利用 descriptor-access teacher 的知识，但在推理阶段只使用 ECFP4 fingerprint？

## 方法概述

主方法为 **Descriptor-Teacher AdapterFusion**。

模型结构保持轻量：

- student 输入：ECFP4 fingerprint
- student 主体：ECFP encoder + pseudo-descriptor adapter + gated fusion
- auxiliary supervision：预测标准化 RDKit descriptors
- teacher：`ECFP4_Desc_RF`
- distillation：student 预测向 descriptor-access RF teacher 预测对齐

训练目标：

```text
task loss
+ descriptor reconstruction loss
+ teacher distillation loss
```

主配置：

```text
lambda_transfer = 0.1
lambda_distill = 1.0
```

推理阶段只用 ECFP4，不使用真实 descriptor。

## 实验设计

主实验聚焦 5 个 TDC regression ADMET 数据集：

- Caco-2 permeability
- lipophilicity
- aqueous solubility
- volume of distribution
- plasma protein binding

低资源设置：

- train ratio: 10%, 20%, 50%
- seeds: 42, 123, 3407
- primary metric: RMSE

主要 baseline：

- ECFP4 MLP
- ECFP4 MLP + descriptor prediction
- AdapterFusion
- descriptor-concat MLP
- ECFP4 RF
- ECFP4+descriptor RF
- distilled AdapterFusion

classification ADMET 结果仅作为 supplementary，不作为主结论。

## 主要结果

1. **Descriptor-access baseline 很强。**

   `ECFP4+Desc RF` 在多数 regression setting 中仍然优于 neural student，说明 descriptor 信息确实有价值。

2. **AdapterFusion 优于简单 DescPred。**

   在 15/15 个 regression dataset-ratio setting 上，结构化 pseudo-descriptor adapter + gated fusion 比简单辅助 descriptor prediction 更好。

3. **Teacher distillation 有效但有限。**

   固定 `lambda_distill = 1.0` 时，distilled AdapterFusion 在 **11/15** 个 regression dataset-ratio setting 中优于原始 AdapterFusion。

   平均 RMSE delta：

   ```text
   -0.0656
   ```

4. **Validation-selected lambda 不是更强默认策略。**

   根据 validation RMSE 选择 lambda 只在 **2/15** 个 setting 中优于固定 `lambda_distill = 1.0`，但在 **12/15** 个 setting 中优于 base AdapterFusion。

5. **Adaptive teacher weighting 是 negative ablation。**

   validation teacher-advantage adaptive weighting：

   - 优于 base AdapterFusion：9/15
   - 优于 fixed lambda=1.0：3/15
   - 优于 best fixed lambda：1/15
   - 平均比 fixed lambda=1.0 差：0.0817 RMSE

## 当前论文主张

支持的主张：

> Descriptor-teacher distillation can improve an ECFP-only AdapterFusion student for low-resource regression ADMET prediction.

不主张：

> 不是 SOTA ADMET predictor。

> 不声称超过 descriptor-access RF。

更准确的定位：

> A controlled study and lightweight method for descriptor-guided multi-to-uni transfer in low-resource regression ADMET.

## 创新点

本文创新点不是提出大模型，而是提出并验证一个受控的轻量 transfer 设定：

> 用 descriptor-access teacher 在训练阶段指导 ECFP-only student，使 student 在推理阶段保持轻量，同时部分吸收 descriptor knowledge。

具体贡献：

- 将 descriptor knowledge transfer 明确放在 multi-to-uni ADMET 低资源设定中。
- 比较 simple auxiliary descriptor prediction 与 structured AdapterFusion。
- 引入 ECFP4+descriptor RF teacher distillation。
- 报告 positive result、lambda sensitivity 和 adaptive negative ablation。

## 局限

- 主结论目前限于 regression ADMET。
- classification 结果不够稳定。
- distilled student 仍普遍弱于 descriptor-access RF。
- 尚未加入大规模 pretrained molecular encoder。
- adaptive teacher weighting 目前是失败尝试，需要更细粒度的 teacher reliability 建模。

## 建议投稿定位

当前更适合：

- AI for drug discovery / molecular ML workshop
- BIBM-style bioinformatics / biomedical AI conference paper
- 后续增强后再考虑 Journal of Cheminformatics

不建议当前直接定位为：

- ICML / NeurIPS / ICLR main conference
- SOTA ADMET benchmark paper

## 希望导师重点反馈的问题

1. 这个问题设定是否足够适合作为博士初期第一篇论文？
2. 当前主张是否应该定位为 method paper，还是 controlled empirical study？
3. 是否需要补 Chemprop 或 pretrained encoder baseline？
4. 是否应该继续强化 regression，还是尝试解释 classification 失败？
5. 更合适的投稿目标是 workshop、BIBM-style conference，还是 journal short communication？

## 当前封版状态

- 英文 LaTeX 初稿已完成投稿前润色。
- 主表已包含 `DescPred`，因此可以直接支撑 AdapterFusion 优于简单 auxiliary descriptor prediction 的判断。
- 当前版本适合导师预审；若目标提高到 Q1/journal，则建议另起扩展版方案，而不是继续在当前短稿上小修。
