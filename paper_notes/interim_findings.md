# Interim Findings

This file is generated from complete three-seed result groups only.

## Current Signal

- DescConcat improves over ECFP_MLP in 28/30 complete settings.
- AdapterFusion improves over DescPred in 24/30 complete settings.
- Regression teacher-distilled AdapterFusion improves over base AdapterFusion
  in 8/15 complete regression settings at `lambda_distill = 0.1`.
- A regression distillation lambda sweep over `0.01`, `0.1`, `0.3`, and `1.0`
  found `lambda_distill = 1.0` to be the strongest setting by aggregate RMSE:
  it improves over base AdapterFusion in 11/15 complete regression settings.
- Regression teacher-distilled AdapterFusion improves over `ECFP4_RF` in 4/15
  complete regression settings, but remains below the `ECFP4_Desc_RF` teacher
  on average.
- Positive values in improvement columns mean better performance.

## Regression Distillation Update

Full regression expansion completed 45/45 seed runs. Mean RMSE delta versus
base AdapterFusion is -0.0221, while the mean RMSE gap to the descriptor-access
`ECFP4_Desc_RF` teacher is 0.7916. This supports teacher distillation as a
useful but insufficient method iteration: it improves some tasks, especially
`caco2_wang`, but it is not stable enough at fixed `lambda_distill = 0.1` to
claim broad superiority.

## Regression Distillation Lambda Sweep

The regression-only distillation sweep completed all planned runs:

- `lambda_distill = 0.01`: 45/45 seed runs, 15/15 complete settings.
- `lambda_distill = 0.1`: 45/45 seed runs, 15/15 complete settings.
- `lambda_distill = 0.3`: 45/45 seed runs, 15/15 complete settings.
- `lambda_distill = 1.0`: 45/45 seed runs, 15/15 complete settings.

Aggregate comparison:

| lambda_distill | complete settings | beats AdapterFusion | beats ECFP4_RF | mean delta vs AdapterFusion | mean gap to ECFP4_Desc_RF |
| --- | --- | --- | --- | ---: | ---: |
| 0.01 | 15/15 | 8/15 | 4/15 | 0.0075 | 0.8211 |
| 0.1 | 15/15 | 8/15 | 4/15 | -0.0221 | 0.7916 |
| 0.3 | 15/15 | 10/15 | 4/15 | -0.0046 | 0.8090 |
| 1.0 | 15/15 | 11/15 | 4/15 | -0.0656 | 0.7481 |

Interpretation:

- `lambda_distill = 1.0` is the current best global setting among tested
  values: it has the largest average RMSE improvement over base AdapterFusion
  and the smallest average gap to the descriptor-access RF teacher.
- The improvement is still not enough to beat `ECFP4_RF` broadly; all lambda
  values beat `ECFP4_RF` in only 4/15 settings.
- Best lambda is dataset-dependent: `caco2_wang` prefers `0.1` or `0.3`,
  `lipophilicity_astrazeneca` is mixed, `ppbr_az` favors stronger distillation
  at low ratios, and `vdss_lombardo` remains marginal.
- The strongest next methodological question is not whether distillation helps,
  but how to make teacher guidance adaptive across tasks and train ratios.

## Distillation Mechanism Diagnostics

A seed-level prediction diagnostic was added to compare base AdapterFusion,
distilled AdapterFusion, and the `ECFP4_Desc_RF` teacher on the same test
molecules.

Main diagnostic results:

| lambda_distill | complete settings | beats base | mean delta RMSE | mean delta teacher RMSE | mean delta teacher corr |
| --- | --- | --- | ---: | ---: | ---: |
| 0.01 | 15/15 | 8/15 | 0.0075 | 0.0392 | -0.0002 |
| 0.1 | 15/15 | 8/15 | -0.0221 | 0.0117 | 0.0042 |
| 0.3 | 15/15 | 10/15 | -0.0046 | 0.0087 | 0.0082 |
| 1.0 | 15/15 | 11/15 | -0.0655 | -0.1664 | 0.0097 |

Interpretation:

- `lambda_distill = 1.0` is the only tested setting that, on average, both
  improves test RMSE and moves the student closer to the RF teacher in
  prediction space.
- This supports the claim that the performance gain is at least partly true
  teacher transfer, not only random regularization.
- The effect is task-dependent. `ppbr_az` benefits strongly from stronger
  teacher matching, while `vdss_lombardo` shows that closer teacher matching can
  fail to improve downstream RMSE.
- The next method iteration should consider adaptive teacher weighting rather
  than a single fixed global lambda.

## Adaptive Distillation Check

A first adaptive teacher-weighting strategy was tested:

> `lambda_effective = lambda_max * max((base_valid_rmse - teacher_valid_rmse) / base_valid_rmse, 0)`

with `lambda_max = 1.0`. This keeps the model structure unchanged and only
changes the per-run distillation weight according to validation teacher
advantage.

Result summary:

| comparison | result |
| --- | --- |
| complete settings | 15/15 |
| adaptive beats base AdapterFusion | 9/15 |
| adaptive beats fixed `lambda_distill = 1.0` | 3/15 |
| adaptive beats best fixed lambda per setting | 1/15 |
| mean delta vs base AdapterFusion | 0.0161 |
| mean delta vs fixed `lambda_distill = 1.0` | 0.0817 |
| mean delta vs best fixed lambda | 0.1264 |

Interpretation:

- This adaptive rule is not better than fixed distillation.
- It often down-weights teacher guidance too aggressively. Mean effective
  lambda is low for several datasets: `lipophilicity_astrazeneca` 0.0775,
  `ppbr_az` 0.1155, `vdss_lombardo` 0.0234.
- This is a useful negative result: raw validation teacher advantage is not a
  sufficient reliability signal for adaptive distillation.
- The next adaptive strategy should not simply scale by teacher advantage
  ratio. Better options are validation-selected lambda, learned gating, or
  sample-level teacher agreement.

## Neural Model Comparison

| dataset | task_type | train_ratio_tag | metric | ecfp_mlp | descpred | adapter | descconcat | descpred_vs_ecfp | adapter_vs_ecfp | adapter_vs_descpred | descconcat_vs_ecfp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bbb_martins | classification | 10 | test_roc_auc | 0.7200 | 0.7075 | 0.7269 | 0.8190 | -0.0125 | 0.0069 | 0.0194 | 0.0990 |
| bbb_martins | classification | 20 | test_roc_auc | 0.8133 | 0.8308 | 0.8204 | 0.8499 | 0.0175 | 0.0071 | -0.0104 | 0.0366 |
| bbb_martins | classification | 50 | test_roc_auc | 0.8428 | 0.8499 | 0.8487 | 0.8768 | 0.0071 | 0.0059 | -0.0012 | 0.0340 |
| bioavailability_ma | classification | 10 | test_roc_auc | 0.6382 | 0.6448 | 0.6182 | 0.6976 | 0.0066 | -0.0200 | -0.0266 | 0.0594 |
| bioavailability_ma | classification | 20 | test_roc_auc | 0.5752 | 0.5745 | 0.5733 | 0.6149 | -0.0007 | -0.0019 | -0.0012 | 0.0397 |
| bioavailability_ma | classification | 50 | test_roc_auc | 0.5810 | 0.5596 | 0.5722 | 0.5569 | -0.0214 | -0.0088 | 0.0126 | -0.0241 |
| caco2_wang | regression | 10 | test_rmse | 3.3683 | 3.3584 | 1.7702 | 2.9221 | 0.0099 | 1.5981 | 1.5882 | 0.4462 |
| caco2_wang | regression | 20 | test_rmse | 1.6070 | 1.6383 | 0.9541 | 1.3699 | -0.0313 | 0.6529 | 0.6842 | 0.2371 |
| caco2_wang | regression | 50 | test_rmse | 0.8039 | 0.7613 | 0.6775 | 0.5722 | 0.0426 | 0.1264 | 0.0838 | 0.2317 |
| herg | classification | 10 | test_roc_auc | 0.7025 | 0.6959 | 0.7008 | 0.7426 | -0.0066 | -0.0017 | 0.0049 | 0.0401 |
| herg | classification | 20 | test_roc_auc | 0.7424 | 0.7431 | 0.7534 | 0.8091 | 0.0007 | 0.0110 | 0.0103 | 0.0667 |
| herg | classification | 50 | test_roc_auc | 0.8111 | 0.8034 | 0.7855 | 0.8512 | -0.0077 | -0.0256 | -0.0179 | 0.0401 |
| hia_hou | classification | 10 | test_roc_auc | 0.6269 | 0.6228 | 0.6702 | 0.7174 | -0.0041 | 0.0433 | 0.0474 | 0.0905 |
| hia_hou | classification | 20 | test_roc_auc | 0.6642 | 0.7199 | 0.7295 | 0.7089 | 0.0557 | 0.0653 | 0.0096 | 0.0447 |
| hia_hou | classification | 50 | test_roc_auc | 0.9165 | 0.9005 | 0.8796 | 0.9176 | -0.0160 | -0.0369 | -0.0209 | 0.0011 |
| lipophilicity_astrazeneca | regression | 10 | test_rmse | 1.2550 | 1.2638 | 1.2325 | 1.0763 | -0.0088 | 0.0225 | 0.0313 | 0.1787 |
| lipophilicity_astrazeneca | regression | 20 | test_rmse | 1.1232 | 1.1232 | 1.0960 | 0.9712 | 0.0000 | 0.0272 | 0.0272 | 0.1520 |
| lipophilicity_astrazeneca | regression | 50 | test_rmse | 0.9912 | 1.0013 | 0.9736 | 0.8492 | -0.0101 | 0.0176 | 0.0277 | 0.1420 |
| pgp_broccatelli | classification | 10 | test_roc_auc | 0.8282 | 0.8161 | 0.8361 | 0.8911 | -0.0121 | 0.0079 | 0.0200 | 0.0629 |
| pgp_broccatelli | classification | 20 | test_roc_auc | 0.8615 | 0.8733 | 0.8740 | 0.9090 | 0.0118 | 0.0125 | 0.0007 | 0.0475 |
| pgp_broccatelli | classification | 50 | test_roc_auc | 0.8858 | 0.8899 | 0.8956 | 0.9235 | 0.0041 | 0.0098 | 0.0057 | 0.0377 |
| ppbr_az | regression | 10 | test_rmse | 65.0653 | 65.1399 | 18.1281 | 64.7225 | -0.0746 | 46.9372 | 47.0118 | 0.3428 |
| ppbr_az | regression | 20 | test_rmse | 26.5570 | 26.9134 | 16.6702 | 24.8199 | -0.3564 | 9.8868 | 10.2432 | 1.7371 |
| ppbr_az | regression | 50 | test_rmse | 14.7315 | 15.3150 | 14.9058 | 13.9929 | -0.5835 | -0.1743 | 0.4092 | 0.7386 |
| solubility_aqsoldb | regression | 10 | test_rmse | 1.8477 | 1.8330 | 1.7832 | 1.5133 | 0.0147 | 0.0645 | 0.0498 | 0.3344 |
| solubility_aqsoldb | regression | 20 | test_rmse | 1.7364 | 1.7247 | 1.7117 | 1.4250 | 0.0117 | 0.0247 | 0.0130 | 0.3114 |
| solubility_aqsoldb | regression | 50 | test_rmse | 1.6426 | 1.6411 | 1.6107 | 1.3540 | 0.0015 | 0.0319 | 0.0304 | 0.2886 |
| vdss_lombardo | regression | 10 | test_rmse | 5.4858 | 5.4896 | 5.4563 | 5.3666 | -0.0038 | 0.0295 | 0.0333 | 0.1192 |
| vdss_lombardo | regression | 20 | test_rmse | 5.6165 | 5.5751 | 5.5086 | 5.5556 | 0.0414 | 0.1079 | 0.0665 | 0.0609 |
| vdss_lombardo | regression | 50 | test_rmse | 5.3960 | 5.3722 | 5.2838 | 5.4127 | 0.0238 | 0.1122 | 0.0884 | -0.0167 |
