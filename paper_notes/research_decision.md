# Research Decision After Full ADMET Runs

## Status

The expanded experiment is complete:

- Neural runs: 360/360
- RF baselines: 270/270
- Datasets: 10 ADMET datasets
- Train ratios: 10%, 20%, 50%
- Seeds: 42, 123, 3407

## Main Finding

The current evidence supports a focused but not overclaimed paper direction:

> Descriptor knowledge is consistently useful for low-resource ADMET. A
> structured ECFP-only transfer mechanism, AdapterFusion, improves over simple
> descriptor prediction in most settings, especially regression, but it does not
> beat strong descriptor-access RF baselines.

## Quantitative Summary

Across 30 dataset-ratio settings:

- `ECFP4_MLP_DescConcat` improves over `ECFP4_MLP` in 28/30 settings.
- `ECFP4_MLP_DescAdapterFusion` improves over `ECFP4_MLP_DescPred` in 24/30 settings.
- On regression tasks, AdapterFusion improves over DescPred in 15/15 settings.
- On classification tasks, AdapterFusion improves over DescPred in 9/15 settings.
- AdapterFusion beats `ECFP4_RF` in only 9/30 settings.
- AdapterFusion beats `ECFP4_Desc_RF` in 0/30 settings.

Best-model counts across the 30 settings:

- `Desc_RF`: 14
- `ECFP4_Desc_RF`: 11
- `ECFP4_MLP_DescConcat`: 5

## Interpretation

The result is scientifically useful but should be framed carefully.

Strong claim that is supported:

> Adapter-based descriptor transfer is a better neural transfer mechanism than
> plain descriptor prediction, especially for low-resource regression ADMET.

Strong claim that is not supported:

> AdapterFusion is a new overall state-of-the-art ADMET predictor.

The strongest empirical story is not "we beat all baselines." It is:

1. Descriptor information is valuable.
2. Direct descriptor access creates a strong upper-bound/control.
3. Simple auxiliary descriptor prediction often under-transfers this knowledge.
4. AdapterFusion transfers descriptor knowledge more effectively, especially in
   regression and lower-resource settings.
5. Traditional descriptor-based RF remains a very strong practical baseline.

## Recommended Next Research Move

The next version should not merely add more datasets. The dataset expansion is
already done. The next step should improve the method against the RF baseline:

1. Add a stronger ECFP-only neural baseline, such as residual MLP or calibrated
   MLP with better regularization.
2. Distill from `ECFP4_Desc_RF` or `Desc_RF` into the ECFP-only neural model.
3. Add a ranking or teacher-student loss, not only descriptor reconstruction.
4. Keep AdapterFusion as the transfer backbone.

The most promising next method name:

> Descriptor-Teacher AdapterFusion

Possible formulation:

- Train descriptor-access teacher: `ECFP4_Desc_RF` or `Desc_RF`.
- Train ECFP-only student with:
  - task loss
  - descriptor reconstruction loss
  - teacher prediction distillation loss
  - AdapterFusion gated representation

This directly targets the current gap: AdapterFusion transfers descriptors
better than DescPred, but not enough to match descriptor-access RF.

## First Teacher-Distillation Check

A first `caco2_wang` smoke/ablation was run with:

- student: `ECFP4_MLP_DescAdapterFusion`
- teacher: `ECFP4_Desc_RF`
- distillation loss weight: `lambda_distill = 0.1`
- train ratios: 10%, 20%, 50%
- seeds: 42, 123, 3407

Results:

| Train ratio | AdapterFusion RMSE | Distilled AdapterFusion RMSE | Teacher RF RMSE |
|---:|---:|---:|---:|
| 10 | 1.7702±0.0304 | 1.7529±0.0340 | 0.5535±0.0017 |
| 20 | 0.9541±0.0563 | 0.9240±0.0362 | 0.4691±0.0083 |
| 50 | 0.6775±0.0995 | 0.6174±0.0371 | 0.4252±0.0121 |

Interpretation:

- Teacher distillation improves AdapterFusion on all three `caco2_wang`
  low-resource settings.
- The improvement is largest at 50% train ratio in this first check.
- The student still does not match the descriptor-access RF teacher.
- This supports continuing with Descriptor-Teacher AdapterFusion as the next
  method iteration.

## Paper Positioning

If writing now, position as a method-analysis paper:

> A controlled study of descriptor-guided multi-to-uni transfer for
> low-resource ADMET prediction.

If adding one more method iteration, position as a stronger method paper:

> Descriptor-teacher guided AdapterFusion for ECFP-only low-resource ADMET
> prediction.

## Regression Teacher-Distillation Expansion

The full regression-only teacher distillation expansion was completed with:

- datasets: `caco2_wang`, `lipophilicity_astrazeneca`, `solubility_aqsoldb`,
  `vdss_lombardo`, `ppbr_az`
- train ratios: 10%, 20%, 50%
- seeds: 42, 123, 3407
- teacher: `ECFP4_Desc_RF`
- student: `ECFP4_MLP_DescAdapterFusion`
- distillation loss weight: `lambda_distill = 0.1`

Result summary across 15 regression dataset-ratio settings:

- Completed distilled runs: 45/45 seed runs, 15/15 complete settings.
- Distilled AdapterFusion improves over base AdapterFusion in 8/15 settings.
- Distilled AdapterFusion improves over `ECFP4_RF` in 4/15 settings.
- Mean RMSE delta vs base AdapterFusion: -0.0221.
- Mean RMSE gap to `ECFP4_Desc_RF` teacher: 0.7916.

Interpretation:

- Distillation is helpful but not uniformly reliable.
- The strongest positive signal remains `caco2_wang`, where all three train
  ratios improve.
- `ppbr_az` improves at 10% and 50%, but not at 20%.
- `lipophilicity_astrazeneca`, `solubility_aqsoldb`, and `vdss_lombardo` show
  mixed or small changes.
- The RF teacher remains substantially stronger in most settings, so the
  method should be framed as a controlled transfer improvement rather than an
  RF-level replacement.

Current decision:

> Descriptor-Teacher AdapterFusion is a useful method iteration, but at fixed
> `lambda_distill = 0.1` it is not yet strong enough to become the sole central
> paper claim. The next experiment should be a small lambda sweep on the five
> regression datasets or a targeted sweep on the unstable datasets.

## Regression Distillation Lambda Sweep Decision

A full regression lambda sweep was completed for:

- datasets: `caco2_wang`, `lipophilicity_astrazeneca`, `solubility_aqsoldb`,
  `vdss_lombardo`, `ppbr_az`
- train ratios: 10%, 20%, 50%
- seeds: 42, 123, 3407
- teacher: `ECFP4_Desc_RF`
- student: `ECFP4_MLP_DescAdapterFusion`
- lambda values: `0.01`, `0.1`, `0.3`, `1.0`

All four lambda settings completed 45/45 seed runs.

| lambda_distill | complete settings | beats AdapterFusion | beats ECFP4_RF | mean delta vs AdapterFusion | mean gap to ECFP4_Desc_RF |
| --- | --- | --- | --- | ---: | ---: |
| 0.01 | 15/15 | 8/15 | 4/15 | 0.0075 | 0.8211 |
| 0.1 | 15/15 | 8/15 | 4/15 | -0.0221 | 0.7916 |
| 0.3 | 15/15 | 10/15 | 4/15 | -0.0046 | 0.8090 |
| 1.0 | 15/15 | 11/15 | 4/15 | -0.0656 | 0.7481 |

Decision:

> Use `lambda_distill = 1.0` as the current default for Descriptor-Teacher
> AdapterFusion in regression experiments, while reporting lambda sensitivity
> as an ablation.

Reasoning:

- `lambda_distill = 1.0` gives the best aggregate performance among tested
  values: it improves over base AdapterFusion in 11/15 settings and has the
  smallest average gap to the descriptor-access RF teacher.
- The result strengthens the method story: teacher prediction supervision is
  more useful than fixed weak distillation at `0.1`.
- The result still does not support a claim of replacing traditional RF
  baselines, because every lambda value beats `ECFP4_RF` in only 4/15 settings.

Updated paper positioning:

> Descriptor-teacher distillation improves ECFP-only AdapterFusion for
> low-resource regression ADMET, but strong descriptor-access RF remains the
> practical upper-bound baseline.

Next research action:

1. Add an adaptive distillation variant or selection rule instead of a single
   global lambda.
2. Prioritize analysis of why `lambda_distill = 1.0` helps `ppbr_az` and
   `lipophilicity_astrazeneca`, while `caco2_wang` prefers `0.1` or `0.3`.
3. Keep classification distillation paused until regression analysis is
   mechanistically clearer.

## Mechanism Diagnostic After Lambda Sweep

A seed-level diagnostic compared each distilled student against the same
`ECFP4_Desc_RF` teacher predictions on the test set. This checks whether
distillation actually moves the ECFP-only student toward the teacher, rather
than only changing the task loss.

Key result:

| lambda_distill | complete settings | beats base | mean delta RMSE | mean delta teacher RMSE | mean delta teacher corr |
| --- | --- | --- | ---: | ---: | ---: |
| 0.01 | 15/15 | 8/15 | 0.0075 | 0.0392 | -0.0002 |
| 0.1 | 15/15 | 8/15 | -0.0221 | 0.0117 | 0.0042 |
| 0.3 | 15/15 | 10/15 | -0.0046 | 0.0087 | 0.0082 |
| 1.0 | 15/15 | 11/15 | -0.0655 | -0.1664 | 0.0097 |

Decision update:

> `lambda_distill = 1.0` remains the default regression setting because it is
> the only tested value that improves aggregate RMSE while also moving the
> student closer to the teacher on average.

But the next paper-level method should not stop at fixed `lambda = 1.0`.
Dataset-level behavior shows a clear adaptive weighting motivation:

- `ppbr_az`: strong teacher advantage, strong benefit from higher distillation.
- `caco2_wang`: moderate distillation works better than the strongest setting.
- `vdss_lombardo`: teacher matching can increase without downstream RMSE
  improvement, so blindly increasing teacher weight is risky.

Next concrete experiment:

> Implement an adaptive teacher-weighting variant that increases distillation
> only when teacher guidance is reliable, measured by validation teacher-student
> agreement or teacher validation advantage over the base student.

## Adaptive Teacher-Weighting Result

The first adaptive strategy was implemented and run on the full regression
grid. It used validation teacher advantage to set the per-run distillation
weight:

> `lambda_effective = lambda_max * max((base_valid_rmse - teacher_valid_rmse) / base_valid_rmse, 0)`

with `lambda_max = 1.0`.

Completed runs:

- 5 regression datasets
- 3 train ratios
- 3 seeds
- 45/45 seed runs complete

Summary:

| comparison | result |
| --- | --- |
| adaptive beats base AdapterFusion | 9/15 |
| adaptive beats fixed `lambda_distill = 1.0` | 3/15 |
| adaptive beats best fixed lambda per setting | 1/15 |
| mean delta vs base AdapterFusion | 0.0161 |
| mean delta vs fixed `lambda_distill = 1.0` | 0.0817 |
| mean delta vs best fixed lambda | 0.1264 |

Decision:

> Do not use this validation-advantage ratio strategy as the paper method.

Reasoning:

- It is weaker than fixed `lambda_distill = 1.0`.
- It is weaker than the best fixed lambda in almost every setting.
- It suppresses teacher guidance too much on datasets where fixed strong
  distillation was useful, especially `ppbr_az`.

Updated next step:

> Keep fixed `lambda_distill = 1.0` as the current main regression method.
> Treat validation-advantage adaptive weighting as a negative ablation.

The next worthwhile adaptive direction is not another hand-scaled ratio. It
should be one of:

1. Select lambda by validation performance from the fixed sweep per
   dataset-ratio.
2. Learn a small gate for teacher loss weighting.
3. Use sample-level teacher-student disagreement or uncertainty, rather than
   dataset-level teacher advantage.
