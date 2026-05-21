# Second Paper Dataset Extension Audit

## Current state

The manuscript's main claim remains based on the completed five-endpoint
regression panel:

- `caco2_wang`
- `lipophilicity_astrazeneca`
- `solubility_aqsoldb`
- `vdss_lombardo`
- `ppbr_az`

The local `admet_group.zip` also contains four additional regression endpoints
that can support a controlled extension panel:

- `clearance_hepatocyte_az`
- `clearance_microsome_az`
- `half_life_obach`
- `ld50_zhu`

These are now registered in `utils/config.py` as regression tasks, but they are
not part of the current manuscript tables until their full teacher, selector,
and student matrices are generated and checked.

As of the current audit, the first three extension stages, the base
AdapterFusion control, and the plain pretrained-selector student matrix have
completed:

- data splits were prepared from the local `admet_group.zip`;
- ECFP4 and descriptor features were generated for all four extension datasets;
- RF teacher models, exported teacher predictions, and cross-fit RF selectors
  were generated for all 36 dataset-ratio-seed settings.
- pretrained selector hard top-1 routing was trained for all 36
  dataset-ratio-seed settings.
- base AdapterFusion was trained for all 36 dataset-ratio-seed settings.

The extension selector quality is comparable to the main regression selector
but not strong enough by itself to justify promoting the extension panel before
student results are known:

| dataset | valid accuracy | test accuracy | valid macro-F1 | test macro-F1 |
| --- | --- | --- | --- | --- |
| `clearance_hepatocyte_az` | 0.4817 | 0.5007 | 0.4695 | 0.4924 |
| `clearance_microsome_az` | 0.4678 | 0.5083 | 0.4393 | 0.4635 |
| `half_life_obach` | 0.5084 | 0.5424 | 0.4199 | 0.4594 |
| `ld50_zhu` | 0.4748 | 0.4567 | 0.4532 | 0.4416 |
| `ALL_EXTENSION` | 0.4832 | 0.5020 | 0.4455 | 0.4643 |

## Recommended extension protocol

Use the extension panel as a robustness check, not as a blocker for the current
submission.

1. Prepare splits and features:

   ```bash
   PYTHON=.miniforge/envs/tdc-admet/bin/python make prepare
   PYTHON=.miniforge/envs/tdc-admet/bin/python make second-paper-extension-features
   ```

2. Train RF teacher models and export teacher predictions:

   ```bash
   PYTHON=.miniforge/envs/tdc-admet/bin/python make ml-baselines-extension-regression
   PYTHON=.miniforge/envs/tdc-admet/bin/python make teacher-predictions-extension-regression
   ```

3. Generate selector labels:

   ```bash
   PYTHON=.miniforge/envs/tdc-admet/bin/python make teacher-selector-extension-regression
   ```

4. Run the required extension student controls and selector variants:

   ```bash
   PYTHON=.miniforge/envs/tdc-admet/bin/python make base-extension-regression
   PYTHON=.miniforge/envs/tdc-admet/bin/python make distill-extension-regression
   PYTHON=.miniforge/envs/tdc-admet/bin/python make multiteacher-top1-extension-regression
   ```

5. Run the two extension selector variants:

   ```bash
   PYTHON=.miniforge/envs/tdc-admet/bin/python make pretrained-selector-top1-extension-regression
   PYTHON=.miniforge/envs/tdc-admet/bin/python make pretrained-selector-top1-high-resource-lambda-extension-regression
   ```

The plain pretrained-selector and base AdapterFusion runs are complete. The
extension panel still needs fixed ECFP4+descriptor RF distillation,
validation-selected top-1 teacher distillation, and the high-resource-decay
selector variant before it can be judged as manuscript evidence.

Plain pretrained selector hard top-1 routing currently gives:

| dataset | train ratio | test RMSE |
| --- | --- | --- |
| `clearance_hepatocyte_az` | 10 | 47.5592 +/- 0.8232 |
| `clearance_hepatocyte_az` | 20 | 46.8430 +/- 0.4507 |
| `clearance_hepatocyte_az` | 50 | 48.5480 +/- 0.8911 |
| `clearance_microsome_az` | 10 | 41.5221 +/- 0.2601 |
| `clearance_microsome_az` | 20 | 41.0119 +/- 0.6310 |
| `clearance_microsome_az` | 50 | 39.5979 +/- 0.5163 |
| `half_life_obach` | 10 | 21.8733 +/- 0.0449 |
| `half_life_obach` | 20 | 20.9064 +/- 0.1061 |
| `half_life_obach` | 50 | 18.1318 +/- 0.3965 |
| `ld50_zhu` | 10 | 1.0797 +/- 0.0192 |
| `ld50_zhu` | 20 | 0.9715 +/- 0.0246 |
| `ld50_zhu` | 50 | 0.9447 +/- 0.0202 |

Against the newly completed base AdapterFusion extension control, plain
pretrained selector top-1 routing currently wins 10 of 12 dataset-ratio
settings and 26 of 36 seed-level runs, with an average seed-level RMSE delta of
-0.4211. This is encouraging enough to continue the extension controls, but not
yet sufficient for manuscript promotion because selector-vs-top1_validation and
selector-vs-fixed-teacher comparisons remain missing.

| dataset | train ratio | base RMSE | selector RMSE | selector - base |
| --- | --- | --- | --- | --- |
| `clearance_hepatocyte_az` | 10 | 48.3505 | 47.5592 | -0.7913 |
| `clearance_hepatocyte_az` | 20 | 47.9350 | 46.8430 | -1.0920 |
| `clearance_hepatocyte_az` | 50 | 49.9505 | 48.5480 | -1.4025 |
| `clearance_microsome_az` | 10 | 41.6288 | 41.5221 | -0.1067 |
| `clearance_microsome_az` | 20 | 41.4849 | 41.0119 | -0.4730 |
| `clearance_microsome_az` | 50 | 40.7581 | 39.5979 | -1.1602 |
| `half_life_obach` | 10 | 21.8344 | 21.8733 | 0.0389 |
| `half_life_obach` | 20 | 20.9409 | 20.9064 | -0.0345 |
| `half_life_obach` | 50 | 18.1671 | 18.1318 | -0.0353 |
| `ld50_zhu` | 10 | 1.1085 | 1.0797 | -0.0288 |
| `ld50_zhu` | 20 | 0.9340 | 0.9715 | 0.0375 |
| `ld50_zhu` | 50 | 0.9502 | 0.9447 | -0.0055 |

## Decision rule

Promote the extension panel into the manuscript only if it supports the current
story without introducing a large unexplained failure:

- the selector should remain competitive with base AdapterFusion and fixed
  descriptor-teacher distillation on average;
- high-resource decay should not create broad degradation;
- failures should be explainable by teacher strength, selector quality, or
  student utilization, not by data preparation artifacts.

If the extension panel is mixed, keep it as a robustness or revision reserve and
do not weaken the current five-endpoint main claim.
