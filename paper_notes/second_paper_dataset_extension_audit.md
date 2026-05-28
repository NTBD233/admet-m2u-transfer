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

These are now registered in `utils/config.py` as regression tasks. Their full
teacher, selector, and student matrices have been generated, so they can be used
as an extension-panel robustness check.

As of the current audit, the extension matrix has completed:

- data splits were prepared from the local `admet_group.zip`;
- ECFP4 and descriptor features were generated for all four extension datasets;
- RF teacher models, exported teacher predictions, and cross-fit RF selectors
  were generated for all 36 dataset-ratio-seed settings;
- base AdapterFusion, fixed ECFP4+descriptor RF distillation,
  validation-selected top-1 teacher distillation, plain pretrained-selector
  hard top-1 routing, and high-resource-decay pretrained-selector routing were
  trained for all 36 dataset-ratio-seed settings.

The extension selector quality is comparable to the main regression selector:

| dataset | valid accuracy | test accuracy | valid macro-F1 | test macro-F1 |
| --- | --- | --- | --- | --- |
| `clearance_hepatocyte_az` | 0.4817 | 0.5007 | 0.4695 | 0.4924 |
| `clearance_microsome_az` | 0.4678 | 0.5083 | 0.4393 | 0.4635 |
| `half_life_obach` | 0.5084 | 0.5424 | 0.4199 | 0.4594 |
| `ld50_zhu` | 0.4748 | 0.4567 | 0.4532 | 0.4416 |
| `ALL_EXTENSION` | 0.4832 | 0.5020 | 0.4455 | 0.4643 |

## Completed extension protocol

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

All commands above have completed. The current extension-panel summary is:

| method | completed settings | mean test RMSE | mean valid RMSE |
| --- | --- | --- | --- |
| `top1_validation` | 36 | 27.3490 | 29.3088 |
| `selector_top1` | 36 | 27.4158 | 29.4364 |
| `high_decay` | 36 | 27.4865 | 29.5728 |
| `fixed_ecfp_desc` | 36 | 27.8143 | 29.7699 |
| `base` | 36 | 27.8369 | 29.8432 |

Relative to base AdapterFusion, plain pretrained selector top-1 routing wins 26
of 36 seed-level runs with a mean seed-level RMSE delta of -0.4211. The
high-resource-decay variant also wins 26 of 36 seed-level runs, but its average
delta is weaker at -0.3504. Fixed single-teacher distillation wins 22 of 36
seed-level runs and is nearly neutral on average (-0.0226).

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

At the dataset-ratio level, plain pretrained selector hard top-1 routing wins 10
of 12 settings against both base AdapterFusion and fixed ECFP4+descriptor RF
distillation. However, it wins only 5 of 12 settings against the strong
`top1_validation` baseline and is slightly worse on average (+0.0668 RMSE).
This makes the extension panel supportive for the main claim that supervised
teacher selection improves ECFP-only low-resource distillation over base and
fixed single-teacher training, but not supportive for a stronger claim that the
selector uniformly dominates validation-selected teacher choice.

| dataset | train ratio | base | fixed | top1 val | selector | high decay | selector - base | selector - fixed | selector - top1 val | best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `clearance_hepatocyte_az` | 10 | 48.3505 | 48.1929 | 47.3457 | 47.5592 | 47.5592 | -0.7913 | -0.6337 | 0.2135 | `top1_validation` |
| `clearance_hepatocyte_az` | 20 | 47.9350 | 48.0685 | 46.8722 | 46.8430 | 46.8430 | -1.0920 | -1.2255 | -0.0292 | `selector_top1` |
| `clearance_hepatocyte_az` | 50 | 49.9505 | 50.0472 | 48.4147 | 48.5480 | 49.0490 | -1.4025 | -1.4992 | 0.1333 | `top1_validation` |
| `clearance_microsome_az` | 10 | 41.6288 | 41.5988 | 41.1514 | 41.5221 | 41.5221 | -0.1067 | -0.0767 | 0.3707 | `top1_validation` |
| `clearance_microsome_az` | 20 | 41.4849 | 41.5354 | 40.9216 | 41.0119 | 41.0119 | -0.4730 | -0.5235 | 0.0903 | `top1_validation` |
| `clearance_microsome_az` | 50 | 40.7581 | 40.3155 | 39.4604 | 39.5979 | 39.9205 | -1.1602 | -0.7176 | 0.1375 | `top1_validation` |
| `half_life_obach` | 10 | 21.8344 | 21.8444 | 21.8432 | 21.8733 | 21.8733 | 0.0389 | 0.0289 | 0.0301 | `base` |
| `half_life_obach` | 20 | 20.9409 | 20.9500 | 20.9801 | 20.9064 | 20.9064 | -0.0345 | -0.0436 | -0.0737 | `selector_top1` |
| `half_life_obach` | 50 | 18.1671 | 18.2039 | 18.1902 | 18.1318 | 18.1561 | -0.0353 | -0.0721 | -0.0584 | `selector_top1` |
| `ld50_zhu` | 10 | 1.1085 | 1.0898 | 1.0657 | 1.0797 | 1.0797 | -0.0288 | -0.0101 | 0.0140 | `top1_validation` |
| `ld50_zhu` | 20 | 0.9340 | 0.9555 | 0.9810 | 0.9715 | 0.9715 | 0.0375 | 0.0160 | -0.0095 | `base` |
| `ld50_zhu` | 50 | 0.9502 | 0.9701 | 0.9614 | 0.9447 | 0.9457 | -0.0055 | -0.0254 | -0.0167 | `selector_top1` |

## Decision rule

The extension panel should be used as a robustness or supplementary panel rather
than replacing the five-endpoint main table:

- it supports the claim that supervised selector routing is more useful than
  base AdapterFusion and fixed single-teacher distillation on additional
  regression endpoints;
- it confirms that validation-selected top-1 teacher choice is a strong and
  sometimes better baseline, so the paper should keep the claim calibrated;
- it suggests that high-resource lambda decay is not a reliable extension-panel
  improvement and should remain secondary or be omitted from the main story.

The manuscript should not claim universal SOTA behavior. A defensible wording is:
pretrained teacher selection improves ECFP-only low-resource regression ADMET
distillation over base and fixed single-teacher training, while remaining
competitive with validation-selected teacher routing and exposing settings where
simple validation choice remains hard to beat.
