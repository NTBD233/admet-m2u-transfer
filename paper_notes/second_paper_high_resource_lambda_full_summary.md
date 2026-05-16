# High-Resource Lambda Full Regression Summary

## Scope

This is the full-matrix expansion of the ratio-aware selector calibration
experiment.

- Method: `pretrained_selector_top1`
- Schedule: `high_resource_decay`
- Lambda factors: `10 -> 1.0`, `20 -> 1.0`, `50 -> 0.3`
- Student: `ECFP4_MLP_DescAdapterFusion`
- Teachers: `ECFP4_RF`, `Desc_RF`, `ECFP4_Desc_RF`
- Selector: `rf_crossfit_train_pseudo_oracle`
- Datasets: `caco2_wang`, `lipophilicity_astrazeneca`,
  `solubility_aqsoldb`, `vdss_lombardo`, `ppbr_az`
- Train ratios: `10`, `20`, `50`
- Seeds: `42`, `123`, `3407`
- Total settings: `45`

Result root:

```text
results_pretrained_selector_top1_high_resource_lambda_regression
```

Comparison entrypoint:

```bash
make selector-calibration-full-comparison
```

## Aggregate Results

Mean test RMSE, lower is better:

| method | completed settings | mean test RMSE |
| --- | ---: | ---: |
| `top1_validation` | 45 | 5.0792 |
| `high_resource_decay` | 45 | 5.0814 |
| plain `pretrained_selector_top1` | 45 | 5.0889 |
| fixed `ECFP4_Desc_RF` | 45 | 5.1621 |
| base AdapterFusion | 45 | 5.1841 |

Pairwise comparisons for `high_resource_decay`:

| reference | wins | total | mean delta |
| --- | ---: | ---: | ---: |
| base AdapterFusion | 34 | 45 | -0.1028 |
| fixed `ECFP4_Desc_RF` | 32 | 45 | -0.0807 |
| `top1_validation` | 27 | 45 | +0.0021 |
| plain selector | 9 | 45 | -0.0075 |

## Ratio-Wise Behavior

Mean test RMSE by train ratio:

| ratio | base | fixed | top1 validation | plain selector | high decay |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 5.6741 | 5.6641 | 5.5832 | 5.6114 | 5.6114 |
| 20 | 5.1881 | 5.2033 | 5.0439 | 5.0552 | 5.0552 |
| 50 | 4.6903 | 4.6188 | 4.6106 | 4.6001 | 4.5774 |

Interpretation:

- At `10%` and `20%`, `high_resource_decay` exactly matches plain selector
  because its factor is `1.0`.
- At `50%`, reducing teacher forcing to `0.3` improves mean test RMSE from
  `4.6001` to `4.5774`.
- The full-matrix result confirms the partial-regression signal: weakening
  routed distillation is useful in the higher-resource regime, but not enough
  to clearly beat `top1_validation` overall.

## Dataset-Wise Behavior

Mean test RMSE by dataset:

| dataset | base | fixed | top1 validation | plain selector | high decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| `caco2_wang` | 1.1339 | 1.0981 | 1.1044 | 1.1048 | 1.0912 |
| `lipophilicity_astrazeneca` | 1.1007 | 1.0946 | 1.0879 | 1.0848 | 1.0832 |
| `ppbr_az` | 16.5680 | 16.4917 | 16.0407 | 16.1134 | 16.0792 |
| `solubility_aqsoldb` | 1.7019 | 1.7041 | 1.6937 | 1.6904 | 1.6910 |
| `vdss_lombardo` | 5.4162 | 5.4219 | 5.4695 | 5.4510 | 5.4622 |

Dataset takeaways:

- Clear improvement over plain selector appears on `caco2_wang`,
  `lipophilicity_astrazeneca`, and `ppbr_az`.
- `solubility_aqsoldb` is essentially unchanged.
- `vdss_lombardo` gets slightly worse than plain selector, though it still
  remains competitive with the other baselines.

## Paper Interpretation

This result strengthens the second-paper story, but does not change the main
claim.

Defensible claim:

> Pretrained selector routing is the main contribution, and train-ratio-aware
> distillation strength is a lightweight calibration that further narrows the
> remaining gap to the strongest setting-level teacher selector.

Do not claim:

> high-resource lambda decay decisively beats `top1_validation`.

Best use in the paper:

- include `high_resource_decay` as a calibrated selector variant in the main or
  ablation table;
- emphasize that it improves plain selector on mean RMSE;
- state that `top1_validation` remains a very strong setting-level oracle-like
  baseline and is still marginally better in aggregate;
- use the ratio-wise result to support the mechanistic point that teacher
  choice and teacher strength are separate axes.

## Next Step

The experiments are now sufficient to freeze the method section and start
turning the notes into manuscript-ready tables and figures.

The next technical task should be table construction, not another routing
variant:

1. build a paper-ready main comparison table;
2. build a selector-supervision diagnostics table;
3. build a compact ablation table covering joint gates, auto reweighting, and
   ratio-aware lambda.
