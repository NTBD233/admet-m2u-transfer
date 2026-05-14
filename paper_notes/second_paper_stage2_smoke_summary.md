# Stage 2 Multi-Teacher Smoke Summary

## Scope

- Dataset: `caco2_wang`
- Train ratio: `10`
- Seed: `42`
- Student: `ECFP4_MLP_DescAdapterFusion`
- Teachers: `ECFP4_RF`, `Desc_RF`, `ECFP4_Desc_RF`
- Distillation strength: `lambda_distill = 1.0`

This smoke run checks whether the Stage 2 multi-teacher baselines are implemented correctly before launching the full 5-dataset x 3-ratio x 3-seed matrix.

## Baselines

| Strategy | Teacher weights | Valid RMSE | Test RMSE |
| --- | --- | ---: | ---: |
| Uniform | `(0.333, 0.333, 0.333)` | `1.6576` | `1.7852` |
| Validation-weighted | `(0.306, 0.367, 0.327)` | `1.6525` | `1.7758` |
| Top-1 validation | `(0.000, 1.000, 0.000)` | `1.5979` | `1.7255` |

Teacher order above is `(ECFP4_RF, Desc_RF, ECFP4_Desc_RF)`.

## Comparison to Existing References

| Reference | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| Base AdapterFusion | `1.6577` | `1.7804` |
| Fixed single-teacher distill (`ECFP4_Desc_RF`, `lambda=1.0`) | `1.6478` | `1.7725` |
| Multi-teacher top-1 validation | `1.5979` | `1.7255` |

## Interpretation

The smoke result is directionally consistent with Stage 1 teacher reliability diagnostics:

- For `caco2_wang`, the setting-level best RF teacher is `Desc_RF`.
- The `top1_validation` baseline therefore collapses to `Desc_RF` only.
- That choice beats both uniform averaging and validation-weighted averaging on this smoke setting.

This is useful because it shows the Stage 1 diagnostics are actionable rather than descriptive only. It also gives a concrete control for Stage 3:

> a sample-level reliability gate should beat a strong setting-level teacher selector, not only uniform averaging.

## Immediate Next Step

Run the full Stage 2 regression matrix:

- 5 datasets
- 3 train ratios (`10`, `20`, `50`)
- 3 seeds (`42`, `123`, `3407`)
- 3 multi-teacher baselines (`uniform`, `validation_weighted`, `top1_validation`)

If `top1_validation` wins many settings, Stage 3 should be framed as:

> moving from setting-level teacher selection to sample-level reliability gating.
