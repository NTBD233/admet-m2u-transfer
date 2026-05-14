# Second Paper Pretrained Selector Smoke

## Goal

Test whether a **pretrained frozen teacher selector** is a better direction than
jointly learned routing gates.

Smoke setting:

- dataset: `caco2_wang`
- train ratio: `10`
- seed: `42`
- student: `ECFP4_MLP_DescAdapterFusion`
- teachers: `ECFP4_RF`, `Desc_RF`, `ECFP4_Desc_RF`
- selector: `rf_crossfit_train_pseudo_oracle`
- distillation strength: `lambda_distill = 1.0`

## Selector Pretraining Snapshot

Selector training uses train-split cross-fit pseudo-oracle labels:

- label source: `crossfit_train_pseudo_oracle`
- primary selector: `RF`
- baseline selector: `logistic`

Current `caco2_wang / train_10 / seed_42` metrics:

| selector | train acc | valid acc | test acc |
| --- | ---: | ---: | ---: |
| `RF` | `0.7414` | `0.4521` | `0.5714` |
| `logistic` | `0.5172` | `0.3836` | `0.5495` |

Interpretation:

1. the cross-fit train selector is learnable
2. RF is clearly better than logistic
3. the selector is not perfect, but it captures signal beyond a weak linear probe

## Frozen Selector Distillation Smoke

Compared methods:

- `top1_validation`
- `pretrained_selector_top1`
- `pretrained_selector_top2`
- `selector_filtered_validation_weighted`

Results:

| method | valid RMSE | test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `1.5979` | `1.7255` |
| `pretrained_selector_top1` | `1.6616` | `1.7337` |
| `selector_filtered_validation_weighted` | `1.6753` | `1.7683` |
| `pretrained_selector_top2` | `1.6777` | `1.7896` |

## Interpretation

This is the strongest Stage 3 result so far.

1. `pretrained_selector_top1` is materially better than the joint soft/hard gate variants.
2. Frozen selector routing is already very close to `top1_validation` on test RMSE.
3. `top2` routing is worse than `top1`, which suggests that reintroducing multiple teachers may still drag the student toward noisy supervision.
4. Selector-based filtering helps, but not as much as direct top-1 routing.

## Consequence For The Main Method

The current evidence favors this ordering:

1. **main method**: pretrained selector + frozen top-1 routing
2. **secondary method**: selector-filtered validation-weighted distillation
3. **ablation / enhancement**: top-2 routing

This means the second paper should now be reframed around:

> teacher-selector supervision and frozen routing, rather than generic reliability gating.

## Next Step

If the next expansion to the full regression matrix preserves this ordering,
the paper's method section should center on:

- cross-fit pseudo-oracle selector labels
- standalone selector pretraining
- frozen selector routing into distillation

Joint gates should remain as negative controls and failure evidence.
