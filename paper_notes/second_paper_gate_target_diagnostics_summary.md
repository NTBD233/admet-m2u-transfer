# Second Paper Gate-Target Diagnostics Summary

## Status

The repository now includes a dedicated oracle teacher-selection diagnostic:

```bash
make gate-target-diagnostics
```

Implementation entrypoint:

- `analyze_gate_targets.py`

Generated outputs are written under `results_gate_targets/summary/` and are not
tracked by Git, so this note records the current result snapshot.

## Diagnostic Question

The learned gates tried so far underperformed `top1_validation`, but that did
not tell us whether the **signals** were bad or whether the **training signal**
was bad.

This diagnostic asks:

> Can current reliability features predict the per-sample oracle best teacher?

Oracle label definition:

- validation sample-level teacher with the lowest absolute prediction error

Features used:

- per-teacher uncertainty
- per-teacher deviation from teacher consensus
- per-teacher setting-level validation prior weight
- shared descriptor-space OOD distance

Evaluation protocol:

- 45 grouped folds
- groups = `dataset | train_ratio | seed`
- leave-one-setting-out cross-validation

Teacher set:

- `ECFP4_RF`
- `Desc_RF`
- `ECFP4_Desc_RF`

## Cross-Setting Probe Results

| model | groups | total samples | mean group accuracy | mean group macro F1 | weighted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rf_gate_probe` | 45 | 27378 | 0.5726 | 0.4646 | 0.5617 |
| `logistic_gate_probe` | 45 | 27378 | 0.5095 | 0.3317 | 0.5044 |
| `majority_train` | 45 | 27378 | 0.4765 | 0.2148 | 0.4818 |
| `setting_prior_top1` | 45 | 27378 | 0.4016 | 0.1874 | 0.4360 |

Group-level accuracy ranges:

- `setting_prior_top1`: min `0.1986`, median `0.4721`, max `0.5426`
- `logistic_gate_probe`: min `0.4088`, median `0.5137`, max `0.6099`
- `rf_gate_probe`: min `0.4641`, median `0.5648`, max `0.6507`

## Interpretation

This result is important:

1. current reliability signals do contain real teacher-selection information
2. those signals are strong enough to beat both a global-majority baseline and a setting-level top-1 prior baseline
3. the bottleneck is therefore **not** that teacher selection is inherently random

This is the first strong piece of evidence that a supervised or semi-supervised
teacher selector is worth trying.

## Immediate Follow-Up Smoke

A minimal supervised gate variant was then tested by adding an auxiliary gate
cross-entropy loss to the learned prior-residual gate during student training.

Smoke setting:

- dataset: `caco2_wang`
- train ratio: `10`
- seed: `42`
- strategy: `learned_prior_residual_gate`
- `lambda_distill = 1.0`
- `lambda_gate_supervision = 0.5`

Comparison:

| method | valid RMSE | test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `1.5979` | `1.7255` |
| `learned_prior_residual_gate` | `1.7155` | `1.8852` |
| `supervised_prior_residual_gate` | `1.7143` | `1.8844` |

## What This Means

The diagnostic is positive, but the first supervised smoke is still negative.

That narrows the real research problem:

1. teacher selection is predictable
2. but directly adding a train-time gate classification loss is not enough to improve student regression
3. therefore the missing piece is probably the **coupling between gate supervision and distillation target formation**, not simply gate predictability

## Hard Top-1 Follow-Up

Since Stage 2 already showed that `top1_validation` is the strongest simple
baseline, the next minimal test was to replace soft teacher weighting with a
supervised **hard top-1 gate** using straight-through one-hot routing.

Smoke setting:

- dataset: `caco2_wang`
- train ratio: `10`
- seed: `42`
- strategy: `supervised_hard_top1_gate`
- `lambda_distill = 1.0`

Results:

| method | valid RMSE | test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `1.5979` | `1.7255` |
| `supervised_prior_residual_gate` | `1.7143` | `1.8844` |
| `supervised_hard_top1_gate` (`lambda_gate_supervision = 0.5`) | `1.6593` | `1.8192` |
| `supervised_hard_top1_gate` (`lambda_gate_supervision = 0.1`) | `1.6593` | `1.8192` |

Interpretation:

1. hard top-1 routing is clearly better than the previous soft supervised gate
2. but it still does not beat the simple `top1_validation` selector
3. changing the gate-supervision weight from `0.5` to `0.1` does not materially change this smoke result

This further supports the view that the main problem is not the loss weight.
The likely bottleneck is that **jointly learning the selector and student from
scratch** is still too unstable or too weakly coupled to the actual best
teacher-choice objective.

## Next Method Direction

The next serious method should likely move away from soft weighting alone and
toward one of these:

1. **hard top-1 / top-2 teacher selection**
   - closer to the diagnostic oracle target
   - but likely after separate selector pretraining rather than pure joint training

2. **teacher-selection pretraining plus frozen/warm-start gate**
   - first train the selector on oracle pseudo-labels
   - then use it to route distillation instead of learning both jointly from scratch

3. **teacher filtering before distillation**
   - use the supervised selector only to eliminate bad teachers
   - distill from the surviving teacher set afterward

The important conclusion is that the project should not keep iterating on naive
soft gates. The evidence now points toward **supervised discrete teacher
selection** as the more credible next innovation.
