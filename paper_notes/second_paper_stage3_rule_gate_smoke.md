# Stage 3 Rule-Based Sample Gate Smoke

## Goal

Before implementing a learned reliability gate, test whether a very simple sample-level rule is already enough to beat the strongest Stage 2 baseline.

Smoke setting:

- Dataset: `caco2_wang`
- Train ratio: `10`
- Seed: `42`
- Student: `ECFP4_MLP_DescAdapterFusion`
- Teachers: `ECFP4_RF`, `Desc_RF`, `ECFP4_Desc_RF`
- Distillation strength: `lambda_distill = 1.0`

## Tried Gates

1. `uncertainty_only`
   - sample-level teacher weights from RF predictive uncertainty only
   - lower uncertainty gets higher weight
   - global initialization uses uniform teacher prior

2. `uncertainty_validation_prior`
   - sample-level teacher weights from RF predictive uncertainty
   - plus a global validation-performance prior

Reference:

- strongest Stage 2 baseline: `top1_validation`

## Smoke Results

| Method | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `1.5979` | `1.7255` |
| `uncertainty_validation_prior` | `1.6666` | `1.7712` |
| `uncertainty_only` | `1.6876` | `1.8392` |

## Second-Round Composite Smoke

To test whether richer rule signals help, several additional sample-level gates were added:

- `uncertainty_teacher_disagreement`
- `uncertainty_teacher_student`
- `uncertainty_student_prior`
- `uncertainty_composite`

Results on the same smoke setting:

| Method | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `1.5979` | `1.7255` |
| `uncertainty_student_prior` | `1.6591` | `1.7867` |
| `uncertainty_teacher_student` | `1.6516` | `1.7994` |
| `uncertainty_validation_prior` | `1.6666` | `1.7712` |
| `uncertainty_composite` | `1.6694` | `1.8304` |
| `uncertainty_only` | `1.6876` | `1.8392` |
| `uncertainty_teacher_disagreement` | `1.6830` | `1.8476` |

Second-round takeaways:

1. `teacher-student disagreement` is a useful signal.
2. Adding a validation prior on top of `teacher-student disagreement` helps a bit more.
3. `teacher-teacher disagreement` does not help on this smoke setting.
4. Even the best rule-based gate still fails to beat `top1_validation`.

## Interpretation

This first sample-level gate does **not** beat the best setting-level selector.

That matters for the paper design because it tells us:

1. sample-level gating is not automatically useful just because uncertainty exists
2. RF predictive uncertainty alone is too weak as a routing signal
3. teacher-student disagreement is more informative than teacher-teacher disagreement here
4. a stronger Stage 3 gate should likely be learned, not purely rule-based

## Consequence for Method Design

The next Stage 3 version should expose at least:

- teacher uncertainty
- teacher-student disagreement
- optional teacher-teacher disagreement
- optional descriptor-space OOD signal

The next serious version should be a learned gate over these signals, for example a small MLP fed with:

- teacher uncertainty
- absolute deviation from teacher consensus
- absolute deviation from current student prediction
- global teacher validation prior

That is a cleaner next step than continuing to hand-tune rule coefficients.

## Paper Value

This failed smoke run is still useful evidence:

> naive sample-level uncertainty gating is not enough, so reliability gating needs richer signals than teacher confidence alone.

That strengthens the motivation for the final method instead of weakening it.
