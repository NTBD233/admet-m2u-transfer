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

## Interpretation

This first sample-level gate does **not** beat the best setting-level selector.

That matters for the paper design because it tells us:

1. sample-level gating is not automatically useful just because uncertainty exists
2. RF predictive uncertainty alone is too weak as a routing signal
3. a stronger Stage 3 gate should use multiple reliability signals, not only uncertainty

## Consequence for Method Design

The next Stage 3 version should combine at least:

- teacher uncertainty
- teacher disagreement
- teacher-student disagreement
- optional descriptor-space OOD signal

A reasonable next step is a rule-based composite score:

`score_i(x) = a * validation_prior_i - b * uncertainty_i(x) - c * disagreement_i(x)`

Then the learned gate can be introduced only after this composite rule shows signal.

## Paper Value

This failed smoke run is still useful evidence:

> naive sample-level uncertainty gating is not enough, so reliability gating needs richer signals than teacher confidence alone.

That strengthens the motivation for the final method instead of weakening it.
