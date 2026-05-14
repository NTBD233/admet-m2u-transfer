# Stage 3 Learned Gate Smoke

## Goal

After rule-based sample gates failed to beat the strongest Stage 2 baseline, test whether a learned sample-level gate can recover the missing signal.

Smoke setting:

- Dataset: `caco2_wang`
- Train ratio: `10`
- Seed: `42`
- Student: `ECFP4_MLP_DescAdapterFusion`
- Teachers: `ECFP4_RF`, `Desc_RF`, `ECFP4_Desc_RF`
- Distillation strength: `lambda_distill = 1.0`

## Tried Learned Gates

1. `learned_reliability_gate`
   - per-teacher MLP scorer over:
     - teacher uncertainty
     - teacher-consensus deviation
     - teacher-student disagreement
     - global validation prior
   - weights produced directly by softmax

2. `learned_linear_gate`
   - linear version of the same feature set
   - added to test whether the learned MLP was simply overfitting

3. `learned_prior_residual_gate`
   - starts from the validation-weighted teacher prior
   - learns only a bounded sample-level residual on top of the prior
   - designed to be more stable in low-resource settings than direct weight prediction

## Smoke Results

| Method | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `1.5979` | `1.7255` |
| `uncertainty_student_prior` | `1.6591` | `1.7867` |
| `learned_reliability_gate` | `1.6962` | `1.7888` |
| `learned_prior_residual_gate` | `1.7155` | `1.8852` |
| `learned_linear_gate` | `1.7584` | `1.9129` |

## Takeaways

1. End-to-end learned gating did **not** beat the strongest setting-level selector.
2. The unconstrained linear gate is clearly too weak and unstable.
3. The MLP gate is slightly better than the linear gate, but still below the best rule-based gate.
4. Even the prior-residual gate, which should have been the most stable learned variant in this regime, still underperforms `top1_validation`.

## Interpretation

At this point the evidence is consistent:

1. sample-level gating is a real problem, but not one that current signals solve well enough
2. simply exposing uncertainty and disagreement features to an end-to-end gate is not sufficient
3. low-resource ADMET likely makes the gate itself hard to train without stronger supervision or stronger regularization

This is an important paper result, not just a failed implementation detail.

The current evidence supports the claim that:

> naive adaptive weighting fails, rule-based uncertainty routing fails, and straightforward end-to-end learned gating also fails under low-resource ADMET.

That sharply defines what the next method version must address.

## Consequence for Next Stage

The next Stage 3 iteration should stop trying to learn teacher weights only from the main task loss.

More plausible next directions are:

1. **oracle-supervised gate targets**
   - derive per-sample pseudo-labels from validation hindsight or teacher residual errors
   - train the gate with an auxiliary supervision signal instead of only indirect task loss

2. **hard selection instead of soft averaging**
   - predict top-1 / top-2 teacher selection rather than a dense softmax over all teachers
   - this better matches the Stage 2 result where `top1_validation` is already strong

3. **descriptor-space region modeling**
   - use train-set neighborhood / OOD region features explicitly
   - current uncertainty and disagreement signals may be too indirect

## Paper Value

This smoke run is still useful for the second paper because it narrows the real method gap:

> the challenge is not merely to parameterize a gate, but to obtain a reliable supervision signal for teacher selection in low-resource ADMET.
