## Selector Calibration And Route Audit

### Purpose

After the `auto` reweight partial regression failed to improve aggregate RMSE, two follow-up questions were tested:

1. Are selector probabilities poorly calibrated?
2. Does the validation-set route decision choose the wrong reweight family?

The audit was run on the same controlled subset:

- datasets: `caco2_wang`, `ppbr_az`
- train ratios: `10 / 20 / 50`
- seeds: `42 / 123 / 3407`
- total settings: `18`

The script is:

- [analyze_selector_calibration_and_routes.py](/Users/yuzibo/Documents/Codex/2026-05-07/files-mentioned-by-the-user-tdc/analyze_selector_calibration_and_routes.py)

Raw outputs are under:

- `results_selector_calibration_route_audit/`

### 1. Selector confidence calibration

Calibration was measured by comparing selector confidence with oracle teacher-selection accuracy.

| Dataset | Split | Mean confidence | Oracle accuracy | ECE |
| --- | --- | ---: | ---: | ---: |
| `caco2_wang` | valid | `0.5255` | `0.4924` | `0.0357` |
| `caco2_wang` | test | `0.5398` | `0.5354` | `0.0387` |
| `ppbr_az` | valid | `0.5517` | `0.5566` | `0.0230` |
| `ppbr_az` | test | `0.5602` | `0.5450` | `0.0267` |

Reading:

- selector confidence is not severely miscalibrated overall;
- `ppbr_az` is actually reasonably calibrated;
- high-confidence bins are mildly overconfident, especially above `0.80`, but this is not large enough to explain the auto-reweight failure by itself.

Fixed-bin calibration pattern:

- low-confidence bins can be underconfident or slightly noisy;
- confidence `0.50-0.80` is fairly reasonable;
- confidence above `0.80` is overconfident, but contains relatively few samples.

### 2. Validation route audit

The `auto` rule selects:

- `global_confidence` when validation selector oracle accuracy beats validation top1 oracle accuracy;
- `disagreement_confidence` otherwise.

This was audited against the equivalent test-split teacher-selection advantage.

| Dataset | Settings | Mode match rate | Auto wins vs plain | Auto wins vs top1 | Mean delta vs plain | Mean delta vs top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `caco2_wang` | `9` | `0.5556` | `4` | `3` | `+0.0196` | `+0.0201` |
| `ppbr_az` | `9` | `0.8889` | `3` | `2` | `+0.3371` | `+0.4098` |
| `ALL` | `18` | `0.7222` | `7` | `5` | `+0.1784` | `+0.2149` |

Reading:

- for `caco2_wang`, route family selection is genuinely noisy;
- for `ppbr_az`, route family selection is mostly correct, but student RMSE still degrades.

This distinction matters. It means there are two different failure modes:

1. `caco2_wang`: route-family selection is unstable;
2. `ppbr_az`: route-family selection is mostly right, but the student does not reliably benefit.

### 3. Important setting examples

#### Local positive cases

`caco2_wang / train_10 / seed_42`

- selected mode: `disagreement_confidence`
- test oracle mode: `disagreement_confidence`
- auto vs plain selector: `-0.0380`
- auto vs `top1_validation`: `-0.0297`

This setting supports the intuition behind conservative disagreement-aware correction.

`ppbr_az / train_20 / seed_3407`

- selected mode: `global_confidence`
- test oracle mode: `global_confidence`
- auto vs plain selector: `-0.2376`
- auto vs `top1_validation`: `-0.2904`

This setting supports strong selector utilization when selector teacher-selection accuracy is clearly better.

#### Failure cases despite correct mode

`ppbr_az / train_50 / seed_42`

- selected mode: `global_confidence`
- test oracle mode: `global_confidence`
- auto vs plain selector: `+1.2722`
- auto vs `top1_validation`: `+0.8630`

`ppbr_az / train_10 / seed_42`

- selected mode: `global_confidence`
- test oracle mode: `global_confidence`
- auto vs plain selector: `+0.5252`
- auto vs `top1_validation`: `+0.9600`

These failures show that correct teacher-selection direction is not enough. The student can still be harmed by stronger routed distillation.

### 4. Scientific interpretation

The current evidence rules out two simple explanations:

1. **It is not mainly a confidence calibration problem.**
   - ECE is modest: roughly `0.02-0.04`.
   - Better calibration may help, but it is unlikely to fully solve the auto-reweight failure.

2. **It is not only a validation route-selection problem.**
   - On `ppbr_az`, route mode matches the test oracle direction in `8/9` settings.
   - Yet auto reweighting still loses on mean RMSE.

The stronger explanation is:

> selector-side correctness and student-side usefulness are not equivalent.

The selector can choose a teacher that is more oracle-like at the sample level, and the route family can be directionally correct, but the student may still overfit, destabilize, or fail to absorb that routed teacher signal.

### 5. Consequence for the second paper

This reinforces the decision **not** to make auto reweighting the main method.

The stronger second-paper story remains:

> pretrained teacher selection makes sample-level routing competitive, but the remaining barrier is the coupling between selector decisions and student optimization.

The reweight experiments should be presented as a mechanism analysis:

- they show local gains;
- they expose the student-utilization problem;
- they justify why naive confidence scaling is insufficient.

### 6. Recommended next move

Do not add more hand-designed reweight rules immediately.

The next technically clean direction is:

1. keep `pretrained_selector_top1` as the main method candidate;
2. add a section analyzing why confidence-based reweighting fails globally;
3. if continuing methods work, test a more principled student-side fix:
   - lower distillation weight at higher train ratios;
   - route-specific validation early stopping;
   - or teacher-loss normalization to prevent strong routed targets from dominating optimization.

Among these, the lowest-risk next experiment is:

> ratio-aware distillation strength for selector-routed training.

Reason:

- auto reweight failure is especially visible at `ppbr_az / train_50`;
- higher-resource settings likely need less teacher forcing, not more.
