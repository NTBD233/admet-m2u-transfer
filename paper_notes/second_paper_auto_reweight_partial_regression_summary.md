## Auto Reweight Partial Regression Summary

### Scope

This controlled expansion was run only on the two most relevant regression datasets:

- `caco2_wang`
- `ppbr_az`

Experimental grid:

- train ratios: `10 / 20 / 50`
- seeds: `42 / 123 / 3407`
- total settings: `18`

Method under test:

> `pretrained_selector_top1 + selector_distill_reweight(mode=auto)`

Reference methods:

- plain `pretrained_selector_top1`
- `top1_validation`
- fixed single-teacher distillation

### Aggregate comparison

Mean test RMSE:

- `auto_reweight`: `8.7875`
- plain selector: `8.6091`
- `top1_validation`: `8.5725`
- fixed teacher: `8.7949`

Wins:

- auto vs plain selector: `7/18`
- auto vs `top1_validation`: `5/18`
- auto vs fixed teacher: `8/18`

Mean deltas:

- auto vs plain selector: `+0.1783`
- auto vs `top1_validation`: `+0.2149`
- auto vs fixed teacher: `-0.0074`

### Dataset-wise behavior

#### `caco2_wang`

- auto mean test RMSE: `1.1245`
- plain selector mean: `1.1048`
- `top1_validation` mean: `1.1044`
- auto wins vs plain selector: `4/9`
- auto wins vs `top1_validation`: `3/9`
- mean delta vs plain selector: `+0.0196`
- mean delta vs `top1_validation`: `+0.0201`

Takeaway:

- the gap is small;
- auto can help on selected settings;
- but it is not a consistent improvement even on the dataset where disagreement-aware routing looked promising in smoke.

#### `ppbr_az`

- auto mean test RMSE: `16.4505`
- plain selector mean: `16.1134`
- `top1_validation` mean: `16.0407`
- auto wins vs plain selector: `3/9`
- auto wins vs `top1_validation`: `2/9`
- mean delta vs plain selector: `+0.3371`
- mean delta vs `top1_validation`: `+0.4098`

Takeaway:

- the `ppbr_az / train_20` gains are real but localized;
- as soon as the setting grid is widened, the auto rule is not robust enough to dominate plain selector routing.

### Important positives

Even though the aggregate result is negative, two things were established:

1. the `auto` mode-selection path is now reproducible;
2. the rule does sometimes pick the right correction:
   - `caco2_wang / train_10 / seed_42`: auto chooses `disagreement_confidence`, test RMSE `1.6957`, which beats `top1_validation` `1.7255`;
   - `ppbr_az / train_20 / seed_42`: auto chooses `global_confidence`, test RMSE `15.9943`, narrowing the plain-selector gap.

So the failure is not due to an implementation artifact anymore. It is a method-quality issue.

### Current conclusion

This is not yet strong enough to become the paper's main upgraded method.

The defensible interpretation is:

- setting-aware distillation reweighting is mechanistically meaningful;
- but the current hand-designed `auto` rule is too weak as a general recipe.

### Implication for the second paper

The paper should **not** currently pivot its main claim to:

> selector auto-reweighting solves the remaining gap.

Instead, the safer role is:

- negative-but-informative follow-up analysis;
- evidence that the remaining difficulty is not just “which teacher” but also “how strongly to enforce routed supervision”.

### Recommended next step

Do not expand `auto` to the full 45-setting regression matrix yet.

The next sensible move is one of:

1. **selector confidence calibration analysis**
   - test whether the selector probabilities are miscalibrated across settings;
   - this would directly explain why `auto` is locally useful but globally unstable.

2. **validation-set route audit**
   - compare chosen mode vs actual best mode per setting;
   - quantify how often the `auto` rule chooses the wrong reweight family.

3. **stop escalating reweight rules**
   - keep auto-reweight as analysis only;
   - return focus to the stronger pretrained-selector baseline as the main second-paper method.
