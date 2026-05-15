## Selector Reweight Follow-up

### What was tested

This follow-up pushed the selector-routed student in three directions:

1. run a partial `ppbr_az` regression expansion for `pretrained_selector_top1 + global confidence reweight`;
2. test a narrower `disagreement_confidence` reweight rule on the two diagnostic smoke settings;
3. add an `auto` reweight mode that chooses between `global_confidence` and `disagreement_confidence` from validation-split oracle teacher accuracy.

### 1. Partial `ppbr_az` regression expansion

Global confidence reweight looked promising on the original smoke:

- `ppbr_az / train_20 / seed_42`
  - plain selector: `16.4242`
  - global reweight: `15.9943`
  - `top1_validation`: `15.6716`

But after expanding to all 9 `ppbr_az` regression settings, the trend did **not** hold globally.

Aggregate comparison on test RMSE:

- plain selector mean: `16.1134`
- global reweight mean: `16.4012`
- `top1_validation` mean: `16.0407`
- fixed teacher mean: `16.4917`

Wins:

- global reweight vs plain selector: `3/9`
- global reweight vs `top1_validation`: `2/9`
- global reweight vs fixed teacher: `4/9`

Interpretation:

- the training-side reweight is real and helps some hard settings;
- but as a **global default**, it is too blunt.

### 2. Disagreement-only reweight

The narrower rule was:

> only upweight the distillation loss when selector top-1 disagrees with the setting-level validation top-1 teacher.

Smoke results:

#### `caco2_wang / train_10 / seed_42`

| Method | Test RMSE |
| --- | ---: |
| `top1_validation` | `1.7255` |
| plain selector | `1.7337` |
| global reweight | `1.7385` |
| disagreement-only reweight | `1.6957` |

This was the first result that beat `top1_validation` on this setting.

#### `ppbr_az / train_20 / seed_42`

| Method | Test RMSE |
| --- | ---: |
| `top1_validation` | `15.6716` |
| plain selector | `16.4242` |
| global reweight | `15.9943` |
| disagreement-only reweight | `17.0289` |

So the two reweight rules are complementary:

- `ppbr_az` prefers stronger global selector utilization;
- `caco2_wang` prefers a more conservative disagreement-only correction.

### 3. Validation-aware `auto` reweight mode

To support a setting-level choice between the two rules, two infrastructure changes were added:

1. `utils/dataset.py` now loads teacher predictions and selector probabilities for `valid` and `test` splits when available;
2. `train.py` now supports:
   - `--selector-distill-reweight-mode global_confidence`
   - `--selector-distill-reweight-mode disagreement_confidence`
   - `--selector-distill-reweight-mode auto`

`auto` uses validation-split oracle teacher accuracy:

- if selector oracle accuracy > setting-level top-1 oracle accuracy, choose `global_confidence`;
- otherwise choose `disagreement_confidence`.

The mode resolution itself behaved as intended:

- `ppbr_az / train_20 / seed_42`
  - selector valid oracle acc: `0.5426`
  - top1 valid oracle acc: `0.2220`
  - resolved mode: `global_confidence`

- `caco2_wang / train_10 / seed_42`
  - selector valid oracle acc: `0.4521`
  - top1 valid oracle acc: `0.5137`
  - resolved mode: `disagreement_confidence`

### Current blocker

Although `auto` chooses the intended mode, the rerun smoke metrics after enabling valid/test teacher-selector loading did **not** reproduce the earlier direct-mode wins:

- `auto_reweight_caco2_v2`: test RMSE `1.8167`
- `auto_reweight_ppbr_v2`: test RMSE `16.9645`

Those numbers are materially worse than the earlier direct-mode smokes.

So the current status is:

1. the *scientific decomposition* is stronger than before;
2. the *engineering behavior* is not stable enough yet;
3. before any larger expansion, the next step must be a deterministic sanity check:
   - rerun direct `global_confidence` and `disagreement_confidence` after the loader change;
   - confirm whether the regression is caused by the new validation-aware plumbing or just normal training variance.

### Decision

Do **not** expand the new reweight variants further yet.

The right next move is:

1. verify reproducibility after the valid/test loader change;
2. only then decide whether the paper should keep:
   - a single global reweight rule,
   - a validation-aware mode switch,
   - or no reweight claim at all.
