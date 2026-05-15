## Selector Prior Blend And Confidence Fallback Smoke

### Motivation

`pretrained_selector_top1` is competitive with `top1_validation`, but still trails it slightly on the full 45-setting regression matrix.

Two simple follow-up hypotheses were tested:

1. blend selector probabilities with setting-level validation prior before top-1 routing;
2. use selector top-1 only when selector confidence is high, otherwise fall back to the setting-level top-1 teacher.

The purpose here was not to introduce a final new method, but to determine whether the remaining gap to `top1_validation` can be closed with a simple routing correction.

### New strategies tested

- `pretrained_selector_validation_blend_top1`
  - route by `argmax(selector_probs * validation_prior)`
- `pretrained_selector_confidence_fallback_top1`
  - route by selector top-1 when `max(selector_probs) >= threshold`
  - otherwise route by the `top1_validation` teacher

### Targeted settings

Two settings were used as probes:

1. `caco2_wang / train_10 / seed_42`
   - the original smoke setting
   - useful to detect whether a modification harms the previously strong case
2. `ppbr_az / train_20 / seed_42`
   - one of the strongest failure cases for `pretrained_selector_top1`
   - full-matrix delta vs `top1_validation`: `+0.7527` RMSE

### Results

#### `caco2_wang / train_10 / seed_42`

| Method | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `1.5979` | `1.7255` |
| `pretrained_selector_top1` | `1.6616` | `1.7337` |
| `validation_blend_top1` | `1.6399` | `1.7419` |
| `confidence_fallback_top1 (t=0.60)` | `1.6241` | `1.7743` |
| `confidence_fallback_top1 (t=0.55)` | `1.6365` | `1.7528` |

Reading:

- both corrections improve validation RMSE relative to plain `pretrained_selector_top1`;
- both corrections worsen test RMSE;
- the fallback threshold `0.55` is less damaging than `0.60`, but still does not beat plain selector routing.

#### `ppbr_az / train_20 / seed_42`

| Method | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `18.1364` | `15.6716` |
| `pretrained_selector_top1` | `18.8200` | `16.4242` |
| `validation_blend_top1` | `18.4804` | `16.4286` |
| `confidence_fallback_top1 (t=0.60)` | `18.3239` | `16.1117` |
| `confidence_fallback_top1 (t=0.55)` | `18.6149` | `16.1216` |

Reading:

- `validation_blend_top1` does not help on the main failure case;
- `confidence_fallback_top1` clearly improves over plain `pretrained_selector_top1`;
- the best targeted result here is threshold `0.60`;
- however, even the improved fallback still does not match `top1_validation`.

### Selector confidence diagnostics used to choose fallback thresholds

Train-set selector confidence statistics:

- `caco2_wang / train_10 / seed_42`
  - mean max probability: `0.5429`
  - median max probability: `0.5356`
  - fraction with `max prob >= 0.6`: `0.3276`
- `ppbr_az / train_20 / seed_42`
  - mean max probability: `0.6000`
  - median max probability: `0.5696`
  - fraction with `max prob >= 0.6`: `0.4482`

This matches the observed tradeoff:

- a high threshold helps on `ppbr_az` by forcing fallback more often;
- the same threshold hurts `caco2_wang` by suppressing selector routing too aggressively.

### Conclusion

The current evidence does **not** support a simple global correction on top of pretrained selector routing:

- prior blending is not enough;
- confidence fallback has real repair signal on `ppbr_az`;
- but a single global threshold introduces a cross-setting tradeoff and does not reliably close the gap to `top1_validation`.

### Implication for the next step

The next step should move away from one global fallback rule and focus on **failure analysis / setting-aware control**.

The most reasonable follow-ups are:

1. analyze which teacher the selector chooses on the `ppbr_az` failure settings and how often that differs from `top1_validation`;
2. test setting-aware fallback thresholds rather than a single global threshold;
3. inspect whether the remaining gap is caused by selector misclassification or by student optimization after routing.
