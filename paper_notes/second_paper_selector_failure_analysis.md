## Selector Failure Analysis

### Why this analysis was needed

After the full regression matrix, `pretrained_selector_top1` was:

- clearly better than `base`;
- better than fixed single-teacher distillation;
- very close to `top1_validation`;
- but still slightly worse overall.

The key unresolved question was:

> Is the remaining gap caused by poor teacher selection, or by the student failing to benefit from selector-based routing?

To answer that, a setting-level failure analysis script was added:

- [analyze_selector_failure_modes.py](/Users/yuzibo/Documents/Codex/2026-05-07/files-mentioned-by-the-user-tdc/analyze_selector_failure_modes.py)

It compares, on a given split:

- selector teacher choice;
- `top1_validation` teacher choice;
- oracle best teacher from sample-level teacher errors;
- final student prediction error for the selector-routed student vs the `top1_validation` student.

### Two probe settings

Two settings were analyzed because they represent different behaviors.

1. `ppbr_az / train_20 / seed_42`
   - strong failure case for `pretrained_selector_top1`
   - full-matrix test delta vs `top1_validation`: `+0.7527`
2. `caco2_wang / train_10 / seed_42`
   - original smoke setting
   - `pretrained_selector_top1` was already close to `top1_validation`

---

## 1. `ppbr_az / train_20 / seed_42`

### Test split summary

| subset | n | selector oracle acc | `top1_validation` oracle acc | selector teacher beats top1 teacher | selector student beats top1 student | mean student abs error delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 559 | `0.5546` | `0.2576` | `0.4490` | `0.3936` | `+1.0748` |
| agree | 109 | `0.5872` | `0.5872` | `0.0000` | `0.3578` | `+1.3831` |
| disagree | 450 | `0.5467` | `0.1778` | `0.5578` | `0.4022` | `+1.0001` |
| disagree, selector better teacher | 251 | `0.9801` | `0.0000` | `1.0000` | `0.3307` | `+1.5003` |
| disagree, selector worse teacher | 199 | `0.0000` | `0.4020` | `0.0000` | `0.4925` | `+0.3693` |

### Interpretation

This is the most important finding so far.

On `ppbr_az`, the selector is **much better** than `top1_validation` at the teacher-selection problem itself:

- selector oracle accuracy: `0.5546`
- `top1_validation` oracle accuracy: `0.2576`

Even more strongly, on the disagreement subset:

- selector picks the lower-error teacher on `55.78%` of samples;
- `top1_validation` oracle accuracy there is only `17.78%`.

But the routed student still underperforms:

- selector-routed student beats the `top1_validation` student on only `39.36%` of test samples;
- mean student absolute-error delta is `+1.0748`, meaning the selector-routed student is worse.

The strongest evidence is this row:

> disagreement + selector better teacher

Here the selector almost perfectly tracks the oracle (`0.9801`), yet the final student still loses badly:

- selector student beats top1 student on only `33.07%` of samples;
- mean student abs-error delta is `+1.5003`.

### Conclusion for `ppbr_az`

The bottleneck is **not** sample-level teacher selection.

The bottleneck is:

> the student does not reliably convert better routing decisions into better predictions.

That points to a **student optimization / routing utilization** problem, not a selector problem.

---

## 2. `caco2_wang / train_10 / seed_42`

### Test split summary

| subset | n | selector oracle acc | `top1_validation` oracle acc | selector teacher beats top1 teacher | selector student beats top1 student | mean student abs error delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 182 | `0.5714` | `0.5989` | `0.0934` | `0.3736` | `+0.0072` |
| agree | 149 | `0.6242` | `0.6242` | `0.0000` | `0.3356` | `+0.0093` |
| disagree | 33 | `0.3333` | `0.4848` | `0.5152` | `0.5455` | `-0.0023` |
| disagree, selector better teacher | 17 | `0.6471` | `0.0000` | `1.0000` | `0.5294` | `-0.0126` |
| disagree, selector worse teacher | 16 | `0.0000` | `1.0000` | `0.0000` | `0.5625` | `+0.0087` |

### Interpretation

`caco2_wang` behaves differently from `ppbr_az`.

Here the selector is not stronger than `top1_validation` on the teacher-selection problem:

- selector oracle accuracy: `0.5714`
- `top1_validation` oracle accuracy: `0.5989`

So this setting is more consistent with:

> selector quality itself is not better than the simple setting-level teacher.

At the same time, on the small disagreement subset, when selector really does choose a better teacher, the student effect is slightly positive:

- disagreement subset mean student abs-error delta: `-0.0023`
- selector-better-teacher disagreement subset: `-0.0126`

That is small, but directionally consistent.

### Conclusion for `caco2_wang`

The main limitation is not routing utilization.

It is:

> the selector itself is not actually better than `top1_validation` on this setting.

---

## Cross-setting conclusion

The same global story does **not** explain every setting.

Instead, we now have two different failure modes:

1. **Selector-good, student-bad** (`ppbr_az`)
   - teacher selection is already better than `top1_validation`
   - the student fails to exploit that better routing

2. **Selector-not-better** (`caco2_wang`)
   - the selector does not beat the simple setting-level teacher choice
   - so there is little room for routed distillation to win

This is a much sharper diagnosis than “sample-level gating is unstable.”

### What this means for the second paper

The second paper should not collapse everything into one claim about selector quality.

A more accurate story is:

> Pretrained teacher selection makes sample-level routing competitive, but the remaining gap comes from two distinct sources: selector quality on some settings, and student utilization of routing decisions on others.

### Immediate next step

The next technical step should focus on the `ppbr_az` failure mode first, because it is the more interesting one:

- there is already selector signal;
- the missing part is how the student absorbs routed teacher supervision.

That makes the next good experiment:

> keep the selector fixed, and modify the distillation/training side rather than the selector side.

Concrete directions:

1. route-strength scaling:
   - increase distillation weight when selector confidence is high;
2. selector-conditioned distillation:
   - make the chosen teacher dominate more strongly than the current hard routing implementation effectively does;
3. routed optimization diagnostics:
   - inspect whether the chosen teacher signal is being diluted by task loss or descriptor transfer loss.
