## Selector-Confidence Distillation Reweight Smoke

### Motivation

The failure analysis showed two different regimes:

- `ppbr_az`: selector routing was already better than `top1_validation` at choosing the lower-error teacher, but the student still failed to benefit;
- `caco2_wang`: selector quality itself was not better than `top1_validation`.

That pointed to a concrete next step:

> keep selector routing fixed, and strengthen the distillation signal on samples where the selector is more confident.

### Training change

A minimal training-side modification was added to [train.py](/Users/yuzibo/Documents/Codex/2026-05-07/files-mentioned-by-the-user-tdc/train.py):

- new flag: `--selector-distill-reweight`
- for selector-based multi-teacher routing, the per-sample distillation loss is multiplied by:

```text
max(selector_probs) * num_teachers
```

Interpretation:

- low-confidence selector decisions are downweighted;
- high-confidence selector decisions are upweighted;
- average scale stays in a reasonable range without changing the routing policy itself.

### Probe settings

Two settings were used again:

1. `ppbr_az / train_20 / seed_42`
2. `caco2_wang / train_10 / seed_42`

### Student-level RMSE comparison

#### `ppbr_az / train_20 / seed_42`

| Method | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `18.1364` | `15.6716` |
| `pretrained_selector_top1` | `18.8200` | `16.4242` |
| `pretrained_selector_top1 + reweight` | `18.3755` | `15.9943` |
| `fixed_teacher` | `18.5822` | `16.0992` |

Reading:

- the reweighted version substantially improves over plain `pretrained_selector_top1`;
- it also beats fixed single-teacher distillation;
- it still does not fully catch `top1_validation`, but the gap shrinks sharply:
  - plain selector gap vs `top1_validation`: `+0.7527`
  - reweighted selector gap vs `top1_validation`: `+0.3227`

#### `caco2_wang / train_10 / seed_42`

| Method | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| `top1_validation` | `1.5979` | `1.7255` |
| `pretrained_selector_top1` | `1.6616` | `1.7337` |
| `pretrained_selector_top1 + reweight` | `1.6331` | `1.7385` |

Reading:

- validation RMSE improves;
- test RMSE degrades slightly;
- this matches the earlier diagnosis that `caco2_wang` is not mainly a routing-utilization problem.

### Failure-analysis re-run

The selector failure analysis was re-run with the reweighted student outputs.

#### `ppbr_az / train_20 / seed_42`, test split

| subset | selector oracle acc | `top1_validation` oracle acc | selector teacher beats top1 teacher | selector student beats top1 student | mean student abs error delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | `0.5546` | `0.2576` | `0.4490` | `0.5206` | `-0.2764` |
| disagree | `0.5467` | `0.1778` | `0.5578` | `0.5156` | `-0.3334` |
| disagree, selector better teacher | `0.9801` | `0.0000` | `1.0000` | `0.5498` | `-0.9757` |

Compare this to the plain selector-routed student before reweighting:

- plain selector student beats top1 student on only `39.36%` of test samples;
- mean student abs-error delta was `+1.0748`.

After reweighting:

- selector student beats top1 student on `52.06%` of test samples;
- mean student abs-error delta becomes `-0.2764`.

This is the strongest evidence so far that the training-side modification addresses the right bottleneck.

#### `caco2_wang / train_10 / seed_42`, test split

| subset | selector oracle acc | `top1_validation` oracle acc | selector teacher beats top1 teacher | selector student beats top1 student | mean student abs error delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | `0.5714` | `0.5989` | `0.0934` | `0.3956` | `+0.0079` |

This remains consistent with the earlier conclusion:

- `caco2_wang` is not mainly constrained by routing utilization;
- reweighting does not produce a meaningful test improvement there.

### Conclusion

This is the first post-selector modification that directly improves the right failure mode.

The evidence now supports:

1. selector quality is already useful on some difficult settings like `ppbr_az`;
2. the remaining problem there is student utilization of routed supervision;
3. confidence-based distillation reweighting makes the routed student use selector decisions more effectively.

### Next step

This result is strong enough to justify a larger follow-up:

- run a partial regression expansion of `pretrained_selector_top1 + reweight`;
- prioritize the settings where plain selector routing lost to `top1_validation`;
- only if the broader trend holds, then launch the full 45-setting matrix.
