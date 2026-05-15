## Stage 3 Full Regression Summary: Pretrained Selector Top-1

### Setup

- Method: `pretrained_selector_top1`
- Student: `ECFP4_MLP_DescAdapterFusion`
- Teachers: `ECFP4_RF`, `Desc_RF`, `ECFP4_Desc_RF`
- Selector: `rf_crossfit_train_pseudo_oracle`
- Datasets: `caco2_wang`, `lipophilicity_astrazeneca`, `solubility_aqsoldb`, `vdss_lombardo`, `ppbr_az`
- Train ratios: `10`, `20`, `50`
- Seeds: `42`, `123`, `3407`
- Total settings: `45`

### Main aggregated result

| Method | Completed settings | Mean test RMSE | Mean valid RMSE |
| --- | ---: | ---: | ---: |
| `top1_validation` | 45 | 5.0792 | 5.6263 |
| `pretrained_selector_top1` | 45 | 5.0889 | 5.6427 |
| `fixed_teacher` | 45 | 5.1621 | 5.7487 |
| `base` | 45 | 5.1841 | 5.7456 |

Immediate reading:

- `pretrained_selector_top1` is clearly better than `base`.
- `pretrained_selector_top1` is also better than fixed single-teacher distillation.
- `pretrained_selector_top1` is very close to `top1_validation`, but does not beat it on mean RMSE.

### Pairwise comparisons on test RMSE

- `pretrained_selector_top1` vs `base`:
  - wins: `34/45`
  - mean delta: `-0.0953`
- `pretrained_selector_top1` vs `fixed_teacher`:
  - wins: `28/45`
  - mean delta: `-0.0732`
- `pretrained_selector_top1` vs `top1_validation`:
  - wins: `25/45`
  - losses: `20/45`
  - mean delta: `+0.0096`
  - median delta: `-0.0010`

Interpretation:

- The pretrained selector has already closed most of the gap to the strongest setting-level selector.
- The remaining gap to `top1_validation` is small in aggregate, but still real.
- The result is good enough to justify keeping pretrained selector routing as the second-paper main method candidate.

### Dataset-wise comparison vs `top1_validation`

Delta is `pretrained_selector_top1 - top1_validation` on test RMSE.

| Dataset | Wins | Losses | Mean delta |
| --- | ---: | ---: | ---: |
| `caco2_wang` | 5 | 4 | `+0.0004` |
| `lipophilicity_astrazeneca` | 5 | 4 | `-0.0031` |
| `ppbr_az` | 4 | 5 | `+0.0727` |
| `solubility_aqsoldb` | 4 | 5 | `-0.0034` |
| `vdss_lombardo` | 7 | 2 | `-0.0185` |

Takeaways:

- `vdss_lombardo` is the clearest positive case for pretrained selector routing.
- `lipophilicity_astrazeneca` and `solubility_aqsoldb` are roughly tied with `top1_validation`.
- `ppbr_az` is the main failure case and currently dominates the residual gap.

### What this means for the paper

This result supports the revised paper narrative:

1. teacher selection is predictable;
2. joint soft/hard gates are weak;
3. pretrained selector routing is much stronger than joint routing;
4. but selector pretraining alone is not yet enough to dominate the strongest setting-level teacher selector.

So the paper should not claim:

> pretrained selector routing decisively beats all simpler selectors.

Instead, the defensible claim is:

> selector supervision is the missing ingredient that makes sample-level teacher routing competitive, and it substantially narrows the gap between unstable joint gates and strong setting-level selection.

### Next decision

The next step should focus on one of two directions:

1. **selector as routing prior, not hard replacement**
   - blend pretrained selector probabilities with setting-level validation prior;
   - this targets the small remaining gap to `top1_validation`.

2. **failure analysis first**
   - inspect why `ppbr_az` remains weak;
   - determine whether the issue is selector error, teacher quality, or student optimization.

At this point, running full-matrix `pretrained_selector_top2` is lower priority than either of the two options above.
