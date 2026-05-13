# Teacher Reliability Diagnostics Summary

## Status

Stage 1 RF-only teacher reliability diagnostics have been implemented and run.

Command used:

```bash
./.miniforge/envs/tdc-admet/bin/python analyze_teacher_reliability.py \
  --teachers ECFP4_RF Desc_RF ECFP4_Desc_RF \
  --output-root results_teacher_reliability/summary
```

The generated CSV/Markdown outputs live under
`results_teacher_reliability/summary/`. That directory is intentionally ignored
by Git, so this note records the current tracked summary.

## Current Teacher Set

Analyzed teachers:

- `ECFP4_RF`
- `Desc_RF`
- `ECFP4_Desc_RF`

Deferred teachers:

- `ECFP4_XGB`
- `Desc_XGB`
- `ECFP4_Desc_XGB`

Reason: XGBoost teacher runs are not present locally yet.

## Aggregate Performance

Across 45 regression test runs per teacher:

| teacher | completed runs | mean test RMSE | median test RMSE | mean test MAE |
| --- | ---: | ---: | ---: | ---: |
| `Desc_RF` | 45 | 4.3087 | 1.3073 | 2.9017 |
| `ECFP4_Desc_RF` | 45 | 4.3705 | 1.3943 | 2.8804 |
| `ECFP4_RF` | 45 | 4.6439 | 1.7666 | 3.1309 |

Interpretation:

- `Desc_RF` is the strongest RF teacher on average.
- `ECFP4_Desc_RF` remains competitive and wins some settings.
- `ECFP4_RF` is weaker globally but still wins a non-trivial fraction of
  validation samples under sample-level oracle selection.

## Setting-Level Teacher Switching

Best test teacher counts across 45 dataset-ratio-seed settings:

| dataset | `Desc_RF` wins | `ECFP4_Desc_RF` wins |
| --- | ---: | ---: |
| `caco2_wang` | 9 | 0 |
| `lipophilicity_astrazeneca` | 0 | 9 |
| `ppbr_az` | 6 | 3 |
| `solubility_aqsoldb` | 9 | 0 |
| `vdss_lombardo` | 6 | 3 |

Interpretation:

- Teacher reliability is endpoint-dependent.
- `lipophilicity_astrazeneca` prefers the combined ECFP+descriptor RF teacher.
- `caco2_wang` and `solubility_aqsoldb` prefer descriptor-only RF.
- `ppbr_az` and `vdss_lombardo` show mixed teacher preference.

This supports the second-paper premise that a single global teacher is too
coarse.

## Sample-Level Oracle Signal

Mean validation sample-win fractions:

| dataset | `Desc_RF` | `ECFP4_Desc_RF` | `ECFP4_RF` |
| --- | ---: | ---: | ---: |
| `caco2_wang` | 0.500 | 0.225 | 0.275 |
| `lipophilicity_astrazeneca` | 0.444 | 0.263 | 0.293 |
| `ppbr_az` | 0.516 | 0.219 | 0.265 |
| `solubility_aqsoldb` | 0.492 | 0.261 | 0.246 |
| `vdss_lombardo` | 0.430 | 0.267 | 0.303 |

Interpretation:

- No teacher wins every sample.
- Even when `Desc_RF` is the strongest aggregate teacher, `ECFP4_RF` and
  `ECFP4_Desc_RF` win a meaningful minority of validation samples.
- This is direct motivation for sample-level reliability gating rather than
  dataset-level teacher selection alone.

## Reliability Feature Signals

Mean validation uncertainty-error correlation:

| teacher | corr |
| --- | ---: |
| `Desc_RF` | 0.410 |
| `ECFP4_Desc_RF` | 0.369 |
| `ECFP4_RF` | 0.329 |

Mean test descriptor-OOD-distance/error correlation:

| teacher | corr |
| --- | ---: |
| `Desc_RF` | 0.099 |
| `ECFP4_Desc_RF` | 0.144 |
| `ECFP4_RF` | 0.180 |

Interpretation:

- RF per-tree uncertainty is positively correlated with prediction error.
- Descriptor-space coverage is also positively correlated with prediction
  error, especially for ECFP-based RF teachers.
- These signals are usable inputs for the reliability gate.

## Current Conclusion

The RF-only diagnostics already support the core second-paper motivation:

> Teacher reliability varies by endpoint and by sample, and uncertainty/OOD
> signals contain information about teacher error.

This justifies moving from fixed single-teacher distillation toward
sample-level reliability-gated distillation.

## Next Actions

1. Train or generate XGBoost teachers:
   `ECFP4_XGB`, `Desc_XGB`, `ECFP4_Desc_XGB`.
2. Re-run `analyze_teacher_reliability.py` with RF+XGB teachers.
3. Implement Stage 2 student baselines:
   uniform multi-teacher, validation-weighted multi-teacher, top-1 validation
   teacher.
4. Implement Stage 3 reliability-gated conflict-aware distillation.

