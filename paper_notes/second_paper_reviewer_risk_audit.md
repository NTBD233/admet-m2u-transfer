# Second Paper Reviewer-Risk Audit

## Current Position

The manuscript is now defensible as a method paper about supervised teacher
selection, not as a paper claiming a universal new state of the art. The main
result is that pretrained selector routing is better than base AdapterFusion
and fixed descriptor-teacher distillation, and it nearly matches the strongest
setting-level teacher selector. The calibrated high-resource schedule narrows
the remaining gap but still does not decisively beat `top1_validation`.

## Most Likely Reviewer Concerns

### 1. The method does not beat `top1_validation` on mean RMSE

Risk: A reviewer may argue that the sample-level selector is unnecessary if a
validation-chosen teacher is marginally better in aggregate.

Defense:

- Report that the selector wins `25/45` runs against `top1_validation`, while
  the calibrated selector wins `27/45`.
- Emphasize that `top1_validation` uses a whole-setting validation decision and
  cannot adapt within a setting.
- Add the failure-mode probe: in `ppbr_az / train_20 / seed_42`, the selector
  is much better than `top1_validation` at choosing the lower-error teacher, but
  the student does not convert those choices into better predictions. This
  reframes the remaining gap as a student-utilization problem, not only a
  selector-quality problem.

### 2. Selector accuracy is moderate

Risk: A reviewer may question whether 0.54 test oracle accuracy is enough.

Defense:

- Avoid calling the selector an oracle.
- Compare it to majority and setting-prior baselines.
- State that the contribution is stable supervised selection signal, not
  perfect teacher identification.

### 3. Regression-only scope may look narrow

Risk: ADMET papers often include classification endpoints, so a regression-only
claim may look incomplete.

Defense:

- Keep the regression scope explicit.
- Use the classification diagnostic as a negative result showing that copying
  regression pseudo-oracle labels to classification is not valid.
- Frame classification as a calibration-specific extension, not missing main
  evidence.

### 4. Teacher pool is limited to RF variants

Risk: A reviewer may ask why graph or pretrained molecular teachers are absent.

Defense:

- Explain that the paper isolates the teacher-selection supervision problem
  before expanding the teacher pool.
- Position XGBoost, Chemprop, and pretrained encoders as natural extensions
  once selector supervision is established.
- Do not claim that RF teachers are the final best teacher pool.

### 5. The high-resource decay schedule is heuristic

Risk: A reviewer may view lambda decay as tuned post hoc.

Defense:

- Treat it as secondary calibration, not the core method.
- Report that the plain pretrained selector is already the main method.
- State that teacher choice and teacher strength are distinct axes, with learned
  teacher-strength calibration left for future work.

## Next Experiments If More Compute Is Available

1. Add a fixed-selector, student-side experiment for the `ppbr_az` failure mode:
   stronger selected-teacher distillation, delayed task-loss mixing, or
   selector-confidence lambda scaling.
2. Add a compact XGBoost teacher-pool extension after the RF-only story is
   stable.
3. For classification, redesign pseudo-oracle labels around ROC-AUC,
   probability calibration, or pairwise ranking behavior instead of absolute
   regression error.
