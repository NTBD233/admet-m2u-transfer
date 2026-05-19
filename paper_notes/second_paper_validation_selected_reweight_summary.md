# Validation-Selected Reweight Summary

## Purpose

The selector-utilization sanity check showed that direct reweight modes can
improve specific failure cases:

- `global_confidence` helps the `ppbr_az / train_20 / seed_42` utilization
  failure;
- `disagreement_confidence` helps the `caco2_wang / train_10 / seed_42` smoke
  setting.

The next question was whether these modes can be selected per setting using
validation RMSE, instead of promoting one fixed reweight rule.

## Protocol

Scope:

- datasets: `caco2_wang`, `ppbr_az`
- train ratios: `10`, `20`, `50`
- seeds: `42`, `123`, `3407`
- total settings: `18`

Candidate modes:

- no reweight: `plain_selector`
- `global_confidence`
- `disagreement_confidence`

Selection rule:

> For each dataset / train-ratio / seed, choose the candidate with the lowest
> validation RMSE, then report its test RMSE.

Entrypoints:

```bash
PYTHON=.miniforge/envs/tdc-admet/bin/python make validation-selected-reweight-partial-regression
PYTHON=.miniforge/envs/tdc-admet/bin/python make validation-selected-reweight-summary
```

Comparison output:

```text
results_validation_selected_reweight_partial_regression/summary/
```

## Aggregate Results

| method | runs | mean test RMSE | mean valid RMSE |
| --- | ---: | ---: | ---: |
| `top1_validation` | 18 | 8.5725 | 9.6313 |
| `plain_selector` | 18 | 8.6091 | 9.6779 |
| `validation_selected_reweight` | 18 | 8.6244 | 9.6038 |
| `global_confidence` | 18 | 8.7540 | 9.6545 |
| `disagreement_confidence` | 18 | 8.9202 | 9.8263 |

Pairwise results for `validation_selected_reweight`:

| reference | wins | total | mean delta |
| --- | ---: | ---: | ---: |
| `plain_selector` | 5 | 18 | +0.0153 |
| `global_confidence` | 10 | 18 | -0.1296 |
| `disagreement_confidence` | 11 | 18 | -0.2958 |
| `top1_validation` | 8 | 18 | +0.0518 |

Mode counts:

| dataset | `plain_selector` | `global_confidence` | `disagreement_confidence` |
| --- | ---: | ---: | ---: |
| `caco2_wang` | 6 | 1 | 2 |
| `ppbr_az` | 4 | 4 | 1 |

## Interpretation

Validation-selected reweighting improves mean validation RMSE relative to plain
selector routing (`9.6038` vs `9.6779`), but this does not transfer to test
RMSE (`8.6244` vs `8.6091`). It also remains worse than `top1_validation`
(`8.5725`).

This is a useful negative result. The reweight modes are mechanistically real
and can improve specific settings, but the current validation-selected policy
is not robust enough to become a second main method.

## Paper Decision

Do not promote validation-selected reweighting to the main method.

Use it as mechanism evidence:

> Student-side utilization of routed teacher choices matters, but selecting a
> reweight rule from low-resource validation RMSE is itself unstable.

The manuscript should keep the main method as:

> selector pretraining + frozen routing distillation

and treat reweighting as a diagnostic showing where future student-side
optimization should focus.
