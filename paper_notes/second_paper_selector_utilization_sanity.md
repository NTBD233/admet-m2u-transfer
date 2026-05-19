# Selector Utilization Sanity Check

## Purpose

The failure-mode analysis showed that the remaining gap to `top1_validation`
has two causes:

- selector-quality failure, where the selector does not choose better teachers;
- student-utilization failure, where the selector chooses better teachers but
  the ECFP-only student does not benefit from those choices.

This sanity check reruns the two direct reweight modes under the current code
state to verify that the earlier student-utilization finding still reproduces.

## Commands

Use the project environment:

```bash
PYTHON=.miniforge/envs/tdc-admet/bin/python make selector-utilization-sanity
PYTHON=.miniforge/envs/tdc-admet/bin/python make selector-utilization-sanity-analysis
```

The executed roots were:

- `results_selector_utilization_sanity_ppbr_global`
- `results_selector_utilization_sanity_caco2_disagreement`
- `results_selector_utilization_sanity_analysis`

## Smoke RMSE Results

### `ppbr_az / train_20 / seed_42`

| Method | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| `top1_validation` | 18.1364 | 15.6716 |
| plain `pretrained_selector_top1` | 18.8200 | 16.4242 |
| selector + `global_confidence` reweight | 18.3755 | 15.9943 |

The direct global-confidence reweight reproduces the previous direction: it
does not beat `top1_validation`, but it closes the test RMSE gap from `+0.7527`
to `+0.3227`.

### `caco2_wang / train_10 / seed_42`

| Method | Valid RMSE | Test RMSE |
| --- | ---: | ---: |
| `top1_validation` | 1.5979 | 1.7255 |
| plain `pretrained_selector_top1` | 1.6616 | 1.7337 |
| selector + `disagreement_confidence` reweight | 1.6002 | 1.6957 |

The disagreement-only reweight also reproduces the previous direction and beats
both plain selector routing and `top1_validation` on this smoke setting.

## Sample-Level Utilization Results

### `ppbr_az / train_20 / seed_42`, test split

| subset | n | selector oracle acc | top1 oracle acc | selector teacher beats top1 | selector student beats top1 | mean student abs error delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 559 | 0.5546 | 0.2576 | 0.4490 | 0.5206 | -0.2764 |
| disagree | 450 | 0.5467 | 0.1778 | 0.5578 | 0.5156 | -0.3334 |
| disagree, selector better teacher | 251 | 0.9801 | 0.0000 | 1.0000 | 0.5498 | -0.9757 |

This confirms the student-utilization interpretation. In the plain selector
failure analysis, the selector chose better teachers on many samples but the
student still lost to the `top1_validation` student. With global-confidence
reweighting, the selector-routed student beats the `top1_validation` student on
52.06% of all test samples, and the mean absolute-error delta becomes negative.

### `caco2_wang / train_10 / seed_42`, test split

| subset | n | selector oracle acc | top1 oracle acc | selector teacher beats top1 | selector student beats top1 | mean student abs error delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 182 | 0.5714 | 0.5989 | 0.0934 | 0.6484 | -0.0341 |
| disagree | 33 | 0.3333 | 0.4848 | 0.5152 | 0.6364 | -0.0607 |
| disagree, selector better teacher | 17 | 0.6471 | 0.0000 | 1.0000 | 0.5882 | -0.0394 |

This setting remains different: selector teacher-selection accuracy is still
weaker than `top1_validation` overall, but disagreement-only reweighting helps
the student-level outcome.

## Interpretation

The direct reweight modes are reproducible under the current code and support a
clear mechanism:

> selector supervision is useful, but routed teacher choices need a
> student-side utilization mechanism.

However, previous partial-matrix results showed that a single global reweight
rule is not robust enough to promote as the main method. The next experiment
should therefore test a small validation-selected policy over reweight modes,
not a fixed universal reweight.

## Next Step

Run a targeted partial expansion over the settings where plain selector routing
loses to `top1_validation`:

- candidate modes: no reweight, `global_confidence`,
  `disagreement_confidence`;
- selection criterion: validation RMSE among these modes, not validation oracle
  teacher accuracy alone;
- first scope: `caco2_wang` and `ppbr_az`, all ratios and seeds.

If validation-selected reweighting improves mean RMSE without hurting
selector-stable settings, it can become a secondary method. Otherwise it should
remain a mechanism analysis rather than a headline result.
