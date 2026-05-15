## Auto Reweight Reproducibility

### Question

After adding validation/test teacher-selector loading, the new `auto` reweight mode initially selected the intended mode but failed to match the direct-mode smoke results.

The key question was:

> did the loader change alter training behavior, or was the divergence specific to the `auto` code path?

### Step 1: direct-mode rerun after loader change

Both direct modes were rerun after the loader update.

#### `caco2_wang / train_10 / seed_42`

- direct `disagreement_confidence`, old:
  - valid RMSE `1.6002`
  - test RMSE `1.6957`
- direct `disagreement_confidence`, rerun:
  - valid RMSE `1.6002`
  - test RMSE `1.6957`

#### `ppbr_az / train_20 / seed_42`

- direct `global_confidence`, old:
  - valid RMSE `18.3755`
  - test RMSE `15.9943`
- direct `global_confidence`, rerun:
  - valid RMSE `18.3755`
  - test RMSE `15.9943`

Conclusion:

- the loader change did **not** perturb the direct-mode training trajectory;
- the divergence came from the `auto` path itself.

### Step 2: identify the cause

The only extra computation in `auto` before training was a validation pass to choose the reweight mode.

That suggested a narrow hypothesis:

> the pre-training validation pass was perturbing the global RNG state, which then changed the shuffled train order and moved the optimization trajectory away from the direct-mode baseline.

### Step 3: fix

`train.py` now snapshots and restores:

- Python RNG state
- NumPy RNG state
- PyTorch CPU RNG state
- PyTorch CUDA RNG state, when available

around the `choose_selector_reweight_mode(...)` call.

### Step 4: verify `auto` against direct mode

After the RNG-state fix, `auto` exactly matched the intended direct-mode runs.

#### `caco2_wang / train_10 / seed_42`

- `auto` chose `disagreement_confidence`
- selector valid oracle acc: `0.4521`
- top1 valid oracle acc: `0.5137`
- `auto`, final:
  - valid RMSE `1.6002`
  - test RMSE `1.6957`
- direct `disagreement_confidence`:
  - valid RMSE `1.6002`
  - test RMSE `1.6957`

#### `ppbr_az / train_20 / seed_42`

- `auto` chose `global_confidence`
- selector valid oracle acc: `0.5426`
- top1 valid oracle acc: `0.2220`
- `auto`, final:
  - valid RMSE `18.3755`
  - test RMSE `15.9943`
- direct `global_confidence`:
  - valid RMSE `18.3755`
  - test RMSE `15.9943`

### Conclusion

This is a meaningful methodological cleanup:

1. the setting-aware mode switch is now reproducible;
2. `auto` no longer introduces hidden optimization drift;
3. the mode-selection logic is now a defensible part of the method, rather than a fragile implementation artifact.

### Immediate next step

The right follow-up is now clear:

- run a controlled partial regression expansion of `auto` on the most relevant subset;
- compare it against:
  - plain pretrained selector routing
  - fixed `top1_validation`
  - direct global/disagreement reweight where appropriate

That expansion was started on `caco2_wang + ppbr_az`, but it was intentionally stopped before full completion to avoid turning this turn into a pure wait cycle.
