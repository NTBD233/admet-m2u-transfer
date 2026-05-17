# Second Paper Dataset Scope And Expansion Plan

## Current Main Scope

The main paper currently uses five regression ADMET endpoints:

| Dataset | Task | Endpoint role | Total split rows |
|---|---:|---|---:|
| `caco2_wang` | regression | absorption / permeability | 913 |
| `lipophilicity_astrazeneca` | regression | physicochemical property | 4203 |
| `solubility_aqsoldb` | regression | physicochemical property | 9985 |
| `vdss_lombardo` | regression | distribution | 1133 |
| `ppbr_az` | regression | distribution / binding | 2793 |

This is sufficient for a method prototype, but the manuscript should avoid a
general "all ADMET" claim. The defensible claim is:

> five representative regression ADMET endpoints covering absorption,
> distribution, and physicochemical properties under an ECFP-only inference
> constraint.

## Dataset Selection Rationale

The main set should be described as selected by four criteria:

1. Regression endpoint, so the teacher selector and distillation loss are
   evaluated under a single metric family.
2. Standardized splits and features are already available in the local TDC-style
   preprocessing pipeline.
3. Each endpoint supports 10%, 20%, and 50% low-resource train ratios with three
   seeds.
4. The endpoints cover distinct ADMET-relevant mechanisms instead of only one
   physicochemical family.

This framing reduces the cherry-picking risk without overstating generality.

## Available Supplementary Classification Scope

The repository already contains prepared splits and M2U features for five
classification endpoints:

| Dataset | Task | Endpoint role | Total split rows |
|---|---:|---|---:|
| `bbb_martins` | classification | blood-brain barrier permeability | 2033 |
| `hia_hou` | classification | human intestinal absorption | 581 |
| `pgp_broccatelli` | classification | P-gp interaction | 1221 |
| `bioavailability_ma` | classification | oral bioavailability | 643 |
| `herg` | classification | cardiotoxicity / ion-channel liability | 658 |

These should not be promoted to the main claim yet because earlier
classification behavior was unstable. They are useful as supplementary
diagnostics:

- Can the RF selector predict pseudo-oracle teacher labels under ROC-AUC tasks?
- Are teacher reliability features still informative?
- Does classification fail because of selector predictability, calibration, or
  downstream student distillation?

## Recommended Next Experiment Order

1. Generate RF-only teacher predictions for the five classification endpoints.
2. Train RF selectors with cross-fit pseudo-oracle labels for those endpoints.
3. Report selector validation/test accuracy and macro-F1 as supplementary
   mechanism evidence.
4. Only if selector diagnostics look reasonable, run a small classification
   routing smoke on `bbb_martins` and `hia_hou`.
5. Keep classification results in appendix unless the full student matrix is
   stable.

## Paper Positioning

The paper should not claim that the method is solved for every ADMET endpoint.
The stronger and more accurate positioning is:

- Main claim: supervised teacher selection improves ECFP-only distillation on
  representative regression ADMET endpoints.
- Secondary evidence: the repository supports classification diagnostics, but
  classification-specific routing requires separate calibration.
- Future extension: add more regression endpoints and graph/pretrained teachers
  after the RF-only selector protocol is stable.
