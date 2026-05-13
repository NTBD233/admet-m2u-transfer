# Paper Outline

Working title:

Lightweight Descriptor-Guided Multi-to-Uni Transfer for Low-Resource ADMET Prediction

## Core Claim

RDKit physicochemical descriptors provide useful auxiliary molecular knowledge
for low-resource ADMET prediction. The paper tests whether this knowledge can
be transferred into an ECFP-only inference model through descriptor prediction
and adapter-based fusion.

## Proposed Structure

1. Introduction
   - Low-resource ADMET prediction is common in early drug discovery.
   - Descriptors are cheap and informative, but direct descriptor access is an
     upper-bound control rather than the desired lightweight inference setting.
   - The goal is descriptor-guided multi-to-uni transfer.

2. Related Work
   - Molecular language models and domain-relevant auxiliary tasks.
   - Auxiliary learning and task-specific adaptation.
   - M2UMol and multi-to-uni molecular representation transfer.
   - ADMET benchmark modeling.

3. Method
   - ECFP4 encoder.
   - Descriptor prediction auxiliary objective.
   - AdapterFusion: pseudo-descriptor adapter plus learned fusion gate.
   - Inference uses only ECFP4 except the DescConcat upper-bound control.

4. Experiments
   - TDC ADMET datasets.
   - Train ratios: 10%, 20%, 50%.
   - Seeds: 42, 123, 3407.
   - Metrics: ROC-AUC/PR-AUC for classification, MAE/RMSE for regression.

5. Results and Analysis
   - Main benchmark table.
   - Low-resource sensitivity.
   - Lambda ablation.
   - Gate distribution and descriptor prediction error.

6. Discussion
   - Descriptor knowledge is useful.
   - Structured transfer can help, especially on regression.
   - Direct descriptor access remains a strong upper bound.
   - Classification gains may be task-dependent.

7. Limitations
   - Current scope is ADMET prediction, not retrosynthesis.
   - Method has not yet been validated against large pretrained encoders.
   - AdapterFusion must be tested across more datasets before being framed as a
     general method improvement.
