# Related Work Notes

## MolBERT / Domain-Relevant Auxiliary Tasks

Use this line of work to motivate molecular representations that benefit from
domain-specific auxiliary objectives rather than generic language modeling
alone.

Connection to this project:

- Descriptor prediction is a domain-relevant auxiliary task.
- The difference is that this project studies a lightweight ECFP-only inference
  setting rather than a large language-model encoder.

## Auxiliary Learning + Task-Specific Adaptation

Use this line of work to frame the distinction between a generic auxiliary loss
and task-specific adaptation.

Connection to this project:

- `ECFP4_MLP_DescPred` is the simple auxiliary-learning baseline.
- `ECFP4_MLP_DescAdapterFusion` is the task-specific adaptation variant.

## M2UMol

Use this line of work as the closest conceptual inspiration.

Connection to this project:

- M2UMol studies multi-to-uni knowledge transfer for molecular representation
  learning.
- This project tests a smaller, controlled version: descriptor-guided transfer
  into an ECFP-only ADMET predictor.

## ADMET Prediction

Use TDC ADMET benchmarks to position the experiments as drug-discovery relevant
and comparable across properties.

Required citations to add later:

- TDC benchmark paper.
- Dataset-specific ADMET references if needed by the venue.
- Descriptor and fingerprint baseline references.
