This directory stores project data artifacts.

Expected inputs from the original notebook:

- `prepared_data_subsample/{dataset}/train_50.csv`
- `prepared_data_subsample/{dataset}/valid.csv`
- `prepared_data_subsample/{dataset}/test.csv`

Generated feature files:

- `features_m2u/{dataset}/train_50_features.npz`
- `features_m2u/{dataset}/valid_features.npz`
- `features_m2u/{dataset}/test_features.npz`
- `features_m2u/{dataset}/desc_scaler.pkl`

The scripts keep the original notebook logic: ECFP4 fingerprints are the main
input, RDKit physicochemical descriptors are auxiliary training knowledge, and
inference uses only ECFP4.
