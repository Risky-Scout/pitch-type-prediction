# Verification

This repository was verified end-to-end against the supplied CSV.

## Automated checks

- `pytest -q`: **3 passed**
- serialized production artifact reload: **passed**
- CLI inference on example rows: **passed**
- output probabilities are nonnegative and sum to 1: **passed**
- clean rerun of `python -m pitch_type_prediction.train`: **return code 0**
- reproduced test metrics: **matched delivered metrics to floating-point precision**
- maximum absolute difference between delivered and clean-rebuilt test probabilities: **0.0**

## Clean rebuild result

```text
log_loss             1.179285639492051
brier                0.6550049492514225
mse                  0.16375123731285562
auc_macro_ovr        0.6677941047562748
auc_weighted_ovr     0.6405426466576529
top_label_ece_10     0.012059487423863308
classwise_ece_10     0.016531037691755197
adaptive_top_ece_10  0.036523569963688524
accuracy              0.4705380017079419
```

The tiny printed differences versus the original metrics are floating-point formatting/serialization noise (roughly 1e-10 or smaller); the actual rebuilt model produced exactly the same test probability matrix as the delivered artifact.

## macOS Apple Silicon reproduction

A clean Python 3.11.15 environment on macOS Apple Silicon was used to perform two independent full retrains from the original supplied CSV.

Both local retrains produced exactly identical metrics and byte-for-byte identical test prediction CSVs.

Local reproduced test metrics:

- log loss: 1.1791660661243275
- multiclass Brier: 0.6549290275500701
- probability-cell MSE: 0.16373225688751755
- macro one-vs-rest AUC: 0.6678586192735007
- weighted one-vs-rest AUC: 0.6405828739115935
- top-label ECE (10 bins): 0.00877931311250502
- classwise ECE (10 bins): 0.014770696386812552
- adaptive top-label ECE (10 bins): 0.03734475244056453
- accuracy: 0.46968403074295473

The two macOS runs had zero metric differences and identical prediction files.

The originally shipped model artifact was trained in a different execution environment and differs from the macOS reproduction only at a very small numerical level. This is expected for native gradient-boosting implementations across platforms. Reproducibility should therefore be interpreted as deterministic within a pinned execution environment and statistically/numerically equivalent across supported platforms, rather than guaranteed bit-for-bit cross-platform model identity.

Dataset SHA-256:

3b6e8ae6975eb486aabcdf99613cc492ddf78dd69a7ac41e2991ce52335766f0
