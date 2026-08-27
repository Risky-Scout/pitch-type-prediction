# Recruiter Summary

## Pitch Type Prediction

I built a reproducible four-class probability model using only the supplied 5,846-pitch dataset.

The final model is a **stacked ensemble of tuned XGBoost and CatBoost classifiers**. Each base learner is trained independently, and a regularized multinomial logistic stacker (`C=0.1`) combines their **out-of-fold log probabilities**. I explicitly tested extra post-hoc calibration; it did not improve validation log loss/Brier, so it was omitted.

### Why this validation design

The dataset does not include a date or game ID. To avoid a naïve row-random split, I construct conservative contiguous blocks using `pitch_count` and `inning_half`, then use `StratifiedGroupKFold` so a block never crosses train/validation/test.

### Sealed test

- Log loss: **1.179286**
- Multiclass Brier: **0.655005**
- Probability-cell MSE: **0.163751**
- Macro OVR AUC: **0.667794**
- Top-label ECE (10 bins): **0.012059**
- Classwise ECE (10 bins): **0.016531**

For comparison, a development-frequency baseline scored **1.274353 log loss**, **0.689122 Brier**, and **0.500000 macro AUC**.

The 5,000-resample grouped bootstrap interval for the stacker-minus-baseline log-loss difference is **[-0.123459, -0.067736]**.

### Reproducibility

The repository contains:

- pinned dependencies;
- all final hyperparameters;
- deterministic split/grouping code;
- model-family search audit;
- exact split manifest;
- validation/test probabilities;
- serialized fitted model;
- command-line train/predict scripts;
- smoke tests and GitHub Actions CI;
- interview walkthrough and model card.

The raw CSV is not included in the repository by default; the expected SHA-256 of the supplied file is recorded in the README/config so the exact data can be verified.
