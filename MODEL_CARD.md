# Model Card

## Model

**Pitch Type Stacked Probability Ensemble v1.0.0**

Base learners:

1. shallow heavily regularized XGBoost;
2. CatBoost with native handling of the three nominal predictors.

Meta learner:

- multinomial logistic regression;
- `C = 0.1`;
- inputs = concatenated log probabilities from the two base learners;
- trained on grouped out-of-fold base predictions.

No post-hoc calibration layer is used.

## Intended use

Estimate the probability distribution over:

- Changeup
- Four-Seam Fastball
- Sinker
- Slider

for rows with the nine predictors in the supplied task schema.

## Not intended for

- other pitchers without validation;
- future seasons without prospective backtesting;
- wagering deployment without additional monitoring/governance;
- causal interpretation.

## Training data

5,846 observations from the supplied CSV, all from one pitcher.

No external data is used.

Data SHA-256:

`3b6e8ae6975eb486aabcdf99613cc492ddf78dd69a7ac41e2991ce52335766f0`

## Grouping/splitting

Because true game IDs are absent, conservative contiguous blocks are inferred when:

- pitch count does not increase; or
- inning half changes.

297 blocks are generated.

Outer split is deterministic `StratifiedGroupKFold(5, shuffle=True, random_state=20260825)`.

## Hyperparameters

### XGBoost

```json
{
  "max_depth": 2,
  "learning_rate": 0.0168322,
  "n_estimators": 800,
  "min_child_weight": 23.2608,
  "reg_lambda": 10.0417,
  "reg_alpha": 1.111,
  "subsample": 0.8631,
  "colsample_bytree": 0.8995,
  "gamma": 0.6703
}
```

Additional fixed parameters:

```json
{
  "objective": "multi:softprob",
  "num_class": 4,
  "eval_metric": "mlogloss",
  "random_state": 20260825,
  "n_jobs": 1,
  "tree_method": "hist",
  "verbosity": 0
}
```

### CatBoost

```json
{
  "iterations": 700,
  "depth": 4,
  "learning_rate": 0.025,
  "l2_leaf_reg": 30,
  "random_strength": 0.8,
  "bagging_temperature": 0.7
}
```

Additional fixed parameters:

```json
{
  "loss_function": "MultiClass",
  "eval_metric": "MultiClass",
  "random_seed": 20260825,
  "thread_count": 1,
  "bootstrap_type": "Bayesian",
  "allow_writing_files": false
}
```

### Stacker

```json
{
  "C": 0.1,
  "solver": "lbfgs",
  "max_iter": 3000,
  "random_state": 20260825
}
```

## Locked validation

```json
{
  "log_loss": 1.1891370405973534,
  "brier": 0.6575286916561821,
  "mse": 0.16438217291404553,
  "auc_macro_ovr": 0.6573234424764693,
  "auc_weighted_ovr": 0.630331288973896,
  "top_label_ece_10": 0.012752861321354902,
  "classwise_ece_10": 0.012829229712704088,
  "adaptive_top_ece_10": 0.05055677425077333,
  "accuracy": 0.4563356164383562
}
```

## Sealed test

```json
{
  "log_loss": 1.179285639634919,
  "brier": 0.6550049493129529,
  "mse": 0.16375123732823824,
  "auc_macro_ovr": 0.6677941047562748,
  "auc_weighted_ovr": 0.6405426466576529,
  "top_label_ece_10": 0.012059487315033682,
  "classwise_ece_10": 0.016531037638253344,
  "adaptive_top_ece_10": 0.0365235703788463,
  "accuracy": 0.4705380017079419
}
```

## Uncertainty

Grouped bootstrap 95% intervals:

- log loss: [1.141644, 1.219142]
- Brier: [0.635031, 0.675494]
- macro OVR AUC: [0.638924, 0.692026]

Additive proper-score intervals use 5,000 cluster-bootstrap resamples. AUC/ECE intervals in `artifacts/metrics.json` use 400 cluster-bootstrap resamples because AUC/ECE are non-additive.

## Test-set integrity

CatBoost's standalone test log loss is slightly lower than the stacker's. The production model remains the pre-selected stacker; no post-test model switching was performed.

## Known limitations

The CSV lacks verified game IDs and timestamps. Inferred blocks reduce leakage risk but do not create a true chronological backtest. All observations come from one pitcher, so external validity is narrow.
