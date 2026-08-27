# Pitch Type Probability Model

A leakage-conscious, reproducible multiclass probability model for the supplied **5,846-pitch / one-pitcher** dataset.

## Executive result

The locked production recipe is:

**XGBoost + CatBoost → out-of-fold log probabilities → L2 multinomial stacker (`C=0.1`)**

Model selection prioritized **multiclass log loss** and **Brier score**, with macro OVR AUC and calibration diagnostics (top-label ECE, classwise ECE, adaptive ECE) as secondary checks. Accuracy is reported but was not optimized.

### Sealed test performance

| Metric | Locked stacker | Development-frequency baseline |
|---|---:|---:|
| Log loss ↓ | **1.179286** | 1.274353 |
| Multiclass Brier ↓ | **0.655005** | 0.689122 |
| Probability-cell MSE ↓ | **0.163751** | 0.172281 |
| Macro OVR AUC ↑ | **0.667794** | 0.500000 |
| Weighted OVR AUC ↑ | **0.640543** | 0.500000 |
| Top-label ECE (10 bins) ↓ | **0.012059** | 0.003659 |
| Classwise ECE (10 bins) ↓ | **0.016531** | 0.001846 |
| Adaptive top-label ECE (10 bins) ↓ | **0.036524** | 0.054675 |
| Accuracy | **47.0538%** | 45.6020% |

Cluster bootstrap (5,000 resamples for additive proper scores) gives:

- test log loss 95% interval: **[1.141644, 1.219142]**
- stacker-minus-baseline log loss 95% interval: **[-0.123459, -0.067736]**
- test Brier 95% interval: **[0.635031, 0.675494]**
- stacker-minus-baseline Brier 95% interval: **[-0.046688, -0.022140]**

## Holdout discipline

The CSV has no date or game ID. A row-random split would risk overstating generalization because nearby pitches can be dependent.

The repository therefore infers conservative contiguous blocks. A new block starts when either:

1. `pitch_count` does not increase, or
2. `inning_half` changes.

This produces **297** blocks. They are proxies, not verified game IDs.

The deterministic outer `StratifiedGroupKFold` split uses seed `20260825`:

| Split | Rows | Groups |
|---|---:|---:|
| Train | 3,507 | 178 |
| Validation | 1,168 | 60 |
| Test | 1,171 | 59 |

- outer fold 0 = sealed test
- outer fold 1 = validation
- outer folds 2–4 = training

Base-model hyperparameters were compared with **train-only 4-fold grouped CV**. Architecture and stacker regularization were chosen on the validation fold. Only after that decision was the test fold opened.

## Model research

The search audit is in [`artifacts/model_search.csv`](artifacts/model_search.csv). Best train-CV log losses among tested families:

| Family | Best CV log loss |
|---|---:|
| CatBoost | **1.186951** |
| XGBoost | **1.190642** |
| LightGBM | 1.195190 |
| Multinomial logistic | 1.202007 |

Manual feature expansion and over-categorizing baseball count variables were investigated and rejected because they reduced out-of-sample proper-score performance. The production input remains the nine supplied predictors.

### Validation architecture decision

| Candidate | Log loss | Brier | Macro AUC | Top-label ECE |
|---|---:|---:|---:|---:|
| XGBoost | 1.191130 | 0.658608 | 0.655251 | 0.022215 |
| CatBoost | 1.190407 | 0.658168 | 0.656528 | 0.015607 |
| Optimized probability blend | 1.189937 | 0.657978 | 0.656914 | 0.017658 |
| **Stacker C=0.1** | **1.189137** | **0.657529** | **0.657323** | 0.012753 |
| Stacker + temperature | 1.189196 | 0.657563 | 0.657328 | **0.011556** |
| Stacker + class-bias temperature | 1.189249 | 0.657601 | 0.657338 | 0.013507 |

Temperature scaling marginally improved ECE but worsened the primary proper scores. It was therefore rejected. The final model has **no post-hoc calibration layer**.

## Important test-set note

After the architecture was locked, CatBoost alone happened to score slightly better than the stacker on this particular test fold:

- locked stacker test log loss: **1.179286**
- CatBoost diagnostic test log loss: **1.178266**

The official model is **not** changed after seeing this result. Promoting CatBoost post hoc would use the test set for model selection and invalidate the holdout.

## Feature importance

Validation-set permutation importance measured by change in log loss ranks the strongest supplied signals as:

1. `hitter_hand`
2. `balls`
3. `strikes`
4. `previous_pitch_type`
5. `ab_pitch_count`

See [`artifacts/feature_importance_validation.csv`](artifacts/feature_importance_validation.csv) and [`reports/validation_feature_importance.png`](reports/validation_feature_importance.png).

## Repository layout

```text
.
├── README.md
├── RECRUITER_SUMMARY.md
├── INTERVIEW_WALKTHROUGH.md
├── MODEL_CARD.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── configs/model_config.json
├── data/README.md
├── src/pitch_type_prediction/
│   ├── core.py
│   ├── train.py
│   ├── predict.py
│   └── research.py
├── tests/test_smoke.py
├── artifacts/
│   ├── pitch_type_model.joblib
│   ├── metrics.json
│   ├── model_search.csv
│   ├── validation_architecture_comparison.csv
│   ├── feature_importance_validation.csv
│   ├── split_manifest.csv
│   ├── validation_predictions.csv
│   └── test_predictions.csv
└── reports/
    ├── validation_architecture.png
    ├── validation_feature_importance.png
    └── test_reliability.png
```

## Setup

Python 3.11–3.13 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The raw task CSV is intentionally **not committed** by default. Place it at:

```text
data/pitch-type-prediction-data.csv
```

The exact supplied file used here has SHA-256:

```text
3b6e8ae6975eb486aabcdf99613cc492ddf78dd69a7ac41e2991ce52335766f0
```

## Reproduce the locked model

```bash
python -m pitch_type_prediction.train \
  --data data/pitch-type-prediction-data.csv \
  --output-dir artifacts/reproduced
```

## Re-run model-family research

```bash
python -m pitch_type_prediction.research \
  --data data/pitch-type-prediction-data.csv \
  --output artifacts/research_rerun.csv
```

## Generate probabilities

```bash
python -m pitch_type_prediction.predict \
  --model artifacts/pitch_type_model.joblib \
  --input new_pitches.csv \
  --output predictions.csv
```

Output contains `predicted_pitch_type` plus four probability columns whose rows sum to 1.

## Test

```bash
pytest -q
```

## Scope

This is the strongest validated model among the tested families and hyperparameter regions under the information actually present in the supplied CSV. The dataset contains one pitcher and no verified game IDs or timestamps, so the evidence does **not** support claims about other pitchers, other seasons, or a true future-time backtest.

## Recommended presentation environment

For macOS Apple Silicon, use Python 3.11. CatBoost 1.2.8 provides a compatible prebuilt wheel for Python 3.11, while newer Python versions may attempt an unsupported source build.

Environment setup:

    /opt/homebrew/bin/python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    python -m pytest -q

Full retraining:

    python -m pitch_type_prediction.train \
      --data data/pitch-type-prediction-data.csv \
      --output-dir reproduced



## Hyperparameter provenance

`configs/model_config.json` records the locked architecture, split/CV seeds,
selected hyperparameters, metrics, and calibration decision.
`artifacts/effective_hyperparameters.json` records the effective fitted-model
parameter sets, including library defaults, for XGBoost, CatBoost, and the
multinomial stacker.

## Presentation-friendly terminal commands

The package includes human-readable terminal views for live review and screen sharing.

Model/backtest report:

    python -m pitch_type_prediction.report

Integrity and leakage audit:

    python -m pitch_type_prediction.audit --data pitch-type-prediction-data.csv

Live prediction table:

    python -m pitch_type_prediction.predict --model artifacts/pitch_type_model.joblib --input demo_input.example.csv --output demo_predictions.csv

These commands do not alter the fitted statistical model or the locked evaluation protocol.
