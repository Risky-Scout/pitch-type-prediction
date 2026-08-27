# Interview Walkthrough

## 1. Frame the objective

> This is a **probability estimation** problem, not primarily an accuracy problem. A sportsbook cares about the entire conditional distribution over pitch types. I therefore optimize multiclass log loss, use Brier as a second proper score, and report AUC plus calibration diagnostics.

## 2. Describe the data constraint

The task provides 5,846 pitches from one pitcher, nine predictors, and four target classes. There is no game ID, date, season, pitcher ID, batter ID, score, base state, or external tracking data.

That means I do **not** add external data or APIs, and I do not claim cross-pitcher or future-season generalization.

## 3. Explain the split before the models

The hardest methodological problem is dependence.

A random row split could put nearby pitches from the same underlying game sequence on both sides of the split.

I infer conservative contiguous blocks:

- new block if `pitch_count <= previous pitch_count`;
- new block if `inning_half` changes.

This creates 297 blocks. Then I use shuffled `StratifiedGroupKFold` with a fixed seed.

Outer allocation:

- train: 3,507 rows / 178 groups
- validation: 1,168 rows / 60 groups
- test: 1,171 rows / 59 groups

I explicitly call these **inferred blocks**, not true games.

## 4. Start with baselines

I benchmark a regularized multinomial logistic model first. It scores about **1.2020 grouped-CV log loss** in the tested configuration.

That tells me the nonlinear tree models need to beat a sensible low-variance baseline.

## 5. Model-family research

Best train-only grouped-CV log loss by family:

- CatBoost: **1.186951**
- XGBoost: **1.190642**
- LightGBM: **1.195190**
- logistic: **1.202007**

I also tested manual feature interactions and aggressive categorical state expansion. Those variants did not improve out-of-sample proper scores, so they were rejected.

The lesson is that with only ~5.8k rows, strong regularization matters more than feature explosion.

## 6. Why CatBoost and XGBoost both survive

CatBoost handles the three nominal fields natively:

- `inning_half`
- `hitter_hand`
- `previous_pitch_type`

XGBoost sees one-hot versions of those variables and the remaining numeric predictors.

They make sufficiently different probability errors that a meta-learner adds value on validation.

## 7. Leakage-safe stacking

I never train the stacker on base-model predictions from models that saw the same labels.

For the training set:

1. split into grouped inner folds;
2. train each base learner on K-1 folds;
3. predict the held-out fold;
4. concatenate the OOF log probabilities;
5. fit regularized multinomial logistic regression on those meta-features.

That gives the stacker honest base-model inputs.

## 8. Validation decision

Validation results:

- XGBoost: **1.191130 log loss**
- CatBoost: **1.190407**
- optimized raw blend: **1.189937**
- stacker `C=0.1`: **1.189137**

So the stacker wins the primary metric and is locked.

## 9. Calibration decision

I then generate cross-fitted stacker predictions on the training set and test:

- scalar temperature scaling;
- temperature plus class-specific logit biases.

Temperature was nearly identity (`T ≈ 0.9938`).

Both calibrated variants slightly worsened validation log loss/Brier. One slightly improved ECE, but I do **not** trade away proper-score performance for a tiny ECE gain.

Final decision: **no extra calibration layer**.

## 10. Sealed test result

The locked stacker scores:

- log loss: **1.179286**
- Brier: **0.655005**
- MSE: **0.163751**
- macro OVR AUC: **0.667794**
- top-label ECE: **0.012059**
- classwise ECE: **0.016531**

Frequency baseline:

- log loss: **1.274353**
- Brier: **0.689122**
- macro AUC: **0.500000**

## 11. The test-set integrity question

CatBoost alone happens to score **1.178266 log loss** on the final test fold, slightly better than the selected stacker at **1.179286**.

I do not switch models.

> The stacker was selected before the test was opened. Promoting CatBoost because of the test realization would convert the test set into another validation set and bias the performance estimate.

That is the correct holdout discipline.

## 12. ECE nuance

A constant frequency forecast can show a deceptively small conventional top-label ECE because its one constant max probability may nearly equal its observed accuracy.

That does **not** imply useful conditional probabilities: its macro AUC is 0.50 and its log loss/Brier are materially worse.

So I use ECE diagnostically, not as the primary objective.

## 13. Strongest observed features

Validation permutation importance by increase in log loss:

1. hitter hand
2. balls
3. strikes
4. previous pitch type
5. plate-appearance pitch count

I would present these as **predictive importance**, not causal effects.

## 14. Limitations

The strongest limitation is data scope:

- one pitcher;
- no true game IDs;
- no timestamps;
- no batter identity;
- no pitch location/velocity/movement;
- no game context beyond supplied fields.

Therefore the correct claim is:

> strongest validated model among the tested approaches under the supplied-data constraint.

Not “universally best pitch model.”

## 15. Live demo

```bash
pip install -e ".[dev]"
pytest -q

python -m pitch_type_prediction.predict \
  --model artifacts/pitch_type_model.joblib \
  --input demo_input.example.csv \
  --output demo_predictions.csv
```

Then open `demo_predictions.csv` and show that the four probabilities are nonnegative and sum to 1.
