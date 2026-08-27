from __future__ import annotations

"""Re-run the train-only model-family search used for the final submission.

This script deliberately does not touch outer fold 0 (test). It is intended as an
audit/research utility, not as the production training entry point.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .core import (
    CAT_FEATURES,
    CLASS_TO_INDEX,
    CLASSES,
    NUM_FEATURES,
    RAW_FEATURES,
    SEED,
    TARGET,
    infer_groups,
    make_outer_folds,
    probability_metrics,
)

XGB_CANDIDATES = [
    {"max_depth":2,"learning_rate":0.02,"n_estimators":700,"min_child_weight":20,"reg_lambda":10,"reg_alpha":1,"subsample":0.9,"colsample_bytree":0.9,"gamma":0.5},
    {"max_depth":2,"learning_rate":0.0168322,"n_estimators":800,"min_child_weight":23.2608,"reg_lambda":10.0417,"reg_alpha":1.111,"subsample":0.8631,"colsample_bytree":0.8995,"gamma":0.6703},
    {"max_depth":3,"learning_rate":0.02,"n_estimators":500,"min_child_weight":15,"reg_lambda":30,"reg_alpha":2,"subsample":0.9,"colsample_bytree":0.9,"gamma":0},
    {"max_depth":2,"learning_rate":0.03,"n_estimators":450,"min_child_weight":15,"reg_lambda":20,"reg_alpha":2,"subsample":0.9,"colsample_bytree":0.9,"gamma":0.5},
    {"max_depth":3,"learning_rate":0.015,"n_estimators":700,"min_child_weight":25,"reg_lambda":30,"reg_alpha":2,"subsample":0.85,"colsample_bytree":0.9,"gamma":0.5},
    {"max_depth":1,"learning_rate":0.025,"n_estimators":700,"min_child_weight":15,"reg_lambda":15,"reg_alpha":1,"subsample":0.9,"colsample_bytree":1.0,"gamma":0},
]
CAT_CANDIDATES = [
    {"iterations":700,"depth":5,"learning_rate":0.0246623,"l2_leaf_reg":27.7875,"random_strength":0.6303,"bagging_temperature":0.5293},
    {"iterations":600,"depth":5,"learning_rate":0.03,"l2_leaf_reg":20,"random_strength":0.5,"bagging_temperature":0.5},
    {"iterations":500,"depth":4,"learning_rate":0.035,"l2_leaf_reg":20,"random_strength":0.5,"bagging_temperature":0.5},
    {"iterations":700,"depth":4,"learning_rate":0.025,"l2_leaf_reg":30,"random_strength":0.8,"bagging_temperature":0.7},
    {"iterations":500,"depth":6,"learning_rate":0.03,"l2_leaf_reg":30,"random_strength":0.7,"bagging_temperature":0.5},
]
LGB_CANDIDATES = [
    {"n_estimators":350,"learning_rate":0.03,"num_leaves":7,"min_child_samples":80,"reg_lambda":10,"reg_alpha":1,"subsample":0.9,"colsample_bytree":0.9},
    {"n_estimators":500,"learning_rate":0.02,"num_leaves":7,"min_child_samples":100,"reg_lambda":20,"reg_alpha":2,"subsample":0.9,"colsample_bytree":0.9},
]


def xgb_model(params):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    base = dict(objective="multi:softprob", num_class=4, eval_metric="mlogloss",
                random_state=SEED, n_jobs=1, tree_method="hist", verbosity=0)
    base.update(params)
    return Pipeline([("prep", prep), ("model", XGBClassifier(**base))])


def cat_model(params):
    base = dict(loss_function="MultiClass", eval_metric="MultiClass", random_seed=SEED,
                verbose=False, thread_count=1, allow_writing_files=False, bootstrap_type="Bayesian")
    base.update(params)
    return CatBoostClassifier(**base)


def lgb_model(params):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    base = dict(objective="multiclass", num_class=4, random_state=SEED, n_jobs=1, verbosity=-1)
    base.update(params)
    return Pipeline([("prep", prep), ("model", LGBMClassifier(**base))])


def logit_model(C):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=C, max_iter=3000, solver="lbfgs", random_state=SEED))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="artifacts/research_rerun.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    fold, y, groups = make_outer_folds(df)
    train_idx = np.where(np.isin(fold, [2, 3, 4]))[0]
    X = df.iloc[train_idx][RAW_FEATURES].reset_index(drop=True)
    yt = y[train_idx]
    gt = groups[train_idx]
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 17)

    rows = []
    families = [
        ("xgb", XGB_CANDIDATES),
        ("cat", CAT_CANDIDATES),
        ("lgb", LGB_CANDIDATES),
        ("logit", [{"C": c} for c in [0.03, 0.1, 0.3]]),
    ]
    for family, candidates in families:
        for i, params in enumerate(candidates):
            mets = []
            for tr, va in cv.split(X, yt, gt):
                if family == "xgb":
                    m = xgb_model(params); m.fit(X.iloc[tr], yt[tr]); p = m.predict_proba(X.iloc[va])
                elif family == "cat":
                    m = cat_model(params); m.fit(X.iloc[tr], yt[tr], cat_features=CAT_FEATURES); p = m.predict_proba(X.iloc[va])
                elif family == "lgb":
                    m = lgb_model(params); m.fit(X.iloc[tr], yt[tr]); p = m.predict_proba(X.iloc[va])
                else:
                    m = logit_model(params["C"]); m.fit(X.iloc[tr], yt[tr]); p = m.predict_proba(X.iloc[va])
                mets.append(probability_metrics(yt[va], p))
            row = {"family": family, "candidate": i, "params": json.dumps(params, sort_keys=True)}
            for key in mets[0]:
                row[key] = float(np.mean([m[key] for m in mets]))
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["log_loss", "brier"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out[["family", "candidate", "log_loss", "brier", "auc_macro_ovr", "top_label_ece_10"]].to_string(index=False))


if __name__ == "__main__":
    main()
