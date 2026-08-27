from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

TARGET = "pitch_type"
RAW_FEATURES = [
    "inning_number",
    "inning_half",
    "balls",
    "strikes",
    "outs",
    "hitter_hand",
    "ab_pitch_count",
    "pitch_count",
    "previous_pitch_type",
]
CAT_FEATURES = ["inning_half", "hitter_hand", "previous_pitch_type"]
NUM_FEATURES = [c for c in RAW_FEATURES if c not in CAT_FEATURES]
CLASSES = ["Changeup", "Four-Seam Fastball", "Sinker", "Slider"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
SEED = 20260825

XGB_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.0168322,
    "n_estimators": 800,
    "min_child_weight": 23.2608,
    "reg_lambda": 10.0417,
    "reg_alpha": 1.111,
    "subsample": 0.8631,
    "colsample_bytree": 0.8995,
    "gamma": 0.6703,
}
CAT_PARAMS = {
    "iterations": 700,
    "depth": 4,
    "learning_rate": 0.025,
    "l2_leaf_reg": 30.0,
    "random_strength": 0.8,
    "bagging_temperature": 0.7,
}
STACKER_C = 0.1
META_CLIP = 1e-9


def validate_frame(df: pd.DataFrame, require_target: bool = False) -> None:
    required = RAW_FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric = [c for c in RAW_FEATURES if c not in CAT_FEATURES]
    for c in numeric:
        values = pd.to_numeric(df[c], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Column {c!r} contains non-numeric or non-finite values.")

    allowed = {
        "inning_half": {"B", "T"},
        "hitter_hand": {"L", "R"},
        "previous_pitch_type": set(CLASSES) | {"na"},
    }
    for c, vals in allowed.items():
        bad = set(df[c].dropna().astype(str).unique()) - vals
        if bad:
            raise ValueError(f"Unexpected values in {c!r}: {sorted(bad)}")

    if require_target:
        bad_target = set(df[TARGET].dropna().astype(str).unique()) - set(CLASSES)
        if bad_target:
            raise ValueError(f"Unexpected target labels: {sorted(bad_target)}")


def infer_groups(df: pd.DataFrame) -> np.ndarray:
    """Conservative contiguous blocks using only fields available in the CSV.

    Start a new block when:
      1) pitch_count does not increase, OR
      2) inning_half changes.

    This is a proxy for game-level grouping, not a true game_id.
    """
    validate_frame(df, require_target=False)
    non_increase = df["pitch_count"].diff().fillna(1) <= 0
    half_change = df["inning_half"].astype(str) != df["inning_half"].astype(str).shift(1)
    start = (non_increase | half_change).to_numpy(dtype=bool)
    start[0] = True
    return np.cumsum(start).astype(int)


def make_outer_folds(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_frame(df, require_target=True)
    y = df[TARGET].map(CLASS_TO_INDEX).to_numpy()
    groups = infer_groups(df)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold = np.empty(len(df), dtype=int)
    for k, (_, te) in enumerate(cv.split(df[RAW_FEATURES], y, groups)):
        fold[te] = k
    return fold, y, groups


def build_xgb() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )
    params = dict(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )
    params.update(XGB_PARAMS)
    return Pipeline([("prep", prep), ("model", XGBClassifier(**params))])


def build_catboost() -> CatBoostClassifier:
    params = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=SEED,
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
        bootstrap_type="Bayesian",
    )
    params.update(CAT_PARAMS)
    return CatBoostClassifier(**params)


def meta_features(p_xgb: np.ndarray, p_cat: np.ndarray) -> np.ndarray:
    a = np.log(np.clip(np.asarray(p_xgb, dtype=float), META_CLIP, 1.0))
    b = np.log(np.clip(np.asarray(p_cat, dtype=float), META_CLIP, 1.0))
    return np.hstack([a, b])


def build_stacker() -> LogisticRegression:
    return LogisticRegression(
        C=STACKER_C,
        max_iter=3000,
        solver="lbfgs",
        random_state=SEED,
    )


def probability_metrics(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    Y = np.eye(len(CLASSES))[y_true]
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    top_ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < (hi if hi < 1 else hi + 1e-12))
        if mask.any():
            top_ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())

    class_ece = 0.0
    for k in range(p.shape[1]):
        yk = (y_true == k).astype(float)
        ek = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p[:, k] >= lo) & (p[:, k] < (hi if hi < 1 else hi + 1e-12))
            if mask.any():
                ek += mask.mean() * abs(yk[mask].mean() - p[mask, k].mean())
        class_ece += ek
    class_ece /= p.shape[1]

    order = np.argsort(conf)
    chunks = np.array_split(order, n_bins)
    adaptive_ece = sum(
        len(c) / len(y_true) * abs(correct[c].mean() - conf[c].mean())
        for c in chunks
        if len(c)
    )

    return {
        "log_loss": float(log_loss(y_true, p, labels=np.arange(len(CLASSES)))),
        "brier": float(np.mean(np.sum((p - Y) ** 2, axis=1))),
        "mse": float(np.mean((p - Y) ** 2)),
        "auc_macro_ovr": float(roc_auc_score(y_true, p, multi_class="ovr", average="macro")),
        "auc_weighted_ovr": float(roc_auc_score(y_true, p, multi_class="ovr", average="weighted")),
        "top_label_ece_10": float(top_ece),
        "classwise_ece_10": float(class_ece),
        "adaptive_top_ece_10": float(adaptive_ece),
        "accuracy": float((pred == y_true).mean()),
    }


def fit_locked_model(df: pd.DataFrame) -> Tuple[dict, dict]:
    """Fit the locked production recipe and score the sealed outer test fold.

    The architecture/hyperparameters are fixed constants in this module.
    """
    validate_frame(df, require_target=True)
    fold, y, groups = make_outer_folds(df)
    dev_idx = np.where(fold != 0)[0]
    test_idx = np.where(fold == 0)[0]

    X_dev = df.iloc[dev_idx][RAW_FEATURES].reset_index(drop=True)
    y_dev = y[dev_idx]
    g_dev = groups[dev_idx]
    X_test = df.iloc[test_idx][RAW_FEATURES].reset_index(drop=True)
    y_test = y[test_idx]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 41)
    oof_x = np.zeros((len(dev_idx), len(CLASSES)))
    oof_c = np.zeros((len(dev_idx), len(CLASSES)))
    for tr, va in cv.split(X_dev, y_dev, g_dev):
        mx = build_xgb()
        mx.fit(X_dev.iloc[tr], y_dev[tr])
        oof_x[va] = mx.predict_proba(X_dev.iloc[va])

        mc = build_catboost()
        mc.fit(X_dev.iloc[tr], y_dev[tr], cat_features=CAT_FEATURES)
        oof_c[va] = mc.predict_proba(X_dev.iloc[va])

    stacker = build_stacker()
    stacker.fit(meta_features(oof_x, oof_c), y_dev)

    xgb_model = build_xgb()
    xgb_model.fit(X_dev, y_dev)
    cat_model = build_catboost()
    cat_model.fit(X_dev, y_dev, cat_features=CAT_FEATURES)

    p_x = xgb_model.predict_proba(X_test)
    p_c = cat_model.predict_proba(X_test)
    p = stacker.predict_proba(meta_features(p_x, p_c))

    bundle = {
        "version": "1.0.0",
        "model_type": "stacked_probability_ensemble",
        "xgb_model": xgb_model,
        "cat_model": cat_model,
        "stacker": stacker,
        "raw_features": RAW_FEATURES,
        "categorical_features": CAT_FEATURES,
        "numeric_features": NUM_FEATURES,
        "classes": CLASSES,
        "class_to_index": CLASS_TO_INDEX,
        "meta_transform": "concatenate log-clipped probabilities from [xgboost, catboost]",
        "meta_clip": META_CLIP,
        "xgb_params": XGB_PARAMS,
        "catboost_params": CAT_PARAMS,
        "stacker_params": {"C": STACKER_C, "solver": "lbfgs", "max_iter": 3000, "random_state": SEED},
        "split_seed": SEED,
        "group_rule": "new group when pitch_count <= previous pitch_count OR inning_half changes",
    }
    report = {
        "fold": fold,
        "groups": groups,
        "test_idx": test_idx,
        "test_probabilities": p,
        "test_metrics": probability_metrics(y_test, p),
        "y_test": y_test,
    }
    return bundle, report


def predict_proba(bundle: dict, raw: pd.DataFrame) -> pd.DataFrame:
    validate_frame(raw, require_target=False)
    x = raw[bundle["raw_features"]].copy()
    p_x = bundle["xgb_model"].predict_proba(x)
    p_c = bundle["cat_model"].predict_proba(x)
    z = meta_features(p_x, p_c)
    p = bundle["stacker"].predict_proba(z)
    return pd.DataFrame(p, columns=[f"p_{c}" for c in bundle["classes"]], index=raw.index)


def save_bundle(bundle: dict, path: str | Path, data_path: str | Path | None = None) -> None:
    bundle = dict(bundle)
    if data_path is not None:
        data_path = Path(data_path)
        bundle["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)
