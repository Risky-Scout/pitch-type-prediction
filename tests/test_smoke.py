from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitch_type_prediction.core import (
    CLASSES,
    RAW_FEATURES,
    infer_groups,
    load_bundle,
    predict_proba,
)

ROOT = Path(__file__).resolve().parents[1]


def sample_row():
    return pd.DataFrame([{
        "inning_number": 5,
        "inning_half": "B",
        "balls": 1,
        "strikes": 1,
        "outs": 1,
        "hitter_hand": "R",
        "ab_pitch_count": 3,
        "pitch_count": 75,
        "previous_pitch_type": "Slider",
    }])


def test_serialized_model_predicts_valid_probabilities():
    bundle = load_bundle(ROOT / "artifacts" / "pitch_type_model.joblib")
    p = predict_proba(bundle, sample_row())
    assert p.shape == (1, 4)
    assert list(p.columns) == [f"p_{c}" for c in CLASSES]
    assert np.isfinite(p.to_numpy()).all()
    assert (p.to_numpy() >= 0).all()
    assert np.allclose(p.sum(axis=1).to_numpy(), 1.0, atol=1e-12)


def test_missing_columns_rejected():
    bundle = load_bundle(ROOT / "artifacts" / "pitch_type_model.joblib")
    with pytest.raises(ValueError):
        predict_proba(bundle, pd.DataFrame([{"inning_number": 5}]))


def test_group_rule_splits_reset_and_half_change():
    x = pd.DataFrame([
        {"inning_number":1,"inning_half":"T","balls":0,"strikes":0,"outs":0,"hitter_hand":"R","ab_pitch_count":1,"pitch_count":1,"previous_pitch_type":"na"},
        {"inning_number":1,"inning_half":"T","balls":1,"strikes":0,"outs":0,"hitter_hand":"R","ab_pitch_count":2,"pitch_count":2,"previous_pitch_type":"Slider"},
        {"inning_number":2,"inning_half":"B","balls":0,"strikes":0,"outs":0,"hitter_hand":"L","ab_pitch_count":1,"pitch_count":3,"previous_pitch_type":"na"},
        {"inning_number":1,"inning_half":"B","balls":0,"strikes":0,"outs":0,"hitter_hand":"R","ab_pitch_count":1,"pitch_count":1,"previous_pitch_type":"na"},
    ])
    g = infer_groups(x)
    assert g.tolist() == [1, 1, 2, 3]
