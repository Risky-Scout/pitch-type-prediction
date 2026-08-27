from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (
    CLASSES,
    TARGET,
    fit_locked_model,
    probability_metrics,
    save_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the locked pitch-type probability model.")
    parser.add_argument("--data", required=True, help="Path to pitch-type-prediction-data.csv")
    parser.add_argument("--output-dir", default="artifacts/reproduced")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)

    bundle, report = fit_locked_model(df)
    save_bundle(bundle, out / "pitch_type_model.joblib", data_path=args.data)

    test_idx = report["test_idx"]
    p = report["test_probabilities"]
    y = report["y_test"]

    pred = pd.DataFrame(
        {
            "source_row": test_idx,
            "actual_pitch_type": df.iloc[test_idx][TARGET].to_numpy(),
            "predicted_pitch_type": np.array(CLASSES)[p.argmax(axis=1)],
        }
    )
    for i, c in enumerate(CLASSES):
        pred[f"p_{c}"] = p[:, i]
    pred.to_csv(out / "test_predictions.csv", index=False)

    (out / "test_metrics.json").write_text(json.dumps(report["test_metrics"], indent=2))
    print(json.dumps(report["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
