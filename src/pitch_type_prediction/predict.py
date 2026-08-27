from __future__ import annotations

import argparse
import pandas as pd

from .core import load_bundle, predict_proba


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calibrated pitch-type probabilities.")
    parser.add_argument("--model", default="artifacts/pitch_type_model.joblib")
    parser.add_argument("--input", required=True, help="CSV containing the nine raw predictor columns")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    bundle = load_bundle(args.model)
    probs = predict_proba(bundle, raw)

    out = raw.copy()
    out["predicted_pitch_type"] = probs.idxmax(axis=1).str.removeprefix("p_")
    out = pd.concat([out, probs], axis=1)
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
