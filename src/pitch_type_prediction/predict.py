from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .core import load_bundle, predict_proba
from .terminal_ui import section, table, title


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pitch-type probabilities.")
    parser.add_argument("--model", default="artifacts/pitch_type_model.joblib")
    parser.add_argument("--input", required=True, help="CSV containing the nine raw predictor columns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=20, help="Maximum prediction rows to print in the terminal")
    parser.add_argument("--quiet", action="store_true", help="Write CSV without presentation table")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    bundle = load_bundle(args.model)
    probs = predict_proba(bundle, raw)

    out = raw.copy()
    out["predicted_pitch_type"] = probs.idxmax(axis=1).str.removeprefix("p_")
    out = pd.concat([out, probs], axis=1)
    out.to_csv(args.output, index=False)

    if args.quiet:
        return

    pcols = list(probs.columns)
    p = probs.to_numpy(dtype=float)
    max_error = float(np.max(np.abs(p.sum(axis=1) - 1.0))) if len(p) else 0.0

    print(title("PITCH TYPE MODEL — LIVE PREDICTIONS"))
    rows = []
    n_show = min(len(out), max(args.max_rows, 0))
    short_names = {
        "p_Changeup": "Changeup",
        "p_Four-Seam Fastball": "Four-Seam",
        "p_Sinker": "Sinker",
        "p_Slider": "Slider",
    }
    for i in range(n_show):
        row = out.iloc[i]
        rows.append(
            (
                i + 1,
                row["predicted_pitch_type"],
                f"{row['p_Changeup']:.3f}",
                f"{row['p_Four-Seam Fastball']:.3f}",
                f"{row['p_Sinker']:.3f}",
                f"{row['p_Slider']:.3f}",
                f"{sum(float(row[c]) for c in pcols):.6f}",
            )
        )

    print(section("PREDICTION PROBABILITIES"))
    print(
        table(
            ("Row", "Prediction", "Changeup", "Four-Seam", "Sinker", "Slider", "Σp"),
            rows,
            right_align={0, 2, 3, 4, 5, 6},
        )
    )
    if len(out) > n_show:
        print(f"… {len(out) - n_show} additional row(s) written to {args.output}")

    valid = (
        np.isfinite(p).all()
        and (p >= 0).all()
        and (p <= 1).all()
        and np.allclose(p.sum(axis=1), 1.0, atol=1e-12)
    )
    print(section("OUTPUT INTEGRITY"))
    print(f"{'✓' if valid else '✗'} Four probability columns are bounded in [0, 1]")
    print(f"{'✓' if valid else '✗'} Every row sums to 1 (max |Σp−1| = {max_error:.2e})")
    print(f"✓ CSV written to: {args.output}")
