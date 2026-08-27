from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .terminal_ui import section, status, table, title


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a presentation-ready integrity and leakage audit.")
    parser.add_argument("--data", default="pitch-type-prediction-data.csv")
    parser.add_argument("--metrics", default="artifacts/metrics.json")
    parser.add_argument("--manifest", default="artifacts/split_manifest.csv")
    parser.add_argument("--predictions", default="artifacts/test_predictions.csv")
    parser.add_argument("--model", default="artifacts/pitch_type_model.joblib")
    parser.add_argument("--source", default="src/pitch_type_prediction/core.py")
    parser.add_argument("--research-source", default="src/pitch_type_prediction/research.py")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    manifest = pd.read_csv(args.manifest)
    pred = pd.read_csv(args.predictions)
    bundle = joblib.load(args.model)
    data_path = Path(args.data)

    checks: list[tuple[str, str, str]] = []

    # Dataset identity.
    if data_path.exists():
        actual_sha = _hash(data_path)
        expected_sha = metrics["data"]["sha256"]
        checks.append(("Dataset SHA-256 matches locked data", _state(actual_sha == expected_sha), actual_sha[:12] + "…"))
        df = pd.read_csv(data_path)
    else:
        df = None
        checks.append(("Dataset available for deep audit", "WARN", f"missing: {data_path}"))

    # Feature contract / target leakage.
    raw_features = list(bundle.get("raw_features", []))
    checks.append(("Exactly nine raw predictors in bundle", _state(len(raw_features) == 9), str(len(raw_features))))
    checks.append(("Current target excluded from raw features", _state("pitch_type" not in raw_features), "pitch_type not in feature list"))

    # Model architecture.
    required_components = {"xgb_model", "cat_model", "stacker"}
    checks.append(("Serialized stack components present", _state(required_components.issubset(bundle)), "XGB + CatBoost + stacker"))
    checks.append(("Bundle identifies stacked probability ensemble", _state(bundle.get("model_type") == "stacked_probability_ensemble"), str(bundle.get("model_type"))))

    # Manifest integrity / group overlap.
    sets = {
        split: set(manifest.loc[manifest["split"] == split, "inferred_group"])
        for split in ("train", "validation", "test")
    }
    checks.append(("Train / validation groups disjoint", _state(sets["train"].isdisjoint(sets["validation"])), f"{len(sets['train'] & sets['validation'])} overlaps"))
    checks.append(("Train / test groups disjoint", _state(sets["train"].isdisjoint(sets["test"])), f"{len(sets['train'] & sets['test'])} overlaps"))
    checks.append(("Validation / test groups disjoint", _state(sets["validation"].isdisjoint(sets["test"])), f"{len(sets['validation'] & sets['test'])} overlaps"))

    # Prediction integrity.
    pcols = [c for c in pred.columns if c.startswith("p_")]
    p = pred[pcols].to_numpy(dtype=float)
    prob_ok = (
        len(pcols) == 4
        and np.isfinite(p).all()
        and (p >= 0).all()
        and (p <= 1).all()
        and np.allclose(p.sum(axis=1), 1.0, atol=1e-12)
    )
    checks.append(("Saved test probabilities valid", _state(prob_ok), f"max |Σp−1|={np.max(np.abs(p.sum(axis=1)-1)):.2e}"))

    # Test rows correspond exactly to manifest test rows.
    test_rows_manifest = set(manifest.loc[manifest["split"] == "test", "source_row"].astype(int))
    test_rows_pred = set(pred["source_row"].astype(int))
    checks.append(("Prediction rows equal sealed test rows", _state(test_rows_manifest == test_rows_pred), f"{len(test_rows_pred)} rows"))

    # Recompute headline metrics from predictions.
    classes = list(bundle.get("classes", ["Changeup", "Four-Seam Fastball", "Sinker", "Slider"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = pred["actual_pitch_type"].map(class_to_idx).to_numpy()
    Y = np.eye(len(classes))[y]
    recomputed = {
        "log_loss": float(log_loss(y, p, labels=np.arange(len(classes)))),
        "brier": float(np.mean(np.sum((p - Y) ** 2, axis=1))),
        "auc_macro_ovr": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
    }
    locked = metrics["test_locked_champion"]
    metric_ok = all(abs(recomputed[k] - locked[k]) < 1e-10 for k in recomputed)
    checks.append(("Saved predictions reproduce locked metrics", _state(metric_ok), f"log loss={recomputed['log_loss']:.6f}"))

    # Deep data checks if raw data is available.
    if df is not None:
        checks.append(("Manifest row count matches raw data", _state(len(df) == len(manifest)), f"{len(df):,} rows"))

        # Recompute inferred groups.
        non_increase = df["pitch_count"].diff().fillna(1) <= 0
        half_change = df["inning_half"].astype(str) != df["inning_half"].astype(str).shift(1)
        starts = (non_increase | half_change).to_numpy(dtype=bool)
        starts[0] = True
        groups = np.cumsum(starts).astype(int)
        group_match = np.array_equal(groups, manifest["inferred_group"].to_numpy(dtype=int))
        checks.append(("Inferred groups reproduce manifest", _state(group_match), f"{len(np.unique(groups))} groups"))

        # Recompute outer folds.
        y_all = df["pitch_type"].map(class_to_idx).to_numpy()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260825)
        fold = np.empty(len(df), dtype=int)
        for k, (_, te) in enumerate(cv.split(df[raw_features], y_all, groups)):
            fold[te] = k
        fold_match = np.array_equal(fold, manifest["outer_fold"].to_numpy(dtype=int))
        checks.append(("Outer grouped folds reproduce manifest", _state(fold_match), "seed=20260825"))

        # Previous-pitch legality / lag consistency.
        first = df["ab_pitch_count"].to_numpy() == 1
        first_ok = (df.loc[first, "previous_pitch_type"].astype(str).str.lower() == "na").all()
        checks.append(("First PA pitches have previous_pitch_type='na'", _state(first_ok), f"{first.sum():,}/{first.sum():,}"))

        ab = df["ab_pitch_count"].to_numpy()
        continuation = (ab[1:] == ab[:-1] + 1) & (groups[1:] == groups[:-1])
        idx = np.where(continuation)[0] + 1
        lag_ok = np.array_equal(
            df.iloc[idx]["previous_pitch_type"].to_numpy(),
            df.iloc[idx - 1]["pitch_type"].to_numpy(),
        )
        checks.append(("Observed PA continuations use true prior pitch as lag", _state(lag_ok), f"{len(idx):,}/{len(idx):,}"))

    # Static implementation checks: precise, visible, and intentionally narrow.
    core_path = Path(args.source)
    if core_path.exists():
        source = core_path.read_text()
        oof_tokens = [
            "oof_x[va] = mx.predict_proba",
            "oof_c[va] = mc.predict_proba",
            "stacker.fit(meta_features(oof_x, oof_c), y_dev)",
        ]
        oof_ok = all(token in source for token in oof_tokens)
        checks.append(("Source implements OOF meta-feature training", _state(oof_ok), "base predictions assigned only to held-out fold"))
    else:
        checks.append(("Source available for OOF implementation audit", "WARN", str(core_path)))

    research_path = Path(args.research_source)
    if research_path.exists():
        research = research_path.read_text()
        train_only_ok = (
            "np.isin(fold, [2, 3, 4])" in research
            or "np.isin(fold,[2,3,4])" in research.replace(" ", "")
        )
        checks.append(("Research source restricts tuning to outer training folds", _state(train_only_ok), "folds 2/3/4"))
    else:
        checks.append(("Research source available for train-only audit", "WARN", str(research_path)))

    print(title("PITCH TYPE MODEL — INTEGRITY & LEAKAGE AUDIT"))
    print(section("AUDIT CHECKS"))
    for label, state_name, detail in checks:
        print(status(label, state_name, detail))

    fail_count = sum(state_name == "FAIL" for _, state_name, _ in checks)
    warn_count = sum(state_name == "WARN" for _, state_name, _ in checks)

    print(section("LOCKED TEST METRICS — RECOMPUTED FROM SAVED PREDICTIONS"))
    rows = [
        ("Log loss ↓", f"{recomputed['log_loss']:.6f}", f"{locked['log_loss']:.6f}"),
        ("Brier ↓", f"{recomputed['brier']:.6f}", f"{locked['brier']:.6f}"),
        ("Macro OVR AUC ↑", f"{recomputed['auc_macro_ovr']:.6f}", f"{locked['auc_macro_ovr']:.6f}"),
    ]
    print(table(("Metric", "Recomputed", "Locked artifact"), rows, right_align={1, 2}))

    print(section("OVERALL STATUS"))
    if fail_count:
        print(status("Integrity audit", "FAIL", f"{fail_count} failed check(s), {warn_count} warning(s)"))
        raise SystemExit(1)
    if warn_count:
        print(status("Integrity audit", "WARN", f"0 failures, {warn_count} warning(s)"))
    else:
        print(status("Integrity audit", "PASS", "all checks passed"))


if __name__ == "__main__":
    main()
