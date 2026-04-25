from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from utils import (
    TARGET_COLUMN,
    add_time_and_lag_features,
    evaluate_models_time_series_detailed,
    find_dataset_path,
    forward_feature_selection,
    load_preprocess_resample,
)


def fit_single_model(
    model_name: str,
    x_selected: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit one model variant on all rows using selected, scaled features."""
    if model_name == "Ridge":
        model = Ridge(alpha=float(params.get("alpha", 1.0)))
        model.fit(x_selected, y)
        return {"model_type": "Ridge", "model": model, "params": params}

    if model_name == "Lasso":
        model = Lasso(alpha=float(params.get("alpha", 0.001)), max_iter=20000, random_state=42)
        model.fit(x_selected, y)
        return {"model_type": "Lasso", "model": model, "params": params}

    if model_name == "PCR":
        n_components = int(params.get("n_components", max(1, min(5, x_selected.shape[1]))))
        alpha = float(params.get("alpha", 1.0))
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x_selected)
        reg = Ridge(alpha=alpha)
        reg.fit(x_pca, y)
        return {"model_type": "PCR", "pca": pca, "model": reg, "params": params}

    if model_name == "PLS":
        n_components = int(params.get("n_components", max(1, min(5, x_selected.shape[1]))))
        model = PLSRegression(n_components=n_components)
        model.fit(x_selected, y)
        return {"model_type": "PLS", "model": model, "params": params}

    raise ValueError(f"Unsupported model type: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train electricity consumption forecasting models.")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Optional path to the household power consumption txt file.",
    )
    args = parser.parse_args()

    dataset_path = find_dataset_path(args.data if args.data else None)

    print(f"Using dataset: {dataset_path}")
    hourly_df = load_preprocess_resample(dataset_path)
    model_df = add_time_and_lag_features(hourly_df)

    if TARGET_COLUMN not in model_df.columns:
        raise KeyError(f"Target column {TARGET_COLUMN} is missing after preprocessing.")

    x = model_df.drop(columns=[TARGET_COLUMN]).copy()
    y = model_df[TARGET_COLUMN].copy()

    scaler = StandardScaler()
    x_scaled_array = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled_array, columns=x.columns, index=x.index)

    selected_features, selection_history = forward_feature_selection(
        x=x_scaled,
        y=y,
        max_features=5,
        cv_splits=5,
    )

    if not selected_features:
        raise RuntimeError("Forward feature selection returned no features.")

    x_selected = x_scaled[selected_features].copy()

    model_scores, model_best_params = evaluate_models_time_series_detailed(x_selected, y, cv_splits=5)
    best_model_name = min(model_scores, key=model_scores.get)

    trained_models: Dict[str, Dict[str, Any]] = {}
    for model_name in ["Ridge", "Lasso", "PCR", "PLS"]:
        trained_models[model_name] = fit_single_model(
            model_name=model_name,
            x_selected=x_selected,
            y=y,
            params=model_best_params.get(model_name, {}),
        )

    fitted_bundle = trained_models[best_model_name]

    model_bundle: Dict[str, Any] = {
        "target_column": TARGET_COLUMN,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_features": selected_features,
        "selection_history": selection_history,
        "all_feature_columns": list(x.columns),
        "feature_defaults": x.mean().to_dict(),
        "training_rows": int(x.shape[0]),
        "model_scores_mse": model_scores,
        "model_best_params": model_best_params,
        "best_model_name": best_model_name,
        "trained_models": trained_models,
        "trained_artifact": fitted_bundle,
    }

    scaler_bundle: Dict[str, Any] = {
        "scaler": scaler,
        "feature_order": list(x.columns),
    }

    with open("model.pkl", "wb") as model_file:
        pickle.dump(model_bundle, model_file)

    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler_bundle, scaler_file)

    usage_chart = model_df[[TARGET_COLUMN]].copy()
    usage_chart.to_csv("hourly_usage.csv")

    score_df = pd.DataFrame(
        [{"model": model_name, "mse": mse_value} for model_name, mse_value in model_scores.items()]
    ).sort_values("mse", ascending=True)
    score_df.to_csv("model_comparison.csv", index=False)

    print("Training completed successfully.")
    print(f"Selected features: {selected_features}")
    print("Model comparison (MSE):")
    for model_name, mse_value in sorted(model_scores.items(), key=lambda item: item[1]):
        print(f"  {model_name}: {mse_value:.6f}")
    print(f"Best model: {best_model_name}")
    print("Saved artifacts: model.pkl, scaler.pkl, hourly_usage.csv, model_comparison.csv")


if __name__ == "__main__":
    main()
