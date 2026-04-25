from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

TARGET_COLUMN = "Global_active_power"
REQUIRED_UI_FEATURES = ["hour", "day", "lag_1", "lag_24", "rolling_mean_24"]


def find_dataset_path(explicit_path: Optional[str] = None) -> Path:
    """Locate the dataset either from an explicit path or common project locations."""
    candidates: List[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    candidates.extend(
        [
            Path("data") / "household_power_consumption.txt",
            Path("household_power_consumption.txt"),
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    looked_up = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dataset file not found. Checked: {looked_up}")


def load_preprocess_resample(dataset_path: Path) -> pd.DataFrame:
    """Load the raw text dataset, clean it, and convert minute-level rows to hourly mean values."""
    df = pd.read_csv(dataset_path, sep=";", low_memory=False)

    df = df.replace("?", np.nan)

    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["Datetime"]).copy()

    numeric_columns = [column for column in df.columns if column not in ["Date", "Time", "Datetime"]]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[numeric_columns] = df[numeric_columns].astype(float)

    # Median imputation keeps timeline continuity before resampling.
    for column in numeric_columns:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

    df = df.set_index("Datetime").sort_index()
    hourly_df = df[numeric_columns].resample("h").mean()

    return hourly_df


def add_time_and_lag_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered time, lag, and rolling features from hourly consumption."""
    if TARGET_COLUMN not in hourly_df.columns:
        raise KeyError(f"Required target column '{TARGET_COLUMN}' is missing from dataset.")

    engineered = hourly_df.copy()

    engineered["hour"] = engineered.index.hour
    engineered["day"] = engineered.index.day
    engineered["lag_1"] = engineered[TARGET_COLUMN].shift(1)
    engineered["lag_24"] = engineered[TARGET_COLUMN].shift(24)
    engineered["rolling_mean_24"] = engineered[TARGET_COLUMN].rolling(window=24).mean()

    # Lag and rolling operations create nulls at the beginning of the series.
    engineered = engineered.dropna().copy()

    return engineered


def forward_feature_selection(
    x: pd.DataFrame,
    y: pd.Series,
    max_features: int = 5,
    cv_splits: int = 5,
) -> Tuple[List[str], List[Dict[str, float]]]:
    """Greedy forward selection using Ridge + TimeSeriesSplit and MSE as objective."""
    selected: List[str] = []
    remaining = list(x.columns)
    history: List[Dict[str, float]] = []

    cv = TimeSeriesSplit(n_splits=cv_splits)

    for _ in range(min(max_features, x.shape[1])):
        best_feature = None
        best_score = float("inf")

        for candidate in remaining:
            trial_features = selected + [candidate]
            fold_mse: List[float] = []

            for train_idx, test_idx in cv.split(x):
                x_train = x.iloc[train_idx][trial_features]
                x_test = x.iloc[test_idx][trial_features]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                model = Ridge(alpha=1.0)
                model.fit(x_train, y_train)
                preds = model.predict(x_test)
                fold_mse.append(mean_squared_error(y_test, preds))

            avg_mse = float(np.mean(fold_mse))
            if avg_mse < best_score:
                best_score = avg_mse
                best_feature = candidate

        if best_feature is None:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        history.append({"step": len(selected), "feature": best_feature, "mse": best_score})

    return selected, history


def evaluate_models_time_series(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate Ridge, Lasso, PCR, and PLS with TimeSeriesSplit + MSE."""
    cv = TimeSeriesSplit(n_splits=cv_splits)

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.001, max_iter=10000, random_state=42)

    scores: Dict[str, List[float]] = {"Ridge": [], "Lasso": [], "PCR": [], "PLS": []}

    for train_idx, test_idx in cv.split(x):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        ridge_model = clone(ridge)
        ridge_model.fit(x_train, y_train)
        ridge_preds = ridge_model.predict(x_test)
        scores["Ridge"].append(mean_squared_error(y_test, ridge_preds))

        lasso_model = clone(lasso)
        lasso_model.fit(x_train, y_train)
        lasso_preds = lasso_model.predict(x_test)
        scores["Lasso"].append(mean_squared_error(y_test, lasso_preds))

        n_components = max(1, min(5, x_train.shape[1]))

        pca = PCA(n_components=n_components)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)
        pcr_reg = Ridge(alpha=1.0)
        pcr_reg.fit(x_train_pca, y_train)
        pcr_preds = pcr_reg.predict(x_test_pca)
        scores["PCR"].append(mean_squared_error(y_test, pcr_preds))

        pls = PLSRegression(n_components=n_components)
        pls.fit(x_train, y_train)
        pls_preds = pls.predict(x_test).ravel()
        scores["PLS"].append(mean_squared_error(y_test, pls_preds))

    return {model_name: float(np.mean(mse_values)) for model_name, mse_values in scores.items()}


def evaluate_models_time_series_detailed(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Tune Ridge/Lasso/PCR/PLS with TimeSeriesSplit and return best MSE + params per model."""
    cv = TimeSeriesSplit(n_splits=cv_splits)

    ridge_alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 25.0]
    lasso_alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    component_candidates = list(range(1, min(8, x.shape[1]) + 1))

    best_scores: Dict[str, float] = {"Ridge": float("inf"), "Lasso": float("inf"), "PCR": float("inf"), "PLS": float("inf")}
    best_params: Dict[str, Dict[str, Any]] = {
        "Ridge": {"alpha": 1.0},
        "Lasso": {"alpha": 0.001},
        "PCR": {"alpha": 1.0, "n_components": max(1, min(5, x.shape[1]))},
        "PLS": {"n_components": max(1, min(5, x.shape[1]))},
    }

    for alpha in ridge_alphas:
        fold_mse: List[float] = []
        for train_idx, test_idx in cv.split(x):
            x_train = x.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = Ridge(alpha=alpha)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            fold_mse.append(mean_squared_error(y_test, preds))

        avg_mse = float(np.mean(fold_mse))
        if avg_mse < best_scores["Ridge"]:
            best_scores["Ridge"] = avg_mse
            best_params["Ridge"] = {"alpha": alpha}

    for alpha in lasso_alphas:
        fold_mse = []
        for train_idx, test_idx in cv.split(x):
            x_train = x.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = Lasso(alpha=alpha, max_iter=20000, random_state=42)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            fold_mse.append(mean_squared_error(y_test, preds))

        avg_mse = float(np.mean(fold_mse))
        if avg_mse < best_scores["Lasso"]:
            best_scores["Lasso"] = avg_mse
            best_params["Lasso"] = {"alpha": alpha}

    for n_components in component_candidates:
        for alpha in ridge_alphas:
            fold_mse = []
            for train_idx, test_idx in cv.split(x):
                x_train = x.iloc[train_idx]
                x_test = x.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                pca = PCA(n_components=n_components)
                x_train_pca = pca.fit_transform(x_train)
                x_test_pca = pca.transform(x_test)
                reg = Ridge(alpha=alpha)
                reg.fit(x_train_pca, y_train)
                preds = reg.predict(x_test_pca)
                fold_mse.append(mean_squared_error(y_test, preds))

            avg_mse = float(np.mean(fold_mse))
            if avg_mse < best_scores["PCR"]:
                best_scores["PCR"] = avg_mse
                best_params["PCR"] = {"alpha": alpha, "n_components": n_components}

    for n_components in component_candidates:
        fold_mse = []
        for train_idx, test_idx in cv.split(x):
            x_train = x.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = PLSRegression(n_components=n_components)
            model.fit(x_train, y_train)
            preds = model.predict(x_test).ravel()
            fold_mse.append(mean_squared_error(y_test, preds))

        avg_mse = float(np.mean(fold_mse))
        if avg_mse < best_scores["PLS"]:
            best_scores["PLS"] = avg_mse
            best_params["PLS"] = {"n_components": n_components}

    return best_scores, best_params


def lag_features_from_recent_usage(
    recent_usage_kwh: Sequence[float],
    fallback_lag_1: float,
    fallback_lag_24: float,
    fallback_rolling_24: float,
) -> Tuple[float, float, float]:
    """Compute lag features from recent hourly usage values provided by the user."""
    cleaned = [float(value) for value in recent_usage_kwh if value is not None and not pd.isna(value)]

    if not cleaned:
        return float(fallback_lag_1), float(fallback_lag_24), float(fallback_rolling_24)

    lag_1 = float(cleaned[-1])
    lag_24 = float(cleaned[-24]) if len(cleaned) >= 24 else float(cleaned[0])
    rolling = float(np.mean(cleaned[-24:]))

    return lag_1, lag_24, rolling


def generate_synthetic_usage(hours: int = 24 * 14) -> pd.DataFrame:
    """Generate a synthetic hourly usage profile for UI demo and fallback visualizations."""
    if hours <= 0:
        return pd.DataFrame(columns=[TARGET_COLUMN])

    index = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=hours, freq="h")
    hour = index.hour.to_numpy()

    # Daily demand pattern: morning and evening peaks + low random noise.
    baseline = 0.8 + 0.4 * np.sin((hour - 7) * np.pi / 12.0) + 0.5 * np.sin((hour - 18) * np.pi / 12.0) ** 2
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(loc=0.0, scale=0.12, size=hours)
    values = np.clip(baseline + noise, a_min=0.15, a_max=None)

    return pd.DataFrame({TARGET_COLUMN: values}, index=index)


def get_current_hour_day() -> Tuple[int, int]:
    now = datetime.now()
    return int(now.hour), int(now.day)
