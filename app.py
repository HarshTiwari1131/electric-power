from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    TARGET_COLUMN,
    add_time_and_lag_features,
    generate_synthetic_usage,
    get_current_hour_day,
    lag_features_from_recent_usage,
    load_preprocess_resample,
)

st.set_page_config(page_title="Electricity Consumption Forecast", layout="wide")
px.defaults.template = "plotly_white"

APPLIANCE_PRESETS = pd.DataFrame(
    [
        {"appliance": "Air Conditioner", "power_w": 1500.0, "quantity": 1, "hours_per_day": 6.0, "start_hour": 18},
        {"appliance": "Refrigerator", "power_w": 150.0, "quantity": 1, "hours_per_day": 24.0, "start_hour": 0},
        {"appliance": "Washing Machine", "power_w": 500.0, "quantity": 1, "hours_per_day": 1.0, "start_hour": 10},
        {"appliance": "Television", "power_w": 120.0, "quantity": 1, "hours_per_day": 4.0, "start_hour": 19},
        {"appliance": "Fan", "power_w": 75.0, "quantity": 3, "hours_per_day": 8.0, "start_hour": 20},
        {"appliance": "Lights", "power_w": 15.0, "quantity": 10, "hours_per_day": 6.0, "start_hour": 18},
    ]
)

APPLIANCE_LIBRARY: Dict[str, Dict[str, float]] = {
    "Air Conditioner": {"power_w": 1500.0, "hours_per_day": 6.0, "start_hour": 18.0},
    "Refrigerator": {"power_w": 150.0, "hours_per_day": 24.0, "start_hour": 0.0},
    "Television": {"power_w": 120.0, "hours_per_day": 4.0, "start_hour": 19.0},
    "Fan": {"power_w": 75.0, "hours_per_day": 8.0, "start_hour": 20.0},
    "Water Heater": {"power_w": 2000.0, "hours_per_day": 1.5, "start_hour": 6.0},
    "Microwave": {"power_w": 1200.0, "hours_per_day": 0.5, "start_hour": 13.0},
    "Laptop": {"power_w": 65.0, "hours_per_day": 8.0, "start_hour": 9.0},
    "Lights": {"power_w": 15.0, "hours_per_day": 6.0, "start_hour": 18.0},
}


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at 15% -10%, #dbeafe 0%, #f6f7fb 35%, #edf7ef 100%);
            }
            [data-testid="stMainBlockContainer"],
            [data-testid="stMainBlockContainer"] * {
                color: #0f172a !important;
            }
            [data-testid="stMainBlockContainer"] p,
            [data-testid="stMainBlockContainer"] label,
            [data-testid="stMainBlockContainer"] h1,
            [data-testid="stMainBlockContainer"] h2,
            [data-testid="stMainBlockContainer"] h3,
            [data-testid="stMainBlockContainer"] h4 {
                opacity: 1 !important;
            }
            [data-testid="stMainBlockContainer"] .stButton > button {
                background: #0ea5e9 !important;
                color: #ffffff !important;
                border: 1px solid #0284c7 !important;
                font-weight: 700 !important;
            }
            [data-testid="stMainBlockContainer"] .stButton > button:hover {
                background: #0284c7 !important;
                color: #ffffff !important;
            }
            [data-testid="stMainBlockContainer"] .stButton > button:disabled {
                background: #94a3b8 !important;
                color: #e2e8f0 !important;
                border-color: #94a3b8 !important;
                opacity: 1 !important;
            }
            [data-testid="stMainBlockContainer"] .stNumberInput input,
            [data-testid="stMainBlockContainer"] .stTextInput input,
            [data-testid="stMainBlockContainer"] .stTextArea textarea,
            [data-testid="stMainBlockContainer"] [data-baseweb="select"] > div {
                background: #ffffff !important;
                color: #0f172a !important;
                border-color: #cbd5e1 !important;
            }
            [data-testid="stMainBlockContainer"] .stRadio label,
            [data-testid="stMainBlockContainer"] .stCheckbox label,
            [data-testid="stMainBlockContainer"] .stToggle label {
                color: #0f172a !important;
                opacity: 1 !important;
            }
            [data-testid="stWidgetLabel"] p {
                color: #0f172a !important;
                opacity: 1 !important;
                font-weight: 600 !important;
            }
            [data-testid="stSidebar"] * {
                color: #f8fafc !important;
            }
            [data-testid="stSidebar"] .stNumberInput input,
            [data-testid="stSidebar"] .stTextInput input,
            [data-testid="stSidebar"] .stSelectbox div,
            [data-testid="stSidebar"] .stSlider div {
                color: #f8fafc !important;
            }
            [data-baseweb="tab"] {
                color: #0f172a !important;
                font-weight: 700 !important;
                opacity: 1 !important;
            }
            [data-baseweb="tab"][aria-selected="true"] {
                color: #0369a1 !important;
            }
            [data-baseweb="tab-panel"] {
                color: #0f172a !important;
                opacity: 1 !important;
            }
            .stMetric label,
            .stMetric [data-testid="stMetricValue"] {
                color: #0f172a !important;
            }
            [data-testid="stDataFrame"] * {
                color: #0f172a !important;
            }
            .hero {
                background: linear-gradient(120deg, #0f766e 0%, #0c4a6e 45%, #1d4ed8 100%);
                border-radius: 16px;
                color: white;
                padding: 20px 24px;
                margin-bottom: 18px;
                box-shadow: 0 10px 30px rgba(16, 24, 40, 0.18);
            }
            .hero h1 {
                margin: 0;
                font-size: 1.9rem;
                letter-spacing: 0.3px;
            }
            .hero p {
                margin: 8px 0 0 0;
                opacity: 0.93;
            }
            .metric-card {
                background: rgba(255, 255, 255, 0.85);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 14px;
                padding: 14px;
                box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts() -> Dict[str, Any]:
    model_path = Path("model.pkl")
    scaler_path = Path("scaler.pkl")

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run python train.py first to generate model.pkl and scaler.pkl."
        )

    with model_path.open("rb") as model_file:
        model_bundle = pickle.load(model_file)

    with scaler_path.open("rb") as scaler_file:
        scaler_bundle = pickle.load(scaler_file)

    return {"model_bundle": model_bundle, "scaler_bundle": scaler_bundle}


@st.cache_data
def load_uploaded_chart_data(uploaded_file: Any) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    hourly_df = load_preprocess_resample(temp_path)
    enhanced_df = add_time_and_lag_features(hourly_df)
    return enhanced_df[[TARGET_COLUMN]].copy()


def get_available_models(model_bundle: Dict[str, Any]) -> List[str]:
    trained_models = model_bundle.get("trained_models", {})
    if isinstance(trained_models, dict) and trained_models:
        names = [name for name in ["Ridge", "Lasso", "PCR", "PLS"] if name in trained_models]
        if names:
            return names
    return [model_bundle.get("best_model_name", "Ridge")]


def load_usage_chart_data() -> pd.DataFrame:
    csv_path = Path("hourly_usage.csv")
    if csv_path.exists():
        chart_df = pd.read_csv(csv_path)
        if "Datetime" in chart_df.columns:
            chart_df["Datetime"] = pd.to_datetime(chart_df["Datetime"], errors="coerce")
            chart_df = chart_df.dropna(subset=["Datetime"]).set_index("Datetime")
        return chart_df

    return generate_synthetic_usage(hours=24 * 21)


def parse_recent_usage_input(raw_text: str) -> List[float]:
    values: List[float] = []
    for token in raw_text.replace("\n", ",").split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        values.append(float(cleaned))
    return values


def build_full_feature_vector(
    defaults: Dict[str, float],
    hour: int,
    day: int,
    lag_1: float,
    lag_24: float,
    rolling_mean_24: float,
) -> Dict[str, float]:
    features = {name: float(value) for name, value in defaults.items()}
    features["hour"] = float(hour)
    features["day"] = float(day)
    features["lag_1"] = float(lag_1)
    features["lag_24"] = float(lag_24)
    features["rolling_mean_24"] = float(rolling_mean_24)
    return features


def predict_next_hour(
    model_bundle: Dict[str, Any],
    scaler_bundle: Dict[str, Any],
    feature_input: Dict[str, float],
    chosen_model_name: str,
) -> float:
    scaler = scaler_bundle["scaler"]
    feature_order = scaler_bundle["feature_order"]

    input_df = pd.DataFrame([feature_input], columns=feature_order)
    scaled_input = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_input, columns=feature_order)

    selected_features = model_bundle["selected_features"]
    x_selected = scaled_df[selected_features]

    artifact = model_bundle.get("trained_models", {}).get(chosen_model_name, model_bundle["trained_artifact"])
    model_type = artifact["model_type"]

    if model_type == "PCR":
        pca = artifact["pca"]
        reg = artifact["model"]
        pred = reg.predict(pca.transform(x_selected))
        return float(pred[0])

    model = artifact["model"]
    pred = model.predict(x_selected)
    if hasattr(pred, "ravel"):
        pred = pred.ravel()
    return float(pred[0])


def forecast_next_24_hours(
    model_bundle: Dict[str, Any],
    scaler_bundle: Dict[str, Any],
    base_input: Dict[str, float],
    chosen_model_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    hour = int(base_input["hour"])
    day = int(base_input["day"])
    last_values = [float(base_input["lag_24"]) for _ in range(23)] + [float(base_input["lag_1"])]

    for step in range(1, 25):
        features = dict(base_input)
        features["hour"] = float(hour)
        features["day"] = float(day)
        features["lag_1"] = float(last_values[-1])
        features["lag_24"] = float(last_values[0])
        features["rolling_mean_24"] = float(np.mean(last_values))

        pred = predict_next_hour(
            model_bundle=model_bundle,
            scaler_bundle=scaler_bundle,
            feature_input=features,
            chosen_model_name=chosen_model_name,
        )

        rows.append({"step": step, "hour": hour, "day": day, TARGET_COLUMN: pred})

        last_values = last_values[1:] + [pred]
        hour += 1
        if hour > 23:
            hour = 0
            day = 1 if day >= 31 else day + 1

    return pd.DataFrame(rows)


def appliance_hourly_profile(appliance_df: pd.DataFrame) -> np.ndarray:
    profile = np.zeros(24, dtype=float)

    for _, row in appliance_df.iterrows():
        try:
            power_w = max(0.0, float(row.get("power_w", 0.0)))
            qty = max(0.0, float(row.get("quantity", 0.0)))
            hours_per_day = max(0.0, float(row.get("hours_per_day", 0.0)))
            start_hour = int(float(row.get("start_hour", 0.0))) % 24
        except Exception:
            continue

        if power_w == 0.0 or qty == 0.0 or hours_per_day == 0.0:
            continue

        kw_total = (power_w * qty) / 1000.0
        full_hours = int(hours_per_day)
        frac_hour = hours_per_day - full_hours

        for h in range(full_hours):
            profile[(start_hour + h) % 24] += kw_total

        if frac_hour > 0:
            profile[(start_hour + full_hours) % 24] += kw_total * frac_hour

    return profile


def usage_metrics_from_profile(profile: Sequence[float]) -> Tuple[float, float]:
    daily_kwh = float(np.sum(profile))
    monthly_kwh = daily_kwh * 30.0
    return daily_kwh, monthly_kwh


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
        title_font=dict(color="#0f172a"),
        legend=dict(font=dict(color="#0f172a")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
    return fig


def render_appliance_quick_add(key_prefix: str) -> None:
    st.markdown("### Quick Add Appliance")
    appliance_options = list(APPLIANCE_LIBRARY.keys()) + ["Other (Custom)"]

    selected_name = st.selectbox(
        "Appliance type",
        options=appliance_options,
        key=f"{key_prefix}_appliance_type",
    )

    preset = APPLIANCE_LIBRARY.get(selected_name, {"power_w": 100.0, "hours_per_day": 1.0, "start_hour": 18.0})
    custom_name = selected_name

    if selected_name == "Other (Custom)":
        custom_name = st.text_input("Custom appliance name", value="Custom Appliance", key=f"{key_prefix}_custom")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        power_w = st.number_input("Power (W)", min_value=1.0, value=float(preset["power_w"]), key=f"{key_prefix}_power")
    with q2:
        quantity = st.number_input("Quantity", min_value=1, value=1, key=f"{key_prefix}_qty")
    with q3:
        hours_per_day = st.number_input(
            "Hours/day",
            min_value=0.1,
            max_value=24.0,
            value=float(preset["hours_per_day"]),
            step=0.1,
            key=f"{key_prefix}_hours",
        )
    with q4:
        start_hour = st.number_input(
            "Start hour",
            min_value=0,
            max_value=23,
            value=int(preset["start_hour"]),
            step=1,
            key=f"{key_prefix}_start",
        )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Add appliance", key=f"{key_prefix}_add_btn"):
            name = str(custom_name).strip() or "Custom Appliance"
            new_row = pd.DataFrame(
                [
                    {
                        "appliance": name,
                        "power_w": float(power_w),
                        "quantity": int(quantity),
                        "hours_per_day": float(hours_per_day),
                        "start_hour": int(start_hour),
                    }
                ]
            )
            st.session_state["appliance_table"] = pd.concat(
                [st.session_state["appliance_table"], new_row],
                ignore_index=True,
            )
            st.success(f"Added {name} to appliance list.")
    with c2:
        if st.button("Reset to defaults", key=f"{key_prefix}_reset_btn"):
            st.session_state["appliance_table"] = APPLIANCE_PRESETS.copy()
            st.info("Appliance list reset to default presets.")


apply_custom_style()

st.markdown(
    """
    <div class='hero'>
        <h1>Electricity Consumption Forecast</h1>
        <p>Usage-aware predictions, 24-hour forecasting, interactive charts, and appliance-level bill insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info("Run python train.py, then refresh this app.")
    st.stop()

model_bundle = artifacts["model_bundle"]
scaler_bundle = artifacts["scaler_bundle"]
feature_defaults = model_bundle.get("feature_defaults", {})
available_models = get_available_models(model_bundle)

if "forecast_df" not in st.session_state:
    st.session_state["forecast_df"] = pd.DataFrame()
if "appliance_table" not in st.session_state:
    st.session_state["appliance_table"] = APPLIANCE_PRESETS.copy()

current_hour, current_day = get_current_hour_day()

st.sidebar.header("Control Panel")
model_name = st.sidebar.selectbox("Model", options=available_models)
auto_time = st.sidebar.toggle("Auto-fill current hour/day", value=True)

hour = st.sidebar.slider("Hour", 0, 23, current_hour if auto_time else 12)
day = st.sidebar.slider("Day", 1, 31, current_day if auto_time else 1)

pricing_col1, pricing_col2 = st.sidebar.columns(2)
with pricing_col1:
    tariff_per_kwh = st.number_input("Tariff / kWh", min_value=0.0, value=8.0, step=0.1)
with pricing_col2:
    fixed_monthly_charge = st.number_input("Fixed Charge", min_value=0.0, value=150.0, step=10.0)

tax_percent = st.sidebar.slider("Tax (%)", 0, 30, 5)

uploaded_dataset = st.sidebar.file_uploader("Upload custom .txt dataset", type=["txt"])

prediction_tab, bill_tab, viz_tab, details_tab = st.tabs(
    ["Prediction Studio", "Usage & Bill", "Visual Analytics", "Model Details"]
)

with prediction_tab:
    st.subheader("Prediction Inputs")
    mode = st.radio(
        "Input Mode",
        options=["Manual features", "Recent usage history", "Appliance-driven"],
        horizontal=True,
    )

    lag_1_default = float(feature_defaults.get("lag_1", 1.0))
    lag_24_default = float(feature_defaults.get("lag_24", lag_1_default))
    rolling_default = float(feature_defaults.get("rolling_mean_24", lag_1_default))

    lag_1 = lag_1_default
    lag_24 = lag_24_default
    rolling_mean_24 = rolling_default

    if mode == "Manual features":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            lag_1 = st.number_input("lag_1", value=lag_1_default)
        with col_b:
            lag_24 = st.number_input("lag_24", value=lag_24_default)
        with col_c:
            rolling_mean_24 = st.number_input("rolling_mean_24", value=rolling_default)

    if mode == "Recent usage history":
        usage_text = st.text_area(
            "Enter recent hourly usage values (kWh), comma-separated",
            value="0.81,0.79,0.76,0.72,0.74,0.88,1.10,1.26,1.32,1.45,1.38,1.24",
            height=120,
        )
        try:
            recent_values = parse_recent_usage_input(usage_text)
            lag_1, lag_24, rolling_mean_24 = lag_features_from_recent_usage(
                recent_usage_kwh=recent_values,
                fallback_lag_1=lag_1_default,
                fallback_lag_24=lag_24_default,
                fallback_rolling_24=rolling_default,
            )
            st.caption(f"Loaded {len(recent_values)} usage points to compute lag features.")
        except Exception as exc:
            st.warning(f"Could not parse usage values. Using defaults. Details: {exc}")

    if mode == "Appliance-driven":
        st.caption("Enter appliance details. The app will derive lag values from estimated hourly household usage.")
        render_appliance_quick_add("pred")
        appliance_df = st.data_editor(
            st.session_state["appliance_table"],
            num_rows="dynamic",
            use_container_width=True,
            key="appliance_editor_prediction",
        )
        st.session_state["appliance_table"] = appliance_df.copy()

        profile = appliance_hourly_profile(appliance_df)
        lag_1 = float(profile[(hour - 1) % 24])
        lag_24 = float(profile[hour % 24])
        rolling_mean_24 = float(np.mean(profile))

        profile_df = pd.DataFrame({"hour": list(range(24)), "estimated_kw": profile})
        fig_profile = px.area(
            profile_df,
            x="hour",
            y="estimated_kw",
            title="Estimated Hourly Load from Appliances",
            markers=True,
        )
        st.plotly_chart(style_figure(fig_profile), use_container_width=True)

    feature_input = build_full_feature_vector(
        defaults=feature_defaults,
        hour=hour,
        day=day,
        lag_1=lag_1,
        lag_24=lag_24,
        rolling_mean_24=rolling_mean_24,
    )

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("lag_1", f"{lag_1:.3f}")
    metric_col2.metric("lag_24", f"{lag_24:.3f}")
    metric_col3.metric("rolling_mean_24", f"{rolling_mean_24:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

    button_col1, button_col2 = st.columns([1, 1])
    with button_col1:
        if st.button("Predict Next Hour", type="primary"):
            prediction = predict_next_hour(
                model_bundle=model_bundle,
                scaler_bundle=scaler_bundle,
                feature_input=feature_input,
                chosen_model_name=model_name,
            )
            next_hour_cost = prediction * tariff_per_kwh
            st.success(f"Predicted next hour {TARGET_COLUMN}: {prediction:.4f} kW")
            st.info(f"Estimated cost for next hour: {next_hour_cost:.2f}")

    with button_col2:
        if st.button("Forecast Next 24 Hours"):
            forecast_df = forecast_next_24_hours(
                model_bundle=model_bundle,
                scaler_bundle=scaler_bundle,
                base_input=feature_input,
                chosen_model_name=model_name,
            )
            st.session_state["forecast_df"] = forecast_df
            st.success("24-hour forecast generated.")

    if not st.session_state["forecast_df"].empty:
        forecast_df = st.session_state["forecast_df"].copy()
        forecast_df["cost"] = forecast_df[TARGET_COLUMN] * tariff_per_kwh
        st.dataframe(forecast_df, use_container_width=True)

        fig_forecast = px.line(
            forecast_df,
            x="step",
            y=TARGET_COLUMN,
            markers=True,
            title="Next 24 Hours Forecast",
        )
        st.plotly_chart(style_figure(fig_forecast), use_container_width=True)

with bill_tab:
    st.subheader("Appliance Usage and Bill Summary")
    render_appliance_quick_add("bill")

    appliance_df = st.data_editor(
        st.session_state["appliance_table"],
        num_rows="dynamic",
        use_container_width=True,
        key="appliance_editor_bill",
    )
    st.session_state["appliance_table"] = appliance_df.copy()

    profile = appliance_hourly_profile(appliance_df)
    daily_kwh, monthly_kwh = usage_metrics_from_profile(profile)

    energy_cost = monthly_kwh * tariff_per_kwh
    subtotal = energy_cost + fixed_monthly_charge
    tax_value = subtotal * (tax_percent / 100.0)
    total_bill = subtotal + tax_value

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Daily Usage (kWh)", f"{daily_kwh:.2f}")
    b2.metric("Monthly Usage (kWh)", f"{monthly_kwh:.2f}")
    b3.metric("Energy Cost", f"{energy_cost:.2f}")
    b4.metric("Estimated Bill", f"{total_bill:.2f}")

    bill_breakdown = pd.DataFrame(
        [
            {"item": "Energy Cost", "amount": energy_cost},
            {"item": "Fixed Charge", "amount": fixed_monthly_charge},
            {"item": "Tax", "amount": tax_value},
            {"item": "Total", "amount": total_bill},
        ]
    )

    col_bill_1, col_bill_2 = st.columns([1.2, 1])
    with col_bill_1:
        fig_bill = px.bar(bill_breakdown, x="item", y="amount", title="Bill Components")
        st.plotly_chart(style_figure(fig_bill), use_container_width=True)

    with col_bill_2:
        profile_df = pd.DataFrame({"hour": list(range(24)), "estimated_kw": profile})
        fig_day = px.line(profile_df, x="hour", y="estimated_kw", markers=True, title="Estimated Daily Load")
        st.plotly_chart(style_figure(fig_day), use_container_width=True)

with viz_tab:
    st.subheader("Usage Visualization")

    chart_data = load_usage_chart_data()
    source_name = "generated/training usage"

    if uploaded_dataset is not None:
        try:
            chart_data = load_uploaded_chart_data(uploaded_dataset)
            source_name = "uploaded dataset"
        except Exception as exc:
            st.warning(f"Could not parse uploaded dataset, using fallback data. Details: {exc}")

    if chart_data.empty or TARGET_COLUMN not in chart_data.columns:
        st.warning("No visualization data available.")
    else:
        recent = chart_data[[TARGET_COLUMN]].tail(24 * 21).copy()

        plot_frame = recent.reset_index()
        x_col = plot_frame.columns[0]
        fig_usage = px.line(plot_frame, x=x_col, y=TARGET_COLUMN, title=f"Last 21 Days Usage ({source_name})")

        if not st.session_state["forecast_df"].empty:
            overlay = st.session_state["forecast_df"][["step", TARGET_COLUMN]].copy()
            overlay["time_marker"] = [f"T+{idx}" for idx in overlay["step"]]
            fig_usage.add_trace(
                go.Scatter(
                    x=overlay["time_marker"],
                    y=overlay[TARGET_COLUMN],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(width=3, dash="dot"),
                )
            )

        st.plotly_chart(style_figure(fig_usage), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(recent, x=TARGET_COLUMN, nbins=40, title="Usage Distribution")
            st.plotly_chart(style_figure(fig_hist), use_container_width=True)
        with c2:
            recent_copy = recent.copy()
            recent_copy["hour"] = recent_copy.index.hour
            fig_hour = px.box(recent_copy, x="hour", y=TARGET_COLUMN, title="Usage by Hour of Day")
            st.plotly_chart(style_figure(fig_hour), use_container_width=True)

with details_tab:
    st.subheader("Model and Feature Details")
    st.write(f"Best model by MSE: {model_bundle.get('best_model_name', 'Unknown')}")
    st.write(f"Selected model for prediction: {model_name}")
    st.write(f"Training rows: {model_bundle.get('training_rows', 'Unknown')}")

    score_map = model_bundle.get("model_scores_mse", {})
    score_df = pd.DataFrame(
        [{"model": m, "mse": s} for m, s in score_map.items()]
    ).sort_values("mse", ascending=True)
    st.dataframe(score_df, use_container_width=True)

    best_params = model_bundle.get("model_best_params", {})
    if best_params:
        st.write("Best tuned parameters:")
        st.json(best_params)

    st.write("Selected features:")
    st.write(model_bundle.get("selected_features", []))

    history = model_bundle.get("selection_history", [])
    if history:
        st.write("Forward selection history:")
        st.dataframe(pd.DataFrame(history), use_container_width=True)
